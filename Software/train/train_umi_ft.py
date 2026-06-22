"""
UMI-FT Diffusion Policy Training Script.

PREREQUISITE: Run `python generate_ee_pose_csv.py` once before training, so each
episode in Data_Raw/data_crop/ has an ee_pose_trajectory.csv produced offline.

Follows the official Diffusion Policy training pattern:
  - Workspace-style class with save/load checkpoint, EMA, LR scheduler
  - TopKCheckpointManager for best-k checkpoints
  - Gradient accumulation
  - Validation via predict_action MSE
  - Resume from checkpoint

Architecture (from UMI-FT paper):
  - CLIP ViT-B/32 for RGB and Depth (224x224)
  - Transformer encoder self-attention fusion
  - TemporalAggregator for EE pose
  - ConditionalUnet1D diffusion policy
  - Action: 7D EE pose [tx, ty, tz, rx, ry, rz, gripper_width]
  - No F/T sensor (removed per user request)

Usage:
    cd d:/UMI_Gripper/train
    python train_umi_ft.py
"""

import os
import sys
import copy
import time
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ---- diffusion_policy library imports ----
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

# ---- local imports ----
from umi_dataset import UMIDataset
from umi_ft_policy import UMIFTDiffusionPolicy


# ============================================================
# CONFIG  (mirrors official Hydra YAML style)
# ============================================================
cfg = dict(
    # ---- data ----
    data_dir        = r"D:\UMI_Gripper\Data_Raw\data_crop",
    calibration_path = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json",
    gripper_min     = 0.015,
    gripper_max     = 0.085,

    # ---- model / horizons (official: shape_meta + task) ----
    img_size        = 224,        # CLIP ViT-B/32 expects 224x224
    n_obs_steps     = 2,
    horizon         = 20,         # official sparse_action_horizon: 20 (3.33s)
    action_dim      = 7,          # [tx, ty, tz, rx, ry, rz, gripper]
    depth_clip_m    = 0.5,

    # ---- down_sample_steps (official task yaml) ----
    rgb_down_sample_steps    = 10,
    lowdim_down_sample_steps = 5,
    action_down_sample_steps = 10,

    # ---- diffusion (official DDIMScheduler) ----
    num_train_timesteps = 50,
    num_infer_steps     = 16,     # official: 16
    beta_start          = 0.0001,
    beta_end            = 0.02,
    beta_schedule       = "squaredcos_cap_v2",
    prediction_type     = "epsilon",
    predict_epsilon     = True,
    clip_sample         = True,
    set_alpha_to_one    = True,
    steps_offset        = 0,
    input_pertub        = 0.1,    # official: 0.1

    # ---- Vision encoder (official: timm ViT CLIP, frozen=False, use_group_norm=True) ----
    vision_model_name   = "vit_base_patch32_clip_224.openai",
    vision_pretrained   = True,
    vision_frozen       = False,
    share_rgb_model     = False,
    use_group_norm      = True,
    downsample_ratio    = 32,
    feature_aggregation = "attention_pool_2d",  # ignored for ViT (uses CLS token)
    position_encoding   = "learnable",
    fuse_mode           = "modality-attention",

    # ---- Augmentation applied inside encoder (official) ----
    aug_random_crop_ratio = 0.95,
    aug_color_jitter      = (0.3, 0.4, 0.5, 0.08),  # b, c, s, h

    # ---- UNet (official) ----
    down_dims           = (256, 512, 1024),
    diffusion_step_embed_dim = 32,   # official: 32
    kernel_size         = 5,
    n_groups            = 8,
    cond_predict_scale  = True,

    # ---- training (official) ----
    seed              = 42,
    num_epochs        = 500,         # official: 300; resuming from epoch 219
    # Physical batch fits in 8GB GPU; gradient_accumulate keeps effective batch = 128
    batch_size        = 32,
    num_workers       = 2,
    lr                = 3.0e-4,      # official: 3e-4 (effective batch unchanged)
    weight_decay      = 1.0e-6,
    adam_betas        = (0.95, 0.999),
    adam_eps          = 1.0e-8,
    gradient_accumulate_every = 4,   # 32 * 4 = 128 effective (matches official)
    val_ratio         = 0.05,       # official: 0.05
    reduce_pretrained_lr = True,   # official: vision backbone uses 0.1 * lr
    resume            = True,      # resume from latest.ckpt (epoch 219)

    # ---- dense/sparse gradient clipping schedule (official) ----
    dense_training_start_epochs = 60,     # after this epoch, dense clipping kicks in
    dense_gradient_norm_limit   = 0.2,
    sparse_gradient_norm_limit  = -1,     # -1 = disabled
    gt_traj_num_epochs          = 80,

    # ---- lr scheduler (official) ----
    lr_scheduler      = "cosine",
    lr_warmup_steps   = 2000,

    # ---- EMA (official) ----
    use_ema           = True,
    ema_update_after_step = 0,
    ema_inv_gamma     = 1.0,
    ema_power         = 0.75,
    ema_min_value     = 0.0,
    ema_max_value     = 0.9999,

    # ---- logging ----
    use_wandb         = False,
    wandb_project     = "hierarchical-policy",
    wandb_name        = None,

    # ---- checkpointing (official) ----
    output_dir        = r"D:\UMI_Gripper\train\checkpoints",
    save_last_ckpt    = True,
    topk_k            = 20,              # official: 20
    topk_monitor_key  = "train_loss",    # official: monitors train_loss
    topk_mode         = "min",
    checkpoint_every  = 10,

    # ---- validation (official) ----
    val_every         = 5,               # epochs (unused; kept for compat)
    sample_every      = 5,               # epochs — naction MSE eval cadence

    # ---- misc ----
    tqdm_interval_sec = 1.0,
    device            = "cuda:0",
)


# ============================================================
# COLLATE
# ============================================================
def _collate(batch_list):
    rgb     = torch.stack([b["obs"]["rgb"]     for b in batch_list], dim=0)
    depth   = torch.stack([b["obs"]["depth"]   for b in batch_list], dim=0)
    ee_pose = torch.stack([b["obs"]["ee_pose"] for b in batch_list], dim=0)
    action  = torch.stack([b["action"]         for b in batch_list], dim=0)
    return {
        "obs": {"rgb": rgb, "depth": depth, "ee_pose": ee_pose},
        "action": action,
    }


# ============================================================
# WORKSPACE
# ============================================================
class TrainUMIFTWorkspace:
    """
    Official-style workspace for UMI-FT Diffusion Policy training.
    """

    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Seed
        seed = cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Build policy
        self.model = self._build_policy()

        # EMA
        self.ema_model = None
        if cfg["use_ema"]:
            self.ema_model = copy.deepcopy(self.model)

        # Optimizer (official: grouped LR — vision backbone uses 0.1 * lr)
        obs_encoder_lr = cfg["lr"] * (0.1 if cfg.get("reduce_pretrained_lr", True) else 1.0)
        pretrained_obs_encoder_params = []
        if hasattr(self.model, "obs_encoder") and hasattr(self.model.obs_encoder, "rgb_keys"):
            for key in self.model.obs_encoder.rgb_keys:
                for p in self.model.obs_encoder.key_model_map[key].parameters():
                    if p.requires_grad:
                        pretrained_obs_encoder_params.append(p)
        pretrained_ids = {id(p) for p in pretrained_obs_encoder_params}
        other_params = [p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in pretrained_ids]

        param_groups = [
            {"params": other_params, "lr": cfg["lr"]},
            {"params": pretrained_obs_encoder_params, "lr": obs_encoder_lr},
        ]

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Parameters: {n_trainable:,} trainable / {n_total:,} total")
        print(f"       obs_encoder backbone params: {len(pretrained_obs_encoder_params)} "
              f"@ lr={obs_encoder_lr:.2e}")
        print(f"       other params:                {len(other_params)} @ lr={cfg['lr']:.2e}")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=cfg["adam_betas"],
            eps=cfg["adam_eps"],
            weight_decay=cfg["weight_decay"],
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Resume behavior (official: skip optimizer when not resuming)
        self.exclude_optimizer = not cfg.get("resume", False)

    def _build_policy(self):
        c = self.cfg
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=c["num_train_timesteps"],
            beta_start=c["beta_start"],
            beta_end=c["beta_end"],
            beta_schedule=c["beta_schedule"],
            clip_sample=c["clip_sample"],
            set_alpha_to_one=c["set_alpha_to_one"],
            steps_offset=c["steps_offset"],
            prediction_type=c["prediction_type"],
        )
        policy = UMIFTDiffusionPolicy(
            action_dim=c["action_dim"],
            ee_pose_dim=c["action_dim"],
            horizon=c["horizon"],
            n_obs_steps=c["n_obs_steps"],
            img_size=c["img_size"],
            vision_model_name=c["vision_model_name"],
            vision_pretrained=c["vision_pretrained"],
            vision_frozen=c["vision_frozen"],
            share_rgb_model=c["share_rgb_model"],
            use_group_norm=c["use_group_norm"],
            downsample_ratio=c["downsample_ratio"],
            feature_aggregation=c["feature_aggregation"],
            position_encoding=c["position_encoding"],
            fuse_mode=c["fuse_mode"],
            aug_random_crop_ratio=c["aug_random_crop_ratio"],
            aug_color_jitter=c["aug_color_jitter"],
            diffusion_step_embed_dim=c["diffusion_step_embed_dim"],
            down_dims=c["down_dims"],
            kernel_size=c["kernel_size"],
            n_groups=c["n_groups"],
            cond_predict_scale=c["cond_predict_scale"],
            noise_scheduler=noise_scheduler,
            num_inference_steps=c["num_infer_steps"],
            predict_epsilon=c["predict_epsilon"],
            input_pertub=c["input_pertub"],
        )
        return policy

    # ─── Checkpoint helpers ──────────────────────────────────────

    def _state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "cfg": self.cfg,
        }
        if not self.exclude_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        if self.ema_model is not None:
            state["ema_model"] = self.ema_model.state_dict()
        return state

    def save_checkpoint(self, path=None, extra=None):
        if path is None:
            path = str(self.output_dir / "latest.ckpt")
        state = self._state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        if self.ema_model is not None and "ema_model" in state:
            self.ema_model.load_state_dict(state["ema_model"])
        print(f"[RESUME] epoch={self.epoch}, global_step={self.global_step}")

    def _save_inference_ckpt(self, path, epoch, train_loss, val_loss):
        """Save lightweight inference-only checkpoint."""
        c = self.cfg
        policy = self.ema_model if self.ema_model is not None else self.model
        state = {
            "policy": policy.state_dict(),
            "config": {
                "img_size": c["img_size"],
                "n_obs_steps": c["n_obs_steps"],
                "horizon": c["horizon"],
                "action_dim": c["action_dim"],
                "depth_clip_m": c["depth_clip_m"],
                "num_train_timesteps": c["num_train_timesteps"],
                "num_infer_steps": c["num_infer_steps"],
                "vision_model_name": c["vision_model_name"],
                "vision_pretrained": False,  # already loaded from ckpt
                "vision_frozen": c["vision_frozen"],
                "share_rgb_model": c["share_rgb_model"],
                "use_group_norm": c["use_group_norm"],
                "downsample_ratio": c["downsample_ratio"],
                "feature_aggregation": c["feature_aggregation"],
                "position_encoding": c["position_encoding"],
                "fuse_mode": c["fuse_mode"],
                "aug_random_crop_ratio": c["aug_random_crop_ratio"],
                "aug_color_jitter": c["aug_color_jitter"],
                "diffusion_step_embed_dim": c["diffusion_step_embed_dim"],
                "cond_predict_scale": c["cond_predict_scale"],
                "input_pertub": c["input_pertub"],
                "gripper_min": c["gripper_min"],
                "gripper_max": c["gripper_max"],
                "rgb_down_sample_steps": c["rgb_down_sample_steps"],
                "lowdim_down_sample_steps": c["lowdim_down_sample_steps"],
                "action_down_sample_steps": c["action_down_sample_steps"],
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, str(path))

    # ─── RUN ─────────────────────────────────────────────────────

    def run(self):
        c = self.cfg

        # Resume (only if cfg.resume=True, matching official)
        if c.get("resume", False):
            resume_path = self.output_dir / "latest.ckpt"
            if resume_path.is_file():
                self.load_checkpoint(str(resume_path))

        # Dataset — episode-based split (official UMI-FT)
        print("[INFO] Loading dataset ...")
        train_ds = UMIDataset(
            data_dir=c["data_dir"],
            calibration_path=c["calibration_path"],
            n_obs_steps=c["n_obs_steps"],
            horizon=c["horizon"],
            img_size=c["img_size"],
            depth_clip_m=c["depth_clip_m"],
            stride=8,  # official: sparse_query_frequency_down_sample_steps
            gripper_min=c["gripper_min"],
            gripper_max=c["gripper_max"],
            rgb_down_sample_steps=c["rgb_down_sample_steps"],
            lowdim_down_sample_steps=c["lowdim_down_sample_steps"],
            action_down_sample_steps=c["action_down_sample_steps"],
            val_ratio=c["val_ratio"],
            split_seed=c["seed"],
            is_validation=False,
        )
        val_ds = train_ds.get_validation_dataset()

        train_loader = DataLoader(
            train_ds, batch_size=c["batch_size"], shuffle=True,
            num_workers=c["num_workers"], pin_memory=True,
            collate_fn=_collate, persistent_workers=c["num_workers"] > 0)
        val_loader = DataLoader(
            val_ds, batch_size=c["batch_size"], shuffle=False,
            num_workers=c["num_workers"], pin_memory=True,
            collate_fn=_collate, persistent_workers=c["num_workers"] > 0)

        # Normalizer (fit on training split; official saves it to pickle)
        import pickle
        print("[INFO] Fitting normalizer ...")
        normalizer = train_ds.get_normalizer(mode="limits")
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        # Save normalizer to disk (official UMI-FT)
        normalizer_path = self.output_dir / "sparse_normalizer.pkl"
        with open(normalizer_path, "wb") as f:
            pickle.dump(normalizer, f)
        print(f"[INFO] Saved normalizer to {normalizer_path}")

        # LR scheduler
        # NOTE: last_epoch must be in OPTIMIZER STEP units, not forward batch.
        # global_step counts forward batches; divide by accumulate to get opt steps.
        opt_step_so_far = self.global_step // c["gradient_accumulate_every"]

        # When resuming with last_epoch >= 0, PyTorch requires `initial_lr`
        # in each param_group (normally set automatically when last_epoch=-1).
        # We add it manually since our ckpt doesn't save optimizer state.
        if opt_step_so_far > 0:
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

        lr_scheduler = get_scheduler(
            c["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=c["lr_warmup_steps"],
            num_training_steps=(
                len(train_loader) * c["num_epochs"]
            ) // c["gradient_accumulate_every"],
            last_epoch=opt_step_so_far - 1,
        )

        # EMA
        ema = None
        if c["use_ema"]:
            ema = EMAModel(
                model=self.ema_model,
                update_after_step=c["ema_update_after_step"],
                inv_gamma=c["ema_inv_gamma"],
                power=c["ema_power"],
                min_value=c["ema_min_value"],
                max_value=c["ema_max_value"],
            )

        # TopK checkpoints (official: monitor train_loss, k=20)
        topk_manager = TopKCheckpointManager(
            save_dir=str(self.ckpt_dir),
            monitor_key=c["topk_monitor_key"],
            mode=c["topk_mode"],
            k=c["topk_k"],
            format_str="epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt",
        )

        # WandB
        wandb_run = None
        if c["use_wandb"]:
            import wandb
            wandb_run = wandb.init(
                project=c["wandb_project"],
                name=c["wandb_name"],
                dir=str(self.output_dir),
                config=c,
            )

        # Device
        # Require GPU (official UMI-FT)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for training.")
        device = torch.device(c["device"])
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Training loop
        print(f"[INFO] Training: {c['num_epochs']} epochs, "
              f"{len(train_loader)} batches/epoch, device={device}")

        best_val = float("inf")
        train_sampling_batch = None
        batch_size = c["batch_size"]

        for _ in range(self.epoch, c["num_epochs"]):
            epoch_t0 = time.time()
            epoch_train_loss = 0.0
            n_batches = 0

            with tqdm.tqdm(
                train_loader,
                desc=f"Epoch {self.epoch}",
                leave=False,
                mininterval=c["tqdm_interval_sec"],
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch,
                        lambda x: x.to(device, non_blocking=True))

                    # Record latest full-size batch for sample-based eval
                    # (official: skip last partial batch)
                    if batch_idx == 0 or batch["action"].shape[0] == batch_size:
                        train_sampling_batch = batch

                    # Forward
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / c["gradient_accumulate_every"]
                    loss.backward()

                    # Optimizer step (official: gradient clipping is DISABLED;
                    # the yaml keys exist but the official code has them
                    # commented out.)
                    if self.global_step % c["gradient_accumulate_every"] == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # EMA
                    if ema is not None:
                        ema.step(self.model)

                    # Logging
                    raw_loss_cpu = raw_loss.item()
                    epoch_train_loss += raw_loss_cpu
                    n_batches += 1

                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    if wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)

                    tepoch.set_postfix(loss=f"{raw_loss_cpu:.4f}")
                    self.global_step += 1

            # End of epoch
            epoch_train_loss /= max(1, n_batches)

            # Sample-based naction MSE eval (official: every sample_every epochs)
            train_naction_mse = float("nan")
            val_naction_mse = float("nan")
            if (self.epoch + 1) % c["sample_every"] == 0:
                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()
                with torch.no_grad():
                    # Train sample
                    if train_sampling_batch is not None:
                        b = dict_apply(train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True))
                        gt = b["action"]
                        pred = policy.predict_action(b["obs"])["action_pred"]
                        n_gt = normalizer["action"].normalize(gt)
                        n_pred = normalizer["action"].normalize(pred)
                        train_naction_mse = F.mse_loss(n_pred, n_gt).item()

                    # Val sample
                    try:
                        val_sampling_batch = next(iter(val_loader))
                        b = dict_apply(val_sampling_batch,
                            lambda x: x.to(device, non_blocking=True))
                        gt = b["action"]
                        pred = policy.predict_action(b["obs"])["action_pred"]
                        n_gt = normalizer["action"].normalize(gt)
                        n_pred = normalizer["action"].normalize(pred)
                        val_naction_mse = F.mse_loss(n_pred, n_gt).item()
                    except StopIteration:
                        pass
                policy.train()

            dt = time.time() - epoch_t0
            print(f"[E{self.epoch:04d}] train={epoch_train_loss:.6f}  "
                  f"train_naction_mse={train_naction_mse:.6f}  "
                  f"val_naction_mse={val_naction_mse:.6f}  "
                  f"lr={lr_scheduler.get_last_lr()[0]:.2e}  time={dt:.1f}s")

            if wandb_run is not None:
                log_d = {
                    "epoch_train_loss": epoch_train_loss,
                    "epoch": self.epoch,
                }
                if not np.isnan(train_naction_mse):
                    log_d["train_sparse_naction_mse_error"] = train_naction_mse
                if not np.isnan(val_naction_mse):
                    log_d["val_sparse_naction_mse_error"] = val_naction_mse
                wandb_run.log(log_d, step=self.global_step)

            # Save latest + inference ckpt every checkpoint_every epochs (official: 10)
            if (self.epoch + 1) % c["checkpoint_every"] == 0:
                if c["save_last_ckpt"]:
                    self.save_checkpoint()
                self._save_inference_ckpt(
                    self.output_dir / "latest.pt",
                    epoch=self.epoch, train_loss=epoch_train_loss,
                    val_loss=val_naction_mse)

                # TopK: official monitors train_loss
                topk_path = topk_manager.get_ckpt_path({
                    "epoch": self.epoch,
                    "train_loss": epoch_train_loss,
                })
                if topk_path is not None:
                    self.save_checkpoint(path=topk_path)

            # Best val checkpoint (by naction MSE)
            if not np.isnan(val_naction_mse) and val_naction_mse < best_val:
                best_val = val_naction_mse
                self._save_inference_ckpt(
                    self.output_dir / "best.pt",
                    epoch=self.epoch, train_loss=epoch_train_loss,
                    val_loss=val_naction_mse)
                print(f"  [BEST] val_naction_mse={val_naction_mse:.6f}")

            self.epoch += 1

        print(f"\n[DONE] Best val_naction_mse={best_val:.6f}")
        print(f"       Output: {self.output_dir}")


# ============================================================
# MAIN
# ============================================================
def main():
    workspace = TrainUMIFTWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
