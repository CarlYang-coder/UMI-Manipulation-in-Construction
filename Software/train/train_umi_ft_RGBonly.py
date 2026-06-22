"""
RGB-only baseline for UMI-FT, mirroring Image_DP train_DP_OfficialVideo.py exactly.

No depth. No transformer fusion. Just ResNet18VideoEncoder(RGB) + ee_pose(lowdim)
-> DiffusionUnetVideoPolicy (same policy class Image_DP uses).

PREREQUISITE: Run `python generate_ee_pose_csv.py` once before training, so each
episode in Data_Raw/data_crop_ds/ has an ee_pose_trajectory.csv.

All hyperparameters, optimizer, scheduler, EMA, loss, and checkpointing match
Image_DP's train_DP_OfficialVideo.py. Only the data source differs:
  - Image_DP: GelloVideoDatasetAbs (16D joint positions, RGB)
  - This:    UMIDataset          ( 7D ee_pose,         RGB + Depth(ignored))

Usage:
    python train_umi_ft_RGBonly.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import copy
import time
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import tqdm

# Add Image_DP video_train to path so we can import the exact same modules Image_DP uses
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"D:\Image_DP\video_train")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ---- diffusion_policy library (local UMI copy) ----
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

# ---- Image_DP modules (reused as-is) ----
from resnet_video_encoder import ResNet18VideoEncoder
from diffusion_unet_video_policy import DiffusionUnetVideoPolicy as _Policy

# ---- UMI dataset ----
from umi_dataset import UMIDataset


# ============================================================
# CONFIG  (mirrors Image_DP train_DP_OfficialVideo.py cfg; data/action dims adapted)
# ============================================================
cfg = dict(
    # ---- data (UMI-specific) ----
    data_dir         = r"D:\UMI_Gripper\Data_Raw\data_crop_ds",
    calibration_path = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json",
    gripper_min      = 0.015,
    gripper_max      = 0.085,

    # ---- model / horizons ----
    img_size        = 128,          # match Image_DP (ResNet18 is happy with 128)
    n_obs_steps     = 2,
    n_action_steps  = 8,
    horizon         = 16,
    action_dim      = 7,            # UMI uses 7D EE pose (differs from Image_DP's 16)

    # ---- diffusion ----
    num_train_timesteps = 50,
    num_infer_steps     = 50,
    beta_schedule       = "squaredcos_cap_v2",
    prediction_type     = "epsilon",
    predict_epsilon     = True,

    # ---- encoder ----
    rgb_out_dim       = 512,
    rgb_pool          = "mean",
    rgb_mlp_hidden    = 512,
    rgb_dropout       = 0.0,
    rgb_pretrained    = True,
    rgb_freeze_backbone = True,

    # ---- training ----
    seed              = 42,
    num_epochs        = 8000,       # same as Image_DP
    batch_size        = 64,
    num_workers       = 2,
    lr                = 1e-4,
    weight_decay      = 1e-4,
    max_grad_norm     = 1.0,
    gradient_accumulate_every = 1,
    val_ratio         = 0.1,

    # ---- lr scheduler ----
    lr_scheduler      = "cosine",
    lr_warmup_steps   = 500,

    # ---- EMA ----
    use_ema           = True,
    ema_inv_gamma     = 1.0,
    ema_power         = 0.75,
    ema_max_value     = 0.9999,

    # ---- logging ----
    use_wandb         = False,
    wandb_project     = "umi_ft_rgbonly",
    wandb_name        = None,

    # ---- checkpointing ----
    output_dir        = r"D:\UMI_Gripper\train\checkpoints_rgbonly",
    save_last_ckpt    = True,
    topk_k            = 5,
    topk_monitor_key  = "val_loss",
    topk_mode         = "min",

    # ---- validation ----
    val_every         = 50,

    # ---- tqdm ----
    tqdm_interval_sec = 5.0,

    # ---- device ----
    device            = "cuda",
)


# ============================================================
# IMAGENET TRANSFORMS (ResNet18 expects ImageNet stats, not CLIP)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def imagenet_rgb_transform_train(img_size: int):
    return T.Compose([
        T.Resize(int(img_size * 1.1), antialias=True),
        T.RandomCrop(img_size),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def imagenet_rgb_transform_eval(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# COLLATE (drop depth; pass only rgb + ee_pose)
# ============================================================
def _collate(batch_list):
    rgb     = torch.stack([b["obs"]["rgb"]     for b in batch_list], dim=0)
    ee_pose = torch.stack([b["obs"]["ee_pose"] for b in batch_list], dim=0)
    action  = torch.stack([b["action"]         for b in batch_list], dim=0)
    return {
        "obs": {"rgb": rgb, "ee_pose": ee_pose},
        "action": action,
    }


# ============================================================
# WORKSPACE (structure identical to Image_DP TrainDiffusionUnetVideoWorkspace)
# ============================================================
class TrainUMIFTRGBonlyWorkspace:
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        seed = cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = self._build_policy()

        self.ema_model = None
        if cfg["use_ema"]:
            self.ema_model = copy.deepcopy(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Parameters: {n_trainable:,} trainable / {n_total:,} total")

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        self.global_step = 0
        self.epoch = 0

    def _build_policy(self):
        c = self.cfg
        rgb_net = ResNet18VideoEncoder(
            out_dim=c["rgb_out_dim"],
            pool=c["rgb_pool"],
            mlp_hidden=c["rgb_mlp_hidden"],
            dropout=c["rgb_dropout"],
            pretrained=c["rgb_pretrained"],
            freeze_backbone=c["rgb_freeze_backbone"],
        )
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=c["num_train_timesteps"],
            beta_schedule=c["beta_schedule"],
            clip_sample=False,
            prediction_type=c["prediction_type"],
        )
        shape_meta = {
            "action": {"shape": [c["action_dim"]]},
            "obs": {
                "rgb":     {"shape": [3, c["img_size"], c["img_size"]], "type": "rgb"},
                "ee_pose": {"shape": [c["action_dim"]],                  "type": "lowdim"},
            }
        }
        policy = _Policy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            rgb_net=rgb_net,
            horizon=c["horizon"],
            n_action_steps=c["n_action_steps"],
            n_obs_steps=c["n_obs_steps"],
            num_inference_steps=c["num_infer_steps"],
            lowdim_as_global_cond=True,
            predict_epsilon=c["predict_epsilon"],
        )
        return policy

    # ---- checkpoint helpers ----
    def _state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "cfg": self.cfg,
        }
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
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        if self.ema_model is not None and "ema_model" in state:
            self.ema_model.load_state_dict(state["ema_model"])
        print(f"[RESUME] epoch={self.epoch}, global_step={self.global_step}")

    def _save_inference_ckpt(self, path, epoch, train_loss, val_loss):
        c = self.cfg
        policy = self.ema_model if self.ema_model is not None else self.model
        state = {
            "policy": policy.state_dict(),
            "config": {
                "img_size": c["img_size"],
                "n_obs_steps": c["n_obs_steps"],
                "n_action_steps": c["n_action_steps"],
                "horizon": c["horizon"],
                "action_dim": c["action_dim"],
                "num_train_timesteps": c["num_train_timesteps"],
                "num_infer_steps": c["num_infer_steps"],
                "rgb_out_dim": c["rgb_out_dim"],
                "rgb_pool": c["rgb_pool"],
                "rgb_mlp_hidden": c["rgb_mlp_hidden"],
                "rgb_dropout": c["rgb_dropout"],
                "rgb_freeze_backbone": c["rgb_freeze_backbone"],
                "gripper_min": c["gripper_min"],
                "gripper_max": c["gripper_max"],
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, str(path))

    # ---- run ----
    def run(self):
        c = self.cfg

        # resume
        resume_path = self.output_dir / "latest.ckpt"
        if resume_path.is_file():
            self.load_checkpoint(str(resume_path))

        # dataset
        print("[INFO] Loading dataset ...")
        ds = UMIDataset(
            data_dir=c["data_dir"],
            calibration_path=c["calibration_path"],
            n_obs_steps=c["n_obs_steps"],
            horizon=c["horizon"],
            img_size=c["img_size"],
            stride=1,
            gripper_min=c["gripper_min"],
            gripper_max=c["gripper_max"],
            rgb_transform=imagenet_rgb_transform_train(c["img_size"]),
            augment=True,  # augment flag retained for depth path (which we ignore)
        )

        n_val = max(1, int(len(ds) * c["val_ratio"]))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(c["seed"]))

        train_loader = DataLoader(
            train_ds, batch_size=c["batch_size"], shuffle=True,
            num_workers=c["num_workers"], pin_memory=True,
            collate_fn=_collate, persistent_workers=c["num_workers"] > 0)
        val_loader = DataLoader(
            val_ds, batch_size=c["batch_size"], shuffle=False,
            num_workers=c["num_workers"], pin_memory=True,
            collate_fn=_collate, persistent_workers=c["num_workers"] > 0)

        # normalizer: use UMI dataset's, but drop depth (policy doesn't need it)
        print("[INFO] Fitting normalizer ...")
        normalizer = ds.get_normalizer(mode="limits")
        # rgb already identity; ee_pose fitted; action fitted; depth is ignored by policy
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = get_scheduler(
            c["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=c["lr_warmup_steps"],
            num_training_steps=(
                len(train_loader) * c["num_epochs"]
            ) // c["gradient_accumulate_every"],
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema = None
        if c["use_ema"]:
            ema = EMAModel(
                model=self.ema_model,
                inv_gamma=c["ema_inv_gamma"],
                power=c["ema_power"],
                max_value=c["ema_max_value"],
            )

        topk_manager = TopKCheckpointManager(
            save_dir=str(self.ckpt_dir),
            monitor_key=c["topk_monitor_key"],
            mode=c["topk_mode"],
            k=c["topk_k"],
            format_str="epoch={epoch:03d}-val_loss={val_loss:.6f}.ckpt",
        )

        wandb_run = None
        if c["use_wandb"]:
            import wandb
            wandb_run = wandb.init(
                project=c["wandb_project"],
                name=c["wandb_name"],
                dir=str(self.output_dir),
                config=c,
            )

        device = torch.device(c["device"] if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        val_batch = next(iter(train_loader))

        print(f"[INFO] Training: {c['num_epochs']} epochs, "
              f"{len(train_loader)} batches/epoch, device={device}")

        best_val = float("inf")

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
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / c["gradient_accumulate_every"]
                    loss.backward()

                    if self.global_step % c["gradient_accumulate_every"] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            c["max_grad_norm"])
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    if ema is not None:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    epoch_train_loss += raw_loss_cpu
                    n_batches += 1

                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    if self.global_step > 0 and self.global_step % c["val_every"] == 0:
                        policy = self.ema_model if self.ema_model is not None else self.model
                        policy.eval()
                        with torch.no_grad():
                            vb = dict_apply(val_batch,
                                lambda x: x.to(device, non_blocking=True))
                            result = policy.predict_action(vb["obs"])
                            pred_action = result["action_pred"]
                            gt_action = vb["action"]
                            mse = F.mse_loss(pred_action, gt_action)
                            step_log["val_action_mse"] = mse.item()
                        policy.train()

                    if wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)

                    self.global_step += 1

            epoch_train_loss /= max(1, n_batches)

            # full validation loss
            val_loss = 0.0
            n_val_batches = 0
            policy = self.ema_model if self.ema_model is not None else self.model
            policy.eval()
            with torch.no_grad():
                for vb in val_loader:
                    vb = dict_apply(vb, lambda x: x.to(device, non_blocking=True))
                    vl = policy.compute_loss(vb)
                    val_loss += vl.item()
                    n_val_batches += 1
            val_loss /= max(1, n_val_batches)
            policy.train()

            dt = time.time() - epoch_t0
            print(f"[E{self.epoch:03d}] train={epoch_train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={lr_scheduler.get_last_lr()[0]:.2e}  "
                  f"time={dt:.1f}s")

            if wandb_run is not None:
                wandb_run.log({
                    "epoch_train_loss": epoch_train_loss,
                    "epoch_val_loss": val_loss,
                    "epoch": self.epoch,
                }, step=self.global_step)

            if c["save_last_ckpt"]:
                self.save_checkpoint()

            self._save_inference_ckpt(
                self.output_dir / "latest.pt",
                epoch=self.epoch, train_loss=epoch_train_loss, val_loss=val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self._save_inference_ckpt(
                    self.output_dir / "best.pt",
                    epoch=self.epoch, train_loss=epoch_train_loss, val_loss=val_loss)
                print(f"  [BEST] val_loss={val_loss:.6f}")

            topk_path = topk_manager.get_ckpt_path({
                "epoch": self.epoch,
                "val_loss": val_loss,
            })
            if topk_path is not None:
                self.save_checkpoint(path=topk_path)

            self.epoch += 1

        print(f"\n[DONE] Best val_loss={best_val:.6f}  Output: {self.output_dir}")


def main():
    workspace = TrainUMIFTRGBonlyWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
