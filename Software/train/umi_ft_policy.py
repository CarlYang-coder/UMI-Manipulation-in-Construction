"""
UMI-FT Diffusion Policy.

Follows the UMI-FT paper architecture:
  1. CLIP ViT-B/32 encodes RGB images (2 timesteps) -> rgb_feat
  2. CLIP ViT-B/32 encodes Depth images (2 timesteps) -> depth_feat
  3. Transformer encoder fuses RGB + Depth tokens via self-attention -> visual_feat
  4. TemporalAggregator processes low-dim EE pose (2 timesteps) -> pose_feat
  5. global_cond = concat(visual_feat, pose_feat)
  6. ConditionalUnet1D denoises action trajectory conditioned on global_cond

Action space: 7D EE pose [tx, ty, tz, rx, ry, rz, gripper_width]
"""

from typing import Dict
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from timm_obs_encoder_umi import TimmObsEncoderUMI


class UMIFTDiffusionPolicy(BaseImagePolicy):
    """
    UMI-FT Diffusion Policy aligned with official DiffusionUnetTimmMod1Policy.

    Architecture:
      - TimmObsEncoderUMI: ViT CLIP backbone with modality-attention fusion
        (same as official TimmObsEncoderWithForce but without force encoder)
      - ConditionalUnet1D denoises sparse action trajectory
      - DDIMScheduler for inference
      - input_pertub=0.1 applied to loss noise
    """

    def __init__(
        self,
        # Dimensions
        action_dim: int = 7,
        ee_pose_dim: int = 7,
        # Horizons
        horizon: int = 20,
        n_obs_steps: int = 2,
        # Vision encoder (ViT CLIP)
        img_size: int = 224,
        vision_model_name: str = "vit_base_patch32_clip_224.openai",
        vision_pretrained: bool = True,
        vision_frozen: bool = False,
        share_rgb_model: bool = False,
        use_group_norm: bool = True,
        downsample_ratio: int = 32,
        feature_aggregation: str = "attention_pool_2d",
        position_encoding: str = "learnable",
        # Fusion
        fuse_mode: str = "modality-attention",
        rgb_keys=("rgb", "depth"),
        lowdim_keys=("ee_pose",),
        # Augmentation (applied inside encoder)
        aug_random_crop_ratio: float = 0.95,
        aug_color_jitter=(0.3, 0.4, 0.5, 0.08),  # brightness, contrast, saturation, hue
        # Diffusion UNet
        diffusion_step_embed_dim: int = 32,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # Noise scheduler
        noise_scheduler=None,
        num_inference_steps: int = None,
        # Misc
        predict_epsilon: bool = True,
        input_pertub: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # ─── Observation encoder (official-style) ───
        self.obs_encoder = TimmObsEncoderUMI(
            img_size=img_size,
            rgb_keys=list(rgb_keys),
            lowdim_keys=list(lowdim_keys),
            lowdim_dims={lowdim_keys[0]: ee_pose_dim},
            lowdim_horizons={lowdim_keys[0]: n_obs_steps},
            rgb_horizons={k: n_obs_steps for k in rgb_keys},
            model_name=vision_model_name,
            pretrained=vision_pretrained,
            frozen=vision_frozen,
            share_rgb_model=share_rgb_model,
            use_group_norm=use_group_norm,
            downsample_ratio=downsample_ratio,
            feature_aggregation=feature_aggregation,
            position_encoding=position_encoding,
            fuse_mode=fuse_mode,
            random_crop_ratio=aug_random_crop_ratio,
            color_jitter=aug_color_jitter,
        )
        obs_feature_dim = self.obs_encoder.output_dim()

        # ─── Diffusion UNet ───
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # ─── Noise scheduler (default DDIM to match official) ───
        if noise_scheduler is None:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=50,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        self.noise_scheduler = noise_scheduler

        # ─── Normalizer ───
        self.normalizer = LinearNormalizer()

        # ─── Store config ───
        self.action_dim = action_dim
        self.ee_pose_dim = ee_pose_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.predict_epsilon = predict_epsilon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ─── Inference ────────────────────────────────────────────────

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        """Standard DDIM/DDPM sampling (same as official)."""
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: {rgb: (B,To,3,H,W), depth: (B,To,3,H,W), ee_pose: (B,To,D)}
        Returns:
            action:      (B, horizon, action_dim) — full predicted horizon
            action_pred: (B, horizon, action_dim) — same as action
        (Aligned with official UMI-FT: no action-chunk slicing.)
        """
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        # Encode via obs_encoder (ViT CLIP + modality-attention fusion + lowdim concat)
        global_cond = self.obs_encoder(nobs)

        # Empty conditioning (no inpainting; all-zero mask, same as official)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Sample
        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            **self.kwargs,
        )

        # Unnormalize — return full horizon (official: no chunk slicing)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        return {
            'action': action_pred,
            'action_pred': action_pred,
        }

    # ─── Training ─────────────────────────────────────────────────

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """Compute diffusion training loss (aligned with official)."""
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # Encode observations
        global_cond = self.obs_encoder(nobs)

        trajectory = nactions

        # Sample noise + input perturbation (official: alleviate exposure bias)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device)

        # Sample random diffusion timesteps
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device,
        ).long()

        # Forward diffusion (use perturbed noise for adding, clean noise as target)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        # Predict
        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=None, global_cond=global_cond,
        )

        # Target
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # MSE loss (official: reduction='mean', no mask)
        loss = F.mse_loss(pred, target, reduction='mean')
        return loss
