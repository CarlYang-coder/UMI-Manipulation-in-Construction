"""Print UMI-FT policy parameter counts without starting training."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from umi_ft_policy import UMIFTDiffusionPolicy

noise_scheduler = DDPMScheduler(
    num_train_timesteps=50,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=False,
    prediction_type="epsilon",
)

policy = UMIFTDiffusionPolicy(
    action_dim=7,
    ee_pose_dim=7,
    horizon=16,
    n_action_steps=8,
    n_obs_steps=2,
    clip_out_dim=512,
    clip_pool="mean",
    clip_mlp_hidden=512,
    clip_dropout=0.0,
    clip_freeze_backbone=True,
    fusion_d_model=512,
    fusion_nhead=8,
    fusion_num_layers=1,
    fusion_dim_feedforward=2048,
    fusion_dropout=0.1,
    diffusion_step_embed_dim=256,
    down_dims=(256, 512, 1024),
    kernel_size=5,
    n_groups=8,
    noise_scheduler=noise_scheduler,
    num_inference_steps=50,
    predict_epsilon=True,
)

# Total counts
n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in policy.parameters())
print(f"\n{'='*60}")
print(f"TOTAL: {n_trainable:>15,} trainable / {n_total:>15,} total")
print(f"{'='*60}\n")

# Per-submodule breakdown
print(f"{'Module':<40} {'Trainable':>15} {'Total':>15}")
print("-" * 75)
for name, module in policy.named_children():
    mod_total = sum(p.numel() for p in module.parameters())
    mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name:<40} {mod_trainable:>15,} {mod_total:>15,}")

# CLIP backbone vs MLP within encoders
print(f"\n{'CLIP RGB encoder breakdown':-^75}")
for name, module in policy.rgb_encoder.named_children():
    mod_total = sum(p.numel() for p in module.parameters())
    mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name:<38} {mod_trainable:>15,} {mod_total:>15,}")
