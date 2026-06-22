"""Smoke test: make sure run_umi_ft_RGBonly.py can load the RGBonly checkpoint
format and do a forward pass. Creates a tiny fake checkpoint using the training
workspace (no actual training).
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, r"D:\Image_DP\video_train")

# Build a fresh training workspace and save an inference checkpoint
from train_umi_ft_RGBonly import TrainUMIFTRGBonlyWorkspace, cfg

ws = TrainUMIFTRGBonlyWorkspace(cfg)

# We need the normalizer fitted before saving; use the UMI dataset
from umi_dataset import UMIDataset
from train_umi_ft_RGBonly import imagenet_rgb_transform_train

ds = UMIDataset(
    data_dir=cfg["data_dir"],
    calibration_path=cfg["calibration_path"],
    n_obs_steps=cfg["n_obs_steps"],
    horizon=cfg["horizon"],
    img_size=cfg["img_size"],
    gripper_min=cfg["gripper_min"],
    gripper_max=cfg["gripper_max"],
    rgb_transform=imagenet_rgb_transform_train(cfg["img_size"]),
    augment=True,
)
normalizer = ds.get_normalizer(mode="limits")
ws.model.set_normalizer(normalizer)
if ws.ema_model is not None:
    ws.ema_model.set_normalizer(normalizer)

# Save fake inference checkpoint
tmp_ckpt = Path(r"D:\UMI_Gripper\train\checkpoints_rgbonly\smoketest.pt")
tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
ws._save_inference_ckpt(str(tmp_ckpt), epoch=0, train_loss=0.0, val_loss=0.0)
print(f"[OK] Saved smoke-test ckpt to {tmp_ckpt}")

# Load it via run_umi_ft_RGBonly's loader
from run_umi_ft_RGBonly import load_policy_rgbonly, imagenet_rgb_transform_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy, config = load_policy_rgbonly(str(tmp_ckpt), device)
print(f"[OK] Loaded with device={device}")

# Build a fake obs matching what the rollout would send
img_size = config["img_size"]
n_obs_steps = config["n_obs_steps"]
obs = {
    "rgb": torch.randn(1, n_obs_steps, 3, img_size, img_size, device=device),
    "ee_pose": torch.randn(1, n_obs_steps, config["action_dim"], device=device),
}
with torch.no_grad():
    result = policy.predict_action(obs)
print(f"[OK] predict_action: action={tuple(result['action'].shape)}, "
      f"action_pred={tuple(result['action_pred'].shape)}")

# Clean up
tmp_ckpt.unlink()
print("[OK] Cleaned up smoke-test ckpt")
print("\nALL OK")
