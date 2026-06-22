"""Quick smoke test for train_umi_ft_RGBonly.py without starting real training."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from train_umi_ft_RGBonly import (
    TrainUMIFTRGBonlyWorkspace, cfg, _collate,
    imagenet_rgb_transform_train,
)

# Build workspace (constructs policy)
ws = TrainUMIFTRGBonlyWorkspace(cfg)
print("[OK] Workspace + policy built")

# Build dataset (1 sample is enough)
from umi_dataset import UMIDataset
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
print(f"[OK] Dataset: {len(ds)} samples")

sample = ds[0]
print(f"[OK] Sample shapes: rgb={tuple(sample['obs']['rgb'].shape)}, "
      f"ee_pose={tuple(sample['obs']['ee_pose'].shape)}, "
      f"action={tuple(sample['action'].shape)}")

# Fit normalizer
normalizer = ds.get_normalizer(mode="limits")
ws.model.set_normalizer(normalizer)
print("[OK] Normalizer fit")

# Build a batch manually and try compute_loss
batch = _collate([ds[0], ds[1]])
print(f"[OK] Batch rgb={tuple(batch['obs']['rgb'].shape)}, "
      f"ee_pose={tuple(batch['obs']['ee_pose'].shape)}, "
      f"action={tuple(batch['action'].shape)}")

ws.model.eval()
with torch.no_grad():
    loss = ws.model.compute_loss(batch)
    print(f"[OK] compute_loss = {loss.item():.4f}")

    # predict_action
    result = ws.model.predict_action(batch["obs"])
    print(f"[OK] predict_action: action={tuple(result['action'].shape)}, "
          f"action_pred={tuple(result['action_pred'].shape)}")

print("\nALL OK")
