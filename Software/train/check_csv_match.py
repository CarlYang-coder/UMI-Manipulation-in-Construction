"""Verify that loaded ee_pose_trajectory.csv values match the old on-the-fly
_compute_ee_poses output, for one episode."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path

from umi_dataset import UMIDataset

DATA_DIR = r"D:\UMI_Gripper\Data_Raw\data_crop"
CAL = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json"

# Build dataset (now reads ee_pose_trajectory.csv)
ds = UMIDataset(
    data_dir=DATA_DIR,
    calibration_path=CAL,
    n_obs_steps=2,
    horizon=16,
    img_size=224,
    augment=False,
)

# Use one episode, compare new (CSV) ee_poses vs old (_compute_ee_poses) path
ep = ds.episodes[0]
new_poses = ep["ee_poses"]  # loaded from CSV
old_poses = ds._compute_ee_poses(ep["frames"])  # recompute on-the-fly

print(f"Episode: {ep['dir'].name}  frames={ep['N']}")
print(f"new shape={new_poses.shape}  old shape={old_poses.shape}")

diff = np.abs(new_poses - old_poses)
print(f"max abs diff = {diff.max():.2e}")
print(f"mean abs diff = {diff.mean():.2e}")

# Spot-check a few frames
for i in [0, ep["N"] // 2, ep["N"] - 1]:
    print(f"\nFrame {i}:")
    print(f"  CSV (new): {new_poses[i]}")
    print(f"  recompute: {old_poses[i]}")

# First frame should be identity (0,0,0,0,0,0,gw)
print(f"\nFirst frame first 6 (should be all 0): {new_poses[0][:6]}")
