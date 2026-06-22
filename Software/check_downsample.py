"""Smoke test: run downsample_episodes.process_episode on one episode only,
then verify outputs."""
from pathlib import Path
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downsample_episodes import (
    process_episode, DOWNSAMPLE_STEP, CALIBRATION_PATH,
)
from generate_ee_pose_csv import process_episode as recompute_ee_pose
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
from umi_dataset import load_hand_eye_calibration

SRC = Path(r"D:\UMI_Gripper\Data_Raw\data_crop\episode_0000_2026-04-13--15-41-01")
DST = Path(r"D:\UMI_Gripper\Data_Raw\data_crop_ds\episode_0000_2026-04-13--15-41-01")

# Clean dst if exists
import shutil
if DST.exists():
    shutil.rmtree(DST)

n_out = process_episode(SRC, DST)
print(f"[OK] process_episode returned {n_out} frames")

T_cam_to_ee = load_hand_eye_calibration(CALIBRATION_PATH)
T_ee_to_cam = np.linalg.inv(T_cam_to_ee)
recompute_ee_pose(DST, T_cam_to_ee, T_ee_to_cam)
print("[OK] recomputed ee_pose_trajectory.csv")

# Source frame count
with open(SRC / "trajectory.csv") as f:
    n_in = sum(1 for _ in f) - 1
print(f"Source frames:     {n_in}")
print(f"Downsampled frames: {n_out}  (expected {(n_in + DOWNSAMPLE_STEP - 1) // DOWNSAMPLE_STEP})")

# Check RGB/depth counts
n_rgb = len(list((DST / "rgb").glob("*.png")))
n_depth = len(list((DST / "depth").glob("*.png")))
print(f"RGB pngs:   {n_rgb}")
print(f"Depth pngs: {n_depth}")
assert n_rgb == n_out == n_depth, "mismatch between CSV rows and images"

# Check trajectory.csv + ee_pose_trajectory.csv length
with open(DST / "trajectory.csv") as f:
    n_traj = sum(1 for _ in f) - 1
with open(DST / "ee_pose_trajectory.csv") as f:
    n_ee = sum(1 for _ in f) - 1
print(f"trajectory.csv rows:         {n_traj}")
print(f"ee_pose_trajectory.csv rows: {n_ee}")
assert n_traj == n_ee == n_out

# Check the first ee_pose row is identity
with open(DST / "ee_pose_trajectory.csv") as f:
    first = next(csv.DictReader(f))
pos = [float(first[k]) for k in ("tx", "ty", "tz", "qx", "qy", "qz")]
print(f"First ee_pose row (should be all 0): {pos}")
assert all(abs(v) < 1e-9 for v in pos), "first ee_pose row is not identity"

# Check video files
for name in ("rgb.mp4", "depth.mp4"):
    p = DST / name
    assert p.is_file() and p.stat().st_size > 0, f"{name} missing or empty"
    print(f"  {name}: {p.stat().st_size / 1024:.1f} KB")

print("\nALL OK")
