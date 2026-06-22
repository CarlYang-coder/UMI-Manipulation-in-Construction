"""
Downsample cropped UMI episodes in time (every Nth frame), mirroring Image_DP's
downsample_and_train.py step 1 but adapted for the UMI file layout.

Input:   Data_Raw/data_crop/episode_*/
           rgb/000000.png ... NNNNNN.png
           depth/000000.png ... NNNNNN.png
           trajectory.csv            (timestamp, tx, ty, tz, qx, qy, qz, gripper_width, rgb_path, depth_path)
Output:  Data_Raw/data_crop_ds/episode_*/
           rgb/000000.png ...        (re-indexed sequentially)
           depth/000000.png ...      (re-indexed sequentially)
           trajectory.csv            (every DOWNSAMPLE_STEP-th row, rgb/depth_path updated)
           rgb.mp4, depth.mp4        (at SRC_FPS / DOWNSAMPLE_STEP)
           ee_pose_trajectory.csv    (recomputed from new trajectory.csv)

Does NOT crop images (UMI iPhone frames are clean; UMIDataset handles resizing).
Does NOT modify source data.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

# Import the ee_pose recomputer
sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
from umi_dataset import load_hand_eye_calibration
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ee_pose_csv import process_episode as recompute_ee_pose


SRC_DIR = Path(r"D:\UMI_Gripper\Data_Raw\data_crop")
DST_DIR = Path(r"D:\UMI_Gripper\Data_Raw\data_crop_ds")
CALIBRATION_PATH = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json"

DOWNSAMPLE_STEP = 3      # keep every 3rd frame (30Hz -> 10Hz)
SRC_FPS = 30             # original Record3D frame rate
DST_FPS = SRC_FPS // DOWNSAMPLE_STEP


def images_to_video(image_dir: Path, video_path: Path, fps: float):
    """Make mp4 from sequential *.png files in image_dir."""
    images = sorted(image_dir.glob("*.png"))
    if not images:
        return
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    for p in images:
        img = cv2.imread(str(p))
        if img is not None:
            writer.write(img)
    writer.release()


def process_episode(src_ep: Path, dst_ep: Path) -> int:
    """Downsample one episode; returns number of output frames, or 0 if skipped."""
    src_csv = src_ep / "trajectory.csv"
    if not src_csv.is_file():
        print(f"  [SKIP] {src_ep.name}: no trajectory.csv")
        return 0

    # Read source CSV
    with open(src_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Keep every DOWNSAMPLE_STEP-th row
    kept = rows[::DOWNSAMPLE_STEP]
    if len(kept) < 2:
        print(f"  [SKIP] {src_ep.name}: only {len(kept)} frames after downsample")
        return 0

    # Prepare output dirs
    dst_rgb = dst_ep / "rgb"
    dst_depth = dst_ep / "depth"
    dst_rgb.mkdir(parents=True, exist_ok=True)
    dst_depth.mkdir(parents=True, exist_ok=True)

    # Copy images with sequential new names, rewrite paths in rows
    new_rows = []
    for i, row in enumerate(kept):
        src_rgb_path = src_ep / row["rgb_path"]
        src_depth_path = src_ep / row["depth_path"]
        new_name = f"{i:06d}.png"

        new_rgb_rel = f"rgb/{new_name}"
        new_depth_rel = f"depth/{new_name}"

        shutil.copy2(src_rgb_path, dst_rgb / new_name)
        shutil.copy2(src_depth_path, dst_depth / new_name)

        new_row = dict(row)
        new_row["rgb_path"] = new_rgb_rel
        new_row["depth_path"] = new_depth_rel
        new_rows.append(new_row)

    # Write downsampled trajectory.csv
    with open(dst_ep / "trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    # Make mp4s at reduced fps
    images_to_video(dst_rgb,   dst_ep / "rgb.mp4",   DST_FPS)
    images_to_video(dst_depth, dst_ep / "depth.mp4", DST_FPS)

    return len(new_rows)


def main():
    assert SRC_DIR.is_dir(), f"Source dir not found: {SRC_DIR}"
    DST_DIR.mkdir(parents=True, exist_ok=True)

    T_cam_to_ee = load_hand_eye_calibration(CALIBRATION_PATH)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    episodes = sorted(d for d in SRC_DIR.iterdir()
                      if d.is_dir() and not d.name.startswith("."))
    print(f"Found {len(episodes)} episodes in {SRC_DIR}")
    print(f"Downsample step={DOWNSAMPLE_STEP}  ({SRC_FPS} Hz -> {DST_FPS} Hz)")
    print(f"Output -> {DST_DIR}\n")

    total_in = 0
    total_out = 0
    for src_ep in episodes:
        dst_ep = DST_DIR / src_ep.name
        n_out = process_episode(src_ep, dst_ep)
        if n_out == 0:
            continue
        # Recompute ee_pose_trajectory.csv on the downsampled trajectory
        recompute_ee_pose(dst_ep, T_cam_to_ee, T_ee_to_cam)

        # Source frame count (for reporting only)
        with open(src_ep / "trajectory.csv") as f:
            n_in = sum(1 for _ in f) - 1  # minus header
        total_in += n_in
        total_out += n_out
        print(f"  {src_ep.name}: {n_in} -> {n_out} frames")

    print(f"\nDone. {total_in} -> {total_out} total frames ({100*total_out/max(1,total_in):.1f}%)")


if __name__ == "__main__":
    main()
