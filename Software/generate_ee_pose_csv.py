"""
Generate ee_pose CSV for each episode in Data_Raw/data_crop_ds/.

For each episode, computes ee_poses using the same pipeline as training
(_compute_ee_poses: body-frame delta accumulation from identity with hand-eye
calibration), and writes ee_pose_trajectory.csv with columns:
  timestamp, tx, ty, tz, rx, ry, rz, gripper_width, rgb_path, depth_path

Where tx,ty,tz are in meters, rx,ry,rz are rotation vector (radians),
and gripper_width is rescaled to full [0,1] using GW_CLOSED_IN / GW_OPEN_IN
from gripper_rescale.py (raw ArUco values only span ~[0.30, 0.90]).
"""

import csv
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from umi_dataset import pose_to_matrix, matrix_to_pose_7d, load_hand_eye_calibration
from gripper_rescale import GW_CLOSED_IN, GW_OPEN_IN

DATA_DIR = Path(r"D:\UMI_Gripper\Data_Raw\data_crop_ds")
CALIBRATION_PATH = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json"


def rescale_gw(raw: float) -> float:
    """Map raw gripper_width ~[0.30, 0.90] to full [0, 1]."""
    norm = (raw - GW_CLOSED_IN) / (GW_OPEN_IN - GW_CLOSED_IN)
    return float(np.clip(norm, 0.0, 1.0))


def process_episode(ep_dir, T_cam_to_ee, T_ee_to_cam):
    csv_path = ep_dir / "trajectory.csv"
    if not csv_path.exists():
        return

    # Read original CSV
    rows = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    N = len(rows)
    cam_matrices = [
        pose_to_matrix(float(r['tx']), float(r['ty']), float(r['tz']),
                       float(r['qx']), float(r['qy']), float(r['qz']))
        for r in rows
    ]

    # Body-frame delta accumulation from identity (same as _compute_ee_poses).
    # gripper_width is rescaled from the raw ArUco range to full [0, 1].
    T_ee = np.eye(4)
    ee_poses = [matrix_to_pose_7d(T_ee, rescale_gw(float(rows[0]['gripper_width'])))]

    for i in range(1, N):
        delta_cam = np.linalg.inv(cam_matrices[i - 1]) @ cam_matrices[i]
        delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
        T_ee = T_ee @ delta_ee
        gw = rescale_gw(float(rows[i]['gripper_width']))
        ee_poses.append(matrix_to_pose_7d(T_ee, gw))

    # Write ee_pose CSV
    out_path = ep_dir / "ee_pose_trajectory.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz',
                         'gripper_width', 'rgb_path', 'depth_path'])
        for i in range(N):
            pose = ee_poses[i]
            writer.writerow([
                rows[i]['timestamp'],
                f"{pose[0]:.8f}", f"{pose[1]:.8f}", f"{pose[2]:.8f}",
                f"{pose[3]:.8f}", f"{pose[4]:.8f}", f"{pose[5]:.8f}",
                f"{pose[6]:.6f}",
                rows[i]['rgb_path'],
                rows[i]['depth_path'],
            ])

    return N


def main():
    T_cam_to_ee = load_hand_eye_calibration(CALIBRATION_PATH)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    episodes = sorted([d for d in DATA_DIR.iterdir()
                        if d.is_dir() and not d.name.startswith(".")])
    print(f"Found {len(episodes)} episodes in {DATA_DIR}")

    for ep_dir in episodes:
        n = process_episode(ep_dir, T_cam_to_ee, T_ee_to_cam)
        if n:
            print(f"  {ep_dir.name}: {n} frames -> ee_pose_trajectory.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
