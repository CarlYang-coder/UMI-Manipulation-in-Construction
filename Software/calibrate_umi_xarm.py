"""
Calibrate UMI camera <-> xArm camera coordinate transform.

Compares the UMI demonstration trajectory (camera poses from handheld UMI gripper)
with the xArm replay trajectory (camera poses from iPhone on xArm) to find the
fixed rigid transform T_umi_cam_to_xarm_cam.

Usage:
    python calibrate_umi_xarm.py
    python calibrate_umi_xarm.py --umi_csv Data/episode_xxxx/trajectory.csv --xarm_csv calibration/xarm_camera_trajectory.csv
"""

import csv
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def load_umi_trajectory(csv_path):
    """Load UMI trajectory (rotation vector format)."""
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec([
                float(row['qx']), float(row['qy']), float(row['qz'])
            ]).as_matrix()
            T[:3, 3] = [float(row['tx']), float(row['ty']), float(row['tz'])]
            frames.append(T)
    return frames


def load_xarm_trajectory(csv_path):
    """Load xArm recorded trajectory (quaternion format)."""
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([
                float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])
            ]).as_matrix()
            T[:3, 3] = [float(row['tx']), float(row['ty']), float(row['tz'])]
            frames.append(T)
    return frames


def compute_body_deltas(frames):
    """Compute body-frame deltas between consecutive frames."""
    deltas = []
    for i in range(len(frames) - 1):
        delta = np.linalg.inv(frames[i]) @ frames[i + 1]
        deltas.append(delta)
    return deltas


def main():
    parser = argparse.ArgumentParser(description="Calibrate UMI <-> xArm camera transform")
    parser.add_argument("--umi_csv", default=r"Data\episode_0016_2026-03-29--19-17-48\trajectory.csv",
                        help="UMI demonstration trajectory CSV")
    parser.add_argument("--xarm_csv", default="calibration/xarm_camera_trajectory.csv",
                        help="xArm replay recorded trajectory CSV")
    parser.add_argument("--output", default="calibration/umi_to_xarm_cam.json",
                        help="Output calibration file")
    args = parser.parse_args()

    # Load trajectories
    print(f"Loading UMI trajectory: {args.umi_csv}")
    umi_frames = load_umi_trajectory(args.umi_csv)
    print(f"  Frames: {len(umi_frames)}")

    print(f"Loading xArm trajectory: {args.xarm_csv}")
    xarm_frames = load_xarm_trajectory(args.xarm_csv)
    print(f"  Frames: {len(xarm_frames)}")

    # Align frame counts (use the shorter one)
    n = min(len(umi_frames), len(xarm_frames))
    print(f"  Using {n} frames for calibration")

    umi_frames = umi_frames[:n]
    xarm_frames = xarm_frames[:n]

    # Method 1: Solve T such that T_xarm_i = T @ T_umi_i for all i
    # This gives T_umi_to_xarm in world frame (not useful directly)
    #
    # Method 2: Compare body-frame deltas
    # delta_xarm = T_transform @ delta_umi @ inv(T_transform)
    # This is the conjugation relationship, same as hand-eye calibration
    #
    # Use body-frame deltas and solve with least squares

    umi_deltas = compute_body_deltas(umi_frames)
    xarm_deltas = compute_body_deltas(xarm_frames)

    n_deltas = min(len(umi_deltas), len(xarm_deltas))

    # Filter: only use deltas with significant motion
    min_translation = 0.001  # 1mm
    valid_pairs = []
    for i in range(n_deltas):
        t_umi = np.linalg.norm(umi_deltas[i][:3, 3])
        t_xarm = np.linalg.norm(xarm_deltas[i][:3, 3])
        if t_umi > min_translation and t_xarm > min_translation:
            valid_pairs.append(i)

    print(f"  Valid delta pairs (motion > {min_translation*1000:.0f}mm): {len(valid_pairs)}")

    if len(valid_pairs) < 10:
        print("WARNING: Very few valid pairs. Results may be inaccurate.")

    # Accumulate deltas over windows for more robust estimation
    # Use accumulated deltas over 30-frame windows
    window = 30
    R_A_list = []  # rotation parts of accumulated umi deltas
    t_A_list = []
    R_B_list = []  # rotation parts of accumulated xarm deltas
    t_B_list = []

    for start in range(0, n_deltas - window, window // 2):
        # Accumulate UMI delta over window
        T_umi_acc = np.eye(4)
        T_xarm_acc = np.eye(4)
        for j in range(start, min(start + window, n_deltas)):
            T_umi_acc = T_umi_acc @ umi_deltas[j]
            T_xarm_acc = T_xarm_acc @ xarm_deltas[j]

        t_umi_norm = np.linalg.norm(T_umi_acc[:3, 3])
        t_xarm_norm = np.linalg.norm(T_xarm_acc[:3, 3])

        if t_umi_norm > 0.005 and t_xarm_norm > 0.005:  # 5mm minimum
            R_A_list.append(T_umi_acc[:3, :3])
            t_A_list.append(T_umi_acc[:3, 3].reshape(3, 1))
            R_B_list.append(T_xarm_acc[:3, :3])
            t_B_list.append(T_xarm_acc[:3, 3].reshape(3, 1))

    print(f"  Accumulated delta pairs (window={window}): {len(R_A_list)}")

    if len(R_A_list) < 3:
        print("ERROR: Not enough data for calibration.")
        return

    # Solve using OpenCV calibrateHandEye
    # We want: delta_xarm = X @ delta_umi @ inv(X)
    # This is: A @ X = X @ B where A=delta_xarm, B=delta_umi, X=T_umi_to_xarm
    import cv2

    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    print("\n=== Calibration Results ===\n")

    results = {}
    for name, method in methods.items():
        try:
            R_x, t_x = cv2.calibrateHandEye(
                R_gripper2base=R_B_list,
                t_gripper2base=t_B_list,
                R_target2cam=R_A_list,
                t_target2cam=t_A_list,
                method=method
            )

            T_x = np.eye(4)
            T_x[:3, :3] = R_x
            T_x[:3, 3] = t_x.squeeze()

            t_mm = t_x.squeeze() * 1000
            rpy = Rotation.from_matrix(R_x).as_euler('xyz', degrees=True)
            det = np.linalg.det(R_x)

            # Compute reprojection error
            errors = []
            for i in range(len(R_A_list)):
                T_umi = np.eye(4)
                T_umi[:3, :3] = R_A_list[i]
                T_umi[:3, 3] = t_A_list[i].squeeze()

                T_xarm = np.eye(4)
                T_xarm[:3, :3] = R_B_list[i]
                T_xarm[:3, 3] = t_B_list[i].squeeze()

                # Should satisfy: T_x @ T_umi = T_xarm @ T_x
                lhs = T_x @ T_umi
                rhs = T_xarm @ T_x
                t_err = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]) * 1000
                errors.append(t_err)

            avg_err = np.mean(errors)

            print(f"  {name}:")
            print(f"    Translation: ({t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}) mm")
            print(f"    Rotation (RPY): ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) deg")
            print(f"    det(R)={det:.4f}, avg_err={avg_err:.2f} mm")

            results[name] = (T_x, avg_err)

        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not results:
        print("\nERROR: All methods failed.")
        return

    # Pick best
    best_name = min(results, key=lambda k: results[k][1])
    T_best, best_err = results[best_name]

    print(f"\n  Best: {best_name} (avg error = {best_err:.2f} mm)")

    # Also compute per-axis scale factors
    print("\n=== Per-axis scale analysis ===")
    umi_translations = np.array([t_A_list[i].squeeze() for i in range(len(t_A_list))])
    xarm_translations = np.array([t_B_list[i].squeeze() for i in range(len(t_B_list))])

    for axis, name in enumerate(['X', 'Y', 'Z']):
        umi_range = np.max(np.abs(umi_translations[:, axis]))
        xarm_range = np.max(np.abs(xarm_translations[:, axis]))
        if umi_range > 0.001:
            scale = xarm_range / umi_range
            print(f"  {name}: UMI range={umi_range*1000:.1f}mm, xArm range={xarm_range*1000:.1f}mm, scale={scale:.3f}")
        else:
            print(f"  {name}: too little motion to estimate scale")

    # Save result
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    output = {
        'method': best_name,
        'T_umi_cam_to_xarm_cam': T_best.tolist(),
        'T_xarm_cam_to_umi_cam': np.linalg.inv(T_best).tolist(),
        'avg_error_mm': float(best_err),
        'num_frames': n,
        'num_delta_pairs': len(R_A_list),
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved calibration to {args.output}")


if __name__ == "__main__":
    main()
