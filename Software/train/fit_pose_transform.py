"""
Fit a transform that maps live ee_pose -> csv ee_pose.

Reads collected (csv_pose, live_pose) pairs from verify_pose.py output.
Uses first N episodes for calibration and remaining for validation.

Usage:
    python fit_pose_transform.py
    python fit_pose_transform.py --calib_episodes 0 1 2 --val_episodes 3 4
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_DATA = r"D:/UMI_Gripper/train/pose_calibration_data.npz"


def fit_rigid_transform(live_pts, csv_pts):
    """Kabsch / Procrustes: find R, t such that csv ≈ R @ live + t."""
    csv_c = csv_pts.mean(axis=0)
    live_c = live_pts.mean(axis=0)
    H = (live_pts - live_c).T @ (csv_pts - csv_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = csv_c - R @ live_c
    return R, t


def fit_rotation_only(live_pts, csv_pts):
    """Find R (no translation) such that csv ≈ R @ live.
    Assumes both start at origin (frame 0 = 0)."""
    H = live_pts.T @ csv_pts  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    return R


def apply_rigid(R, t, pts):
    return (R @ pts.T).T + t


def evaluate(R, t, live_pts, csv_pts, label):
    pred = apply_rigid(R, t, live_pts)
    err = np.linalg.norm(pred - csv_pts, axis=1)
    print(f"  {label}:  N={len(live_pts):4d}  "
          f"mean={err.mean():.1f}mm  median={np.median(err):.1f}mm  "
          f"max={err.max():.1f}mm")
    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--calib_episodes", nargs='+', type=int, default=None,
                        help="Episode indices for calibration (default: first half)")
    parser.add_argument("--val_episodes", nargs='+', type=int, default=None,
                        help="Episode indices for validation (default: second half)")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    csv_pose = data['csv_pose']    # (N, 7)
    live_pose = data['live_pose']  # (N, 7)
    episode = data['episode']      # (N,)
    n_episodes = int(episode.max() + 1)
    print(f"Loaded {len(csv_pose)} pairs from {n_episodes} episodes")

    # Default split: first half = calib, second half = val
    if args.calib_episodes is None:
        n_calib = max(1, n_episodes // 2 + (n_episodes % 2))
        args.calib_episodes = list(range(n_calib))
    if args.val_episodes is None:
        args.val_episodes = [e for e in range(n_episodes) if e not in args.calib_episodes]

    print(f"Calibration episodes: {args.calib_episodes}")
    print(f"Validation episodes:  {args.val_episodes}")

    # Build calib/val masks
    calib_mask = np.isin(episode, args.calib_episodes)
    val_mask = np.isin(episode, args.val_episodes)

    csv_calib = csv_pose[calib_mask, :3] * 1000  # to mm
    live_calib = live_pose[calib_mask, :3] * 1000
    csv_val = csv_pose[val_mask, :3] * 1000
    live_val = live_pose[val_mask, :3] * 1000

    print(f"\nCalib pairs: {len(csv_calib)}")
    print(f"Val pairs:   {len(csv_val)}")

    # Fit on calibration data — ROTATION ONLY (no translation)
    # Both live and csv start from origin, so rotation is the only unknown.
    print("\n=== Fitting rotation-only transform on calibration data ===")
    R = fit_rotation_only(live_calib, csv_calib)
    t = np.zeros(3)
    rot = Rotation.from_matrix(R)
    print(f"R =\n{R}")
    print(f"t = {t} (mm) - forced to zero (both frames start at origin)")
    print(f"Rotation Euler XYZ (deg): {rot.as_euler('xyz', degrees=True)}")
    print(f"Rotation as quat (xyzw):  {rot.as_quat()}")

    # Evaluate on calib (training error)
    print("\n=== Errors ===")
    evaluate(R, t, live_calib, csv_calib, "Calib (train)")

    # Evaluate on val (generalization)
    if len(csv_val) > 0:
        evaluate(R, t, live_val, csv_val, "Val   (test) ")

    # Per-episode breakdown
    print("\n=== Per-episode validation errors ===")
    for ep in range(n_episodes):
        mask = episode == ep
        if not mask.any():
            continue
        csv_ep = csv_pose[mask, :3] * 1000
        live_ep = live_pose[mask, :3] * 1000
        pred = apply_rigid(R, t, live_ep)
        err = np.linalg.norm(pred - csv_ep, axis=1)
        tag = "calib" if ep in args.calib_episodes else "val  "
        print(f"  Episode {ep} ({tag}): N={len(csv_ep):4d}  "
              f"mean={err.mean():.1f}mm  max={err.max():.1f}mm")

    # Save the transform
    out_path = args.data.replace('.npz', '_transform.npz')
    np.savez(out_path, R=R, t=t)
    print(f"\n[INFO] Transform saved to {out_path}")


if __name__ == "__main__":
    main()
