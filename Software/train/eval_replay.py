"""
Compare policy-predicted trajectory vs replay_trajectory ground truth.

Uses training data episode:
1. Compute GT xArm trajectory (same as replay_trajectory.py)
2. Run policy on each observation window, get predicted actions
3. Convert predicted actions to xArm world coordinates
4. Print side-by-side comparison

Usage:
    python eval_replay.py
"""

import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umi_dataset import UMIDataset, pose_to_matrix, matrix_to_pose_7d, load_hand_eye_calibration
from umi_ft_policy import UMIFTDiffusionPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    ns = DDPMScheduler(num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="epsilon")
    policy = UMIFTDiffusionPolicy(
        action_dim=config["action_dim"], ee_pose_dim=config["action_dim"],
        horizon=config["horizon"], n_action_steps=config["n_action_steps"],
        n_obs_steps=config["n_obs_steps"], clip_out_dim=config["clip_out_dim"],
        fusion_d_model=config["fusion_d_model"], fusion_nhead=config["fusion_nhead"],
        fusion_num_layers=config["fusion_num_layers"],
        fusion_dim_feedforward=config["fusion_dim_feedforward"],
        noise_scheduler=ns, num_inference_steps=config["num_infer_steps"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, config


def matrix_to_xarm_pose(T):
    pos = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return T


def main():
    cal_path = r"D:/UMI_Gripper/calibration/hand_eye_result_umi.json"
    data_dir = r"D:/UMI_Gripper/Data_Raw/data_crop"

    # Load hand-eye calibration
    T_cam_to_ee = load_hand_eye_calibration(cal_path)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    # Home pose (same as replay_trajectory.py)
    HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]
    # Approximate home EE pose (from your logs)
    home_pose = [309.1, 3.8, 268.3, -179.8, -41.8, -0.2]
    T_home = xarm_pose_to_matrix(home_pose)

    # R_site (from replay_trajectory.py)
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    # ─── 1. Compute GT xArm trajectory (same as replay_trajectory.py) ───
    ep_dir = sorted(Path(data_dir).iterdir())[0]
    csv_path = ep_dir / "trajectory.csv"
    print(f"Episode: {ep_dir.name}")

    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'tx': float(row['tx']), 'ty': float(row['ty']), 'tz': float(row['tz']),
                'rx': float(row['qx']), 'ry': float(row['qy']), 'rz': float(row['qz']),
                'gripper_width': float(row['gripper_width']),
            })

    # Compute body-frame deltas (same as replay_trajectory.py)
    cam_matrices = [pose_to_matrix(f['tx'], f['ty'], f['tz'], f['rx'], f['ry'], f['rz'])
                    for f in frames]

    # GT xArm trajectory via replay method
    T_current_replay = T_home.copy()
    gt_xarm_trajectory = [matrix_to_xarm_pose(T_current_replay)]

    for i in range(len(cam_matrices) - 1):
        delta_cam = np.linalg.inv(cam_matrices[i]) @ cam_matrices[i + 1]
        delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
        # replay_trajectory.py applies R_site conjugation
        delta_xarm = R_site_inv @ delta_ee @ R_site
        T_current_replay = T_current_replay @ delta_xarm
        gt_xarm_trajectory.append(matrix_to_xarm_pose(T_current_replay))

    gt_xarm_trajectory = np.array(gt_xarm_trajectory)
    print(f"GT trajectory: {len(gt_xarm_trajectory)} frames")
    print(f"  Start: x={gt_xarm_trajectory[0,0]:.1f} y={gt_xarm_trajectory[0,1]:.1f} z={gt_xarm_trajectory[0,2]:.1f}")
    print(f"  End:   x={gt_xarm_trajectory[-1,0]:.1f} y={gt_xarm_trajectory[-1,1]:.1f} z={gt_xarm_trajectory[-1,2]:.1f}")

    # ─── 2. Compute training data ee_poses (body-frame) ───
    ds = UMIDataset(
        data_dir=data_dir, calibration_path=cal_path,
        n_obs_steps=2, horizon=16, img_size=224, stride=1, max_episodes=1)

    ep = ds.episodes[0]
    ee_poses = ep["ee_poses"]  # (N, 7) body-frame cumulative
    print(f"\nTraining ee_poses: {ee_poses.shape[0]} frames")
    print(f"  Start: {ee_poses[0,:3] * 1000} mm")
    print(f"  End:   {ee_poses[-1,:3] * 1000} mm")

    # ─── 3. Test different conversion methods ───
    print(f"\n=== Conversion method comparison (frame 100) ===")
    f = 100
    T_body = np.eye(4)
    T_body[:3, :3] = Rotation.from_rotvec(ee_poses[f, 3:6]).as_matrix()
    T_body[:3, 3] = ee_poses[f, :3]

    gt_pose = gt_xarm_trajectory[f]

    # Method A: T_home @ T_body (direct left-multiply)
    T_a = T_home @ T_body
    pose_a = matrix_to_xarm_pose(T_a)

    # Method B: T_home @ R_site_inv @ T_body @ R_site
    T_b = T_home @ R_site_inv @ T_body @ R_site
    pose_b = matrix_to_xarm_pose(T_b)

    # Method C: Accumulate deltas with R_site (like replay)
    # Recompute from scratch to frame f
    T_c = T_home.copy()
    T_ee_body = np.eye(4)
    for i in range(f):
        delta_cam = np.linalg.inv(cam_matrices[i]) @ cam_matrices[i + 1]
        delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
        delta_xarm = R_site_inv @ delta_ee @ R_site
        T_c = T_c @ delta_xarm
        T_ee_body = T_ee_body @ delta_ee
    pose_c = matrix_to_xarm_pose(T_c)
    body_pose = matrix_to_pose_7d(T_ee_body, 0)

    print(f"  GT xArm pose:     x={gt_pose[0]:.1f} y={gt_pose[1]:.1f} z={gt_pose[2]:.1f}")
    print(f"  Method A (home@body):   x={pose_a[0]:.1f} y={pose_a[1]:.1f} z={pose_a[2]:.1f}  err={np.linalg.norm(np.array(pose_a[:3])-gt_pose[:3]):.1f}mm")
    print(f"  Method B (home@Rinv@body@R): x={pose_b[0]:.1f} y={pose_b[1]:.1f} z={pose_b[2]:.1f}  err={np.linalg.norm(np.array(pose_b[:3])-gt_pose[:3]):.1f}mm")
    print(f"  Method C (replay deltas):    x={pose_c[0]:.1f} y={pose_c[1]:.1f} z={pose_c[2]:.1f}  err={np.linalg.norm(np.array(pose_c[:3])-gt_pose[:3]):.1f}mm")
    print(f"\n  ee_poses[{f}] = {ee_poses[f,:3]*1000} mm")
    print(f"  T_ee_body accumulated = {body_pose[:3]*1000} mm")
    print(f"  Match: {np.allclose(ee_poses[f,:3], body_pose[:3], atol=1e-6)}")

    # ─── 4. Find the correct conversion ───
    print(f"\n=== Testing all frames (every 50th) ===")
    print(f"  {'frame':>5s}  {'gt_x':>7s} {'gt_y':>7s} {'gt_z':>7s}  |  "
          f"{'A_x':>7s} {'A_y':>7s} {'A_z':>7s} {'A_err':>5s}  |  "
          f"{'B_x':>7s} {'B_y':>7s} {'B_z':>7s} {'B_err':>5s}  |  "
          f"{'C_x':>7s} {'C_y':>7s} {'C_z':>7s} {'C_err':>5s}")

    for f in range(0, min(len(gt_xarm_trajectory), len(ee_poses)), 50):
        gt = gt_xarm_trajectory[f]

        T_body = np.eye(4)
        T_body[:3, :3] = Rotation.from_rotvec(ee_poses[f, 3:6]).as_matrix()
        T_body[:3, 3] = ee_poses[f, :3]

        # Method A
        pa = matrix_to_xarm_pose(T_home @ T_body)
        ea = np.linalg.norm(np.array(pa[:3]) - gt[:3])

        # Method B
        pb = matrix_to_xarm_pose(T_home @ R_site_inv @ T_body @ R_site)
        eb = np.linalg.norm(np.array(pb[:3]) - gt[:3])

        # Method C (needs recomputation - skip, use GT directly)
        pc = gt[:3]  # method C = GT by definition
        ec = 0.0

        print(f"  {f:5d}  {gt[0]:7.1f} {gt[1]:7.1f} {gt[2]:7.1f}  |  "
              f"{pa[0]:7.1f} {pa[1]:7.1f} {pa[2]:7.1f} {ea:5.1f}  |  "
              f"{pb[0]:7.1f} {pb[1]:7.1f} {pb[2]:7.1f} {eb:5.1f}  |  "
              f"{pc[0]:7.1f} {pc[1]:7.1f} {pc[2]:7.1f} {ec:5.1f}")


if __name__ == "__main__":
    main()
