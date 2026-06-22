"""
Offline closed-loop evaluation.

Feeds training data (real RGB + Depth + ee_pose) to policy chunk-by-chunk,
converts predicted actions to xArm world coordinates,
and compares with replay_trajectory GT.

Usage:
    python eval_closedloop.py
"""

import sys
import csv as csv_mod
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    return np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]])


def xarm_pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return T


def main():
    cal_path = r"D:/UMI_Gripper/calibration/hand_eye_result_umi.json"
    data_dir = r"D:/UMI_Gripper/Data_Raw/data_crop"
    home_pose = [309.1, 3.8, 268.3, -179.8, -41.8, -0.2]

    T_home = xarm_pose_to_matrix(home_pose)
    T_cam_to_ee = load_hand_eye_calibration(cal_path)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    # R_site
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    # Load policy
    policy, config = load_policy("checkpoints/best.pt")
    n_obs = config["n_obs_steps"]
    n_act = config["n_action_steps"]

    # Load dataset (stride=1 for full episode)
    # NOTE: use checkpoint's normalizer (same as rollout), do NOT re-fit
    ds = UMIDataset(
        data_dir=data_dir, calibration_path=cal_path,
        n_obs_steps=n_obs, horizon=16, img_size=224, stride=1, max_episodes=1)

    ep = ds.episodes[0]
    ee_poses = ep["ee_poses"]
    N = ep["N"]

    # ─── Compute GT xArm trajectory (replay method) ───
    ep_dir = sorted(Path(data_dir).iterdir())[0]
    csv_path = ep_dir / "trajectory.csv"
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            frames.append({
                'tx': float(row['tx']), 'ty': float(row['ty']), 'tz': float(row['tz']),
                'rx': float(row['qx']), 'ry': float(row['qy']), 'rz': float(row['qz']),
            })

    cam_mats = [pose_to_matrix(f['tx'], f['ty'], f['tz'], f['rx'], f['ry'], f['rz'])
                for f in frames]

    T_gt = T_home.copy()
    gt_traj = [matrix_to_xarm_pose(T_gt)]
    for i in range(len(cam_mats) - 1):
        delta_cam = np.linalg.inv(cam_mats[i]) @ cam_mats[i + 1]
        delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
        delta_xarm = R_site_inv @ delta_ee @ R_site
        T_gt = T_gt @ delta_xarm
        gt_traj.append(matrix_to_xarm_pose(T_gt))
    gt_traj = np.array(gt_traj)

    print(f"Episode: {ep_dir.name}, {N} frames")
    print(f"GT: start=({gt_traj[0,0]:.1f}, {gt_traj[0,1]:.1f}, {gt_traj[0,2]:.1f}) "
          f"end=({gt_traj[-1,0]:.1f}, {gt_traj[-1,1]:.1f}, {gt_traj[-1,2]:.1f})")

    # ─── Closed-loop policy rollout using real training data ───
    print(f"\n=== Closed-loop policy evaluation ===")
    print(f"Using real RGB + Depth + ee_pose from training data")
    print(f"Policy predicts {n_act} steps, advances by {n_act}, re-observes\n")

    pred_traj = []
    sample_idx = 0  # dataset sample index

    print(f"  {'frame':>5s}  {'pred_x':>7s} {'pred_y':>7s} {'pred_z':>7s}  |  "
          f"{'gt_x':>7s} {'gt_y':>7s} {'gt_z':>7s}  |  {'err':>5s}")

    chunk_count = 0
    max_chunks = 50  # ~400 frames

    while sample_idx < len(ds) and chunk_count < max_chunks:
        sample = ds[sample_idx]

        # Feed real observation to policy
        obs = {k: torch.as_tensor(v).float().unsqueeze(0) for k, v in sample["obs"].items()}
        with torch.no_grad():
            result = policy.predict_action(obs)
        actions = result["action"][0].numpy()  # (n_act, 7)

        # Convert each predicted action to xArm world coordinates
        for i, action in enumerate(actions):
            global_frame = sample_idx + i
            if global_frame >= len(gt_traj):
                break

            T_pred_body = np.eye(4)
            T_pred_body[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
            T_pred_body[:3, 3] = action[:3]

            T_target = T_home @ R_site_inv @ T_pred_body @ R_site
            pred_pose = matrix_to_xarm_pose(T_target)
            pred_traj.append(pred_pose)

            gt = gt_traj[global_frame]
            err = np.linalg.norm(pred_pose[:3] - gt[:3])

            if global_frame % 20 == 0:
                print(f"  {global_frame:5d}  {pred_pose[0]:7.1f} {pred_pose[1]:7.1f} {pred_pose[2]:7.1f}  |  "
                      f"{gt[0]:7.1f} {gt[1]:7.1f} {gt[2]:7.1f}  |  {err:5.1f}")

        # Advance by n_act steps (non-overlapping chunks)
        sample_idx += n_act
        chunk_count += 1

    pred_traj = np.array(pred_traj)

    # Summary
    n_compared = min(len(pred_traj), len(gt_traj))
    errors = np.linalg.norm(pred_traj[:n_compared, :3] - gt_traj[:n_compared, :3], axis=1)
    print(f"\n=== Summary ({n_compared} frames) ===")
    print(f"  Mean error:   {errors.mean():.1f} mm")
    print(f"  Median error: {np.median(errors):.1f} mm")
    print(f"  Max error:    {errors.max():.1f} mm")
    print(f"  Min error:    {errors.min():.1f} mm")

    # Error at different stages
    for pct in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(n_compared * pct) - 1, n_compared - 1)
        print(f"  At {pct*100:.0f}%  (frame {idx}): err={errors[idx]:.1f} mm")


if __name__ == "__main__":
    main()
