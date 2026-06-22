"""
Offline evaluation: compare policy predicted actions vs ground truth.
No coordinate transforms - directly compare in training data space.

Usage:
    python eval_offline.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from umi_dataset import UMIDataset, pose_to_matrix, load_hand_eye_calibration
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


def main():
    policy, config = load_policy("checkpoints/best.pt")
    n_obs = config["n_obs_steps"]

    ds = UMIDataset(
        data_dir=r"D:/UMI_Gripper/Data_Raw/data_crop",
        calibration_path=r"D:/UMI_Gripper/calibration/hand_eye_result_umi.json",
        n_obs_steps=n_obs, horizon=16, img_size=224, stride=50, max_episodes=1)
    normalizer = ds.get_normalizer()
    policy.set_normalizer(normalizer)

    # Test on several samples spread across the episode
    print("=== Direct Action Comparison (training data space, mm) ===")
    print("Comparing predicted vs ground truth actions at different points in episode\n")

    test_indices = [0, 3, 6, 9, 12]
    for idx in test_indices:
        if idx >= len(ds):
            break
        sample = ds[idx]
        obs = {k: torch.as_tensor(v).float().unsqueeze(0) for k, v in sample["obs"].items()}
        gt_action = sample["action"].numpy()  # (16, 7)

        with torch.no_grad():
            result = policy.predict_action(obs)
        pred_action = result["action_pred"][0].numpy()  # (16, 7)

        obs_pose = sample["obs"]["ee_pose"][-1].numpy()
        print(f"--- Sample {idx} (obs ee_pose: tx={obs_pose[0]*1000:.1f} ty={obs_pose[1]*1000:.1f} tz={obs_pose[2]*1000:.1f} mm) ---")

        print(f"  {'step':>4s}  {'pred_tx':>8s} {'pred_ty':>8s} {'pred_tz':>8s}  |  {'gt_tx':>8s} {'gt_ty':>8s} {'gt_tz':>8s}  |  {'err_mm':>6s}")
        for i in range(8):
            p = pred_action[i] * 1000  # to mm
            g = gt_action[i] * 1000
            err = np.linalg.norm(p[:3] - g[:3])
            print(f"  {i:4d}  {p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f}  |  {g[0]:8.2f} {g[1]:8.2f} {g[2]:8.2f}  |  {err:6.2f}")
        print()

    # Simulated rollout using DIRECT conversion (matching run_umi_ft.py)
    # T_target = T_home @ R_site_inv @ T_pred_body @ R_site
    # Here T_home = identity since we simulate from home
    print("\n=== Simulated Rollout (direct conversion, matching run_umi_ft.py) ===")
    print("Method: T_target = T_home @ R_site_inv @ T_pred_body @ R_site\n")

    ep = ds.episodes[0]
    gt_poses = ep["ee_poses"]

    T_home = np.eye(4)  # home = identity in body-frame space

    # R_site (from replay_trajectory.py)
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    print(f"  {'step':>4s}  {'pred_x':>8s} {'pred_y':>8s} {'pred_z':>8s}  |  "
          f"{'gt_x':>8s} {'gt_y':>8s} {'gt_z':>8s}  |  {'err_mm':>6s}  "
          f"{'no_rsite_x':>10s} {'no_rsite_y':>10s} {'no_rsite_z':>10s}  |  {'err2_mm':>7s}")

    frame_idx = n_obs - 1
    chunk_count = 0

    while frame_idx < len(ds) and chunk_count < 5:
        sample = ds[frame_idx]
        obs = {k: torch.as_tensor(v).float().unsqueeze(0) for k, v in sample["obs"].items()}

        with torch.no_grad():
            result = policy.predict_action(obs)
        pred_actions = result["action"][0].numpy()

        for i, action in enumerate(pred_actions):
            global_step = frame_idx + i
            if global_step >= len(gt_poses):
                break

            # Policy prediction as body-frame SE(3)
            T_pred_body = np.eye(4)
            T_pred_body[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
            T_pred_body[:3, 3] = action[:3]

            # Method A: with R_site conjugation
            T_target_rsite = T_home @ R_site_inv @ T_pred_body @ R_site
            pred_rsite = T_target_rsite[:3, 3] * 1000

            # Method B: direct (no R_site)
            T_target_direct = T_home @ T_pred_body
            pred_direct = T_target_direct[:3, 3] * 1000

            # GT
            gt_pos = gt_poses[global_step, :3] * 1000
            err_rsite = np.linalg.norm(pred_rsite - gt_pos)
            err_direct = np.linalg.norm(pred_direct - gt_pos)

            print(f"  {global_step:4d}  {pred_rsite[0]:8.2f} {pred_rsite[1]:8.2f} {pred_rsite[2]:8.2f}  |  "
                  f"{gt_pos[0]:8.2f} {gt_pos[1]:8.2f} {gt_pos[2]:8.2f}  |  {err_rsite:6.2f}  "
                  f"{pred_direct[0]:10.2f} {pred_direct[1]:10.2f} {pred_direct[2]:10.2f}  |  {err_direct:7.2f}")

        frame_idx += len(pred_actions)
        chunk_count += 1


if __name__ == "__main__":
    main()
