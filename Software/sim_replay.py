"""
MuJoCo Simulation Replay for xArm7

Replays a recorded trajectory (iPhone ARKit camera poses) in MuJoCo simulation.
The iPhone and EE are rigidly connected, so delta camera pose in world frame
equals delta EE pose in world frame. We compute world-frame deltas from the
iPhone trajectory and apply them directly to the simulated EE.

Gripper: gripper_width in CSV is ArUco marker distance (meters).
  Larger value = more open, smaller value = more closed.
  Mapped to xArm gripper range and then to MuJoCo actuator [0, 255].

Usage:
    python sim_replay.py --csv Data/episode_xxxx/trajectory.csv
    python sim_replay.py --csv Data/episode_xxxx/trajectory.csv --speed_scale 0.3

Requires Python 3.13 (mujoco not yet compatible with 3.14):
    "C:\\Users\\20492\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" sim_replay.py --csv ...
"""

import csv
import time
import argparse
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation

import mujoco
import mujoco.viewer


# ─── Trajectory loading ─────────────────────────────────────────────

CSV_PATH = r"Data\episode_0015_2026-03-29--18-13-04\trajectory.csv"


def load_trajectory(csv_path):
    """Load trajectory CSV and return list of frame dicts."""
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'timestamp': float(row['timestamp']),
                'tx': float(row['tx']),
                'ty': float(row['ty']),
                'tz': float(row['tz']),
                'rx': float(row['qx']),  # rotation vector, not quaternion
                'ry': float(row['qy']),
                'rz': float(row['qz']),
                'gripper_width': float(row['gripper_width']),
            })
    return frames


def pose_to_matrix(tx, ty, tz, rx, ry, rz):
    """Convert translation + rotation vector to 4x4 SE(3) matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def load_hand_eye_calibration():
    """Load T_cam_to_ee from hand-eye calibration result."""
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration", "hand_eye_result_umi.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        T = np.array(cal['tx_cam_to_ee'])
        print(f"  Loaded hand-eye calibration from {cal_path}")
        return T
    else:
        print(f"  WARNING: No hand-eye calibration found at {cal_path}")
        return np.eye(4)


def compute_body_deltas(frames, T_cam_to_ee=None):
    """
    Compute EE body-frame deltas from camera trajectory.

    1. Compute camera body-frame delta: delta_cam = inv(T_cam_i) @ T_cam_{i+1}
    2. Convert to EE body-frame delta via conjugation:
       delta_ee = T_cam_to_ee @ delta_cam @ inv(T_cam_to_ee)

    Apply as: T_ee_new = T_ee_current @ delta_ee
    """
    if T_cam_to_ee is None:
        T_cam_to_ee = np.eye(4)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    matrices = [pose_to_matrix(f['tx'], f['ty'], f['tz'], f['rx'], f['ry'], f['rz'])
                for f in frames]
    deltas = []
    for i in range(len(matrices) - 1):
        delta_cam = np.linalg.inv(matrices[i]) @ matrices[i + 1]
        delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
        deltas.append(delta_ee)
    return deltas


# ─── MuJoCo helpers ─────────────────────────────────────────────────

def get_ee_pose(model, data, site_name="link_tcp"):
    """Get current end-effector pose as 4x4 SE(3) matrix."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site_id].copy()
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3).copy()
    return T


def ik_step(model, data, target_pos, target_mat, site_name="link_tcp",
            joint_names=None, damping=1e-3, max_steps=200, tol=1e-4):
    """
    Iterative IK using damped least squares (Levenberg-Marquardt).
    Solves for joint angles that place the site at the target pose.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    if joint_names is None:
        joint_names = [f"joint{i}" for i in range(1, 8)]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in joint_names]
    dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]
    ndof = len(dof_ids)

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    for _ in range(max_steps):
        mujoco.mj_forward(model, data)

        # Position error
        pos_err = target_pos - data.site_xpos[site_id]
        # Orientation error (from rotation matrix difference)
        current_mat = data.site_xmat[site_id].reshape(3, 3)
        rot_err_mat = target_mat @ current_mat.T
        rot_err = Rotation.from_matrix(rot_err_mat).as_rotvec()

        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err) < tol:
            break

        # Compute Jacobian
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)

        # Extract columns for our joints only
        J = np.zeros((6, ndof))
        for i, dof in enumerate(dof_ids):
            J[:3, i] = jac_pos[:, dof]
            J[3:, i] = jac_rot[:, dof]

        # Damped least squares: dq = J^T (J J^T + lambda I)^{-1} err
        JJT = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Apply
        for i, jid in enumerate(joint_ids):
            data.qpos[model.jnt_qposadr[jid]] += dq[i]

        # Clip to joint limits
        for jid in joint_ids:
            qadr = model.jnt_qposadr[jid]
            lo = model.jnt_range[jid, 0]
            hi = model.jnt_range[jid, 1]
            if model.jnt_limited[jid]:
                data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)

    mujoco.mj_forward(model, data)
    return np.linalg.norm(err)


def gripper_width_to_ctrl(width_m, gripper_min, gripper_max):
    """
    Map gripper width (meters, from ArUco) to MuJoCo control value.

    gripper_width: ArUco marker distance in meters.
      Large = open, small = closed.
    xArm gripper: 0-850 range, large = open.
    MuJoCo actuator: 0-255, 0=open, 255=closed.

    Steps:
      1. Normalize width to [0, 1] where 0=closed, 1=open
      2. Map to MuJoCo: open_ratio=1 -> ctrl=0 (open), open_ratio=0 -> ctrl=255 (closed)
    """
    if gripper_max <= gripper_min:
        return 0.0  # no calibration, default open

    # Normalize: 0=closed, 1=open
    open_ratio = np.clip((width_m - gripper_min) / (gripper_max - gripper_min), 0.0, 1.0)

    # MuJoCo: 0=open, 255=closed
    return (1.0 - open_ratio) * 255.0


# ─── Main replay ────────────────────────────────────────────────────

def trim_imu_frames(frames, threshold=0.5, min_closed=30):
    """Trim IMU adjustment frames: find first gripper closed->open transition."""
    closed_count = 0
    for i, f in enumerate(frames):
        if f['gripper_width'] < threshold:
            closed_count += 1
        else:
            if closed_count >= min_closed:
                print(f"  Trimmed {i} IMU adjustment frames (gripper opened at frame {i})")
                return frames[i:]
            closed_count = 0
    return frames


def replay(args):
    # Load trajectory
    print(f"Loading trajectory: {args.csv}")
    frames = load_trajectory(args.csv)

    # Auto-trim IMU adjustment frames at the beginning
    frames = trim_imu_frames(frames)

    if args.num_frames and args.num_frames < len(frames):
        frames = frames[:args.num_frames]

    duration = frames[-1]['timestamp'] - frames[0]['timestamp']
    print(f"  Frames: {len(frames)}, Duration: {duration:.2f}s")
    print(f"  Speed scale: {args.speed_scale}x")

    # Load hand-eye calibration and compute EE body-frame deltas
    T_cam_to_ee = load_hand_eye_calibration()
    body_deltas = compute_body_deltas(frames, T_cam_to_ee)
    dts = [frames[i + 1]['timestamp'] - frames[i]['timestamp'] for i in range(len(frames) - 1)]

    # Gripper calibration: determine min/max from data or gripper_range.json
    gripper_min = args.gripper_min
    gripper_max = args.gripper_max

    # Try to load gripper_range.json if min/max not specified
    if gripper_min is None or gripper_max is None:
        cal_path = os.path.join(os.path.dirname(args.csv), "..", "gripper_range.json")
        cal_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_range.json")
        for p in [cal_path, cal_path2]:
            if os.path.exists(p):
                with open(p) as f:
                    cal = json.load(f)
                gripper_min = cal.get("min_width", 0.0)
                gripper_max = cal.get("max_width", 0.085)
                print(f"  Gripper calibration from {p}: min={gripper_min:.4f}m, max={gripper_max:.4f}m")
                break

    # Fallback: auto-detect from trajectory data
    if gripper_min is None or gripper_max is None:
        widths = [f['gripper_width'] for f in frames if f['gripper_width'] > 0]
        if widths:
            gripper_min = min(widths) * 0.9  # add some margin
            gripper_max = max(widths) * 1.1
            print(f"  Gripper auto-detected: min={gripper_min:.4f}m, max={gripper_max:.4f}m")
        else:
            gripper_min = 0.0
            gripper_max = 0.085  # typical xArm gripper max opening
            print(f"  Gripper defaults (no data): min={gripper_min:.4f}m, max={gripper_max:.4f}m")

    # Load MuJoCo model
    model_path = os.path.join(os.path.dirname(__file__), "models", "ufactory_xarm7", "scene.xml")
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Reset to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Set joint actuator controls to hold home position
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_act_names = [f"act{i}" for i in range(1, 8)]
    for act_name, jnt_name in zip(arm_act_names, arm_joint_names):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        data.ctrl[act_id] = data.qpos[model.jnt_qposadr[jnt_id]]

    # Find target visualization mocap body
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_vis")
    target_mocap_id = -1
    if target_body_id >= 0 and model.body_mocapid[target_body_id] >= 0:
        target_mocap_id = model.body_mocapid[target_body_id]

    # Get initial EE pose
    T_ee = get_ee_pose(model, data)
    print(f"  Initial EE pos: ({T_ee[0,3]*1000:.1f}, {T_ee[1,3]*1000:.1f}, {T_ee[2,3]*1000:.1f}) mm")

    scale = args.translation_scale
    print(f"  Translation scale: {scale}x")
    print(f"\n  Press Ctrl+C in viewer to stop.\n")

    # Gripper actuator
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_idx = 0

        while viewer.is_running() and step_idx < len(body_deltas):
            t_start = time.perf_counter()

            # World-frame delta: since iPhone & EE are rigidly connected,
            # the world-frame motion of the iPhone IS the world-frame motion of the EE.
            delta = body_deltas[step_idx].copy()
            delta[:3, 3] *= scale  # scale translation component

            # Apply body-frame delta: T_ee_new = T_ee @ delta
            T_ee = T_ee @ delta

            target_pos = T_ee[:3, 3]
            target_mat = T_ee[:3, :3]

            # Update target visualization sphere
            if target_mocap_id >= 0:
                data.mocap_pos[target_mocap_id] = target_pos.copy()

            # Solve IK
            err = ik_step(model, data, target_pos, target_mat)

            # Set arm actuators to solved joint positions
            for act_name, jnt_name in zip(arm_act_names, arm_joint_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                data.ctrl[act_id] = data.qpos[model.jnt_qposadr[jnt_id]]

            # Gripper: map ArUco width (meters) to MuJoCo ctrl
            gripper_width_m = frames[step_idx + 1]['gripper_width']
            ctrl_val = gripper_width_to_ctrl(gripper_width_m, gripper_min, gripper_max)
            data.ctrl[gripper_act_id] = ctrl_val

            # Step simulation forward to match dt
            dt_target = dts[step_idx] / args.speed_scale
            sim_steps = max(1, int(dt_target / model.opt.timestep))
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)

            viewer.sync()

            if step_idx % 30 == 0:
                ee = get_ee_pose(model, data)
                print(f"  [{step_idx:4d}/{len(body_deltas)}] "
                      f"pos=({ee[0,3]*1000:7.1f}, {ee[1,3]*1000:7.1f}, {ee[2,3]*1000:7.1f}) mm  "
                      f"ik_err={err:.6f}  gripper_w={gripper_width_m:.4f}m  ctrl={ctrl_val:.0f}")

            # Real-time pacing
            elapsed = time.perf_counter() - t_start
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            step_idx += 1

        print(f"\n=== Replay finished ({step_idx} steps) ===")

        # Keep viewer open
        if viewer.is_running():
            print("Replay done. Close viewer window to exit.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Simulation Replay for xArm7")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to trajectory.csv")
    parser.add_argument("--speed_scale", type=float, default=0.5, help="Speed multiplier (default: 0.5)")
    parser.add_argument("--num_frames", type=int, default=None, help="Only replay first N frames")
    parser.add_argument("--translation_scale", type=float, default=1.0,
                        help="Scale factor for translation (default: 1.0)")

    # Gripper calibration
    parser.add_argument("--gripper_min", type=float, default=None,
                        help="Gripper min width in meters (closed). Auto-detected if not set.")
    parser.add_argument("--gripper_max", type=float, default=None,
                        help="Gripper max width in meters (open). Auto-detected if not set.")

    args = parser.parse_args()
    replay(args)


if __name__ == "__main__":
    main()
