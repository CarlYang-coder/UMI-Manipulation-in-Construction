"""
Replay training ee_pose trajectory on xArm7.

Loads a cropped episode, computes ee_poses using the same pipeline as training
(_compute_ee_poses: body-frame delta accumulation from identity), then replays
them on xArm via: T_target = T_home @ R_site_inv @ T_body @ R_site.

Before real robot execution, runs a MuJoCo simulation preview.
After simulation ends, press Enter to proceed or 'q' to abort.

Usage:
    # Replay a specific CSV:
    python replay_ee_pose.py

    # Replay at half speed:
    python replay_ee_pose.py --speed_scale 0.5

    # Skip MuJoCo preview:
    python replay_ee_pose.py --no_sim

    # Dry run (print only):
    python replay_ee_pose.py --dry_run
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import csv as csv_mod
import time
import signal
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# ee_pose_trajectory.csv already has gripper in full [0, 1] (rescaled by
# generate_ee_pose_csv.py), so no inference-side rescale is needed.

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

# ─── Config ───
DATA_DIR = r"D:\UMI_Gripper\Data_Raw\data_crop\episode_0005_2026-04-13--15-45-45\ee_pose_trajectory.csv"
HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]

WORKSPACE_BOUNDS = [
    (100, 700),   # X mm
    (-400, 400),  # Y mm
    (50, 600),    # Z mm
]


# ─── Pose utilities ───

def matrix_to_xarm_pose(T):
    pos = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = np.array(pose[:3]) / 1000.0
    return T


def check_bounds(T, bounds):
    pos_mm = T[:3, 3] * 1000.0
    for axis, (lo, hi) in enumerate(bounds):
        if pos_mm[axis] < lo or pos_mm[axis] > hi:
            return False, pos_mm
    return True, pos_mm


def check_speed(delta, dt, max_trans, max_rot):
    if dt <= 0:
        return False, float('inf'), float('inf')
    trans = np.linalg.norm(delta[:3, 3]) * 1000.0
    trans_speed = trans / dt
    angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
    rot_speed = np.degrees(angle) / dt
    return (trans_speed <= max_trans and rot_speed <= max_rot, trans_speed, rot_speed)


# ─── Load episode ───

def load_episode(csv_path):
    """Load ee_poses directly from an ee_pose_trajectory.csv (produced by
    generate_ee_pose_csv.py). Each CSV row is already in body-frame EE space,
    so no hand-eye calibration is needed here.

    Returns (ee_matrices, body_deltas, gripper_widths, timestamps, dts).
    """
    ee_matrices = []
    gripper_widths = []
    timestamps = []

    with open(csv_path, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec([
                float(row['qx']), float(row['qy']), float(row['qz'])
            ]).as_matrix()
            T[:3, 3] = [float(row['tx']), float(row['ty']), float(row['tz'])]
            ee_matrices.append(T)
            gripper_widths.append(float(row['gripper_width']))
            timestamps.append(float(row['timestamp']))

    # Body-frame deltas: delta[i] = inv(ee[i]) @ ee[i+1]  (still needed by sim_preview)
    body_deltas = [
        np.linalg.inv(ee_matrices[i]) @ ee_matrices[i + 1]
        for i in range(len(ee_matrices) - 1)
    ]

    dts = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    return ee_matrices, body_deltas, gripper_widths, timestamps, dts


# ─── MuJoCo helpers (from replay_trajectory.py) ───

def get_ee_pose(model, data, site_name="link_tcp"):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site_id].copy()
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3).copy()
    return T


def ik_step(model, data, target_pos, target_mat, site_name="link_tcp",
            joint_names=None, damping=1e-3, max_steps=200, tol=1e-4):
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
        pos_err = target_pos - data.site_xpos[site_id]
        current_mat = data.site_xmat[site_id].reshape(3, 3)
        rot_err_mat = target_mat @ current_mat.T
        rot_err = Rotation.from_matrix(rot_err_mat).as_rotvec()

        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        J = np.zeros((6, ndof))
        for i, dof in enumerate(dof_ids):
            J[:3, i] = jac_pos[:, dof]
            J[3:, i] = jac_rot[:, dof]

        JJT = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        for i, jid in enumerate(joint_ids):
            data.qpos[model.jnt_qposadr[jid]] += dq[i]
        for jid in joint_ids:
            qadr = model.jnt_qposadr[jid]
            lo = model.jnt_range[jid, 0]
            hi = model.jnt_range[jid, 1]
            if model.jnt_limited[jid]:
                data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)

    mujoco.mj_forward(model, data)
    return np.linalg.norm(err)


def gripper_width_to_ctrl(gw_normalized):
    """Map normalized gripper width [0,1] to MuJoCo control value [0-255].
    0=open, 255=closed in MuJoCo; 0=closed, 1=open in training data."""
    return (1.0 - np.clip(gw_normalized, 0, 1)) * 255.0


# ─── MuJoCo simulation preview ───

def sim_preview(args, ee_matrices, body_deltas, gripper_widths, dts):
    if not HAS_MUJOCO:
        print("  WARNING: mujoco not installed, skipping simulation preview.")
        return

    model_path = os.path.join(os.path.dirname(__file__), "models", "ufactory_xarm7", "scene.xml")
    print(f"  Loading MuJoCo model: {model_path}")
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

    # Get initial EE pose (in MuJoCo/TCP frame)
    T_ee = get_ee_pose(model, data)
    print(f"  Initial EE pos: ({T_ee[0,3]*1000:.1f}, {T_ee[1,3]*1000:.1f}, {T_ee[2,3]*1000:.1f}) mm")

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

    print("\n=== Simulation Preview ===")
    print("  Close viewer window to finish preview.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_idx = 0
        while viewer.is_running() and step_idx < len(body_deltas):
            t_start = time.perf_counter()

            # Apply body-frame delta (same delta used in training accumulation)
            delta = body_deltas[step_idx]
            T_ee = T_ee @ delta

            target_pos = T_ee[:3, 3]
            target_mat = T_ee[:3, :3]

            if target_mocap_id >= 0:
                data.mocap_pos[target_mocap_id] = target_pos.copy()

            err = ik_step(model, data, target_pos, target_mat)

            for act_name, jnt_name in zip(arm_act_names, arm_joint_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                data.ctrl[act_id] = data.qpos[model.jnt_qposadr[jnt_id]]

            # Gripper (already normalized [0,1])
            gw = gripper_widths[step_idx + 1]
            ctrl_val = gripper_width_to_ctrl(gw)
            data.ctrl[gripper_act_id] = ctrl_val

            # Step simulation
            dt_target = dts[step_idx] / args.speed_scale
            sim_steps = max(1, int(dt_target / model.opt.timestep))
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)
            viewer.sync()

            if step_idx % 30 == 0:
                ee = get_ee_pose(model, data)
                print(f"  [{step_idx:4d}/{len(body_deltas)}] "
                      f"pos=({ee[0,3]*1000:7.1f}, {ee[1,3]*1000:7.1f}, {ee[2,3]*1000:7.1f}) mm  "
                      f"ik_err={err:.6f}  gripper={gw:.3f}")

            elapsed = time.perf_counter() - t_start
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            step_idx += 1

    print(f"\n=== Simulation Preview finished ({step_idx} steps) ===")


# ─── Real robot replay ───

def robot_replay(args, ee_matrices, gripper_widths, timestamps, dts):
    from xarm.wrapper import XArmAPI

    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    N = len(ee_matrices)
    home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]

    print(f"\n[INFO] Connecting to xArm at {args.ip} ...")
    arm = XArmAPI(args.ip)

    def signal_handler(sig, frame):
        print("\n!!! EMERGENCY STOP !!!")
        arm.set_state(4)
        arm.disconnect()
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)

    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    confirm = input("\n  Type 'yes' to move to home and start replay: ")
    if confirm.strip().lower() != 'yes':
        print("Cancelled.")
        arm.disconnect()
        return

    print("\n=== Moving to home position ===")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    print("  Reached home.")

    code, home_pose = arm.get_position()
    if code != 0:
        print(f"ERROR: Failed to get position (code={code})")
        arm.disconnect()
        return
    T_home = xarm_pose_to_matrix(home_pose)
    print(f"  Home EE pose: {[f'{x:.1f}' for x in home_pose]}")

    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(5000)

    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.5)

    dt_default = 1.0 / args.hz

    # Pre-compute all xArm targets and deltas to avoid per-frame overhead
    xarm_targets = []
    for i in range(N):
        T_target = T_home @ R_site_inv @ ee_matrices[i] @ R_site
        xarm_targets.append(T_target)

    print(f"\n=== Replay started ({N} frames, speed_scale={args.speed_scale}) ===")

    for i in range(N):
        t_start = time.perf_counter()

        T_target = xarm_targets[i]

        ok_bounds, pos_mm = check_bounds(T_target, WORKSPACE_BOUNDS)
        if not ok_bounds:
            print(f"\n  [STOP] Out of bounds at frame {i}: "
                  f"({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}) mm")
            break

        xarm_pose = matrix_to_xarm_pose(T_target)
        arm.set_servo_cartesian(xarm_pose)

        gw = float(np.clip(gripper_widths[i], 0.0, 1.0))
        gripper_pos = gw * 850.0
        arm.set_gripper_position(gripper_pos, wait=False)

        if i % 30 == 0:
            print(f"  [{i:4d}/{N}] pos=({xarm_pose[0]:7.1f}, {xarm_pose[1]:7.1f}, {xarm_pose[2]:7.1f}) mm  "
                  f"gw={gw:.3f} -> gripper_pos={gripper_pos:.0f}")

        # Timing: use dts[i] (interval from frame i to i+1), same as replay_trajectory.py
        if i < len(dts):
            dt_target = dts[i] / args.speed_scale
        else:
            dt_target = dt_default
        elapsed = time.perf_counter() - t_start
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)

    print("\n=== Replay finished ===")

    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print("Returning to home ...")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    print("  Reached home.")
    arm.disconnect()
    print("Disconnected.")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Replay training ee_pose on xArm7")
    parser.add_argument("--csv", default=DATA_DIR, help="Path to trajectory.csv")
    parser.add_argument("--ip", default="192.168.1.224", help="xArm IP address")
    parser.add_argument("--speed_scale", type=float, default=0.3, help="Speed multiplier")
    parser.add_argument("--max_speed", type=float, default=500, help="Max translational speed mm/s")
    parser.add_argument("--max_rot_speed", type=float, default=60, help="Max rotational speed deg/s")
    parser.add_argument("--num_frames", type=int, default=None, help="Only replay first N frames")
    parser.add_argument("--dry_run", action="store_true", help="Print without moving robot")
    parser.add_argument("--no_sim", action="store_true", help="Skip MuJoCo simulation preview")
    parser.add_argument("--hz", type=float, default=30.0, help="Replay frequency")
    args = parser.parse_args()

    csv_path = args.csv
    print(f"CSV: {csv_path}")

    # Load ee_poses directly from ee_pose_trajectory.csv (preprocessed)
    ee_matrices, body_deltas, gripper_widths, timestamps, dts = load_episode(csv_path)
    N = len(ee_matrices)
    print(f"  {N} frames loaded")

    if args.num_frames is not None:
        N = min(N, args.num_frames)
        ee_matrices = ee_matrices[:N]
        body_deltas = body_deltas[:N - 1]
        gripper_widths = gripper_widths[:N]
        timestamps = timestamps[:N]
        dts = dts[:N - 1]
        print(f"  Replaying first {N} frames")

    # Print ee_pose range
    positions = np.array([T[:3, 3] * 1000 for T in ee_matrices])
    print(f"  EE pose range (mm): X=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}], "
          f"Y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}], "
          f"Z=[{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")

    # MuJoCo simulation preview
    if not args.no_sim and not args.dry_run:
        sim_preview(args, ee_matrices, body_deltas, gripper_widths, dts)
        confirm = input("\n  Press Enter to proceed to real robot, or 'q' to abort: ")
        if confirm.strip().lower() == 'q':
            print("Aborted.")
            return

    if args.dry_run:
        # Dry run with default T_home
        R_site = np.eye(4)
        R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
        R_site_inv = np.linalg.inv(R_site)

        T_home = np.eye(4)
        T_home[:3, :3] = Rotation.from_euler('xyz', [-179.8, -41.8, -0.2], degrees=True).as_matrix()
        T_home[:3, 3] = np.array([309.1, 3.8, 268.3]) / 1000.0

        print(f"\n=== Dry run ===")
        for i in range(0, N, max(1, N // 20)):
            T_target = T_home @ R_site_inv @ ee_matrices[i] @ R_site
            xarm_pose = matrix_to_xarm_pose(T_target)
            gw = gripper_widths[i]
            print(f"  [{i:4d}/{N}] pos=({xarm_pose[0]:7.1f}, {xarm_pose[1]:7.1f}, {xarm_pose[2]:7.1f}) mm  "
                  f"gripper={gw:.3f}")

        for i in range(N):
            T_target = T_home @ R_site_inv @ ee_matrices[i] @ R_site
            ok, pos_mm = check_bounds(T_target, WORKSPACE_BOUNDS)
            if not ok:
                print(f"\n  [WARN] Frame {i} out of bounds: ({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f})")
                break
        else:
            print(f"\n  All {N} frames within workspace bounds.")
        return

    # Real robot
    robot_replay(args, ee_matrices, gripper_widths, timestamps, dts)


if __name__ == "__main__":
    main()
