"""
UMI-FT Trajectory Replay on xArm7

Replays a recorded trajectory (camera poses from iPhone ARKit) on xArm7
using world-frame SE(3) deltas (consistent with sim_replay.py).

Before real robot execution, runs a MuJoCo simulation preview.
After simulation ends, press Enter to proceed or 'q' to abort.

Usage:
    # Dry run (no robot motion):
    python replay_trajectory.py --csv Data/episode_xxxx/trajectory.csv --dry_run

    # Slow replay (half speed, first 20 frames):
    python replay_trajectory.py --csv Data/episode_xxxx/trajectory.csv --ip 192.168.1.xxx --speed_scale 0.5 --num_frames 20

    # Full replay (skip simulation preview):
    python replay_trajectory.py --csv Data/episode_xxxx/trajectory.csv --ip 192.168.1.xxx --no_sim
"""

import csv
import time
import signal
import argparse
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


CSV_PATH = r"D:\UMI_Gripper\Data_Raw_1\data_crop\episode_0027_2026-04-02--16-30-54\trajectory.csv"


def pose_to_matrix(tx, ty, tz, rx, ry, rz):
    """Convert translation + rotation vector to 4x4 SE(3) matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_xarm_pose(T):
    """Convert 4x4 SE(3) matrix to xArm pose [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg].
    xArm uses intrinsic xyz (= extrinsic ZYX) Euler angles."""
    pos = T[:3, 3] * 1000.0  # meters to mm
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    """Convert xArm pose [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg] to 4x4 SE(3).
    xArm uses intrinsic xyz (= extrinsic ZYX) Euler angles."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]  # mm to meters
    return T


def load_trajectory(csv_path):
    """Load trajectory CSV and return frames list."""
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


def check_speed(delta, dt, max_trans_speed, max_rot_speed):
    """Check if delta transform exceeds speed limits. Returns (ok, trans_speed, rot_speed)."""
    if dt <= 0:
        return False, float('inf'), float('inf')

    trans = np.linalg.norm(delta[:3, 3]) * 1000.0  # meters to mm
    trans_speed = trans / dt  # mm/s

    angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
    rot_speed = np.degrees(angle) / dt  # deg/s

    ok = trans_speed <= max_trans_speed and rot_speed <= max_rot_speed
    return ok, trans_speed, rot_speed


def check_bounds(T, bounds):
    """Check if pose is within workspace bounds (in mm)."""
    pos_mm = T[:3, 3] * 1000.0
    for axis, (lo, hi) in enumerate(bounds):
        if pos_mm[axis] < lo or pos_mm[axis] > hi:
            return False, pos_mm
    return True, pos_mm


# ─── MuJoCo helpers (from sim_replay.py) ─────────────────────────────

def get_ee_pose(model, data, site_name="link_tcp"):
    """Get current end-effector pose as 4x4 SE(3) matrix."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site_id].copy()
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3).copy()
    return T


def ik_step(model, data, target_pos, target_mat, site_name="link_tcp",
            joint_names=None, damping=1e-3, max_steps=200, tol=1e-4):
    """Iterative IK using damped least squares (Levenberg-Marquardt)."""
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


def gripper_width_to_ctrl(width_m, gripper_min, gripper_max):
    """Map gripper width (meters, from ArUco) to MuJoCo control value [0-255]."""
    if gripper_max <= gripper_min:
        return 0.0
    open_ratio = np.clip((width_m - gripper_min) / (gripper_max - gripper_min), 0.0, 1.0)
    return (1.0 - open_ratio) * 255.0


def load_gripper_calibration(csv_path, gripper_min, gripper_max):
    """Load gripper calibration from gripper_range.json or auto-detect from trajectory."""
    if gripper_min is not None and gripper_max is not None:
        return gripper_min, gripper_max

    # Try gripper_range.json
    cal_path = os.path.join(os.path.dirname(csv_path), "..", "gripper_range.json")
    cal_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_range.json")
    for p in [cal_path, cal_path2]:
        if os.path.exists(p):
            with open(p) as f:
                cal = json.load(f)
            gripper_min = cal.get("min_width", 0.0)
            gripper_max = cal.get("max_width", 0.085)
            print(f"  Gripper calibration from {p}: min={gripper_min:.4f}m, max={gripper_max:.4f}m")
            return gripper_min, gripper_max

    return None, None


def auto_detect_gripper_range(frames):
    """Auto-detect gripper min/max from trajectory data."""
    widths = [f['gripper_width'] for f in frames if f['gripper_width'] > 0]
    if widths:
        gripper_min = min(widths) * 0.9
        gripper_max = max(widths) * 1.1
        print(f"  Gripper auto-detected: min={gripper_min:.4f}m, max={gripper_max:.4f}m")
    else:
        gripper_min = 0.0
        gripper_max = 0.085
        print(f"  Gripper defaults: min={gripper_min:.4f}m, max={gripper_max:.4f}m")
    return gripper_min, gripper_max


# ─── Simulation preview ──────────────────────────────────────────────

def sim_preview(args, frames, body_deltas, dts):
    """Run MuJoCo simulation preview (same logic as sim_replay.py)."""
    if not HAS_MUJOCO:
        print("  WARNING: mujoco not installed, skipping simulation preview.")
        return

    # Gripper calibration
    gripper_min, gripper_max = load_gripper_calibration(args.csv, args.gripper_min, args.gripper_max)
    if gripper_min is None or gripper_max is None:
        gripper_min, gripper_max = auto_detect_gripper_range(frames)

    # Load MuJoCo model
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

    # Get initial EE pose
    T_ee = get_ee_pose(model, data)
    print(f"  Initial EE pos: ({T_ee[0,3]*1000:.1f}, {T_ee[1,3]*1000:.1f}, {T_ee[2,3]*1000:.1f}) mm")

    scale = args.translation_scale
    print(f"  Translation scale: {scale}x")

    # Gripper actuator
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

    print("\n=== Simulation Preview ===")
    print("  Close viewer window to finish preview.\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_idx = 0

        while viewer.is_running() and step_idx < len(body_deltas):
            t_start = time.perf_counter()

            # World-frame delta
            delta = body_deltas[step_idx].copy()
            delta[:3, 3] *= scale

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

            # Gripper
            gripper_width_m = frames[step_idx + 1]['gripper_width']
            ctrl_val = gripper_width_to_ctrl(gripper_width_m, gripper_min, gripper_max)
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
                      f"ik_err={err:.6f}  gripper_w={gripper_width_m:.4f}m  ctrl={ctrl_val:.0f}")

            # Real-time pacing
            elapsed = time.perf_counter() - t_start
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            step_idx += 1

        print(f"\n=== Simulation Preview finished ({step_idx} steps) ===")


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

    print(f"  Frames: {len(frames)}")
    duration = frames[-1]['timestamp'] - frames[0]['timestamp']
    print(f"  Duration: {duration:.2f}s")
    print(f"  Speed scale: {args.speed_scale}x (replay duration: {duration / args.speed_scale:.2f}s)")

    # Load hand-eye calibration and compute EE body-frame deltas
    T_cam_to_ee = load_hand_eye_calibration()
    body_deltas = compute_body_deltas(frames, T_cam_to_ee)
    print(f"  Body-frame deltas: {len(body_deltas)}")

    # Compute dt array
    dts = [frames[i + 1]['timestamp'] - frames[i]['timestamp'] for i in range(len(frames) - 1)]

    # Workspace bounds [x, y, z] in mm
    bounds = [
        (args.x_min, args.x_max),
        (args.y_min, args.y_max),
        (args.z_min, args.z_max),
    ]

    if args.dry_run:
        print("\n=== DRY RUN (no robot connection) ===")
        T_current = np.eye(4)
        T_current[:3, 3] = [0.3, 0.0, 0.3]  # 300mm, 0, 300mm in meters

        for i in range(len(body_deltas)):
            delta = body_deltas[i].copy()
            delta[:3, 3] *= args.translation_scale
            T_current = T_current @ delta

            xarm_pose = matrix_to_xarm_pose(T_current)
            gripper = frames[i + 1]['gripper_width']

            ok_speed, t_spd, r_spd = check_speed(body_deltas[i], dts[i] / args.speed_scale, args.max_speed, args.max_rot_speed)
            ok_bounds, pos_mm = check_bounds(T_current, bounds)

            flag = ""
            if not ok_speed:
                flag += " [SPEED LIMIT]"
            if not ok_bounds:
                flag += " [OUT OF BOUNDS]"

            if i % 10 == 0 or flag:
                print(f"  [{i:4d}] pos=({xarm_pose[0]:7.1f}, {xarm_pose[1]:7.1f}, {xarm_pose[2]:7.1f}) mm  "
                      f"rpy=({xarm_pose[3]:6.1f}, {xarm_pose[4]:6.1f}, {xarm_pose[5]:6.1f}) deg  "
                      f"gripper={gripper:.2f}  v={t_spd:.1f}mm/s{flag}")

        print("\nDry run complete.")
        return

    # --- Simulation preview ---
    if not args.no_sim:
        if HAS_MUJOCO:
            sim_preview(args, frames, body_deltas, dts)
            confirm = input("\nPress Enter to proceed to real robot, or 'q' to quit: ")
            if confirm.strip().lower() == 'q':
                print("Cancelled.")
                return
        else:
            print("\n  WARNING: mujoco not installed, skipping simulation preview.")
            print("  Install with: pip install mujoco")

    # --- Real robot ---
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        print("ERROR: xarm-python-sdk not found. Install with: pip install xarm-python-sdk")
        return

    print(f"\nConnecting to xArm at {args.ip}...")
    arm = XArmAPI(args.ip)

    # Emergency stop handler
    def signal_handler(sig, frame):
        print("\n!!! EMERGENCY STOP !!!")
        arm.emergency_stop()
        arm.disconnect()
        exit(1)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize
    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.clean_warn()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    # Get current pose
    code, current_pose = arm.get_position()
    if code != 0:
        print(f"ERROR: Failed to get position (code={code})")
        arm.disconnect()
        return

    print(f"  Current pose: x={current_pose[0]:.1f}, y={current_pose[1]:.1f}, z={current_pose[2]:.1f} mm")
    print(f"                roll={current_pose[3]:.1f}, pitch={current_pose[4]:.1f}, yaw={current_pose[5]:.1f} deg")
    print(f"\n  Trajectory: {len(body_deltas)} steps, {duration / args.speed_scale:.2f}s at {args.speed_scale}x speed")
    print(f"  Max speed: {args.max_speed} mm/s, {args.max_rot_speed} deg/s")
    print(f"  Workspace: x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}], z=[{args.z_min}, {args.z_max}] mm")

    # Home joint angles (from MuJoCo keyframe, radians -> degrees)
    HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]
    home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]
    print(f"\n  Home joints (deg): [{', '.join(f'{a:.2f}' for a in home_joints_deg)}]")

    confirm = input("\n  Type 'yes' to move to home and start replay: ")
    if confirm.strip().lower() != 'yes':
        print("Cancelled.")
        arm.disconnect()
        return

    # Move to home position slowly (position mode)
    print("\n=== Moving to home position ===")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    print("  Reached home position.")

    # Read EE pose at home as the starting point for replay
    code, home_pose = arm.get_position()
    if code != 0:
        print(f"ERROR: Failed to get position at home (code={code})")
        arm.disconnect()
        return
    print(f"  Home EE pose: x={home_pose[0]:.1f}, y={home_pose[1]:.1f}, z={home_pose[2]:.1f} mm")
    print(f"                roll={home_pose[3]:.1f}, pitch={home_pose[4]:.1f}, yaw={home_pose[5]:.1f} deg")

    # Enable and configure gripper
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(5000)

    # Gripper calibration (same as simulation)
    gripper_min, gripper_max = load_gripper_calibration(args.csv, args.gripper_min, args.gripper_max)
    if gripper_min is None or gripper_max is None:
        gripper_min, gripper_max = auto_detect_gripper_range(frames)

    # Switch to servo mode for trajectory replay
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.5)

    # xArm EE frame and MuJoCo TCP (UMI convention) differ by -90 deg around Z.
    # Body-frame deltas are computed in MuJoCo TCP convention,
    # so we track T_current in that convention and convert when sending to xArm.
    T_current = xarm_pose_to_matrix(home_pose)

    # MuJoCo TCP site has a local rotation (quat="0.7071 0 0 -0.7071") relative
    # to the xArm EE (flange) frame. Body-frame deltas are computed in TCP site
    # convention. To apply them in xArm EE convention, conjugate by R_site:
    #   delta_xarm = R_site @ delta_tcp @ inv(R_site)
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()  # scipy xyzw
    R_site_inv = np.linalg.inv(R_site)

    print("\n=== Replay started ===")
    for i in range(len(body_deltas)):
        t_start = time.perf_counter()

        # Convert delta from TCP site frame to xArm EE frame
        delta = R_site_inv @ body_deltas[i] @ R_site
        delta[:3, 3] *= args.translation_scale
        T_current = T_current @ delta

        # Safety: speed check (use scaled dt to match actual replay speed)
        dt_scaled = dts[i] / args.speed_scale
        ok_speed, t_spd, r_spd = check_speed(body_deltas[i], dt_scaled, args.max_speed, args.max_rot_speed)
        if not ok_speed:
            print(f"\n!!! Speed limit exceeded at frame {i}: {t_spd:.1f} mm/s, {r_spd:.1f} deg/s")
            print("Stopping for safety.")
            break

        # Safety: bounds check
        ok_bounds, pos_mm = check_bounds(T_current, bounds)
        if not ok_bounds:
            print(f"\n!!! Out of bounds at frame {i}: ({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}) mm")
            print("Stopping for safety.")
            break

        # Send directly in xArm EE convention
        xarm_pose = matrix_to_xarm_pose(T_current)
        arm.set_servo_cartesian(xarm_pose)

        # Gripper: normalize with same calibration as simulation, then map to xArm range
        gripper_width = frames[i + 1]['gripper_width']
        open_ratio = np.clip((gripper_width - gripper_min) / (gripper_max - gripper_min), 0.0, 1.0)
        gripper_pos = open_ratio * 850  # xArm: 0=closed, 850=open
        arm.set_gripper_position(gripper_pos, wait=False)

        # Status output
        if i % 30 == 0:
            print(f"  [{i:4d}/{len(body_deltas)}] pos=({xarm_pose[0]:7.1f}, {xarm_pose[1]:7.1f}, {xarm_pose[2]:7.1f}) mm  "
                  f"gripper={gripper_width:.2f}")

        # Maintain timing
        dt_target = dts[i] / args.speed_scale
        elapsed = time.perf_counter() - t_start
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n=== Replay finished ===")

    # Return to position mode and move back to home
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print("Returning to home position...")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    print("  Reached home position.")

    arm.disconnect()
    print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(description="UMI-FT Trajectory Replay on xArm7")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to trajectory.csv")
    parser.add_argument("--ip", default="192.168.1.224", help="xArm IP address")
    parser.add_argument("--speed_scale", type=float, default=0.3, help="Speed multiplier (default: 0.3)")
    parser.add_argument("--max_speed", type=float, default=500, help="Max translational speed mm/s")
    parser.add_argument("--max_rot_speed", type=float, default=60, help="Max rotational speed deg/s")
    parser.add_argument("--num_frames", type=int, default=None, help="Only replay first N frames")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without moving robot")
    parser.add_argument("--translation_scale", type=float, default=1.0, help="Scale factor for translation (default: 0.867)")
    parser.add_argument("--no_sim", action="store_true", help="Skip MuJoCo simulation preview")

    # Gripper calibration (for simulation)
    parser.add_argument("--gripper_min", type=float, default=None, help="Gripper min width in meters (closed)")
    parser.add_argument("--gripper_max", type=float, default=None, help="Gripper max width in meters (open)")

    # Workspace bounds (mm)
    parser.add_argument("--x_min", type=float, default=100, help="Workspace X min (mm)")
    parser.add_argument("--x_max", type=float, default=700, help="Workspace X max (mm)")
    parser.add_argument("--y_min", type=float, default=-400, help="Workspace Y min (mm)")
    parser.add_argument("--y_max", type=float, default=400, help="Workspace Y max (mm)")
    parser.add_argument("--z_min", type=float, default=50, help="Workspace Z min (mm)")
    parser.add_argument("--z_max", type=float, default=600, help="Workspace Z max (mm)")

    args = parser.parse_args()
    replay(args)


if __name__ == "__main__":
    main()
