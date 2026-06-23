"""
Open-loop trajectory replay on xArm7.

Reads a recorded trajectory.csv (camera poses + gripper width),
transforms camera poses to end-effector poses using a known
camera-to-EE calibration, and replays on xArm7.

Usage:
    # Dry run (no robot needed):
    python replay_trajectory.py --episode_dir data/episode_0007_... --dry_run

    # Real replay at half speed:
    python replay_trajectory.py --episode_dir data/episode_0007_... \
        --config config/replay_config.yaml --speed_scale 0.5

    # Full speed:
    python replay_trajectory.py --episode_dir data/episode_0007_... \
        --config config/replay_config.yaml
"""

import os
import csv
import math
import time
import signal
import argparse

import yaml
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory(csv_path: str) -> dict:
    """
    Load trajectory.csv and reconstruct full quaternions.

    Returns dict with:
        timestamps: (N,) float array, seconds
        poses: (N, 4, 4) array of homogeneous transforms T_world_cam
        gripper_widths: (N,) float array, normalized [0, 1]
    """
    timestamps = []
    positions = []
    quaternions = []
    gripper_widths = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            tx, ty, tz = float(row["tx"]), float(row["ty"]), float(row["tz"])
            qx, qy, qz = float(row["qx"]), float(row["qy"]), float(row["qz"])
            # Reconstruct qw (ARKit convention: qw >= 0)
            qw = math.sqrt(max(0.0, 1.0 - qx**2 - qy**2 - qz**2))
            positions.append([tx, ty, tz])
            quaternions.append([qx, qy, qz, qw])  # scipy uses [x, y, z, w]
            gripper_widths.append(float(row["gripper_width"]))

    timestamps = np.array(timestamps)
    positions = np.array(positions)       # (N, 3) meters
    quaternions = np.array(quaternions)   # (N, 4) [x, y, z, w]

    # Build 4x4 homogeneous transforms
    N = len(timestamps)
    poses = np.zeros((N, 4, 4))
    rotations = Rotation.from_quat(quaternions)  # scipy expects [x, y, z, w]
    for i in range(N):
        poses[i, :3, :3] = rotations[i].as_matrix()
        poses[i, :3, 3] = positions[i]
        poses[i, 3, 3] = 1.0

    return {
        "timestamps": timestamps,
        "poses": poses,
        "gripper_widths": np.array(gripper_widths),
    }


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def build_transform(translation: list, quaternion: list) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from translation + quaternion.

    Args:
        translation: [x, y, z] in meters
        quaternion: [qw, qx, qy, qz] (config convention)

    Returns:
        4x4 ndarray
    """
    T = np.eye(4)
    # Config stores [qw, qx, qy, qz], scipy expects [qx, qy, qz, qw]
    qw, qx, qy, qz = quaternion
    rot = Rotation.from_quat([qx, qy, qz, qw])
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = translation
    return T


def compute_ee_trajectory(cam_poses: np.ndarray,
                          T_cam_to_ee: np.ndarray,
                          robot_start_pose: np.ndarray) -> np.ndarray:
    """
    Convert camera poses to EE poses using relative-pose method.

    1. T_world_ee_i = T_world_cam_i @ T_cam_to_ee
    2. delta_i = inv(T_world_ee_0) @ T_world_ee_i
    3. T_target_i = robot_start_pose @ delta_i

    Args:
        cam_poses: (N, 4, 4) camera poses in ARKit world frame
        T_cam_to_ee: (4, 4) camera-to-EE rigid transform
        robot_start_pose: (4, 4) robot EE pose at start (in robot base frame)

    Returns:
        (N, 4, 4) EE target poses in robot base frame
    """
    N = cam_poses.shape[0]

    # Camera poses → EE poses (still in ARKit world frame)
    ee_poses_world = np.array([cam_poses[i] @ T_cam_to_ee for i in range(N)])

    # Relative to first frame
    T_world_ee_0_inv = np.linalg.inv(ee_poses_world[0])
    ee_targets = np.zeros_like(ee_poses_world)
    for i in range(N):
        delta = T_world_ee_0_inv @ ee_poses_world[i]
        ee_targets[i] = robot_start_pose @ delta

    return ee_targets


def pose_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 homogeneous transform to [x, y, z, roll, pitch, yaw].

    Returns:
        [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]
    """
    xyz_mm = T[:3, 3] * 1000.0  # meters → millimeters
    rot = Rotation.from_matrix(T[:3, :3])
    rpy = rot.as_euler("xyz")  # radians
    return np.concatenate([xyz_mm, rpy])


# ---------------------------------------------------------------------------
# Trajectory resampling
# ---------------------------------------------------------------------------

def resample_trajectory(timestamps: np.ndarray,
                        ee_poses: np.ndarray,
                        gripper_widths: np.ndarray,
                        rate_hz: float,
                        speed_scale: float = 1.0) -> dict:
    """
    Resample trajectory to uniform rate using SLERP + linear interpolation.

    Args:
        timestamps: (N,) original timestamps
        ee_poses: (N, 4, 4) EE poses
        gripper_widths: (N,) gripper widths
        rate_hz: target frequency in Hz
        speed_scale: replay speed multiplier (0.5 = half speed)

    Returns dict with:
        timestamps: (M,) resampled relative timestamps (seconds)
        poses_xyzrpy: (M, 6) [x_mm, y_mm, z_mm, roll, pitch, yaw]
        gripper_widths: (M,) resampled gripper widths
    """
    # Relative timestamps, scaled
    t_rel = (timestamps - timestamps[0]) / speed_scale
    duration = t_rel[-1]
    dt = 1.0 / rate_hz
    t_new = np.arange(0, duration + dt, dt)

    # Extract translations (mm) and rotations
    N = len(timestamps)
    translations_mm = np.array([ee_poses[i, :3, 3] * 1000.0 for i in range(N)])
    rotations = Rotation.from_matrix(ee_poses[:, :3, :3])

    # SLERP for rotations
    slerp = Slerp(t_rel, rotations)
    rot_new = slerp(t_new)

    # Linear interpolation for translations
    trans_new = np.zeros((len(t_new), 3))
    for axis in range(3):
        trans_new[:, axis] = np.interp(t_new, t_rel, translations_mm[:, axis])

    # Linear interpolation for gripper
    gw_new = np.interp(t_new, t_rel, gripper_widths)

    # Combine to xyzrpy
    rpy_new = rot_new.as_euler("xyz")  # (M, 3) radians
    poses_xyzrpy = np.hstack([trans_new, rpy_new])

    return {
        "timestamps": t_new,
        "poses_xyzrpy": poses_xyzrpy,
        "gripper_widths": gw_new,
    }


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyChecker:
    """Validates EE poses against workspace and motion limits."""

    def __init__(self, config: dict):
        safety = config["safety"]
        self.ws_min = np.array(safety["workspace_min"], dtype=float)
        self.ws_max = np.array(safety["workspace_max"], dtype=float)
        self.max_step = safety["max_step_distance"]
        self.max_speed = safety["max_speed"]

    def check_workspace(self, xyz_mm: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        return bool(np.all(xyz_mm >= self.ws_min) and np.all(xyz_mm <= self.ws_max))

    def check_step(self, prev_xyz: np.ndarray, curr_xyz: np.ndarray) -> bool:
        """Check if step distance is within limit."""
        dist = np.linalg.norm(curr_xyz - prev_xyz)
        return dist <= self.max_step

    def check_speed(self, prev_xyz: np.ndarray, curr_xyz: np.ndarray, dt: float) -> bool:
        """Check if Cartesian speed is within limit."""
        if dt <= 0:
            return False
        speed = np.linalg.norm(curr_xyz - prev_xyz) / dt
        return speed <= self.max_speed

    def validate_trajectory(self, poses_xyzrpy: np.ndarray,
                            timestamps: np.ndarray) -> tuple:
        """
        Pre-validate entire trajectory.

        Returns:
            (ok: bool, message: str)
        """
        for i in range(len(poses_xyzrpy)):
            xyz = poses_xyzrpy[i, :3]
            if not self.check_workspace(xyz):
                return False, (
                    f"Frame {i}: position [{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm "
                    f"outside workspace bounds"
                )
            if i > 0:
                prev_xyz = poses_xyzrpy[i - 1, :3]
                dt = timestamps[i] - timestamps[i - 1]
                if not self.check_step(prev_xyz, xyz):
                    dist = np.linalg.norm(xyz - prev_xyz)
                    return False, (
                        f"Frame {i}: step distance {dist:.1f} mm exceeds limit "
                        f"{self.max_step} mm"
                    )
                if not self.check_speed(prev_xyz, xyz, dt):
                    speed = np.linalg.norm(xyz - prev_xyz) / dt
                    return False, (
                        f"Frame {i}: speed {speed:.1f} mm/s exceeds limit "
                        f"{self.max_speed} mm/s"
                    )
        return True, "Trajectory passed all safety checks"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_dry_run(resampled: dict, safety: SafetyChecker):
    """Print trajectory summary without commanding robot."""
    poses = resampled["poses_xyzrpy"]
    ts = resampled["timestamps"]
    gw = resampled["gripper_widths"]

    print("\n=== DRY RUN ===")
    print(f"Total frames: {len(poses)}")
    print(f"Duration: {ts[-1]:.2f} s")
    print(f"Rate: {1.0 / (ts[1] - ts[0]):.1f} Hz")

    xyz = poses[:, :3]
    print(f"\nPosition range (mm):")
    print(f"  X: [{xyz[:, 0].min():.1f}, {xyz[:, 0].max():.1f}]")
    print(f"  Y: [{xyz[:, 1].min():.1f}, {xyz[:, 1].max():.1f}]")
    print(f"  Z: [{xyz[:, 2].min():.1f}, {xyz[:, 2].max():.1f}]")

    rpy_deg = np.degrees(poses[:, 3:])
    print(f"\nOrientation range (deg):")
    print(f"  Roll:  [{rpy_deg[:, 0].min():.1f}, {rpy_deg[:, 0].max():.1f}]")
    print(f"  Pitch: [{rpy_deg[:, 1].min():.1f}, {rpy_deg[:, 1].max():.1f}]")
    print(f"  Yaw:   [{rpy_deg[:, 2].min():.1f}, {rpy_deg[:, 2].max():.1f}]")

    print(f"\nGripper width: [{gw.min():.3f}, {gw.max():.3f}]")

    # Step distances
    diffs = np.diff(xyz, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    print(f"\nStep distances (mm): mean={step_dists.mean():.2f}, "
          f"max={step_dists.max():.2f}")

    # Safety check
    ok, msg = safety.validate_trajectory(poses, ts)
    print(f"\nSafety check: {'PASS' if ok else 'FAIL'} — {msg}")

    # Print first/last few poses
    print(f"\nFirst 3 poses (x, y, z mm | roll, pitch, yaw deg | gripper):")
    for i in range(min(3, len(poses))):
        p = poses[i]
        r = np.degrees(p[3:])
        print(f"  [{i:4d}] {p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f} | "
              f"{r[0]:7.1f} {r[1]:7.1f} {r[2]:7.1f} | {gw[i]:.3f}")
    print("  ...")
    for i in range(max(0, len(poses) - 3), len(poses)):
        p = poses[i]
        r = np.degrees(p[3:])
        print(f"  [{i:4d}] {p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f} | "
              f"{r[0]:7.1f} {r[1]:7.1f} {r[2]:7.1f} | {gw[i]:.3f}")

    return ok


def replay_on_xarm(resampled: dict, config: dict):
    """
    Execute trajectory on xArm7 using servo mode.

    Requires xarm-python-sdk: pip install xarm-python-sdk
    """
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        print("Error: xarm-python-sdk not installed. Run: pip install xarm-python-sdk")
        return

    poses = resampled["poses_xyzrpy"]
    ts = resampled["timestamps"]
    gw = resampled["gripper_widths"]
    xarm_cfg = config["xarm"]
    gripper_cfg = config["gripper"]

    # Connect
    arm = XArmAPI(xarm_cfg["ip"])
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    # Register emergency stop on Ctrl+C
    def emergency_handler(signum, frame):
        print("\n!!! Emergency stop !!!")
        try:
            arm.set_mode(0)
            arm.set_state(state=0)
            arm.emergency_stop()
        except Exception:
            pass
        raise SystemExit(1)

    signal.signal(signal.SIGINT, emergency_handler)

    # Enable gripper
    arm.set_gripper_enable(True)
    arm.set_gripper_speed(gripper_cfg["speed"])

    # Move to start pose (blocking, slow)
    start = poses[0]
    print(f"Moving to start pose: "
          f"[{start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f}] mm, "
          f"[{np.degrees(start[3]):.1f}, {np.degrees(start[4]):.1f}, "
          f"{np.degrees(start[5]):.1f}] deg")
    arm.set_position(
        x=start[0], y=start[1], z=start[2],
        roll=np.degrees(start[3]),
        pitch=np.degrees(start[4]),
        yaw=np.degrees(start[5]),
        speed=50, wait=True
    )

    # Set gripper to initial width
    gp = int(gw[0] * (gripper_cfg["max_pos"] - gripper_cfg["min_pos"])
             + gripper_cfg["min_pos"])
    arm.set_gripper_position(gp, wait=True)

    input("\nPress Enter to start replay...")

    # Switch to servo mode
    arm.set_mode(1)
    arm.set_state(state=0)
    time.sleep(0.1)

    rate_hz = xarm_cfg["servo_rate_hz"]
    dt = 1.0 / rate_hz

    print(f"Replaying {len(poses)} frames at {rate_hz} Hz...")
    t_start = time.perf_counter()

    for i in range(len(poses)):
        t_target = t_start + ts[i]

        # Send EE pose
        p = poses[i]
        arm.set_servo_cartesian(
            [p[0], p[1], p[2], p[3], p[4], p[5]],
            is_radian=True
        )

        # Send gripper command (non-blocking, every 5th frame to avoid flooding)
        if i % 5 == 0:
            gp = int(gw[i] * (gripper_cfg["max_pos"] - gripper_cfg["min_pos"])
                     + gripper_cfg["min_pos"])
            arm.set_gripper_position(gp, wait=False)

        # Check arm error state
        if arm.has_err_warn:
            print(f"\n!!! Arm error at frame {i}, stopping !!!")
            break

        # Precise timing: spin-wait for target time
        while time.perf_counter() < t_target:
            pass

    # Cleanup
    elapsed = time.perf_counter() - t_start
    print(f"\nReplay complete. {len(poses)} frames in {elapsed:.2f}s")

    arm.set_mode(0)
    arm.set_state(state=0)
    arm.disconnect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay trajectory on xArm7")
    parser.add_argument("--episode_dir", required=True,
                        help="Path to episode directory (contains trajectory.csv)")
    parser.add_argument("--config", default="config/replay_config.yaml",
                        help="Path to replay config YAML")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print trajectory info without commanding robot")
    parser.add_argument("--speed_scale", type=float, default=1.0,
                        help="Replay speed multiplier (0.5 = half speed)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start replay from this frame index")
    parser.add_argument("--end_frame", type=int, default=-1,
                        help="End replay at this frame index (-1 = all)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load trajectory
    csv_path = os.path.join(args.episode_dir, "trajectory.csv")
    if not os.path.exists(csv_path):
        print(f"Error: trajectory.csv not found in {args.episode_dir}")
        return
    print(f"Loading trajectory: {csv_path}")
    traj = load_trajectory(csv_path)
    print(f"Loaded {len(traj['timestamps'])} frames")

    # Slice frame range
    s = args.start_frame
    e = args.end_frame if args.end_frame > 0 else len(traj["timestamps"])
    traj["timestamps"] = traj["timestamps"][s:e]
    traj["poses"] = traj["poses"][s:e]
    traj["gripper_widths"] = traj["gripper_widths"][s:e]
    print(f"Using frames [{s}:{e}] ({len(traj['timestamps'])} frames)")

    # Build camera-to-EE transform
    cam_ee_cfg = config["camera_to_ee"]
    T_cam_to_ee = build_transform(cam_ee_cfg["translation"], cam_ee_cfg["quaternion"])
    print(f"Camera-to-EE transform:\n{T_cam_to_ee}")

    # Determine robot start pose
    if args.dry_run:
        # In dry run, use identity as robot start pose (relative motion only)
        robot_start = np.eye(4)
        # Place at a reasonable default position for visualization
        robot_start[:3, 3] = [0.4, 0.0, 0.3]  # 400mm forward, 300mm up
    else:
        # In real mode, will read from arm after connecting
        # For now, compute with identity; actual pose read happens in replay_on_xarm
        robot_start = np.eye(4)
        robot_start[:3, 3] = [0.4, 0.0, 0.3]  # placeholder

    # Compute EE trajectory
    ee_poses = compute_ee_trajectory(traj["poses"], T_cam_to_ee, robot_start)

    # Resample
    rate_hz = config["xarm"]["servo_rate_hz"]
    resampled = resample_trajectory(
        traj["timestamps"], ee_poses, traj["gripper_widths"],
        rate_hz, args.speed_scale
    )
    print(f"Resampled to {len(resampled['timestamps'])} frames at {rate_hz} Hz")

    # Safety checker
    safety = SafetyChecker(config)

    if args.dry_run or config["safety"]["dry_run"]:
        replay_dry_run(resampled, safety)
    else:
        # Pre-validate
        ok, msg = safety.validate_trajectory(
            resampled["poses_xyzrpy"], resampled["timestamps"]
        )
        if not ok:
            print(f"Safety check FAILED: {msg}")
            print("Aborting. Fix the issue or adjust safety limits in config.")
            return
        print(f"Safety check passed: {msg}")
        replay_on_xarm(resampled, config)


if __name__ == "__main__":
    main()
