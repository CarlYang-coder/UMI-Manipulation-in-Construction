"""
Replay trajectory on xArm while recording iPhone ARKit poses.

This script replays a demonstration on the real xArm while simultaneously
recording the ARKit camera pose from an iPhone mounted on the xArm.
The recorded poses are saved to a CSV file for later calibration.

Prerequisites:
    - iPhone mounted on xArm EE (USB connected, Record3D USB Streaming enabled)
    - xArm connected via Ethernet

Usage:
    python replay_with_recording.py
    python replay_with_recording.py --csv Data/episode_xxxx/trajectory.csv --output calibration/xarm_camera_trajectory.csv
"""

import csv
import time
import signal
import argparse
import os
import json
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from record3d import Record3DStream


# ─── ARKit Reader (from calibrate_hand_eye_collect.py) ───────────────

class ARKitReader:
    """Minimal Record3D reader to get ARKit camera poses."""

    def __init__(self):
        self.event = threading.Event()
        self.session = None

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("[Record3D] Stream stopped.")

    def connect(self):
        print("Searching for Record3D devices...")
        devs = Record3DStream.get_connected_devices()
        print(f"  Found {len(devs)} device(s)")
        if len(devs) == 0:
            raise RuntimeError(
                "No devices found. Make sure:\n"
                "  1. iPhone is connected via USB\n"
                "  2. Record3D app is open\n"
                "  3. USB Streaming mode is enabled in Record3D Settings"
            )
        dev = devs[0]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)
        print("  Connected to iPhone.")

    def get_pose(self):
        """Wait for a new frame and return ARKit camera pose as dict."""
        self.event.clear()
        self.event.wait(timeout=5.0)
        if self.session is None:
            return None
        pose = self.session.get_camera_pose()
        return {
            'tx': float(pose.tx),
            'ty': float(pose.ty),
            'tz': float(pose.tz),
            'qx': float(pose.qx),
            'qy': float(pose.qy),
            'qz': float(pose.qz),
            'qw': float(pose.qw),
        }


# ─── Trajectory helpers (from replay_trajectory.py) ─────────────────

CSV_PATH = r"Data\episode_0016_2026-03-29--19-17-48\trajectory.csv"


def pose_to_matrix(tx, ty, tz, rx, ry, rz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_xarm_pose(T):
    pos = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return T


def load_trajectory(csv_path):
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'timestamp': float(row['timestamp']),
                'tx': float(row['tx']),
                'ty': float(row['ty']),
                'tz': float(row['tz']),
                'rx': float(row['qx']),
                'ry': float(row['qy']),
                'rz': float(row['qz']),
                'gripper_width': float(row['gripper_width']),
            })
    return frames


def load_hand_eye_calibration():
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration", "hand_eye_result_umi.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        return np.array(cal['tx_cam_to_ee'])
    cal_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "calibration", "hand_eye_result.json")
    if os.path.exists(cal_path2):
        with open(cal_path2) as f:
            cal = json.load(f)
        return np.array(cal['tx_cam_to_ee'])
    return np.eye(4)


def compute_body_deltas(frames, T_cam_to_ee=None):
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


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Replay on xArm while recording ARKit poses")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to UMI trajectory.csv")
    parser.add_argument("--ip", default="192.168.1.224", help="xArm IP address")
    parser.add_argument("--speed_scale", type=float, default=0.3, help="Speed multiplier")
    parser.add_argument("--translation_scale", type=float, default=0.867, help="Translation scale")
    parser.add_argument("--output", default="calibration/xarm_camera_trajectory.csv",
                        help="Output CSV for recorded ARKit poses")
    parser.add_argument("--num_frames", type=int, default=None, help="Only replay first N frames")
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory: {args.csv}")
    frames = load_trajectory(args.csv)
    if args.num_frames and args.num_frames < len(frames):
        frames = frames[:args.num_frames]

    duration = frames[-1]['timestamp'] - frames[0]['timestamp']
    print(f"  Frames: {len(frames)}, Duration: {duration:.2f}s")

    # Compute body deltas
    T_cam_to_ee = load_hand_eye_calibration()
    body_deltas = compute_body_deltas(frames, T_cam_to_ee)
    dts = [frames[i + 1]['timestamp'] - frames[i]['timestamp'] for i in range(len(frames) - 1)]

    # Connect to iPhone
    arkit = ARKitReader()
    arkit.connect()
    print("Waiting for ARKit to stabilize (2s)...")
    time.sleep(2.0)
    test_pose = arkit.get_pose()
    if test_pose is None:
        print("ERROR: Cannot read ARKit pose.")
        return
    print(f"  ARKit OK: tx={test_pose['tx']:.4f}")

    # Connect to xArm
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        print("ERROR: xarm-python-sdk not found.")
        return

    print(f"\nConnecting to xArm at {args.ip}...")
    arm = XArmAPI(args.ip)

    def signal_handler(sig, frame):
        print("\n!!! EMERGENCY STOP !!!")
        arm.emergency_stop()
        arm.disconnect()
        exit(1)
    signal.signal(signal.SIGINT, signal_handler)

    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.clean_warn()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    # Move to home
    HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]
    home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]

    confirm = input("\n  Type 'yes' to move to home and start replay with recording: ")
    if confirm.strip().lower() != 'yes':
        arm.disconnect()
        return

    print("\n=== Moving to home position ===")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    print("  Reached home position.")

    code, home_pose = arm.get_position()
    if code != 0:
        print(f"ERROR: Failed to get position (code={code})")
        arm.disconnect()
        return
    print(f"  Home EE: ({home_pose[0]:.1f}, {home_pose[1]:.1f}, {home_pose[2]:.1f}) mm")

    # Enable gripper
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(5000)

    # Switch to servo mode
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.5)

    T_current = xarm_pose_to_matrix(home_pose)

    # R_site for delta conversion
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    # Record ARKit poses
    recorded_poses = []

    print("\n=== Replay + Recording started ===")
    for i in range(len(body_deltas)):
        t_start = time.perf_counter()

        # Apply delta
        delta = R_site_inv @ body_deltas[i] @ R_site
        delta[:3, 3] *= args.translation_scale
        T_current = T_current @ delta

        # Send to xArm
        xarm_pose = matrix_to_xarm_pose(T_current)
        arm.set_servo_cartesian(xarm_pose)

        # Record ARKit pose
        arkit_pose = arkit.get_pose()
        if arkit_pose is not None:
            recorded_poses.append({
                'frame_idx': i,
                'timestamp': time.time(),
                **arkit_pose
            })

        if i % 30 == 0:
            print(f"  [{i:4d}/{len(body_deltas)}] pos=({xarm_pose[0]:7.1f}, {xarm_pose[1]:7.1f}, {xarm_pose[2]:7.1f}) mm  "
                  f"arkit_frames={len(recorded_poses)}")

        # Timing
        dt_target = dts[i] / args.speed_scale
        elapsed = time.perf_counter() - t_start
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"\n=== Replay finished ({len(recorded_poses)} ARKit frames recorded) ===")

    # Return to home
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    print("Returning to home position...")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    arm.disconnect()
    print("Disconnected.")

    # Save recorded poses
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_idx', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        writer.writeheader()
        writer.writerows(recorded_poses)

    print(f"Saved {len(recorded_poses)} ARKit poses to {args.output}")


if __name__ == "__main__":
    main()
