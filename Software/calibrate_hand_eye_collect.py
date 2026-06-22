"""
Hand-Eye Calibration: Data Collection

Moves xArm7 to a series of predefined poses while recording ARKit camera
poses from iPhone via Record3D. Saves paired data (xArm EE pose + ARKit
camera pose) for offline hand-eye calibration.

Prerequisites:
    - iPhone mounted rigidly on xArm EE (same as demonstration setup)
    - iPhone connected via USB, Record3D open with USB Streaming enabled
    - An ArUco tag (DICT_4X4_50, ID 2, size >= 100mm) placed on the table
      in view of the iPhone camera

Usage:
    python calibrate_hand_eye_collect.py
    python calibrate_hand_eye_collect.py --ip 192.168.1.224 --output calibration/hand_eye_data.json
"""

import os
import time
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from record3d import Record3DStream
import threading


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

    def get_stable_pose(self, num_samples=10, interval=0.05):
        """Average multiple ARKit poses for stability."""
        txs, tys, tzs = [], [], []
        quats = []
        for _ in range(num_samples):
            p = self.get_pose()
            if p is None:
                continue
            txs.append(p['tx'])
            tys.append(p['ty'])
            tzs.append(p['tz'])
            quats.append([p['qx'], p['qy'], p['qz'], p['qw']])
            time.sleep(interval)

        if len(txs) == 0:
            return None

        # Average translation
        avg_t = [np.mean(txs), np.mean(tys), np.mean(tzs)]

        # Average quaternion (simple mean + normalize, fine for small spread)
        avg_q = np.mean(quats, axis=0)
        avg_q /= np.linalg.norm(avg_q)

        return {
            'tx': avg_t[0], 'ty': avg_t[1], 'tz': avg_t[2],
            'qx': avg_q[0], 'qy': avg_q[1], 'qz': avg_q[2], 'qw': avg_q[3],
        }


# Predefined calibration poses: [joint1, ..., joint7] in degrees
# These cover a range of positions and orientations for good calibration
CALIBRATION_POSES_DEG = [
    # Home
    [0, -14.15, 0, 52.08, 0, 66.26, 0],
    # Vary joint 1 (base rotation)
    [20, -14.15, 0, 52.08, 0, 66.26, 0],
    [-20, -14.15, 0, 52.08, 0, 66.26, 0],
    [40, -14.15, 0, 52.08, 0, 66.26, 0],
    [-40, -14.15, 0, 52.08, 0, 66.26, 0],
    # Vary joint 2 (shoulder)
    [0, -30, 0, 52.08, 0, 66.26, 0],
    [0, 5, 0, 52.08, 0, 66.26, 0],
    # Vary joint 4 (elbow)
    [0, -14.15, 0, 30, 0, 66.26, 0],
    [0, -14.15, 0, 70, 0, 66.26, 0],
    # Vary joint 6 (wrist)
    [0, -14.15, 0, 52.08, 0, 40, 0],
    [0, -14.15, 0, 52.08, 0, 90, 0],
    # Combined variations
    [20, -25, 0, 40, 0, 80, 0],
    [-20, -5, 0, 60, 0, 50, 0],
    [30, -20, 0, 45, 0, 70, 15],
    [-30, -10, 0, 55, 0, 60, -15],
]


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration: Data Collection")
    parser.add_argument("--ip", default="192.168.1.224", help="xArm IP address")
    parser.add_argument("--output", default="calibration/hand_eye_data.json",
                        help="Output file for paired pose data")
    parser.add_argument("--speed", type=float, default=15, help="Joint move speed (deg/s)")
    parser.add_argument("--settle_time", type=float, default=2.0,
                        help="Wait time after reaching each pose (seconds)")
    args = parser.parse_args()

    # Connect to iPhone
    arkit = ARKitReader()
    arkit.connect()

    # Wait for ARKit to initialize
    print("Waiting for ARKit to stabilize (3s)...")
    time.sleep(3.0)
    test_pose = arkit.get_pose()
    if test_pose is None:
        print("ERROR: Cannot read ARKit pose.")
        return
    print(f"  ARKit pose OK: tx={test_pose['tx']:.4f}, ty={test_pose['ty']:.4f}, tz={test_pose['tz']:.4f}")

    # Connect to xArm
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        print("ERROR: xarm-python-sdk not found. Install with: pip install xarm-python-sdk")
        return

    print(f"\nConnecting to xArm at {args.ip}...")
    arm = XArmAPI(args.ip)
    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.clean_warn()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    # Collect paired data
    paired_data = []
    num_poses = len(CALIBRATION_POSES_DEG)

    print(f"\n=== Collecting {num_poses} calibration poses ===")
    print(f"  Speed: {args.speed} deg/s, settle time: {args.settle_time}s\n")

    for idx, joints_deg in enumerate(CALIBRATION_POSES_DEG):
        print(f"  [{idx+1}/{num_poses}] Moving to pose: {joints_deg}")

        # Move to pose
        arm.set_servo_angle(angle=joints_deg, speed=args.speed, wait=True)
        print(f"    Reached. Settling for {args.settle_time}s...")
        time.sleep(args.settle_time)

        # Read xArm EE pose
        code, ee_pose = arm.get_position()
        if code != 0:
            print(f"    WARNING: Failed to read EE pose (code={code}), skipping.")
            continue

        # Read xArm joint angles
        code, joint_angles = arm.get_servo_angle()
        if code != 0:
            print(f"    WARNING: Failed to read joint angles (code={code}), skipping.")
            continue

        # Read ARKit pose (averaged for stability)
        arkit_pose = arkit.get_stable_pose(num_samples=20, interval=0.05)
        if arkit_pose is None:
            print(f"    WARNING: Failed to read ARKit pose, skipping.")
            continue

        pair = {
            'index': idx,
            'joint_angles_deg': joint_angles,
            'ee_pose': {
                'x_mm': ee_pose[0], 'y_mm': ee_pose[1], 'z_mm': ee_pose[2],
                'roll_deg': ee_pose[3], 'pitch_deg': ee_pose[4], 'yaw_deg': ee_pose[5],
            },
            'arkit_pose': arkit_pose,
        }
        paired_data.append(pair)

        print(f"    EE: ({ee_pose[0]:.1f}, {ee_pose[1]:.1f}, {ee_pose[2]:.1f}) mm")
        print(f"    ARKit: ({arkit_pose['tx']:.4f}, {arkit_pose['ty']:.4f}, {arkit_pose['tz']:.4f})")

    # Return to home
    print("\nReturning to home position...")
    arm.set_servo_angle(angle=CALIBRATION_POSES_DEG[0], speed=args.speed, wait=True)

    arm.disconnect()
    print("xArm disconnected.")

    # Save data
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(paired_data, f, indent=2)
    print(f"\nSaved {len(paired_data)} paired poses to {args.output}")

    if len(paired_data) < 5:
        print("WARNING: Less than 5 valid poses collected. Calibration may be inaccurate.")
    else:
        print(f"Data collection complete. Run calibrate_hand_eye_solve.py to compute transform.")


if __name__ == "__main__":
    main()
