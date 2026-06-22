"""
Pose verification script.

While replaying a CSV trajectory on xArm, simultaneously read iPhone camera pose
via Record3D USB stream. Compare frame-by-frame:
  1. CSV ee_pose (computed from CSV camera trajectory)
  2. Live ee_pose (computed from iPhone Record3D camera trajectory)

Both should be in the same body-frame space as training data.
If they match, the live camera path is correct for policy input.

Usage:
    python verify_pose.py --csv D:/UMI_Gripper/Data_Raw/data_crop/episode_0000_2026-04-02--16-05-58/trajectory.csv --ip 192.168.1.224
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import signal
import argparse
import csv
import json
from threading import Event
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umi_dataset import (
    pose_to_matrix, matrix_to_pose_7d, load_hand_eye_calibration,
)
from imu_init import imu_init_sequence


# ─── xArm helpers (from replay_trajectory.py) ──────────────────────

def matrix_to_xarm_pose(T):
    pos = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return T


# ─── Record3D streamer ──────────────────────────────────────────────

class Record3DStreamer:
    def __init__(self):
        from record3d import Record3DStream
        self.event = Event()
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped

    def _on_new_frame(self):
        self.event.set()

    def _on_stream_stopped(self):
        print("[Record3D] Stream stopped")

    def connect(self):
        from record3d import Record3DStream
        devs = Record3DStream.get_connected_devices()
        if len(devs) == 0:
            raise RuntimeError("No Record3D device found")
        self.session.connect(devs[0])
        print("[Record3D] Connected")

    def wait_for_frame(self, timeout=1.0):
        got = self.event.wait(timeout=timeout)
        self.event.clear()
        return got

    def get_camera_pose(self):
        return self.session.get_camera_pose()

    def get_rgb_frame(self):
        import cv2
        rgb = self.session.get_rgb_frame()
        if rgb is None:
            return None
        if self.session.get_device_type() == 0:  # TrueDepth
            rgb = cv2.flip(rgb, 1)
        # Rotate 90 deg counter-clockwise to convert portrait -> landscape
        rgb = np.rot90(rgb, k=1).copy()
        return rgb


# ─── Default paths (edit here) ───────────────────────────────────────

# List of CSVs to replay sequentially (calibration + validation set)
# Sorted by safety margin (biggest margin = safest first)
DEFAULT_CSVS = [
    r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0005_2026-04-02--16-09-40/trajectory.csv",  # margin=30mm
    r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0010_2026-04-02--16-13-34/trajectory.csv",  # margin=26mm
    r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0012_2026-04-02--16-14-52/trajectory.csv",  # margin=13mm
    r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0009_2026-04-02--16-12-50/trajectory.csv",  # margin=11mm
    r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0019_2026-04-02--16-21-04/trajectory.csv",  # margin=7mm
    r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0000_2026-04-02--16-05-58/trajectory.csv",  # margin=6mm
]
DEFAULT_IP = "192.168.1.224"
DEFAULT_CALIBRATION = r"D:/UMI_Gripper/calibration/hand_eye_result_umi.json"
DEFAULT_OUTPUT = r"D:/UMI_Gripper/train/pose_calibration_data.npz"


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs='+', default=DEFAULT_CSVS,
                        help="CSV trajectories to replay sequentially")
    parser.add_argument("--ip", default=DEFAULT_IP, help="xArm IP")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output npz file with collected pairs")
    parser.add_argument("--num_frames", type=int, default=-1,
                        help="Number of frames per episode (default: all)")
    parser.add_argument("--hz", type=float, default=15.0)
    args = parser.parse_args()

    # ─── Load calibration ───
    T_cam_to_ee = load_hand_eye_calibration(args.calibration)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    # R_site
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    # ─── Connect Record3D ───
    streamer = Record3DStreamer()
    streamer.connect()
    time.sleep(0.5)

    # ─── Connect xArm ───
    from xarm.wrapper import XArmAPI
    print(f"Connecting to xArm at {args.ip}")
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

    HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]
    home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]

    # Enable gripper once
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(5000)

    confirm = input(f"Will replay {len(args.csvs)} episode(s). Type 'yes' to start: ")
    if confirm.strip().lower() != 'yes':
        arm.disconnect()
        return

    # IMU init: move to fixed pose, show live feed, confirm, then go to home
    if not imu_init_sequence(arm, streamer, home_joints_deg, show_display=True):
        print("[INFO] IMU init aborted or failed")
        arm.disconnect()
        return

    # Fully close any residual OpenCV windows
    import cv2
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    time.sleep(0.5)

    dt = 1.0 / args.hz

    # Storage for all collected pairs across episodes
    all_pairs = []  # list of dicts {episode, frame, csv_pose, live_pose}

    # Output dir for saved RGB frames
    rgb_save_root = Path(args.output).parent / "verify_pose_rgb"
    rgb_save_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] RGB frames will be saved to: {rgb_save_root}")

    for ep_idx, csv_path in enumerate(args.csvs):
        print(f"\n{'='*60}")
        print(f"=== Episode {ep_idx + 1}/{len(args.csvs)}: {Path(csv_path).name} ===")
        print(f"{'='*60}")

        ep_rgb_dir = rgb_save_root / f"episode_{ep_idx:02d}"
        ep_rgb_dir.mkdir(parents=True, exist_ok=True)

        # Load CSV
        csv_frames = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_frames.append({
                    'tx': float(row['tx']), 'ty': float(row['ty']), 'tz': float(row['tz']),
                    'rx': float(row['qx']), 'ry': float(row['qy']), 'rz': float(row['qz']),
                    'gripper_width': float(row['gripper_width']),
                })

        n_frames = len(csv_frames) if args.num_frames < 0 else min(args.num_frames, len(csv_frames))
        print(f"Loaded {len(csv_frames)} CSV frames, will replay {n_frames}")

        # Compute CSV ee_poses
        cam_mats_csv = [pose_to_matrix(f['tx'], f['ty'], f['tz'], f['rx'], f['ry'], f['rz'])
                        for f in csv_frames[:n_frames]]
        csv_ee_poses = []
        T_ee = np.eye(4)
        csv_ee_poses.append(matrix_to_pose_7d(T_ee, csv_frames[0]['gripper_width']))
        for i in range(1, n_frames):
            delta_cam = np.linalg.inv(cam_mats_csv[i - 1]) @ cam_mats_csv[i]
            delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
            T_ee = T_ee @ delta_ee
            csv_ee_poses.append(matrix_to_pose_7d(T_ee, csv_frames[i]['gripper_width']))

        # Compute body-frame deltas
        body_deltas = []
        for i in range(n_frames - 1):
            delta_cam = np.linalg.inv(cam_mats_csv[i]) @ cam_mats_csv[i + 1]
            delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
            body_deltas.append(delta_ee)

        # Reset any error state and move to home
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        print("Moving to home ...")
        ret = arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
        if ret != 0:
            print(f"  [WARN] set_servo_angle returned {ret}, retrying ...")
            arm.clean_error()
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
        time.sleep(0.5)

        _, home_pose = arm.get_position()
        T_current = xarm_pose_to_matrix(home_pose)
        print(f"  Home reached: ({home_pose[0]:.1f}, {home_pose[1]:.1f}, {home_pose[2]:.1f})")

        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.5)

        # Wait for first valid camera frame
        print("Waiting for first Record3D frame ...")
        for _ in range(20):
            if streamer.wait_for_frame(timeout=1.0):
                cp = streamer.get_camera_pose()
                if cp is not None:
                    break
            time.sleep(0.1)

        cp_first = streamer.get_camera_pose()
        T_cam_live_prev = np.eye(4)
        T_cam_live_prev[:3, :3] = Rotation.from_quat(
            [cp_first.qx, cp_first.qy, cp_first.qz, cp_first.qw]).as_matrix()
        T_cam_live_prev[:3, 3] = [cp_first.tx, cp_first.ty, cp_first.tz]

        T_ee_live = np.eye(4)

        # Initial pair (frame 0)
        all_pairs.append({
            'episode': ep_idx,
            'frame': 0,
            'csv_pose': csv_ee_poses[0].copy(),
            'live_pose': matrix_to_pose_7d(T_ee_live, csv_frames[0]['gripper_width']),
        })

        # Replay loop
        for i in range(n_frames - 1):
            t_start = time.perf_counter()

            # Send xArm command
            delta = R_site_inv @ body_deltas[i] @ R_site
            T_current = T_current @ delta
            xarm_pose = matrix_to_xarm_pose(T_current)
            arm.set_servo_cartesian(xarm_pose)

            # Gripper
            gw = float(np.clip(csv_frames[i + 1]['gripper_width'], 0, 1))
            arm.set_gripper_position(gw * 850, wait=False)

            # Read live camera pose
            if streamer.wait_for_frame(timeout=0.5):
                cp = streamer.get_camera_pose()
                if cp is not None:
                    T_cam_live = np.eye(4)
                    T_cam_live[:3, :3] = Rotation.from_quat(
                        [cp.qx, cp.qy, cp.qz, cp.qw]).as_matrix()
                    T_cam_live[:3, 3] = [cp.tx, cp.ty, cp.tz]
                    delta_cam_live = np.linalg.inv(T_cam_live_prev) @ T_cam_live
                    delta_ee_live = T_cam_to_ee @ delta_cam_live @ T_ee_to_cam
                    T_ee_live = T_ee_live @ delta_ee_live
                    T_cam_live_prev = T_cam_live.copy()

                # Save live RGB to disk (no display to avoid GUI issues)
                import cv2
                rgb_np = streamer.get_rgb_frame()
                if rgb_np is not None:
                    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    out_path = ep_rgb_dir / f"{i+1:06d}.png"
                    cv2.imwrite(str(out_path), bgr)

            # Save pair
            csv_ee = csv_ee_poses[i + 1]
            live_ee = matrix_to_pose_7d(T_ee_live, csv_frames[i + 1]['gripper_width'])
            all_pairs.append({
                'episode': ep_idx,
                'frame': i + 1,
                'csv_pose': csv_ee.copy(),
                'live_pose': live_ee.copy(),
            })

            if i % 50 == 0:
                diff_mm = np.linalg.norm(csv_ee[:3] - live_ee[:3]) * 1000
                print(f"  [E{ep_idx} {i:4d}] csv=({csv_ee[0]*1000:.0f},{csv_ee[1]*1000:.0f},{csv_ee[2]*1000:.0f})  "
                      f"live=({live_ee[0]*1000:.0f},{live_ee[1]*1000:.0f},{live_ee[2]*1000:.0f})  diff={diff_mm:.0f}mm")

            elapsed = time.perf_counter() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        print(f"  Episode {ep_idx + 1} done: {n_frames} frames collected")

        # Return to home before next episode (if any)
        if ep_idx < len(args.csvs) - 1:
            print("  Returning to home before next episode ...")

            # Retry forever until success
            attempt = 0
            while True:
                attempt += 1
                # Full reset sequence
                arm.clean_error()
                arm.clean_warn()
                time.sleep(0.3)
                arm.motion_enable(enable=True)
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.8)

                # Check state
                ret, state = arm.get_state()
                if state != 0:
                    print(f"  [Attempt {attempt}] state={state}, forcing state=0")
                    arm.set_state(0)
                    time.sleep(0.5)

                # Try to move home
                speed = 20 if attempt == 1 else 10
                ret = arm.set_servo_angle(angle=home_joints_deg, speed=speed, wait=True)

                if ret == 0:
                    time.sleep(0.5)
                    _, verify_pose = arm.get_position()
                    err_mm = np.linalg.norm(
                        np.array(verify_pose[:3]) - np.array([309.1, 3.8, 268.3]))
                    if err_mm < 20:
                        print(f"  Home OK (attempt {attempt}): {[f'{x:.1f}' for x in verify_pose[:3]]}")
                        break
                    else:
                        print(f"  [Attempt {attempt}] position wrong: {verify_pose[:3]}, retrying ...")
                else:
                    print(f"  [Attempt {attempt}] set_servo_angle ret={ret}, retrying ...")

                time.sleep(1.0)

    # ─── Save all collected pairs ───
    csv_arr = np.array([p['csv_pose'] for p in all_pairs])    # (N, 7)
    live_arr = np.array([p['live_pose'] for p in all_pairs])  # (N, 7)
    ep_arr = np.array([p['episode'] for p in all_pairs])      # (N,)
    frame_arr = np.array([p['frame'] for p in all_pairs])     # (N,)

    np.savez(args.output,
             csv_pose=csv_arr, live_pose=live_arr,
             episode=ep_arr, frame=frame_arr,
             csv_paths=args.csvs)
    print(f"\n[INFO] Saved {len(all_pairs)} pairs to {args.output}")
    print(f"       Episodes: {len(args.csvs)}")

    # Cleanup
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    print("Returning to home ...")
    arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    arm.disconnect()
    import cv2
    cv2.destroyAllWindows()
    print("Disconnected")


if __name__ == "__main__":
    main()
