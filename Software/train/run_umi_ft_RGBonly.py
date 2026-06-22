"""
RGB-only rollout for UMI-FT (mirrors run_umi_ft.py, but loads the checkpoint
produced by train_umi_ft_RGBonly.py: ResNet18 + ee_pose, no depth, no fusion).

Closed-loop control:
  1. Record3D USB streaming: real-time RGB + Camera Pose from iPhone
     (depth stream is still read for intrinsic info but NOT fed to the policy)
  2. DiffusionUnetVideoPolicy (ResNet18 RGB encoder + TemporalAggregator on ee_pose)
  3. xArm7 servo-mode execution with safety checks

Usage:
    python run_umi_ft_RGBonly.py
    python run_umi_ft_RGBonly.py --replay_frames 30
    python run_umi_ft_RGBonly.py --no_pose_calib
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import signal
import argparse
import json
import collections
from pathlib import Path
from threading import Event, Thread, Lock

import numpy as np
import torch
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms as T

# Paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"D:\Image_DP\video_train")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Image_DP policy + encoder (same as training)
from resnet_video_encoder import ResNet18VideoEncoder
from diffusion_unet_video_policy import DiffusionUnetVideoPolicy

# UMI helpers
from umi_dataset import load_hand_eye_calibration, pose_to_matrix, matrix_to_pose_7d
from imu_init import imu_init_sequence

# ee_pose_trajectory.csv (used for training + policy obs) already has gripper
# rescaled to full [0, 1] by generate_ee_pose_csv.py, so the policy-output path
# uses the simple 0..1 -> 0..850 mapping below.
# The GT replay path still reads the raw trajectory.csv (ArUco-native ~[0.3, 0.9]),
# so we keep gw_to_xarm_pos for that one spot.
from gripper_rescale import gw_to_xarm_pos


# ─── Default Config ─────────────────────────────────────────────────

CKPT_PATH = r"D:\UMI_Gripper\train\checkpoints_rgbonly\best.pt"
ROBOT_IP = "192.168.1.224"
CALIBRATION_PATH = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json"
GRIPPER_CAL_PATH = r"D:\UMI_Gripper\gripper_range.json"
POSE_CALIB_PATH = r"D:\UMI_Gripper\train\pose_calibration_data_transform.npz"

# Safety (same as run_umi_ft.py)
MAX_TRANS_SPEED = 20      # mm/s
MAX_ROT_SPEED = 60.0      # deg/s
WORKSPACE_BOUNDS = [
    (100, 700),   # X mm
    (-400, 400),  # Y mm
    (50, 600),    # Z mm
]

HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]
HZ = 10.0

# ImageNet normalization (matches training — ResNet18 pretrain)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Safety checks ─────────────────────────────────────────────────

def check_speed(delta, dt, max_trans_speed, max_rot_speed):
    if dt <= 0:
        return False, float('inf'), float('inf')
    trans = np.linalg.norm(delta[:3, 3]) * 1000.0
    trans_speed = trans / dt
    angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
    rot_speed = np.degrees(angle) / dt
    return (trans_speed <= max_trans_speed and rot_speed <= max_rot_speed,
            trans_speed, rot_speed)


def check_bounds(T, bounds):
    pos_mm = T[:3, 3] * 1000.0
    for axis, (lo, hi) in enumerate(bounds):
        if pos_mm[axis] < lo or pos_mm[axis] > hi:
            return False, pos_mm
    return True, pos_mm


def matrix_to_xarm_pose(T):
    pos = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return T


def load_gripper_calibration(path):
    if os.path.exists(path):
        with open(path) as f:
            cal = json.load(f)
        return cal["min_width"], cal["max_width"]
    return 0.015, 0.085


def imagenet_rgb_transform_eval(img_size: int):
    """Deterministic transform matching train_umi_ft_RGBonly.py eval mode."""
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Load policy ────────────────────────────────────────────────────

def load_policy_rgbonly(ckpt_path, device="cuda"):
    """Load the RGB-only DiffusionUnetVideoPolicy from a train_umi_ft_RGBonly.py
    inference checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        prediction_type="epsilon",
    )

    rgb_net = ResNet18VideoEncoder(
        out_dim=config["rgb_out_dim"],
        pool=config["rgb_pool"],
        mlp_hidden=config["rgb_mlp_hidden"],
        dropout=config["rgb_dropout"],
        pretrained=True,
        freeze_backbone=config["rgb_freeze_backbone"],
    )

    shape_meta = {
        "action": {"shape": [config["action_dim"]]},
        "obs": {
            "rgb":     {"shape": [3, config["img_size"], config["img_size"]], "type": "rgb"},
            "ee_pose": {"shape": [config["action_dim"]], "type": "lowdim"},
        }
    }

    policy = DiffusionUnetVideoPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        rgb_net=rgb_net,
        horizon=config["horizon"],
        n_action_steps=config["n_action_steps"],
        n_obs_steps=config["n_obs_steps"],
        num_inference_steps=config["num_infer_steps"],
        lowdim_as_global_cond=True,
        predict_epsilon=True,
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    policy.to(device)

    print(f"[INFO] Loaded policy from {ckpt_path}")
    print(f"       epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"       action_dim={config['action_dim']}, horizon={config['horizon']}, "
          f"n_action_steps={config['n_action_steps']}, img_size={config['img_size']}")

    return policy, config


# ─── Record3D streaming (copied from run_umi_ft.py) ─────────────────

class Record3DStreamer:
    DEVICE_TYPE_TRUEDEPTH = 0
    DEVICE_TYPE_LIDAR = 1

    def __init__(self):
        from record3d import Record3DStream
        self.event = Event()
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped
        self.connected = False

    def _on_new_frame(self):
        self.event.set()

    def _on_stream_stopped(self):
        print("[Record3D] Stream stopped.")
        self.connected = False

    def connect(self, dev_idx=0):
        from record3d import Record3DStream
        devs = Record3DStream.get_connected_devices()
        print(f"[Record3D] Found {len(devs)} device(s)")
        if len(devs) == 0:
            raise RuntimeError("No Record3D devices. Check USB connection.")
        dev = devs[dev_idx]
        self.session.connect(dev)
        self.connected = True
        print(f"[Record3D] Connected to device [{dev_idx}]")

    def wait_for_frame(self, timeout=1.0):
        got = self.event.wait(timeout=timeout)
        self.event.clear()
        return got

    def get_frame(self):
        rgb = self.session.get_rgb_frame()
        depth = self.session.get_depth_frame()
        camera_pose = self.session.get_camera_pose()

        if rgb is None:
            return rgb, depth, camera_pose

        if self.session.get_device_type() == self.DEVICE_TYPE_TRUEDEPTH:
            rgb = cv2.flip(rgb, 1)
            if depth is not None:
                depth = cv2.flip(depth, 1)

        rgb = np.rot90(rgb, k=1).copy()
        if depth is not None:
            depth = np.rot90(depth, k=1).copy()

        return rgb, depth, camera_pose

    def get_intrinsic_mat(self):
        coeffs = self.session.get_intrinsic_mat()
        return np.array([
            [coeffs.fx, 0, coeffs.tx],
            [0, coeffs.fy, coeffs.ty],
            [0, 0, 1]
        ])


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UMI-FT RGB-only Online Rollout on xArm7")
    parser.add_argument("--replay_frames", type=int, default=0,
                        help="Number of GT frames to replay before policy")
    parser.add_argument("--no_pose_calib", action="store_true",
                        help="Disable pose calibration R (diagnostic)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run without connecting to robot or iPhone")
    args = parser.parse_args()

    # Hard-coded defaults (same pattern as run_umi_ft.py)
    args.ckpt = CKPT_PATH
    args.ip = ROBOT_IP
    args.calibration = CALIBRATION_PATH
    args.gripper_cal = GRIPPER_CAL_PATH
    args.hz = HZ
    args.max_speed = MAX_TRANS_SPEED
    args.max_rot_speed = MAX_ROT_SPEED
    args.replay_csv = r"D:\UMI_Gripper\Data_Raw\data_crop_ds/episode_0005_2026-04-13--15-45-45/trajectory.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 1.0 / args.hz

    # ─── Load policy ───
    policy, config = load_policy_rgbonly(args.ckpt, device)
    n_obs_steps = config["n_obs_steps"]
    n_action_steps = config["n_action_steps"]
    img_size = config["img_size"]

    rgb_transform = imagenet_rgb_transform_eval(img_size)

    # ─── Load calibrations ───
    T_cam_to_ee = load_hand_eye_calibration(args.calibration)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)
    gripper_min, gripper_max = load_gripper_calibration(args.gripper_cal)
    gripper_range = gripper_max - gripper_min
    print(f"[INFO] Gripper range: [{gripper_min:.4f}, {gripper_max:.4f}] m")

    USE_POSE_CALIB = not args.no_pose_calib
    pose_calib_R = None
    if USE_POSE_CALIB and POSE_CALIB_PATH and os.path.exists(POSE_CALIB_PATH):
        calib_data = np.load(POSE_CALIB_PATH)
        pose_calib_R = calib_data['R']
        print(f"[INFO] Loaded pose calibration from {POSE_CALIB_PATH}")
        print(f"       R={pose_calib_R.flatten()}")
    else:
        if not USE_POSE_CALIB:
            print("[DIAG] Pose calibration DISABLED")
        else:
            print(f"[WARN] No pose calibration at {POSE_CALIB_PATH}, using identity")

    # ─── Record3D (RGB only in the observation; depth stream ignored) ───
    streamer = None
    if not args.dry_run:
        streamer = Record3DStreamer()
        streamer.connect()
        streamer.wait_for_frame()
        K = streamer.get_intrinsic_mat()
        print(f"[INFO] Camera intrinsics:\n{K}")

    # ─── Initialize xArm7 ───
    arm = None
    T_current = np.eye(4)
    T_home = np.eye(4)

    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    if not args.dry_run:
        from xarm.wrapper import XArmAPI
        print(f"\n[INFO] Connecting to xArm at {args.ip} ...")
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

        home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]
        code, current_pose = arm.get_position()
        print(f"[INFO] Current pose: {[f'{x:.1f}' for x in current_pose]}")

        confirm = input("\n  Type 'yes' to move to home and start rollout: ")
        if confirm.strip().lower() != 'yes':
            print("Cancelled.")
            arm.disconnect()
            return

        if not imu_init_sequence(arm, streamer, home_joints_deg, show_display=True):
            print("[INFO] IMU init aborted or failed")
            arm.disconnect()
            return

        code, home_pose = arm.get_position()
        T_current = xarm_pose_to_matrix(home_pose)
        T_home = T_current.copy()
        print(f"[INFO] Home EE pose: {[f'{x:.1f}' for x in home_pose]}")

        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        arm.set_gripper_speed(5000)
        arm.set_gripper_position(850, wait=True)  # start fully open

        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.5)
    else:
        running = True
        def signal_handler(sig, frame):
            nonlocal running
            running = False
            print("\n[INFO] Stopping ...")
        signal.signal(signal.SIGINT, signal_handler)

    # ─── EE pose accumulator ───
    T_ee_accum = np.eye(4)
    T_cam_prev = None

    # ─── Observation buffers (rgb + ee_pose only) ───
    dummy_rgb = torch.zeros(3, img_size, img_size)
    dummy_pose = np.zeros(7, dtype=np.float32)

    rgb_buffer = collections.deque(maxlen=n_obs_steps)
    pose_buffer = collections.deque(maxlen=n_obs_steps)

    if args.dry_run:
        for _ in range(n_obs_steps):
            rgb_buffer.append(dummy_rgb)
            pose_buffer.append(dummy_pose)

    # ─── Action queue ───
    action_queue = collections.deque()
    action_lock = Lock()

    inference_running = Event()
    inference_obs = [None]

    def _inference_worker():
        obs = inference_obs[0]
        if obs is None:
            return
        t_infer = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(obs)
            actions = result["action"][0].cpu().numpy()  # (n_action_steps, 7)
        infer_time = time.perf_counter() - t_infer
        with action_lock:
            for a in actions:
                action_queue.append(a)
        print(f"  [Policy] inference={infer_time*1000:.0f}ms, "
              f"chunk={len(actions)} actions", flush=True)
        inference_running.clear()

    EMA_ALPHA = 0.7
    last_xarm_pose = None

    print(f"\n[INFO] Device: {device}")
    if device.type == 'cuda':
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    # ─── Wait for Record3D ───
    if streamer is not None:
        print("\n[INFO] Waiting for Record3D frames ...")
        for _ in range(10):
            streamer.wait_for_frame(timeout=2.0)
            rgb_np, _, _ = streamer.get_frame()
            if rgb_np is not None:
                break
            time.sleep(0.1)
        print("  Record3D streaming OK.")

    # ─── GT Replay phase ───
    if args.replay_csv is not None and args.replay_frames > 0:
        import csv as csv_mod
        print(f"\n=== GT Replay phase ({args.replay_frames} frames) ===")
        print(f"    CSV: {args.replay_csv}")

        replay_frames = []
        with open(args.replay_csv, 'r') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                replay_frames.append({
                    'tx': float(row['tx']), 'ty': float(row['ty']), 'tz': float(row['tz']),
                    'rx': float(row['qx']), 'ry': float(row['qy']), 'rz': float(row['qz']),
                    'gripper_width': float(row['gripper_width']),
                })

        n_replay = min(args.replay_frames, len(replay_frames) - 1)

        cam_mats = [pose_to_matrix(f['tx'], f['ty'], f['tz'], f['rx'], f['ry'], f['rz'])
                    for f in replay_frames[:n_replay + 1]]

        gripper_range_val = gripper_max - gripper_min
        replay_dt = 1.0 / args.hz
        print(f"    Replaying {n_replay} frames at {args.hz} Hz ...")

        for i in range(n_replay):
            t_start = time.perf_counter()
            delta_cam = np.linalg.inv(cam_mats[i]) @ cam_mats[i + 1]
            delta_ee = T_cam_to_ee @ delta_cam @ T_ee_to_cam
            T_ee_accum = T_ee_accum @ delta_ee

            delta_xarm = R_site_inv @ delta_ee @ R_site
            T_current = T_current @ delta_xarm
            xarm_pose = matrix_to_xarm_pose(T_current)

            gw_raw = replay_frames[i + 1]['gripper_width']
            gw_norm = np.clip((gw_raw - gripper_min) / gripper_range_val, 0, 1)
            # gw_norm is in training-data space ([~0.29, ~0.97]); feed as-is to policy obs
            # but rescale to full [0, 850] when commanding the real gripper
            gripper_pos = gw_to_xarm_pos(gw_norm)

            ee_pose_csv = matrix_to_pose_7d(T_ee_accum, gw_norm)

            if not args.dry_run:
                arm.set_servo_cartesian(xarm_pose)
                arm.set_gripper_position(gripper_pos, wait=False)

            ee_pose_7d = ee_pose_csv

            if streamer is not None and streamer.wait_for_frame(timeout=0.1):
                rgb_np, _, _ = streamer.get_frame()
                rgb_pil = Image.fromarray(rgb_np)
                rgb_tensor = rgb_transform(rgb_pil)
            else:
                rgb_tensor = dummy_rgb

            rgb_buffer.append(rgb_tensor)
            pose_buffer.append(ee_pose_7d)

            last_xarm_pose = list(xarm_pose)

            elapsed = time.perf_counter() - t_start
            if elapsed < replay_dt:
                time.sleep(replay_dt - elapsed)

        T_ee_replay_end = T_ee_accum.copy()
        print(f"    GT Replay done. T_ee_accum pos = {T_ee_accum[:3,3]*1000} mm")
        print(f"    Handing off to policy ...\n")

        T_ee_accum = np.eye(4)
        T_cam_prev = None
    else:
        T_ee_replay_end = None

    print(f"=== Policy rollout at {args.hz} Hz ===")
    print(f"    dry_run={args.dry_run}")
    print(f"    max_speed={args.max_speed} mm/s, max_rot_speed={args.max_rot_speed} deg/s")
    print("    Press Ctrl+C to stop\n")

    step = 0
    if args.dry_run:
        running_ref = [True]
    else:
        running_ref = [True]

    while True:
        if args.dry_run and not running:
            break
        if not args.dry_run and not running_ref[0]:
            break

        t0 = time.perf_counter()

        # ─── 1. Capture observation (RGB + camera pose; no depth used by policy) ───
        if streamer is not None:
            if not streamer.wait_for_frame(timeout=0.5):
                print("[WARN] Frame timeout, reusing last observation")
                step += 1
                continue

            rgb_np, _, camera_pose = streamer.get_frame()

            if arm is not None:
                _, gw_xarm = arm.get_gripper_position()
                # Training data has gripper in [0, 1]; xArm feedback matches directly.
                gw_normalized = float(np.clip(gw_xarm / 850.0, 0.0, 1.0))
            else:
                gw_normalized = 1.0

            rgb_pil = Image.fromarray(rgb_np)
            rgb_tensor = rgb_transform(rgb_pil)

            # Accumulate body-frame ee_pose from camera delta (same as training)
            T_cam_live = np.eye(4)
            T_cam_live[:3, :3] = Rotation.from_quat([
                camera_pose.qx, camera_pose.qy,
                camera_pose.qz, camera_pose.qw]).as_matrix()
            T_cam_live[:3, 3] = [camera_pose.tx, camera_pose.ty, camera_pose.tz]

            if T_cam_prev is None:
                T_cam_prev = T_cam_live.copy()
            else:
                delta_cam_live = np.linalg.inv(T_cam_prev) @ T_cam_live
                delta_ee_live = T_cam_to_ee @ delta_cam_live @ T_ee_to_cam
                T_ee_accum = T_ee_accum @ delta_ee_live
                T_cam_prev = T_cam_live.copy()

            if pose_calib_R is not None:
                T_body = np.eye(4)
                T_body[:3, 3] = pose_calib_R @ T_ee_accum[:3, 3]
                T_body[:3, :3] = pose_calib_R @ T_ee_accum[:3, :3]
            else:
                T_body = T_ee_accum

            if T_ee_replay_end is not None:
                T_body = T_ee_replay_end @ T_body

            ee_pose_7d = matrix_to_pose_7d(T_body, gw_normalized)

            if step < 5:
                print(f"  [DEBUG ee_pose] step={step} body_pos={T_body[:3,3]*1000} "
                      f"ee_7d={ee_pose_7d[:3]*1000}")
                if arm is not None:
                    _, dbg_pose = arm.get_position()
                    print(f"  [DEBUG xarm] cur={dbg_pose[:3]}")
        else:
            rgb_tensor = dummy_rgb
            ee_pose_7d = dummy_pose

        rgb_buffer.append(rgb_tensor)
        pose_buffer.append(ee_pose_7d)

        # ─── Buffer warmup ───
        if len(rgb_buffer) < n_obs_steps:
            if step == 0:
                print("  [INFO] Warming up observation buffers ...", flush=True)
            step += 1
            if last_xarm_pose is not None and arm is not None:
                arm.set_servo_cartesian(last_xarm_pose)
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            continue
        elif step == n_obs_steps:
            print(f"  [INFO] Observation buffers ready ({n_obs_steps} real frames).",
                  flush=True)

        # ─── 2. Launch async inference ───
        with action_lock:
            queue_len = len(action_queue)
        if queue_len <= 1 and not inference_running.is_set():
            inference_obs[0] = {
                "rgb": torch.stack(list(rgb_buffer), dim=0).unsqueeze(0).to(device),
                "ee_pose": torch.from_numpy(
                    np.stack(list(pose_buffer), axis=0)
                ).float().unsqueeze(0).to(device),
            }
            inference_running.set()
            Thread(target=_inference_worker, daemon=True).start()

        # ─── 3. Execute next action ───
        with action_lock:
            has_action = len(action_queue) > 0
            action = action_queue.popleft() if has_action else None

        if action is not None:
            T_pred_body = np.eye(4)
            T_pred_body[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
            T_pred_body[:3, 3] = action[:3]

            T_target = T_home @ R_site_inv @ T_pred_body @ R_site
            delta = np.linalg.inv(T_current) @ T_target

            dt_actual = max(dt, 0.001)
            ok_speed, t_spd, r_spd = check_speed(
                delta, dt_actual, args.max_speed, args.max_rot_speed)

            if not ok_speed:
                ratio = 1.0
                if t_spd > args.max_speed:
                    ratio = min(ratio, args.max_speed / t_spd)
                if r_spd > args.max_rot_speed:
                    ratio = min(ratio, args.max_rot_speed / r_spd)

                delta_trans = delta[:3, 3] * ratio
                delta_rot_full = Rotation.from_matrix(delta[:3, :3])
                delta_rotvec = delta_rot_full.as_rotvec() * ratio
                delta_clamped = np.eye(4)
                delta_clamped[:3, :3] = Rotation.from_rotvec(delta_rotvec).as_matrix()
                delta_clamped[:3, 3] = delta_trans
                T_target = T_current @ delta_clamped

                if step < 10:
                    print(f"  [CLAMP] step={step} speed={t_spd:.0f}mm/s "
                          f"ratio={ratio:.2f}", flush=True)

            # Policy output is already in rescaled [0, 1]; map directly to [0, 850].
            gripper_pos = float(np.clip(action[6], 0.0, 1.0)) * 850.0

            xarm_pose_raw = matrix_to_xarm_pose(T_target)

            if last_xarm_pose is not None:
                xarm_pose_smoothed = [
                    EMA_ALPHA * last + (1.0 - EMA_ALPHA) * raw
                    for last, raw in zip(last_xarm_pose, xarm_pose_raw)
                ]
            else:
                xarm_pose_smoothed = list(xarm_pose_raw)

            if args.dry_run:
                T_current = T_target.copy()
                last_xarm_pose = xarm_pose_smoothed
                if step % 10 == 0:
                    print(f"  [{step:04d}] pose=[{xarm_pose_smoothed[0]:.1f}, "
                          f"{xarm_pose_smoothed[1]:.1f}, {xarm_pose_smoothed[2]:.1f}]  "
                          f"gripper={gripper_pos:.0f}")
            else:
                ok_bounds, pos_mm = check_bounds(T_target, WORKSPACE_BOUNDS)
                if not ok_bounds:
                    print(f"  [WARN] Out of bounds at step {step} — skipping")
                    step += 1
                    continue

                T_current = T_target.copy()
                last_xarm_pose = xarm_pose_smoothed
                arm.set_servo_cartesian(xarm_pose_smoothed)
                arm.set_gripper_position(gripper_pos, wait=False)

                if step % 30 == 0:
                    elapsed_now = time.perf_counter() - t0
                    hz_actual = 1.0 / max(elapsed_now, 1e-6)
                    print(f"  [{step:04d}] pos=({xarm_pose_smoothed[0]:.1f}, "
                          f"{xarm_pose_smoothed[1]:.1f}, {xarm_pose_smoothed[2]:.1f})  "
                          f"gripper={gripper_pos:.0f}  hz={hz_actual:.1f}")
        else:
            if last_xarm_pose is not None and arm is not None:
                arm.set_servo_cartesian(last_xarm_pose)

        step += 1

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # ─── Cleanup ───
    print("\n=== Rollout finished ===")

    if arm is not None:
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)

        print("Returning to home position ...")
        home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]
        arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
        print("  Reached home.")

        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
