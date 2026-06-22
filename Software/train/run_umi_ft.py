"""
UMI-FT Diffusion Policy Online Rollout on xArm7.

Closed-loop control using trained UMI-FT diffusion policy:
  1. Record3D USB streaming: real-time RGB + Depth + Camera Pose from iPhone
  2. ArUco marker detection for gripper width (same as Demonstration.py)
  3. CLIP ViT-B/32 + Transformer fusion observation encoding
  4. Diffusion Policy action chunk prediction
  5. xArm7 servo-mode execution with safety checks (same as replay_trajectory.py)

Usage:
    # Dry run (no robot, no iPhone - dummy data):
    python run_umi_ft.py --ckpt checkpoints/best.pt --dry_run

    # Full rollout:
    python run_umi_ft.py --ckpt checkpoints/best.pt --ip 192.168.1.224
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

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umi_ft_policy import UMIFTDiffusionPolicy
from umi_dataset import (
    get_rgb_transform, load_hand_eye_calibration, matrix_to_pose_7d,
)
from umi_ft.aruco_detector import GripperWidthTracker
from imu_init import imu_init_sequence


# ─── Default Config ─────────────────────────────────────────────────

CKPT_PATH = r"D:\UMI_Gripper\train\checkpoints\best.pt"
ROBOT_IP = "192.168.1.224"
CALIBRATION_PATH = r"D:\UMI_Gripper\calibration\hand_eye_result_umi.json"
GRIPPER_CAL_PATH = r"D:\UMI_Gripper\gripper_range.json"
ARUCO_CONFIG_PATH = r"D:\UMI_Gripper\config\aruco_config.yaml"
POSE_CALIB_PATH = r"D:\UMI_Gripper\train\pose_calibration_data_transform.npz"

# Safety (same as replay_trajectory.py)
MAX_TRANS_SPEED = 80  # mm/s
MAX_ROT_SPEED = 60.0      # deg/s
WORKSPACE_BOUNDS = [
    (100, 700),   # X mm
    (-400, 400),  # Y mm
    (50, 600),    # Z mm
]

# Home joint angles (from replay_trajectory.py, radians -> degrees)
HOME_JOINTS_RAD = [0, -0.294, 0, 0.174533, 0, -0.261799, 0]

# Control
HZ = 15.0


# ─── Safety checks (from replay_trajectory.py) ─────────────────────

def check_speed(delta, dt, max_trans_speed, max_rot_speed):
    """Check if delta exceeds speed limits. Returns (ok, trans_speed, rot_speed)."""
    if dt <= 0:
        return False, float('inf'), float('inf')
    trans = np.linalg.norm(delta[:3, 3]) * 1000.0  # m -> mm
    trans_speed = trans / dt
    angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
    rot_speed = np.degrees(angle) / dt
    return (trans_speed <= max_trans_speed and rot_speed <= max_rot_speed,
            trans_speed, rot_speed)


def check_bounds(T, bounds):
    """Check if pose is within workspace bounds (mm)."""
    pos_mm = T[:3, 3] * 1000.0
    for axis, (lo, hi) in enumerate(bounds):
        if pos_mm[axis] < lo or pos_mm[axis] > hi:
            return False, pos_mm
    return True, pos_mm


def matrix_to_xarm_pose(T):
    """4x4 SE(3) -> [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]."""
    pos = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]


def xarm_pose_to_matrix(pose):
    """xArm pose -> 4x4 SE(3). pose in [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return T


# ─── Depth processing (from umi_dataset.py, paper spec) ────────────

def process_depth_for_clip(depth_np, depth_clip_m=0.5, img_size=224):
    """Process raw depth (meters, float) to CLIP-compatible 3-channel tensor.
    Paper: clip at 0.5m, copy to 3 channels, CLIP normalize."""
    depth_np = depth_np.astype(np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np.mean(axis=2)

    # Record3D depth may contain NaN/inf for invalid pixels — replace with 0
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Record3D gives depth in meters already
    depth_np = np.clip(depth_np, 0, depth_clip_m) / depth_clip_m  # -> [0, 1]

    # Resize
    depth_resized = cv2.resize(depth_np, (img_size, img_size),
                               interpolation=cv2.INTER_LINEAR)

    # To 3-channel tensor (paper: "emulating a grayscale RGB image")
    depth_tensor = torch.from_numpy(depth_resized).float().unsqueeze(0)  # (1,H,W)
    depth_3ch = depth_tensor.repeat(3, 1, 1)  # (3,H,W)

    # CLIP normalization (same as RGB encoder)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    depth_3ch = (depth_3ch - mean) / std

    return depth_3ch


# ─── ArUco config loading (from Demonstration.py) ──────────────────

def load_aruco_config(config_path):
    """Load ArUco configuration from YAML."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dict_name = config["aruco_dict"]["predefined"]
    aruco_enum = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_enum)

    marker_size_map = {}
    for k, v in config["marker_size_map"].items():
        if k == "default":
            continue
        marker_size_map[int(k)] = float(v)

    gripper_cfg = config["gripper"]
    detection_cfg = config.get("detection", {})

    return {
        "aruco_dict": aruco_dict,
        "marker_size_map": marker_size_map,
        "left_finger_id": gripper_cfg["left_finger_id"],
        "right_finger_id": gripper_cfg["right_finger_id"],
        "nominal_z": gripper_cfg["nominal_z"],
        "z_tolerance": gripper_cfg["z_tolerance"],
        "hold_timeout": detection_cfg.get("hold_last_value_timeout", 0.5),
    }


def load_gripper_calibration(path):
    """Load gripper min/max from calibration JSON."""
    if os.path.exists(path):
        with open(path) as f:
            cal = json.load(f)
        return cal["min_width"], cal["max_width"]
    return 0.015, 0.085  # defaults


# ─── Load policy ────────────────────────────────────────────────────

def load_policy(ckpt_path, device="cuda"):
    """Load trained UMI-FT policy from inference checkpoint.

    Matches the new UMIFTDiffusionPolicy architecture (timm ViT CLIP obs encoder,
    DDIM scheduler, no action-chunk slicing).
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt["config"]
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )

    policy = UMIFTDiffusionPolicy(
        action_dim=config["action_dim"],
        ee_pose_dim=config["action_dim"],
        horizon=config["horizon"],
        n_obs_steps=config["n_obs_steps"],
        img_size=config["img_size"],
        vision_model_name=config.get(
            "vision_model_name", "vit_base_patch32_clip_224.openai"),
        vision_pretrained=config.get("vision_pretrained", False),
        vision_frozen=config.get("vision_frozen", False),
        share_rgb_model=config.get("share_rgb_model", False),
        use_group_norm=config.get("use_group_norm", True),
        downsample_ratio=config.get("downsample_ratio", 32),
        feature_aggregation=config.get("feature_aggregation", "attention_pool_2d"),
        position_encoding=config.get("position_encoding", "learnable"),
        fuse_mode=config.get("fuse_mode", "modality-attention"),
        aug_random_crop_ratio=config.get("aug_random_crop_ratio", 0.95),
        aug_color_jitter=config.get("aug_color_jitter", (0.3, 0.4, 0.5, 0.08)),
        diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 32),
        down_dims=config.get("down_dims", (256, 512, 1024)),
        kernel_size=config.get("kernel_size", 5),
        n_groups=config.get("n_groups", 8),
        cond_predict_scale=config.get("cond_predict_scale", True),
        noise_scheduler=noise_scheduler,
        num_inference_steps=config["num_infer_steps"],
        predict_epsilon=True,
        input_pertub=config.get("input_pertub", 0.1),
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    policy.to(device)

    print(f"[INFO] Loaded policy from {ckpt_path}")
    print(f"       epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"       action_dim={config['action_dim']}, horizon={config['horizon']}")

    return policy, config


# ─── Record3D streaming (from Demonstration.py) ────────────────────

class Record3DStreamer:
    """Record3D USB streaming with event-based frame waiting."""

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
        """Connect to iPhone via Record3D USB."""
        from record3d import Record3DStream
        devs = Record3DStream.get_connected_devices()
        print(f"[Record3D] Found {len(devs)} device(s)")

        if len(devs) == 0:
            raise RuntimeError(
                "No Record3D devices. Check:\n"
                "  1. iPhone connected via USB\n"
                "  2. Record3D app open\n"
                "  3. USB Streaming enabled in Record3D Settings"
            )

        dev = devs[dev_idx]
        self.session.connect(dev)
        self.connected = True
        print(f"[Record3D] Connected to device [{dev_idx}]")

    def wait_for_frame(self, timeout=1.0):
        """Wait for next frame. Returns True if frame available."""
        got = self.event.wait(timeout=timeout)
        self.event.clear()
        return got

    def get_frame(self):
        """Get current RGB, Depth, Camera Pose. Handles TrueDepth flip + rotation."""
        rgb = self.session.get_rgb_frame()       # RGB numpy
        depth = self.session.get_depth_frame()    # float, meters
        camera_pose = self.session.get_camera_pose()

        if rgb is None or depth is None:
            return rgb, depth, camera_pose

        # Flip if TrueDepth camera (front camera)
        if self.session.get_device_type() == self.DEVICE_TYPE_TRUEDEPTH:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)

        # Rotate 90 deg CCW to convert portrait -> landscape
        rgb = np.rot90(rgb, k=1).copy()
        depth = np.rot90(depth, k=1).copy()

        return rgb, depth, camera_pose

    def get_rgb_frame(self):
        """Convenience: return only the processed RGB frame."""
        rgb, _, _ = self.get_frame()
        return rgb

    def get_intrinsic_mat(self):
        """Get camera intrinsic matrix (3x3)."""
        coeffs = self.session.get_intrinsic_mat()
        return np.array([
            [coeffs.fx, 0, coeffs.tx],
            [0, coeffs.fy, coeffs.ty],
            [0, 0, 1]
        ])


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UMI-FT Online Rollout on xArm7")
    parser.add_argument("--replay_frames", type=int, default=0,
                        help="Number of GT frames to replay before policy (default: 100)")
    parser.add_argument("--no_pose_calib", action="store_true",
                        help="Disable pose calibration R (diagnostic mode)")
    args = parser.parse_args()

    # Defaults (edit these directly in the script)
    args.ckpt = CKPT_PATH
    args.ip = ROBOT_IP
    args.calibration = CALIBRATION_PATH
    args.gripper_cal = GRIPPER_CAL_PATH
    args.aruco_config = ARUCO_CONFIG_PATH
    args.dry_run = False
    args.hz = HZ
    args.max_speed = MAX_TRANS_SPEED
    args.max_rot_speed = MAX_ROT_SPEED
    args.show_display = True
    # Replay first N frames of a GT episode before policy takes over
    args.replay_csv = r"D:/UMI_Gripper/Data_Raw/data_crop/episode_0005_2026-04-02--16-09-40/trajectory.csv"

    # Require GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rollout.")
    device = torch.device("cuda")
    dt = 1.0 / args.hz

    # ─── Load policy ───
    policy, config = load_policy(args.ckpt, device)
    n_obs_steps = config["n_obs_steps"]
    img_size = config["img_size"]
    depth_clip_m = config["depth_clip_m"]

    rgb_transform = get_rgb_transform(img_size)

    # ─── Load calibrations ───
    T_cam_to_ee = load_hand_eye_calibration(args.calibration)
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)
    gripper_min, gripper_max = load_gripper_calibration(args.gripper_cal)
    gripper_range = gripper_max - gripper_min
    print(f"[INFO] Gripper range: [{gripper_min:.4f}, {gripper_max:.4f}] m")

    # Load pose calibration transform (live -> csv space)
    # Set USE_POSE_CALIB = False to disable calibration (diagnostic mode)
    USE_POSE_CALIB = not args.no_pose_calib
    pose_calib_R = None
    pose_calib_t = None
    if USE_POSE_CALIB and os.path.exists(POSE_CALIB_PATH):
        calib_data = np.load(POSE_CALIB_PATH)
        pose_calib_R = calib_data['R']
        pose_calib_t = calib_data['t']
        print(f"[INFO] Loaded pose calibration from {POSE_CALIB_PATH}")
        print(f"       R={pose_calib_R.flatten()}")
        print(f"       t={pose_calib_t} mm")
    else:
        if not USE_POSE_CALIB:
            print(f"[DIAG] Pose calibration DISABLED (USE_POSE_CALIB=False)")
        else:
            print(f"[WARN] No pose calibration at {POSE_CALIB_PATH}, using identity")

    # ─── Initialize Record3D + ArUco tracker ───
    streamer = None
    tracker = None
    if not args.dry_run:
        # Record3D
        streamer = Record3DStreamer()
        streamer.connect()

        # Wait for first frame to get intrinsics
        streamer.wait_for_frame()
        K = streamer.get_intrinsic_mat()
        print(f"[INFO] Camera intrinsics:\n{K}")

        # ArUco gripper tracker (same as Demonstration.py)
        aruco_cfg = load_aruco_config(args.aruco_config)
        tracker = GripperWidthTracker(
            aruco_dict=aruco_cfg["aruco_dict"],
            marker_size_map=aruco_cfg["marker_size_map"],
            left_id=aruco_cfg["left_finger_id"],
            right_id=aruco_cfg["right_finger_id"],
            nominal_z=aruco_cfg["nominal_z"],
            z_tolerance=aruco_cfg["z_tolerance"],
            hold_timeout=aruco_cfg["hold_timeout"],
        )

    # ─── Initialize xArm7 (from replay_trajectory.py) ───
    arm = None
    T_current = np.eye(4)  # Current EE pose in world frame (meters)

    # T_home: episode start reference (set after homing, or identity for dry_run)
    T_home = np.eye(4)

    # R_site conjugation: body-frame pose -> xArm world coordinates
    # Verified: T_target = T_home @ R_site_inv @ T_body @ R_site gives 0.0mm error vs GT
    R_site = np.eye(4)
    R_site[:3, :3] = Rotation.from_quat([0, 0, -0.7071, 0.7071]).as_matrix()
    R_site_inv = np.linalg.inv(R_site)

    if not args.dry_run:
        from xarm.wrapper import XArmAPI
        print(f"\n[INFO] Connecting to xArm at {args.ip} ...")
        arm = XArmAPI(args.ip)

        # Emergency stop handler (from replay_trajectory.py)
        def signal_handler(sig, frame):
            print("\n!!! EMERGENCY STOP !!!")
            arm.emergency_stop()
            arm.disconnect()
            exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize (from replay_trajectory.py:472-477)
        arm.motion_enable(enable=True)
        arm.clean_error()
        arm.clean_warn()
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)

        # Home joint angles
        home_joints_deg = [np.degrees(r) for r in HOME_JOINTS_RAD]

        # Move to home
        code, current_pose = arm.get_position()
        print(f"[INFO] Current pose: {[f'{x:.1f}' for x in current_pose]}")

        confirm = input("\n  Type 'yes' to move to home and start rollout: ")
        if confirm.strip().lower() != 'yes':
            print("Cancelled.")
            arm.disconnect()
            return

        # IMU init: move to fixed pose, show live feed, confirm, then go to home
        if not imu_init_sequence(arm, streamer, home_joints_deg, show_display=True):
            print("[INFO] IMU init aborted or failed")
            arm.disconnect()
            return

        # Read home EE pose — this is the reference for relative predictions
        code, home_pose = arm.get_position()
        T_current = xarm_pose_to_matrix(home_pose)
        T_home = T_current.copy()  # Episode start reference
        print(f"[INFO] Home EE pose: {[f'{x:.1f}' for x in home_pose]}")

        # Enable gripper (from replay_trajectory.py:518-520)
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        arm.set_gripper_speed(5000)

        # Switch to servo mode (from replay_trajectory.py:528-530)
        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.5)
    else:
        # Dry run: graceful exit
        running = True
        def signal_handler(sig, frame):
            nonlocal running
            running = False
            print("\n[INFO] Stopping ...")
        signal.signal(signal.SIGINT, signal_handler)

    # ─── EE pose from iPhone camera (body-frame accumulation, same as training) ───
    T_ee_accum = np.eye(4)  # accumulated body-frame ee_pose
    T_cam_prev = None        # previous iPhone camera pose (for delta)

    # ─── Save RGB frames for debugging (disabled by default) ───
    SAVE_ROLLOUT_RGB = False
    rollout_rgb_dir = None
    if SAVE_ROLLOUT_RGB and not args.dry_run:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rollout_rgb_dir = Path(f"D:/UMI_Gripper/train/rollout_rgb/{ts}")
        rollout_rgb_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving RGB to {rollout_rgb_dir}")

    # ─── Observation buffers (start empty, fill with real data) ───
    dummy_rgb = torch.zeros(3, img_size, img_size)
    dummy_depth = torch.zeros(3, img_size, img_size)
    dummy_pose = np.zeros(7, dtype=np.float32)

    rgb_buffer = collections.deque(maxlen=n_obs_steps)
    depth_buffer = collections.deque(maxlen=n_obs_steps)
    pose_buffer = collections.deque(maxlen=n_obs_steps)

    # Pre-fill only for dry_run (no real streamer)
    if args.dry_run:
        for _ in range(n_obs_steps):
            rgb_buffer.append(dummy_rgb)
            depth_buffer.append(dummy_depth)
            pose_buffer.append(dummy_pose)

    # ─── Action queue (receding horizon) ───
    action_queue = collections.deque()
    action_lock = Lock()  # protect action_queue between main loop and inference thread

    # ─── Async inference state ───
    inference_running = Event()  # set while inference thread is active
    inference_obs = [None]       # shared slot for observation dict

    # Receding horizon: policy predicts `horizon` steps (e.g. 20),
    # but we only execute the first EXECUTION_HORIZON and discard the rest,
    # then re-plan with fresh observations (aligned with official UMI-FT).
    EXECUTION_HORIZON = 4  # official: sparse_execution_horizon=4 (overlap ratio ≈ 20%)

    def _inference_worker():
        """Background thread: run policy inference and push actions to queue."""
        obs = inference_obs[0]
        if obs is None:
            return
        t_infer = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(obs)
            # Policy returns full horizon (B, horizon, action_dim); keep only
            # the first EXECUTION_HORIZON steps for receding horizon control.
            actions = result["action"][0, :EXECUTION_HORIZON].cpu().numpy()
        infer_time = time.perf_counter() - t_infer

        with action_lock:
            for a in actions:
                action_queue.append(a)

        print(f"  [Policy] inference={infer_time*1000:.0f}ms, "
              f"executing {len(actions)} of full horizon", flush=True)
        inference_running.clear()

    # ─── EMA smoothing (from Image_DP run_DP_Official.py) ───
    EMA_ALPHA = 0.7  # higher = smoother
    last_xarm_pose = None  # last sent pose, for hold-position during inference

    # Print device info
    print(f"\n[INFO] Device: {device}")
    if device.type == 'cuda':
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    # ─── Warm-up: wait for Record3D to start delivering frames ───
    if streamer is not None:
        print("\n[INFO] Waiting for Record3D frames ...")
        for _ in range(10):
            streamer.wait_for_frame(timeout=2.0)
            rgb_np, _, _ = streamer.get_frame()
            if rgb_np is not None:
                break
            time.sleep(0.1)
        print("  Record3D streaming OK.")

    print(f"=== Policy rollout at {args.hz} Hz ===")
    print(f"    dry_run={args.dry_run}")
    print(f"    max_speed={args.max_speed} mm/s, max_rot_speed={args.max_rot_speed} deg/s")
    print("    Press Ctrl+C to stop\n")

    step = 0
    if args.dry_run:
        running_ref = [True]
        def _check_running():
            return running
    else:
        running_ref = [True]
        def _check_running():
            return running_ref[0]

    while True:
        if args.dry_run and not running:
            break
        if not args.dry_run and not running_ref[0]:
            break

        t0 = time.perf_counter()

        # ─── 1. Capture observation ───
        if streamer is not None:
            if not streamer.wait_for_frame(timeout=0.5):
                print("[WARN] Frame timeout, reusing last observation")
                step += 1
                continue

            rgb_np, depth_np, camera_pose = streamer.get_frame()
            K = streamer.get_intrinsic_mat()

            # Gripper width from xArm (0-850 range -> 0-1 normalized)
            if arm is not None:
                _, gw_xarm = arm.get_gripper_position()
                gw_normalized = np.clip(gw_xarm / 850.0, 0.0, 1.0)
            else:
                gw_normalized = 1.0  # assume open in dry-run

            # Process RGB for CLIP
            rgb_pil = Image.fromarray(rgb_np)  # Record3D gives RGB directly
            rgb_tensor = rgb_transform(rgb_pil)

            # Process Depth for CLIP
            depth_tensor = process_depth_for_clip(depth_np, depth_clip_m, img_size)

            # Compute ee_pose from iPhone camera via body-frame accumulation
            # (same as training data construction in umi_dataset._compute_ee_poses)
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

            # Apply rotation-only calibration: live -> csv space
            # Both live and csv start from origin (identity), only rotation differs.
            # csv_pos = R_calib @ live_pos
            # csv_rot = R_calib @ live_rot
            if pose_calib_R is not None:
                T_body = np.eye(4)
                T_body[:3, 3] = pose_calib_R @ T_ee_accum[:3, 3]
                T_body[:3, :3] = pose_calib_R @ T_ee_accum[:3, :3]
            else:
                T_body = T_ee_accum

            ee_pose_7d = matrix_to_pose_7d(T_body, gw_normalized)

            # Debug: print ee_pose for first few steps
            if step < 5:
                print(f"  [DEBUG ee_pose] step={step} body_pos={T_body[:3,3]*1000} "
                      f"ee_7d={ee_pose_7d[:3]*1000}")
                if arm is not None:
                    _, dbg_pose = arm.get_position()
                    print(f"  [DEBUG xarm] cur={dbg_pose[:3]}")

            # Save live RGB to disk (instead of display to avoid GUI issues)
            if rollout_rgb_dir is not None and rgb_np is not None:
                bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(rollout_rgb_dir / f"{step:06d}.png"), bgr)
        else:
            # Dry run: dummy observations
            rgb_tensor = dummy_rgb
            depth_tensor = dummy_depth
            ee_pose_7d = dummy_pose

        rgb_buffer.append(rgb_tensor)
        depth_buffer.append(depth_tensor)
        pose_buffer.append(ee_pose_7d)

        # ─── Buffer warmup: skip policy until we have n_obs_steps real frames ───
        if len(rgb_buffer) < n_obs_steps:
            if step == 0:
                print("  [INFO] Warming up observation buffers ...", flush=True)
            step += 1
            # Hold position while warming up
            if last_xarm_pose is not None and arm is not None:
                arm.set_servo_cartesian(last_xarm_pose)
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            continue
        elif step == n_obs_steps:
            print(f"  [INFO] Observation buffers ready ({n_obs_steps} real frames).",
                  flush=True)

        # ─── 2. Launch async inference when action queue is running low ───
        with action_lock:
            queue_len = len(action_queue)
        if queue_len <= 1 and not inference_running.is_set():
            # Prepare observation tensors (on main thread, fast)
            inference_obs[0] = {
                "rgb": torch.stack(list(rgb_buffer), dim=0).unsqueeze(0).to(device),
                "depth": torch.stack(list(depth_buffer), dim=0).unsqueeze(0).to(device),
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
            # Policy predicts EE body-frame accumulated pose (from identity).
            # Convert to xArm world coordinates with R_site conjugation:
            #   T_target = T_home @ R_site_inv @ T_pred_body @ R_site
            # (verified in eval_replay.py: 0.0mm error vs GT)
            T_pred_body = np.eye(4)
            T_pred_body[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
            T_pred_body[:3, 3] = action[:3]

            T_target = T_home @ R_site_inv @ T_pred_body @ R_site

            # Delta from current for speed check
            delta = np.linalg.inv(T_current) @ T_target

            # --- Real speed clamping: interpolate toward target ---
            dt_actual = max(dt, 0.001)
            ok_speed, t_spd, r_spd = check_speed(
                delta, dt_actual, args.max_speed, args.max_rot_speed)

            if not ok_speed:
                # Compute clamp ratio: fraction of delta we can safely execute
                ratio = 1.0
                if t_spd > args.max_speed:
                    ratio = min(ratio, args.max_speed / t_spd)
                if r_spd > args.max_rot_speed:
                    ratio = min(ratio, args.max_rot_speed / r_spd)

                # Interpolate: T_clamped = T_current @ slerp(I, delta, ratio)
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

            # Gripper
            gripper_normalized = float(np.clip(action[6], 0, 1))
            gripper_pos = gripper_normalized * 850

            # Compute xarm pose for target
            xarm_pose_raw = matrix_to_xarm_pose(T_target)

            # EMA smoothing (from Image_DP run_DP_Official.py)
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
                # Safety: bounds check
                ok_bounds, pos_mm = check_bounds(T_target, WORKSPACE_BOUNDS)
                if not ok_bounds:
                    print(f"  [WARN] Out of bounds at step {step} — skipping")
                    step += 1
                    continue

                # Send smoothed pose to robot
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
            # No action available (waiting for inference) — hold last position
            if last_xarm_pose is not None and arm is not None:
                arm.set_servo_cartesian(last_xarm_pose)

        step += 1

        # Maintain loop rate
        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # ─── Cleanup ───
    print("\n=== Rollout finished ===")

    if args.show_display:
        cv2.destroyAllWindows()

    if arm is not None:
        # Return to position mode and go home
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
