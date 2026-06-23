"""
Offline R3D Processing Script
Extracts RGB, Depth, Camera Pose, and Gripper Width from Record3D .r3d files.

Usage:
    1. Record on iPhone with Record3D (LiDAR mode)
    2. Export as .r3d, AirDrop/transfer to Mac into r3d_data/ folder
    3. Run: python process_r3d.py

    All .r3d files in r3d_data/ will be processed into data/ as episodes.
"""

import os
import io
import csv
import glob
import json
import zipfile
import argparse
import time

import av
import cv2
import yaml
import lzfse
import numpy as np

from umi_ft.aruco_detector import GripperWidthTracker
from umi_ft.gripper_calibration import GripperCalibrator


def load_aruco_config(config_path: str) -> dict:
    """Load ArUco configuration from YAML file."""
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


def process_r3d(r3d_path, episode_dir, aruco_config_path, calibration_path,
                rotate=None, zoom=1.0):
    """Process a single .r3d file into an episode directory."""

    print(f"\nProcessing: {r3d_path}")

    # -- Load ArUco tracker and calibration --
    config = load_aruco_config(aruco_config_path)
    tracker = GripperWidthTracker(
        aruco_dict=config["aruco_dict"],
        marker_size_map=config["marker_size_map"],
        left_id=config["left_finger_id"],
        right_id=config["right_finger_id"],
        nominal_z=config["nominal_z"],
        z_tolerance=config["z_tolerance"],
        hold_timeout=config["hold_timeout"],
    )

    gripper_cal = None
    if os.path.exists(calibration_path):
        gripper_cal = GripperCalibrator.load(calibration_path)
        print(f"  Gripper calibration: min={gripper_cal['min_width']:.4f}, "
              f"max={gripper_cal['max_width']:.4f}")

    # -- Open .r3d (ZIP) and read metadata --
    with zipfile.ZipFile(r3d_path, 'r') as z:
        meta = json.loads(z.read('metadata'))

        num_frames = len(meta['poses'])
        rgb_w, rgb_h = meta['w'], meta['h']
        depth_w, depth_h = meta['dw'], meta['dh']
        fps = meta['fps']
        timestamps = meta['frameTimestamps']

        print(f"  RGB: {rgb_w}x{rgb_h}, Depth: {depth_w}x{depth_h}")
        print(f"  FPS: {fps}, Frames: {num_frames}")
        print(f"  Duration: {timestamps[-1]:.1f}s")

        # Create episode directory
        os.makedirs(os.path.join(episode_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(episode_dir, "depth"), exist_ok=True)

        # Open CSV
        csv_path = os.path.join(episode_dir, "trajectory.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "gripper_width",
            "rgb_path", "depth_path"
        ])

        last_valid_gripper_width = 0.0
        base_timestamp = time.time()  # Use current time as base

        for i in range(num_frames):
            if i % 100 == 0:
                print(f"  Processing frame {i}/{num_frames}...")

            # -- Read RGB --
            jpg_data = z.read(f'rgbd/{i}.jpg')
            rgb_bgr = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)

            # -- Read Depth --
            depth_data = z.read(f'rgbd/{i}.depth')
            depth_dec = lzfse.decompress(depth_data)
            depth = np.frombuffer(depth_dec, dtype=np.float32).reshape((depth_h, depth_w))

            # -- Read Pose: [qx, qy, qz, qw, tx, ty, tz] --
            pose = meta['poses'][i]
            tx, ty, tz = pose[4], pose[5], pose[6]
            qx, qy, qz = pose[0], pose[1], pose[2]

            # -- Get per-frame intrinsics --
            intr = meta['perFrameIntrinsicCoeffs'][i]
            K = np.array([
                [intr[0], 0, intr[2]],
                [0, intr[1], intr[3]],
                [0, 0, 1]
            ], dtype=np.float64)

            # -- Apply rotation --
            if rotate == 'cw':
                H, W = rgb_bgr.shape[:2]
                rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_90_CLOCKWISE)
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                K = np.array([
                    [K[1, 1], 0, H - 1 - K[1, 2]],
                    [0, K[0, 0], K[0, 2]],
                    [0, 0, 1]
                ], dtype=np.float64)
            elif rotate == 'ccw':
                H, W = rgb_bgr.shape[:2]
                rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                K = np.array([
                    [K[1, 1], 0, K[1, 2]],
                    [0, K[0, 0], W - 1 - K[0, 2]],
                    [0, 0, 1]
                ], dtype=np.float64)

            # -- Apply zoom --
            if zoom > 1.0:
                H, W = rgb_bgr.shape[:2]
                crop_h, crop_w = int(H / zoom), int(W / zoom)
                y0, x0 = (H - crop_h) // 2, (W - crop_w) // 2
                rgb_bgr = cv2.resize(rgb_bgr[y0:y0+crop_h, x0:x0+crop_w], (W, H))
                depth = cv2.resize(depth[y0:y0+crop_h, x0:x0+crop_w],
                                   (W, H), interpolation=cv2.INTER_NEAREST)
                K_new = K.copy()
                K_new[0, 0] *= zoom
                K_new[1, 1] *= zoom
                K_new[0, 2] = K[0, 2] * zoom - W * (zoom - 1) / 2
                K_new[1, 2] = K[1, 2] * zoom - H * (zoom - 1) / 2
                K = K_new

            # -- ArUco gripper width detection --
            rgb_for_aruco = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            gripper_width = tracker.update(rgb_for_aruco, K)

            if gripper_width is not None:
                last_valid_gripper_width = gripper_width
                if gripper_cal:
                    min_w = gripper_cal["min_width"]
                    max_w = gripper_cal["max_width"]
                    if max_w > min_w:
                        gripper_width = max(0.0, min(1.0,
                            (gripper_width - min_w) / (max_w - min_w)))
                    last_valid_gripper_width = gripper_width

            gw = gripper_width if gripper_width is not None else last_valid_gripper_width

            # -- Save RGB --
            cv2.imwrite(os.path.join(episode_dir, "rgb", f"{i:06d}.png"), rgb_bgr)

            # -- Save Depth as grayscale (fixed range 0-2m, native resolution) --
            depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            gray = np.clip(depth_clean / 2.0 * 255, 0, 255).astype(np.uint8)
            depth_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(episode_dir, "depth", f"{i:06d}.png"), depth_bgr)

            # -- Write CSV row --
            ts = base_timestamp + timestamps[i]
            csv_writer.writerow([
                f"{ts:.6f}",
                f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}",
                f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}",
                f"{gw:.6f}",
                f"rgb/{i:06d}.png",
                f"depth/{i:06d}.png",
            ])

        csv_file.close()

    # -- Compile videos --
    print(f"  Compiling videos (fps={fps})...")
    time.sleep(1.0)

    rgb_files = [os.path.join(episode_dir, "rgb", f"{i:06d}.png") for i in range(num_frames)]
    depth_files = [os.path.join(episode_dir, "depth", f"{i:06d}.png") for i in range(num_frames)]

    compile_video(rgb_files, os.path.join(episode_dir, "rgb.mp4"), fps=fps)
    compile_video(depth_files, os.path.join(episode_dir, "depth.mp4"), fps=fps)

    print(f"  Done: {num_frames} frames → {episode_dir}")


def compile_video(img_files, output_path, fps=30.0):
    """Compile image files into mp4 using PyAV."""
    if not img_files:
        return
    first = cv2.imread(img_files[0])
    if first is None:
        return
    h, w = first.shape[:2]
    w_even = w if w % 2 == 0 else w - 1
    h_even = h if h % 2 == 0 else h - 1
    need_crop = (w_even != w or h_even != h)

    container = av.open(output_path, mode='w')
    stream = container.add_stream('mpeg4', rate=round(fps))
    stream.width = w_even
    stream.height = h_even
    stream.pix_fmt = 'yuv420p'

    for p in img_files:
        img = cv2.imread(p)
        if img is None:
            continue
        if need_crop:
            img = img[:h_even, :w_even]
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process Record3D .r3d files into episode data"
    )
    parser.add_argument("--r3d_dir", type=str, default="r3d_data",
                        help="Directory containing .r3d files")
    parser.add_argument("--file", type=str, default=None,
                        help="Process a single .r3d file instead of entire directory")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="Output directory for episodes")
    parser.add_argument("--aruco_config", type=str, default="config/aruco_config.yaml",
                        help="Path to ArUco configuration")
    parser.add_argument("--calibration", type=str, default="gripper_range.json",
                        help="Path to gripper calibration JSON")
    parser.add_argument("--rotate", choices=["cw", "ccw", "none"], default="ccw",
                        help="Rotate frames 90° (default: ccw for landscape recording, none=no rotation)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Digital zoom factor")
    args = parser.parse_args()

    # Find .r3d files
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        r3d_files = [args.file]
    else:
        r3d_files = sorted(glob.glob(os.path.join(args.r3d_dir, "*.r3d")))
        if not r3d_files:
            print(f"No .r3d files found in {args.r3d_dir}/")
            return

    print(f"Found {len(r3d_files)} .r3d file(s) in {args.r3d_dir}/")
    os.makedirs(args.save_dir, exist_ok=True)

    # Find next episode index
    existing = glob.glob(os.path.join(args.save_dir, "episode_*"))
    start_idx = len(existing)

    for file_idx, r3d_path in enumerate(r3d_files):
        episode_idx = start_idx + file_idx
        # Use r3d filename as timestamp
        r3d_name = os.path.splitext(os.path.basename(r3d_path))[0]
        episode_name = f"episode_{episode_idx:04d}_{r3d_name}"
        episode_dir = os.path.join(args.save_dir, episode_name)

        if os.path.exists(episode_dir):
            print(f"\nSkipping (already exists): {episode_dir}")
            continue

        process_r3d(
            r3d_path=r3d_path,
            episode_dir=episode_dir,
            aruco_config_path=args.aruco_config,
            calibration_path=args.calibration,
            rotate=args.rotate if args.rotate != "none" else None,
            zoom=args.zoom,
        )

    print(f"\nAll done! {len(r3d_files)} episodes processed.")


if __name__ == "__main__":
    main()
