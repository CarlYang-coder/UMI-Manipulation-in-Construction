"""
UMI-FT Data Collection Script
Collects RGB, Depth, Camera Pose, Timestamp, and Gripper Width
from an iPhone via Record3D USB streaming.

Gripper width is measured via ArUco markers (DICT_4X4_50) on the gripper
fingers, following UMI's visual-based approach.

Usage:
    1. Install: pip install record3d opencv-python opencv-contrib-python numpy pyyaml
    2. Print ArUco markers (DICT_4X4_50, ID 0 and 1) and attach to gripper fingers.
    3. On iPhone: Open Record3D -> Settings -> Enable "USB Streaming mode"
    4. Connect iPhone via USB cable.

    Calibrate gripper range (first time):
        python Demonstration.py --calibrate

    Collect data:
        python Demonstration.py

    Controls:
        r : Start / Stop recording
        q : Quit
"""

import os
import csv
import glob
import time
import argparse
import numpy as np
import cv2
import yaml
from threading import Event
from datetime import datetime

try:
    from record3d import Record3DStream
except ImportError:
    raise ImportError(
        "record3d package not found. Install with: pip install record3d"
    )

from umi_ft.aruco_detector import GripperWidthTracker
from umi_ft.gripper_calibration import GripperCalibrator


def load_aruco_config(config_path: str) -> dict:
    """Load ArUco configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build OpenCV ArUco dictionary
    dict_name = config["aruco_dict"]["predefined"]
    aruco_enum = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_enum)

    # Build marker size map with integer keys
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


class DataCollector:
    """
    Streams RGB, Depth, Camera Pose from iPhone via Record3D,
    detects gripper width via ArUco markers, and records all data.
    """

    def __init__(self, save_dir="data", dev_idx=0,
                 aruco_config_path="config/aruco_config.yaml",
                 calibration_path="gripper_range.json",
                 show_aruco_debug=False, rotate=None, zoom=1.0):
        self.save_dir = save_dir
        self.dev_idx = dev_idx
        self.show_aruco_debug = show_aruco_debug
        self.calibration_path = calibration_path
        self.rotate = rotate  # 'cw', 'ccw', or None
        self.zoom = zoom      # digital zoom factor (1.0 = no zoom)

        # Record3D stream
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

        # Recording state
        self.is_recording = False
        self.episode_idx = 0
        self.frame_idx = 0
        self.episode_dir = None
        self.episode_data = []

        # Load ArUco config and create tracker
        config = load_aruco_config(aruco_config_path)
        self.tracker = GripperWidthTracker(
            aruco_dict=config["aruco_dict"],
            marker_size_map=config["marker_size_map"],
            left_id=config["left_finger_id"],
            right_id=config["right_finger_id"],
            nominal_z=config["nominal_z"],
            z_tolerance=config["z_tolerance"],
            hold_timeout=config["hold_timeout"],
        )

        # Load gripper calibration if available
        self.gripper_cal = None
        if os.path.exists(calibration_path):
            self.gripper_cal = GripperCalibrator.load(calibration_path)
            print(f"Loaded gripper calibration: min={self.gripper_cal['min_width']:.4f}, "
                  f"max={self.gripper_cal['max_width']:.4f}")

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("[Record3D] Stream stopped.")

    def connect(self):
        """Connect to iPhone via Record3D USB."""
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

        for i, dev in enumerate(devs):
            print(f"  [{i}] ID: {dev.product_id}, UDID: {dev.udid}")

        if len(devs) <= self.dev_idx:
            raise RuntimeError(f"Device index {self.dev_idx} not available.")

        dev = devs[self.dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)
        print(f"  Connected to device [{self.dev_idx}]")

    def _apply_rotation(self, rgb, depth, K):
        """Rotate rgb+depth 90° and update intrinsic matrix K accordingly."""
        if self.rotate is None:
            return rgb, depth, K
        H, W = rgb.shape[:2]
        if self.rotate == 'cw':
            code = cv2.ROTATE_90_CLOCKWISE
            K_new = np.array([
                [K[1,1], 0,      H - 1 - K[1,2]],
                [0,      K[0,0], K[0,2]         ],
                [0,      0,      1               ]
            ], dtype=np.float64)
        else:  # ccw
            code = cv2.ROTATE_90_COUNTERCLOCKWISE
            K_new = np.array([
                [K[1,1], 0,      K[1,2]         ],
                [0,      K[0,0], W - 1 - K[0,2] ],
                [0,      0,      1               ]
            ], dtype=np.float64)
        return cv2.rotate(rgb, code), cv2.rotate(depth, code), K_new

    def _apply_zoom(self, rgb, depth, K):
        """Center-crop and upscale to simulate digital zoom. Updates K accordingly."""
        z = self.zoom
        if z <= 1.0:
            return rgb, depth, K
        H, W = rgb.shape[:2]
        crop_h, crop_w = int(H / z), int(W / z)
        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2
        rgb = cv2.resize(rgb[y0:y0+crop_h, x0:x0+crop_w], (W, H))
        depth = cv2.resize(depth[y0:y0+crop_h, x0:x0+crop_w], (W, H),
                           interpolation=cv2.INTER_NEAREST)
        K_new = K.copy().astype(np.float64)
        K_new[0, 0] = K[0, 0] * z          # fx
        K_new[1, 1] = K[1, 1] * z          # fy
        K_new[0, 2] = K[0, 2] * z - W * (z - 1) / 2  # cx
        K_new[1, 2] = K[1, 2] * z - H * (z - 1) / 2  # cy
        return rgb, depth, K_new

    def get_intrinsic_mat(self, coeffs):
        return np.array([
            [coeffs.fx, 0, coeffs.tx],
            [0, coeffs.fy, coeffs.ty],
            [0, 0, 1]
        ])

    def get_K(self):
        """Get current intrinsic matrix from session."""
        return self.get_intrinsic_mat(self.session.get_intrinsic_mat())

    def normalize_width(self, width: float) -> float:
        """Normalize gripper width to [0, 1] using calibration."""
        if self.gripper_cal is None:
            return width
        min_w = self.gripper_cal["min_width"]
        max_w = self.gripper_cal["max_width"]
        if max_w <= min_w:
            return 0.0
        return max(0.0, min(1.0, (width - min_w) / (max_w - min_w)))

    def start_recording(self):
        """Start a new recording episode."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = os.path.join(
            self.save_dir, f"episode_{self.episode_idx:04d}_{timestamp}"
        )
        os.makedirs(os.path.join(self.episode_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.episode_dir, "depth"), exist_ok=True)

        self.frame_idx = 0
        self.episode_data = []

        # CSV file
        csv_path = os.path.join(self.episode_dir, "trajectory.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "gripper_width",
            "rgb_path", "depth_path"
        ])

        self.is_recording = True
        print(f"\n>>> Recording started: {self.episode_dir}")

    def stop_recording(self):
        """Stop recording, close CSV, and compile videos from saved images."""
        self.is_recording = False

        # Close CSV
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

        if not self.episode_data:
            print(">>> No frames recorded, discarding episode.")
            return

        # Compile images into videos
        print(">>> Compiling videos...")
        self._compile_video(
            os.path.join(self.episode_dir, "rgb"),
            os.path.join(self.episode_dir, "rgb.mp4"))
        self._compile_video(
            os.path.join(self.episode_dir, "depth"),
            os.path.join(self.episode_dir, "depth.mp4"))

        print(f">>> Recording stopped: {len(self.episode_data)} frames saved.")
        print(f"    RGB video:   {self.episode_dir}/rgb.mp4")
        print(f"    Depth video: {self.episode_dir}/depth.mp4")
        print(f"    Trajectory:  {self.episode_dir}/trajectory.csv")
        self.episode_idx += 1

    def _compile_video(self, img_dir, output_path, fps=30.0):
        """Compile a folder of PNG images into an mp4 video."""
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not imgs:
            return
        frame = cv2.imread(imgs[0])
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in imgs:
            writer.write(cv2.imread(p))
        writer.release()

    def _depth_to_grayscale(self, depth):
        """Convert depth to grayscale uint8 BGR image, fixed range 0-2m."""
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        gray = np.clip(depth_clean / 2.0 * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def save_frame(self, rgb_bgr, depth, camera_pose, gripper_width):
        """Save one RGB + depth image and one CSV row."""
        timestamp = time.time()

        # Save RGB image
        cv2.imwrite(
            os.path.join(self.episode_dir, "rgb", f"{self.frame_idx:06d}.png"),
            rgb_bgr)

        # Save depth as grayscale (Record3D style)
        depth_bgr = self._depth_to_grayscale(depth)
        if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
            depth_bgr = cv2.resize(depth_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
        cv2.imwrite(
            os.path.join(self.episode_dir, "depth", f"{self.frame_idx:06d}.png"),
            depth_bgr)

        # Write CSV row (use last valid width if detection failed)
        if gripper_width is not None:
            self._last_valid_gripper_width = gripper_width
        gw = gripper_width if gripper_width is not None else getattr(self, '_last_valid_gripper_width', 0.0)
        self.csv_writer.writerow([
            f"{timestamp:.6f}",
            f"{camera_pose.tx:.6f}",
            f"{camera_pose.ty:.6f}",
            f"{camera_pose.tz:.6f}",
            f"{camera_pose.qx:.6f}",
            f"{camera_pose.qy:.6f}",
            f"{camera_pose.qz:.6f}",
            f"{gw:.6f}",
            f"rgb/{self.frame_idx:06d}.png",
            f"depth/{self.frame_idx:06d}.png",
        ])

        self.episode_data.append(self.frame_idx)
        self.frame_idx += 1

    def run(self):
        """Main loop: stream, detect ArUco, display, and optionally record."""
        self.connect()

        print("\n=== Controls ===")
        print("  r : Start / Stop recording")
        print("  q : Quit")
        print("================\n")

        if self.gripper_cal is None:
            print("WARNING: No gripper calibration found.")
            print("  Run with --calibrate first, or gripper_width_normalized will be raw values.\n")

        while True:
            self.event.wait()

            # Get frame data from Record3D
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            intrinsic_mat = self.get_K()
            camera_pose = self.session.get_camera_pose()

            # Flip if TrueDepth camera
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            # Rotate to landscape if requested
            rgb, depth, intrinsic_mat = self._apply_rotation(rgb, depth, intrinsic_mat)

            # Digital zoom if requested
            rgb, depth, intrinsic_mat = self._apply_zoom(rgb, depth, intrinsic_mat)

            # Detect gripper width via ArUco markers (on RGB, before BGR conversion)
            gripper_width = self.tracker.update(rgb, intrinsic_mat)
            aruco_status = self.tracker.last_status

            # Convert to BGR for display
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Save frame if recording
            if self.is_recording:
                self.save_frame(rgb_bgr, depth, camera_pose, gripper_width)

            # --- Display ---
            display = rgb_bgr.copy()

            # Draw ArUco debug overlay
            if self.show_aruco_debug:
                display = self.tracker.draw_debug(display, intrinsic_mat)

            # Recording status
            status = "REC" if self.is_recording else "STANDBY"
            color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Gripper width
            if gripper_width is not None:
                cv2.putText(display, f"Gripper: {gripper_width:.4f} m", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display, "Gripper: N/A", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ArUco detection status
            status_colors = {
                "both": (0, 255, 0),
                "left_only": (0, 255, 255),
                "right_only": (0, 255, 255),
                "none": (0, 0, 255),
            }
            cv2.putText(display, f"ArUco: {aruco_status}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        status_colors.get(aruco_status, (200, 200, 200)), 2)

            if self.is_recording:
                cv2.putText(display, f"Frame: {self.frame_idx}", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Camera pose
            pose_text = (f"Pose: ({camera_pose.tx:.3f}, {camera_pose.ty:.3f}, "
                         f"{camera_pose.tz:.3f})")
            cv2.putText(display, pose_text, (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("UMI-FT Data Collection - RGB",
                       cv2.resize(display, (0, 0), fx=0.5, fy=0.5))

            # Depth visualization (handles NaN/inf)
            depth_vis = self._depth_to_grayscale(depth)
            cv2.imshow("UMI-FT Data Collection - Depth",
                       cv2.resize(depth_vis, (0, 0), fx=0.5, fy=0.5))

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.is_recording:
                    self.stop_recording()
                break
            elif key == ord('r'):
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()

            self.event.clear()

        cv2.destroyAllWindows()
        print("Done.")

    def run_calibration(self):
        """Run gripper calibration mode."""
        self.connect()

        calibrator = GripperCalibrator(self.tracker)
        result = calibrator.run_live(
            session=self.session,
            get_intrinsic_mat_fn=self.get_K,
            get_device_type_fn=self.session.get_device_type,
            DEVICE_TYPE_TRUEDEPTH=self.DEVICE_TYPE__TRUEDEPTH,
            rotate=self.rotate,
            zoom=self.zoom,
        )

        if result is not None:
            GripperCalibrator.save(result, self.calibration_path)

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="UMI-FT Data Collection via Record3D")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="Directory to save recorded episodes")
    parser.add_argument("--dev_idx", type=int, default=0,
                        help="Record3D device index")
    parser.add_argument("--aruco_config", type=str, default="config/aruco_config.yaml",
                        help="Path to ArUco configuration YAML")
    parser.add_argument("--calibration", type=str, default="gripper_range.json",
                        help="Path to gripper calibration JSON")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enter gripper calibration mode")
    parser.add_argument("--show_aruco", action="store_true",
                        help="Show ArUco detection debug overlay")
    parser.add_argument("--rotate", choices=["cw", "ccw"], default="ccw",
                        help="Rotate frames 90°: cw=clockwise, ccw=counterclockwise")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Digital zoom factor, e.g. 2.0 = crop center and 2x upscale")
    args = parser.parse_args()

    collector = DataCollector(
        save_dir=args.save_dir,
        dev_idx=args.dev_idx,
        aruco_config_path=args.aruco_config,
        calibration_path=args.calibration,
        show_aruco_debug=args.show_aruco,
        rotate=args.rotate,
        zoom=args.zoom,
    )

    if args.calibrate:
        collector.run_calibration()
    else:
        collector.run()


if __name__ == "__main__":
    main()
