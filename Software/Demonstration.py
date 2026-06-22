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
import json
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
                 show_aruco_debug=False):
        self.save_dir = save_dir
        self.dev_idx = dev_idx
        self.show_aruco_debug = show_aruco_debug
        self.calibration_path = calibration_path

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
        self.is_recording = True
        print(f"\n>>> Recording started: {self.episode_dir}")

    def stop_recording(self):
        """Stop recording and save metadata."""
        self.is_recording = False

        if not self.episode_data:
            print(">>> No frames recorded, discarding episode.")
            return

        metadata = {
            "num_frames": len(self.episode_data),
            "gripper_calibration": self.gripper_cal,
            "frames": self.episode_data,
        }
        metadata_path = os.path.join(self.episode_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f">>> Recording stopped: {len(self.episode_data)} frames saved.")
        self.episode_idx += 1

    def save_frame(self, rgb_bgr, depth, camera_pose, intrinsic_mat,
                   gripper_width, aruco_status):
        """Save a single frame's data to disk."""
        timestamp = time.time()

        # Save RGB image
        rgb_path = os.path.join(self.episode_dir, "rgb", f"{self.frame_idx:06d}.png")
        cv2.imwrite(rgb_path, rgb_bgr)

        # Save depth as 16-bit PNG (depth in mm)
        depth_mm = (depth * 1000).astype(np.uint16)
        depth_path = os.path.join(self.episode_dir, "depth", f"{self.frame_idx:06d}.png")
        cv2.imwrite(depth_path, depth_mm)

        # Build frame metadata
        frame_data = {
            "frame_idx": self.frame_idx,
            "timestamp": timestamp,
            "camera_pose": {
                "qx": float(camera_pose.qx),
                "qy": float(camera_pose.qy),
                "qz": float(camera_pose.qz),
                "qw": float(camera_pose.qw),
                "tx": float(camera_pose.tx),
                "ty": float(camera_pose.ty),
                "tz": float(camera_pose.tz),
            },
            "intrinsic_matrix": intrinsic_mat.tolist(),
            "gripper_width_raw": gripper_width,
            "gripper_width_normalized": (
                self.normalize_width(gripper_width) if gripper_width is not None else None
            ),
            "aruco_status": aruco_status,
        }

        self.episode_data.append(frame_data)
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

            # Detect gripper width via ArUco markers (on RGB, before BGR conversion)
            gripper_width = self.tracker.update(rgb, intrinsic_mat)
            aruco_status = self.tracker.last_status

            # Convert to BGR for display
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Save frame if recording
            if self.is_recording:
                self.save_frame(rgb_bgr, depth, camera_pose, intrinsic_mat,
                                gripper_width, aruco_status)

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

            cv2.imshow("UMI-FT Data Collection - RGB", display)

            # Depth visualization
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("UMI-FT Data Collection - Depth", depth_vis)

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
    args = parser.parse_args()

    collector = DataCollector(
        save_dir=args.save_dir,
        dev_idx=args.dev_idx,
        aruco_config_path=args.aruco_config,
        calibration_path=args.calibration,
        show_aruco_debug=args.show_aruco,
    )

    if args.calibrate:
        collector.run_calibration()
    else:
        collector.run()


if __name__ == "__main__":
    main()
