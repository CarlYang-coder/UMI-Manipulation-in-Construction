"""
UMI-FT Wireless Data Collection Script
Collects RGB, Depth, Camera Pose, Timestamp, and Gripper Width
from an iPhone via Record3D WiFi streaming (WebRTC).

Control: Use Record3D's red button on iPhone.
  - Press red button → start WiFi streaming → Mac auto-records
  - Press red button again → stop streaming → Mac auto-saves episode
  - Repeat for next episode
  - Ctrl+C on Mac to quit

Usage:
    python demonstration_wireless.py --ip <iphone_ip>

    Example:
    python demonstration_wireless.py --ip 10.207.112.39
"""

import os
import csv
import glob
import json
import time
import asyncio
import argparse
import threading
from collections import namedtuple
from datetime import datetime

import av
import cv2
import numpy as np
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

from Demonstration import load_aruco_config
from umi_ft.aruco_detector import GripperWidthTracker
from umi_ft.gripper_calibration import GripperCalibrator


# Compatible with Record3D SDK's CameraPose
CameraPose = namedtuple("CameraPose", ["tx", "ty", "tz", "qx", "qy", "qz"])


class WirelessDataCollector:
    """
    Collects data from Record3D via WiFi (WebRTC).
    Recording is controlled by Record3D's red button:
    - Stream starts → auto-start recording
    - Stream stops → auto-stop recording and save
    """

    def __init__(self, iphone_ip, save_dir="data", headless=False,
                 aruco_config_path="config/aruco_config.yaml",
                 calibration_path="gripper_range_wifi.json",
                 show_aruco_debug=False, rotate=None, zoom=1.0,
                 phone=None):
        self.iphone_ip = iphone_ip
        self.base_url = f"http://{iphone_ip}"
        self.save_dir = save_dir
        self.headless = headless
        self.phone = phone
        self.show_aruco_debug = show_aruco_debug
        self.rotate = rotate
        self.zoom = zoom

        # Frame data (shared between WebRTC thread and main thread)
        self._frame_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_pose = None
        self._latest_K = None

        # Resolution scaling: K matrix from Record3D is for original resolution,
        # but WiFi frames are smaller. We compute scale factors on first frame.
        self._original_size = None  # (width, height) from metadata
        self._actual_size = None    # (width, height) of actual received frame
        self._k_scale = None        # (scale_x, scale_y)

        # Connection state events
        self._connected_event = threading.Event()
        self._disconnected_event = threading.Event()

        # Recording state
        self.is_recording = False
        self.episode_idx = 0
        self.frame_idx = 0
        self.episode_dir = None
        self.episode_data = []
        self.csv_file = None
        self.csv_writer = None

        # ArUco gripper tracking
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

        # Gripper calibration
        self.gripper_cal = None
        if os.path.exists(calibration_path):
            self.gripper_cal = GripperCalibrator.load(calibration_path)
            print(f"Loaded gripper calibration: min={self.gripper_cal['min_width']:.4f}, "
                  f"max={self.gripper_cal['max_width']:.4f}")

    # -- Frame processing (called from WebRTC async thread) ----------------

    def _process_video_frame(self, frame):
        """Split combined WebRTC frame into RGB and depth."""
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            half_w = w // 2

            # Record actual frame size on first frame (for K matrix scaling)
            if self._actual_size is None:
                self._actual_size = (half_w, h)  # each half is one image
                print(f"  Frame size: {half_w}x{h} (combined: {w}x{h})")
                if self._original_size is not None:
                    sx = half_w / self._original_size[0]
                    sy = h / self._original_size[1]
                    self._k_scale = (sx, sy)
                    print(f"  K matrix scale: {sx:.3f} x {sy:.3f}")

            rgb_bgr = img[:, half_w:].copy()

            depth_hsv_bgr = img[:, :half_w]
            hsv = cv2.cvtColor(depth_hsv_bgr, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.float32) / 180.0
            depth = 3.0 * hue

            with self._frame_lock:
                self._latest_rgb = rgb_bgr
                self._latest_depth = depth

            self._frame_event.set()
        except Exception as e:
            if not hasattr(self, '_frame_error_logged'):
                print(f"[WiFi] Frame processing error: {e}")
                self._frame_error_logged = True

    def _process_data_message(self, message):
        """Parse data channel JSON for pose and intrinsics."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        pose_arr = data.get("pose")
        if pose_arr and len(pose_arr) >= 7:
            # WiFi format: [qx, qy, qz, qw, tx, ty, tz]
            pose = CameraPose(
                tx=pose_arr[4], ty=pose_arr[5], tz=pose_arr[6],
                qx=pose_arr[0], qy=pose_arr[1], qz=pose_arr[2],
            )
            with self._frame_lock:
                self._latest_pose = pose

        K_arr = data.get("intrinsicMatrix")
        if K_arr and len(K_arr) == 9:
            K = np.array([
                [K_arr[0], K_arr[3], K_arr[6]],
                [K_arr[1], K_arr[4], K_arr[7]],
                [K_arr[2], K_arr[5], K_arr[8]],
            ], dtype=np.float64)
            # Scale K to match actual WiFi frame size
            if self._k_scale is not None:
                sx, sy = self._k_scale
                K[0, 0] *= sx  # fx
                K[0, 2] *= sx  # cx
                K[1, 1] *= sy  # fy
                K[1, 2] *= sy  # cy
            with self._frame_lock:
                self._latest_K = K

    # -- WebRTC connection -------------------------------------------------

    async def _connect_and_stream(self):
        """Connect to Record3D WiFi stream via WebRTC. Returns when disconnected."""
        async with aiohttp.ClientSession() as session:
            # Fetch metadata (original resolution + initial K)
            async with session.get(f"{self.base_url}/metadata") as resp:
                metadata = await resp.json()
                orig_size = metadata.get("originalSize", [])
                if len(orig_size) == 2:
                    self._original_size = (orig_size[0], orig_size[1])  # (width, height)
                    print(f"  Original size: {orig_size[0]}x{orig_size[1]}")
                K_arr = metadata.get("K", [])
                if len(K_arr) == 9:
                    K = np.array([
                        [K_arr[0], K_arr[3], K_arr[6]],
                        [K_arr[1], K_arr[4], K_arr[7]],
                        [K_arr[2], K_arr[5], K_arr[8]],
                    ], dtype=np.float64)
                    # Note: this K is for original resolution; will be scaled
                    # after first video frame arrives and _k_scale is computed
                    with self._frame_lock:
                        self._latest_K = K

            # Get WebRTC offer
            async with session.get(f"{self.base_url}/getOffer") as resp:
                if resp.status == 403:
                    raise RuntimeError("Record3D 403: another client connected")
                offer_data = await resp.json()

            # Create peer connection
            pc = RTCPeerConnection()
            disconnected = asyncio.Event()

            @pc.on("track")
            def on_track(track):
                if track.kind == "video":
                    asyncio.ensure_future(self._receive_video(track, disconnected))

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(message):
                    if isinstance(message, str):
                        self._process_data_message(message)

            @pc.on("connectionstatechange")
            async def on_state_change():
                state = pc.connectionState
                if state == "connected":
                    self._connected_event.set()
                elif state in ("failed", "closed", "disconnected"):
                    disconnected.set()
                    self._disconnected_event.set()

            # Set remote description
            offer = RTCSessionDescription(
                sdp=offer_data.get("sdp", offer_data.get("data", "")),
                type=offer_data.get("type", "offer"),
            )
            await pc.setRemoteDescription(offer)

            # Create and send answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            while pc.iceGatheringState != "complete":
                await asyncio.sleep(0.1)

            answer_payload = {
                "type": "answer",
                "data": pc.localDescription.sdp,
            }
            async with session.post(
                f"{self.base_url}/answer",
                json=answer_payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"  Warning: /answer returned {resp.status}: {text}")

            # Wait until disconnected
            await disconnected.wait()
            await pc.close()

    async def _receive_video(self, track, disconnected):
        """Receive video frames from WebRTC track."""
        while not disconnected.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=10.0)
                self._process_video_frame(frame)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
        disconnected.set()
        self._disconnected_event.set()

    def _run_webrtc_session(self):
        """Run one WebRTC session in a new event loop. Returns when disconnected."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_and_stream())
        except Exception as e:
            print(f"  Connection error: {e}")
        finally:
            loop.close()
            self._disconnected_event.set()

    # -- Image transforms --------------------------------------------------

    def _apply_rotation(self, rgb, depth, K):
        if self.rotate is None:
            return rgb, depth, K
        H, W = rgb.shape[:2]
        if self.rotate == 'cw':
            code = cv2.ROTATE_90_CLOCKWISE
            K_new = np.array([
                [K[1, 1], 0, H - 1 - K[1, 2]],
                [0, K[0, 0], K[0, 2]],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            code = cv2.ROTATE_90_COUNTERCLOCKWISE
            K_new = np.array([
                [K[1, 1], 0, K[1, 2]],
                [0, K[0, 0], W - 1 - K[0, 2]],
                [0, 0, 1]
            ], dtype=np.float64)
        return cv2.rotate(rgb, code), cv2.rotate(depth, code), K_new

    def _apply_zoom(self, rgb, depth, K):
        z = self.zoom
        if z <= 1.0:
            return rgb, depth, K
        H, W = rgb.shape[:2]
        crop_h, crop_w = int(H / z), int(W / z)
        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2
        rgb = cv2.resize(rgb[y0:y0 + crop_h, x0:x0 + crop_w], (W, H))
        depth = cv2.resize(depth[y0:y0 + crop_h, x0:x0 + crop_w], (W, H),
                           interpolation=cv2.INTER_NEAREST)
        K_new = K.copy().astype(np.float64)
        K_new[0, 0] = K[0, 0] * z
        K_new[1, 1] = K[1, 1] * z
        K_new[0, 2] = K[0, 2] * z - W * (z - 1) / 2
        K_new[1, 2] = K[1, 2] * z - H * (z - 1) / 2
        return rgb, depth, K_new

    # -- Recording ---------------------------------------------------------

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = os.path.join(
            self.save_dir, f"episode_{self.episode_idx:04d}_{timestamp}"
        )
        os.makedirs(os.path.join(self.episode_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.episode_dir, "depth"), exist_ok=True)

        self.frame_idx = 0
        self.episode_data = []
        self._frame_timestamps = []

        csv_path = os.path.join(self.episode_dir, "trajectory.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "gripper_width",
            "rgb_path", "depth_path"
        ])

        self.is_recording = True
        print(f">>> Recording started: {self.episode_dir}")

    def stop_recording(self):
        self.is_recording = False

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

        if not self.episode_data:
            print(">>> No frames recorded, discarding episode.")
            return

        # Calculate actual fps from saved timestamps
        actual_fps = 30.0
        if len(self._frame_timestamps) >= 2:
            duration = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if duration > 0:
                actual_fps = (len(self._frame_timestamps) - 1) / duration

        num_frames = len(self.episode_data)
        print(f">>> {num_frames} frames saved, compiling videos (fps={actual_fps:.1f})...")

        # Wait for filesystem to flush all written files
        time.sleep(3.0)

        # Build file lists from known frame indices (don't rely on glob)
        rgb_files = [
            os.path.join(self.episode_dir, "rgb", f"{i:06d}.png")
            for i in range(num_frames)
        ]
        depth_files = [
            os.path.join(self.episode_dir, "depth", f"{i:06d}.png")
            for i in range(num_frames)
        ]

        # Verify all files exist
        rgb_exist = [f for f in rgb_files if os.path.exists(f)]
        depth_exist = [f for f in depth_files if os.path.exists(f)]
        print(f"    RGB images found: {len(rgb_exist)}/{num_frames}")
        print(f"    Depth images found: {len(depth_exist)}/{num_frames}")

        self._compile_video(rgb_exist,
                            os.path.join(self.episode_dir, "rgb.mp4"),
                            fps=actual_fps)
        self._compile_video(depth_exist,
                            os.path.join(self.episode_dir, "depth.mp4"),
                            fps=actual_fps)

        print(f">>> Episode {self.episode_idx} saved: {num_frames} frames")
        print(f"    {self.episode_dir}")
        self.episode_idx += 1

    def _compile_video(self, img_files, output_path, fps=30.0):
        """Compile a list of image file paths into an mp4 video using PyAV."""
        if not img_files:
            return
        first = cv2.imread(img_files[0])
        if first is None:
            return
        h, w = first.shape[:2]
        # mpeg4 codec requires even dimensions
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

        # Flush
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def _depth_to_grayscale(self, depth):
        """Convert depth to grayscale uint8 BGR image, fixed range 0-2m."""
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        gray = np.clip(depth_clean / 2.0 * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def save_frame(self, rgb_bgr, depth, camera_pose, gripper_width):
        timestamp = time.time()
        self._frame_timestamps.append(timestamp)

        cv2.imwrite(
            os.path.join(self.episode_dir, "rgb", f"{self.frame_idx:06d}.png"),
            rgb_bgr)

        depth_bgr = self._depth_to_grayscale(depth)
        if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
            depth_bgr = cv2.resize(depth_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
        cv2.imwrite(
            os.path.join(self.episode_dir, "depth", f"{self.frame_idx:06d}.png"),
            depth_bgr)

        # Use last valid width if detection failed
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

    # -- Main loop ---------------------------------------------------------

    def _wait_for_streaming(self):
        """Poll iPhone until Record3D WiFi streaming is available."""
        import urllib.request
        import urllib.error

        while True:
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/metadata", method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1)

    def run(self):
        """Main loop: wait for connection → record → save → repeat."""
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"  UMI Wireless Data Collection")
        print(f"  iPhone IP: {self.iphone_ip}")
        print(f"{'='*50}")
        print(f"\n  How to use:")
        print(f"    1. Press red button in Record3D → recording starts")
        print(f"    2. Press red button again → recording stops & saves")
        print(f"    3. Repeat for next episode")
        print(f"    4. Ctrl+C on Mac to quit\n")

        if self.gripper_cal is None:
            print("WARNING: No gripper calibration found.")
            print("  Run Demonstration.py --calibrate first.\n")

        try:
            while True:
                # Reset state for new session
                self._connected_event.clear()
                self._disconnected_event.clear()
                self._frame_event.clear()
                self._latest_rgb = None
                self._latest_depth = None
                self._latest_pose = None
                if hasattr(self, '_first_frame_logged'):
                    del self._first_frame_logged
                if hasattr(self, '_windows_positioned'):
                    del self._windows_positioned

                # Wait for Record3D to start streaming
                print("Waiting for Record3D WiFi streaming...")
                print("  (Press red button in Record3D on iPhone)")
                self._wait_for_streaming()
                print("  Record3D detected! Connecting...")

                # Start WebRTC session in background thread
                webrtc_thread = threading.Thread(
                    target=self._run_webrtc_session, daemon=True)
                webrtc_thread.start()

                # Wait for connection
                if not self._connected_event.wait(timeout=15):
                    print("  Connection timeout. Retrying...\n")
                    continue

                print("  Connected! Stabilizing stream...")
                time.sleep(15)

                # Send iMessage notification
                if self.phone:
                    os.system(
                        f'osascript -e \'tell application "Messages" to send '
                        f'"Ready to record! (Episode {self.episode_idx})" '
                        f'to buddy "{self.phone}"\''
                    )
                    print(f"  iMessage sent to {self.phone}")

                print("  Stream stable. Auto-recording...\n")

                # Auto-start recording
                self.start_recording()

                # Process frames until disconnected
                while not self._disconnected_event.is_set():
                    got_frame = self._frame_event.wait(timeout=1.0)
                    if not got_frame:
                        continue
                    self._frame_event.clear()

                    with self._frame_lock:
                        rgb_bgr = self._latest_rgb
                        depth = self._latest_depth
                        camera_pose = self._latest_pose
                        intrinsic_mat = self._latest_K

                    if rgb_bgr is None or camera_pose is None or intrinsic_mat is None:
                        continue

                    # Apply transforms
                    rgb_bgr, depth, intrinsic_mat = self._apply_rotation(
                        rgb_bgr, depth, intrinsic_mat)
                    rgb_bgr, depth, intrinsic_mat = self._apply_zoom(
                        rgb_bgr, depth, intrinsic_mat)

                    # ArUco detection
                    rgb_for_aruco = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    gripper_width = self.tracker.update(rgb_for_aruco, intrinsic_mat)

                    # Save frame
                    if self.is_recording:
                        self.save_frame(rgb_bgr, depth, camera_pose, gripper_width)

                    # Display (unless headless)
                    if not self.headless:
                        display = rgb_bgr.copy()
                        aruco_status = self.tracker.last_status

                        if self.show_aruco_debug:
                            display = self.tracker.draw_debug(display, intrinsic_mat)

                        cv2.putText(display, "REC", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                        if gripper_width is not None:
                            cv2.putText(display, f"Gripper: {gripper_width:.4f} m",
                                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 255, 255), 2)

                        status_colors = {
                            "both": (0, 255, 0), "left_only": (0, 255, 255),
                            "right_only": (0, 255, 255), "none": (0, 0, 255),
                        }
                        cv2.putText(display, f"ArUco: {aruco_status}", (10, 95),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    status_colors.get(aruco_status, (200, 200, 200)), 2)

                        cv2.putText(display, f"Frame: {self.frame_idx}", (10, 125),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        pose_text = (f"Pose: ({camera_pose.tx:.3f}, "
                                     f"{camera_pose.ty:.3f}, {camera_pose.tz:.3f})")
                        cv2.putText(display, pose_text, (10, 155),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                        rgb_display = cv2.resize(display, (0, 0), fx=0.5, fy=0.5)
                        cv2.imshow("UMI Wireless - RGB", rgb_display)

                        depth_vis = self._depth_to_grayscale(depth)
                        depth_display = cv2.resize(depth_vis, (0, 0), fx=0.5, fy=0.5)
                        cv2.imshow("UMI Wireless - Depth", depth_display)

                        if not hasattr(self, '_windows_positioned'):
                            cv2.moveWindow("UMI Wireless - RGB", 50, 50)
                            cv2.moveWindow("UMI Wireless - Depth",
                                           50 + rgb_display.shape[1] + 20, 50)
                            self._windows_positioned = True

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            if self.is_recording:
                                self.stop_recording()
                            return

                # Stream disconnected → auto-stop recording
                print("\n  Stream disconnected (red button pressed).")
                if self.is_recording:
                    self.stop_recording()

                if not self.headless:
                    cv2.destroyAllWindows()

                print()  # blank line before next waiting message

        except KeyboardInterrupt:
            print("\n\nCtrl+C received.")
            if self.is_recording:
                self.stop_recording()

        if not self.headless:
            cv2.destroyAllWindows()
        print("Done.")

    def run_calibration(self, calibration_path="gripper_range_wifi.json"):
        """Calibrate gripper range via WiFi streaming. Press 's' to save, 'q' to quit."""
        print(f"\n{'='*50}")
        print(f"  UMI Wireless Gripper Calibration")
        print(f"  iPhone IP: {self.iphone_ip}")
        print(f"{'='*50}")
        print(f"\n  1. Press red button in Record3D to start streaming")
        print(f"  2. Open and close gripper 10+ times in front of camera")
        print(f"  3. Press 's' to save calibration")
        print(f"  4. Press 'q' to quit without saving\n")

        # Wait for streaming
        print("Waiting for Record3D WiFi streaming...")
        self._wait_for_streaming()
        print("  Record3D detected! Connecting...")

        # Start WebRTC
        self._connected_event.clear()
        self._disconnected_event.clear()
        webrtc_thread = threading.Thread(target=self._run_webrtc_session, daemon=True)
        webrtc_thread.start()

        if not self._connected_event.wait(timeout=15):
            print("  Connection timeout.")
            return

        print("  Connected! Start calibration...\n")

        widths = []
        min_w = None
        max_w = None

        try:
            while not self._disconnected_event.is_set():
                got_frame = self._frame_event.wait(timeout=1.0)
                if not got_frame:
                    continue
                self._frame_event.clear()

                with self._frame_lock:
                    rgb_bgr = self._latest_rgb
                    intrinsic_mat = self._latest_K

                if rgb_bgr is None or intrinsic_mat is None:
                    continue

                rgb_bgr, _, intrinsic_mat = self._apply_rotation(
                    rgb_bgr, np.zeros_like(rgb_bgr[:, :, 0], dtype=np.float32),
                    intrinsic_mat)

                # ArUco detection
                rgb_for_aruco = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                width = self.tracker.update(rgb_for_aruco, intrinsic_mat)

                if width is not None:
                    widths.append(width)
                    min_w = float(np.nanmin(widths))
                    max_w = float(np.nanmax(widths))

                # Display
                display = rgb_bgr.copy()
                display = self.tracker.draw_debug(display, intrinsic_mat)

                if width is not None:
                    cv2.putText(display, f"Width: {width:.4f} m", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "No marker detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if min_w is not None:
                    cv2.putText(display,
                                f"Min: {min_w:.4f}  Max: {max_w:.4f}",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.putText(display, f"Samples: {len(widths)}", (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # Width bar
                    bar_x, bar_y, bar_w, bar_h = 10, 110, 300, 20
                    cv2.rectangle(display, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h),
                                  (100, 100, 100), -1)
                    if width is not None and max_w > min_w:
                        ratio = max(0.0, min(1.0, (width - min_w) / (max_w - min_w)))
                        fill_w = int(bar_w * ratio)
                        cv2.rectangle(display, (bar_x, bar_y),
                                      (bar_x + fill_w, bar_y + bar_h),
                                      (0, 255, 0), -1)

                cv2.putText(display, "CALIBRATION MODE", (10, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("WiFi Gripper Calibration",
                           cv2.resize(display, (0, 0), fx=0.5, fy=0.5))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if len(widths) > 0:
                        result = {
                            "min_width": min_w,
                            "max_width": max_w,
                            "num_samples": len(widths),
                        }
                        GripperCalibrator.save(result, calibration_path)
                        print(f"\nCalibration saved: min={min_w:.4f}, max={max_w:.4f}, "
                              f"samples={len(widths)}")
                        break
                    else:
                        print("No samples yet. Keep calibrating.")
                elif key == ord('q'):
                    print("\nCalibration cancelled.")
                    break

        except KeyboardInterrupt:
            print("\n\nCtrl+C received.")

        self._disconnected_event.set()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="UMI-FT Wireless Data Collection via Record3D WiFi"
    )
    parser.add_argument("--ip", type=str, required=True,
                        help="iPhone IP address (shown in Record3D WiFi streaming)")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="Directory to save recorded episodes")
    parser.add_argument("--aruco_config", type=str, default="config/aruco_config.yaml",
                        help="Path to ArUco configuration YAML")
    parser.add_argument("--calibration", type=str, default="gripper_range_wifi.json",
                        help="Path to gripper calibration JSON")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enter gripper calibration mode")
    parser.add_argument("--show_aruco", action="store_true",
                        help="Show ArUco detection debug overlay")
    parser.add_argument("--rotate", choices=["cw", "ccw"], default="ccw",
                        help="Rotate frames 90°")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Digital zoom factor")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display windows")
    parser.add_argument("--phone", type=str, default=None,
                        help="Phone number for iMessage notification when ready (e.g. +86xxx)")
    args = parser.parse_args()

    collector = WirelessDataCollector(
        iphone_ip=args.ip,
        save_dir=args.save_dir,
        headless=args.headless,
        phone=args.phone,
        aruco_config_path=args.aruco_config,
        calibration_path=args.calibration,
        show_aruco_debug=args.show_aruco,
        rotate=args.rotate,
        zoom=args.zoom,
    )

    if args.calibrate:
        collector.run_calibration(calibration_path=args.calibration)
    else:
        collector.run()


if __name__ == "__main__":
    main()
