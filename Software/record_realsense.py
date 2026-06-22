"""
RealSense Recording Script
Records RGB stream and saves as an .mp4 file.

Usage:
    python record_realsense.py

Controls:
    Enter : Start recording
    Ctrl+C: Stop recording and exit
"""

import os
import signal
import sys
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

SAVE_DIR = r"D:\Downloads"


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start pipeline to verify camera is connected
    profile = pipeline.start(config)
    device = profile.get_device().get_info(rs.camera_info.name)
    print(f"Camera detected: {device}")
    pipeline.stop()

    input("Press Enter to start recording...")

    # Set up recording filename
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    filename = os.path.join(SAVE_DIR, f"realsense_{timestamp}.mp4")

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))

    # Start pipeline
    pipeline.start(config)

    # Handle Ctrl+C gracefully
    stopped = False

    def on_sigint(sig, frame):
        nonlocal stopped
        if not stopped:
            stopped = True
            print("\nStopping recording...")
            pipeline.stop()
            writer.release()
            print(f"Saved to: {filename}")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    print(f"Recording to: {filename}")
    print("Press Ctrl+C to stop.")

    frame_count = 0
    while not stopped:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        writer.write(color_image)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"\r  Frames recorded: {frame_count}", end="", flush=True)


if __name__ == "__main__":
    main()
