"""
Gripper range calibration via ArUco marker detection.
Adapted from UMI's scripts/calibrate_gripper_range.py.

Usage:
    1. Attach ArUco markers (DICT_4X4_50, ID 0 & 1) to gripper fingers.
    2. Connect iPhone via USB, open Record3D with USB Streaming enabled.
    3. Run calibration: open and close the gripper 10+ times in front of camera.
    4. Press 's' to save calibration, 'q' to quit without saving.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np

from umi_ft.aruco_detector import GripperWidthTracker


class GripperCalibrator:
    """
    Determines gripper min/max width range from live ArUco detection.

    The user opens and closes the gripper repeatedly while streaming
    from Record3D. The calibrator tracks the min and max detected widths
    and saves them as gripper_range.json.
    """

    def __init__(self, tracker: GripperWidthTracker):
        self.tracker = tracker
        self.widths: list = []
        self.min_width: Optional[float] = None
        self.max_width: Optional[float] = None

    def run_live(self, session, get_intrinsic_mat_fn, get_device_type_fn,
                 DEVICE_TYPE_TRUEDEPTH: int = 0):
        """
        Run live calibration from a Record3D session.

        Args:
            session: Active Record3D session (already streaming).
            get_intrinsic_mat_fn: Callable that returns 3x3 K matrix.
            get_device_type_fn: Callable that returns device type.
            DEVICE_TYPE_TRUEDEPTH: Constant for TrueDepth device.
        """
        from threading import Event
        event = Event()

        original_callback = session.on_new_frame
        session.on_new_frame = lambda: event.set()

        print("\n=== Gripper Calibration ===")
        print("  Open and close the gripper 10+ times.")
        print("  Press 's' to save calibration.")
        print("  Press 'q' to quit without saving.")
        print("===========================\n")

        try:
            while True:
                event.wait()

                rgb = session.get_rgb_frame()
                K = get_intrinsic_mat_fn()

                if get_device_type_fn() == DEVICE_TYPE_TRUEDEPTH:
                    rgb = cv2.flip(rgb, 1)

                width = self.tracker.update(rgb, K)

                if width is not None:
                    self.widths.append(width)
                    self.min_width = float(np.nanmin(self.widths))
                    self.max_width = float(np.nanmax(self.widths))

                # Display
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                display = self.tracker.draw_debug(display, K)

                # Status text
                if width is not None:
                    cv2.putText(display, f"Width: {width:.4f} m", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "No marker detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if self.min_width is not None:
                    cv2.putText(display,
                                f"Min: {self.min_width:.4f}  Max: {self.max_width:.4f}",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.putText(display, f"Samples: {len(self.widths)}", (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # Draw width bar
                    bar_x, bar_y, bar_w, bar_h = 10, 110, 300, 20
                    cv2.rectangle(display, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h),
                                  (100, 100, 100), -1)
                    if width is not None and self.max_width > self.min_width:
                        ratio = (width - self.min_width) / (self.max_width - self.min_width)
                        ratio = max(0.0, min(1.0, ratio))
                        fill_w = int(bar_w * ratio)
                        cv2.rectangle(display, (bar_x, bar_y),
                                      (bar_x + fill_w, bar_y + bar_h),
                                      (0, 255, 0), -1)

                cv2.putText(display, "CALIBRATION MODE", (10, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Gripper Calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if len(self.widths) > 0:
                        print(f"Calibration saved: min={self.min_width:.4f}, "
                              f"max={self.max_width:.4f}, samples={len(self.widths)}")
                        break
                    else:
                        print("No samples collected yet. Keep calibrating.")
                elif key == ord('q'):
                    print("Calibration cancelled.")
                    cv2.destroyAllWindows()
                    return None

                event.clear()
        finally:
            session.on_new_frame = original_callback

        cv2.destroyAllWindows()
        return self.get_result()

    def get_result(self) -> Optional[dict]:
        """Return calibration result as a dict."""
        if not self.widths:
            return None
        return {
            "min_width": self.min_width,
            "max_width": self.max_width,
            "left_tag_id": self.tracker.left_id,
            "right_tag_id": self.tracker.right_id,
            "num_samples": len(self.widths),
        }

    @staticmethod
    def save(result: dict, path: str):
        """Save calibration result to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Calibration saved to: {path}")

    @staticmethod
    def load(path: str) -> dict:
        """Load calibration result from JSON."""
        with open(path, "r") as f:
            return json.load(f)
