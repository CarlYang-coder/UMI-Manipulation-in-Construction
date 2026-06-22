"""
ArUco marker detection and gripper width computation.
Adapted from UMI's umi/common/cv_util.py for iPhone pinhole camera model.

UMI original uses GoPro fisheye lens with cv2.fisheye.undistortPoints.
iPhone uses a standard pinhole model, so we skip fisheye undistortion
and pass corners directly to estimatePoseSingleMarkers with zero distortion.
"""

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def detect_aruco_tags(
    img: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_size_map: Dict[int, float],
    K: np.ndarray,
    refine_subpix: bool = True,
) -> dict:
    """
    Detect and localize ArUco tags in an image.

    Adapted from UMI's detect_localize_aruco_tags() in cv_util.py.
    Key difference: no fisheye undistortion (iPhone = pinhole model).

    Args:
        img: Input image (RGB or BGR).
        aruco_dict: OpenCV ArUco dictionary.
        marker_size_map: Mapping from marker ID to physical size in meters.
        K: 3x3 camera intrinsic matrix.
        refine_subpix: Whether to use subpixel corner refinement.

    Returns:
        Dict mapping marker ID to {'rvec', 'tvec', 'corners'}.
    """
    zero_dist = np.zeros((1, 5))

    # Set up detector parameters
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Detect markers — handle both old and new OpenCV ArUco API
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, param)
        corners, ids, _ = detector.detectMarkers(img)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img, dictionary=aruco_dict, parameters=param
        )

    if ids is None or len(corners) == 0:
        return {}

    tag_dict = {}
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]

        # iPhone pinhole model: no fisheye undistortion needed.
        # Pass corners directly with zero distortion coefficients.
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            this_corners, marker_size_m, K, zero_dist
        )

        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }

    return tag_dict


def get_gripper_width(
    tag_dict: dict,
    left_id: int,
    right_id: int,
    nominal_z: float = 0.072,
    z_tolerance: float = 0.008,
) -> Optional[float]:
    """
    Calculate gripper width from detected ArUco tags on left and right fingers.

    Directly ported from UMI's get_gripper_width() in cv_util.py.
    Uses the x-coordinate separation of the two finger markers' tvec.

    Args:
        tag_dict: Output from detect_aruco_tags().
        left_id: ArUco marker ID on left finger.
        right_id: ArUco marker ID on right finger.
        nominal_z: Expected depth (m) from camera to markers.
        z_tolerance: Acceptable depth deviation (m).

    Returns:
        Gripper width in meters, or None if detection is invalid.
    """
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]["tvec"]
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]["tvec"]
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2

    return width


class GripperWidthTracker:
    """
    Wraps ArUco detection + gripper width computation with hold-last-value logic.

    When detection fails briefly (e.g., motion blur), the last valid width is held
    for up to `hold_timeout` seconds before reporting None.
    """

    def __init__(
        self,
        aruco_dict: cv2.aruco.Dictionary,
        marker_size_map: Dict[int, float],
        left_id: int = 0,
        right_id: int = 1,
        nominal_z: float = 0.072,
        z_tolerance: float = 0.008,
        hold_timeout: float = 0.5,
    ):
        self.aruco_dict = aruco_dict
        self.marker_size_map = marker_size_map
        self.left_id = left_id
        self.right_id = right_id
        self.nominal_z = nominal_z
        self.z_tolerance = z_tolerance
        self.hold_timeout = hold_timeout

        self.last_valid_width: Optional[float] = None
        self.last_valid_time: float = 0.0
        self.last_tag_dict: dict = {}
        self.last_status: str = "none"  # "both", "left_only", "right_only", "none"

    def update(self, img: np.ndarray, K: np.ndarray) -> Optional[float]:
        """
        Run ArUco detection on a frame and return gripper width.

        Args:
            img: Input image (RGB recommended, BGR also works).
            K: 3x3 camera intrinsic matrix.

        Returns:
            Gripper width in meters, or None if unavailable.
        """
        now = time.time()

        tag_dict = detect_aruco_tags(
            img, self.aruco_dict, self.marker_size_map, K
        )
        self.last_tag_dict = tag_dict

        # Determine detection status
        has_left = self.left_id in tag_dict
        has_right = self.right_id in tag_dict
        if has_left and has_right:
            self.last_status = "both"
        elif has_left:
            self.last_status = "left_only"
        elif has_right:
            self.last_status = "right_only"
        else:
            self.last_status = "none"

        width = get_gripper_width(
            tag_dict, self.left_id, self.right_id,
            self.nominal_z, self.z_tolerance,
        )

        if width is not None:
            self.last_valid_width = width
            self.last_valid_time = now
            return width

        # Hold last value within timeout
        if self.last_valid_width is not None:
            if (now - self.last_valid_time) < self.hold_timeout:
                return self.last_valid_width

        return None

    def draw_debug(self, img: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Draw detected ArUco markers and pose axes on the image.

        Args:
            img: BGR image for display.
            K: 3x3 camera intrinsic matrix.

        Returns:
            Image with debug overlay drawn.
        """
        out = img.copy()
        zero_dist = np.zeros((1, 5))

        if not self.last_tag_dict:
            return out

        # Draw detected marker outlines
        corners_list = []
        ids_list = []
        for mid, info in self.last_tag_dict.items():
            corners_list.append(info["corners"].reshape(1, 4, 2))
            ids_list.append([mid])

        if corners_list:
            corners_arr = np.array(corners_list)
            ids_arr = np.array(ids_list)
            cv2.aruco.drawDetectedMarkers(out, corners_arr, ids_arr)

        # Draw pose axes for each detected marker
        for mid, info in self.last_tag_dict.items():
            marker_size = self.marker_size_map.get(mid, 0.016)
            cv2.drawFrameAxes(
                out, K, zero_dist,
                info["rvec"], info["tvec"],
                marker_size * 0.5,
            )

        return out
