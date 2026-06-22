"""
Hand-Eye Calibration: Solve

Reads paired data (xArm EE pose + ARKit camera pose) collected by
calibrate_hand_eye_collect.py and solves for the rigid transform
between the iPhone camera and the robot TCP (T_cam_to_ee).

Uses OpenCV's cv2.calibrateHandEye() with the Tsai method.

The relationship is:
    T_base_to_ee = T_base_to_arkit_world @ T_arkit_world_to_cam @ T_cam_to_ee

We solve for T_cam_to_ee (also called T_gripper2camera in OpenCV convention).

Output: calibration/hand_eye_result.json containing:
    - tx_cam_to_ee: 4x4 transform from camera frame to EE frame
    - tx_ee_to_cam: 4x4 transform from EE frame to camera frame (inverse)

Usage:
    python calibrate_hand_eye_solve.py
    python calibrate_hand_eye_solve.py --input calibration/hand_eye_data.json --output calibration/hand_eye_result.json
"""

import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def arkit_pose_to_matrix(pose):
    """Convert ARKit pose (quaternion + translation) to 4x4 SE(3) matrix."""
    T = np.eye(4)
    quat = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]  # scalar-last
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = [pose['tx'], pose['ty'], pose['tz']]
    return T


def ee_pose_to_matrix(pose):
    """Convert xArm EE pose to 4x4 SE(3) matrix.
    xArm uses intrinsic xyz (= extrinsic ZYX) Euler angles."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz',
        [pose['roll_deg'], pose['pitch_deg'], pose['yaw_deg']], degrees=True).as_matrix()
    T[:3, 3] = [pose['x_mm'] / 1000.0, pose['y_mm'] / 1000.0, pose['z_mm'] / 1000.0]
    return T


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration: Solve")
    parser.add_argument("--input", default="calibration/hand_eye_data.json",
                        help="Paired pose data from collect step")
    parser.add_argument("--output", default="calibration/hand_eye_result.json",
                        help="Output calibration result")
    args = parser.parse_args()

    # Load paired data
    with open(args.input, 'r') as f:
        paired_data = json.load(f)

    print(f"Loaded {len(paired_data)} paired poses from {args.input}")

    if len(paired_data) < 3:
        print("ERROR: Need at least 3 paired poses for calibration.")
        return

    # Convert to matrices
    T_base_ee_list = []   # T_base_to_ee for each pose
    T_world_cam_list = []  # T_arkit_world_to_cam for each pose

    for pair in paired_data:
        T_base_ee = ee_pose_to_matrix(pair['ee_pose'])
        T_world_cam = arkit_pose_to_matrix(pair['arkit_pose'])

        T_base_ee_list.append(T_base_ee)
        T_world_cam_list.append(T_world_cam)

    # cv2.calibrateHandEye solves: A_i @ X = X @ B_i
    # Where:
    #   A_i = T_base_ee_i^{-1} @ T_base_ee_j  (relative EE motion)
    #   B_i = T_world_cam_i^{-1} @ T_world_cam_j  (relative camera motion)
    #   X = T_cam_to_ee (what we want)
    #
    # OpenCV expects:
    #   R_gripper2base, t_gripper2base: list of (R, t) for T_ee_to_base = inv(T_base_to_ee)
    #   R_target2cam, t_target2cam: list of (R, t) for T_world_to_cam = inv(T_cam_to_world)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(len(paired_data)):
        # gripper2base = T_base_to_ee (EE pose in base frame, direct from robot API)
        T_base_ee = T_base_ee_list[i]
        R_gripper2base.append(T_base_ee[:3, :3])
        t_gripper2base.append(T_base_ee[:3, 3].reshape(3, 1))

        # target2cam = inv(ARKit camera_pose)
        # ARKit camera_pose = T_cam_to_world (camera in world frame)
        # We need T_world_to_cam = inv(camera_pose)
        T_world_to_cam = np.linalg.inv(T_world_cam_list[i])
        R_target2cam.append(T_world_to_cam[:3, :3])
        t_target2cam.append(T_world_to_cam[:3, 3].reshape(3, 1))

    # Solve with multiple methods and compare
    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    print("\n=== Calibration Results ===\n")

    for name, method in methods.items():
        try:
            R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=method
            )

            T_cam_to_ee = np.eye(4)
            T_cam_to_ee[:3, :3] = R_cam2ee
            T_cam_to_ee[:3, 3] = t_cam2ee.squeeze()

            # Check if rotation matrix is valid
            det = np.linalg.det(R_cam2ee)
            rpy = Rotation.from_matrix(R_cam2ee).as_euler('XYZ', degrees=True)
            t_mm = t_cam2ee.squeeze() * 1000

            print(f"  {name}:")
            print(f"    Translation: ({t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}) mm")
            print(f"    Rotation (RPY): ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) deg")
            print(f"    det(R) = {det:.6f}")

            results[name] = T_cam_to_ee

        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not results:
        print("\nERROR: All methods failed.")
        return

    # Validate: compute reprojection error for each method
    print("\n=== Reprojection Error ===\n")

    best_method = None
    best_error = float('inf')

    for name, T_cam_to_ee in results.items():
        errors = []
        for i in range(len(paired_data)):
            for j in range(i + 1, len(paired_data)):
                # Relative EE motion
                T_rel_ee = np.linalg.inv(T_base_ee_list[i]) @ T_base_ee_list[j]
                # Relative camera motion
                T_rel_cam = np.linalg.inv(T_world_cam_list[i]) @ T_world_cam_list[j]
                # Should satisfy: T_cam_to_ee @ T_rel_cam = T_rel_ee @ T_cam_to_ee
                lhs = T_cam_to_ee @ T_rel_cam
                rhs = T_rel_ee @ T_cam_to_ee
                # Error in translation (mm) and rotation (deg)
                t_err = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]) * 1000
                R_err = Rotation.from_matrix(lhs[:3, :3].T @ rhs[:3, :3]).magnitude()
                R_err_deg = np.degrees(R_err)
                errors.append((t_err, R_err_deg))

        avg_t_err = np.mean([e[0] for e in errors])
        avg_r_err = np.mean([e[1] for e in errors])
        print(f"  {name}: avg translation error = {avg_t_err:.2f} mm, avg rotation error = {avg_r_err:.2f} deg")

        if avg_t_err < best_error:
            best_error = avg_t_err
            best_method = name

    print(f"\n  Best method: {best_method} (avg translation error = {best_error:.2f} mm)")

    # Save best result
    T_cam_to_ee = results[best_method]
    T_ee_to_cam = np.linalg.inv(T_cam_to_ee)

    output = {
        'method': best_method,
        'tx_cam_to_ee': T_cam_to_ee.tolist(),
        'tx_ee_to_cam': T_ee_to_cam.tolist(),
        'num_poses': len(paired_data),
        'avg_translation_error_mm': float(best_error),
    }

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved calibration to {args.output}")
    print(f"\nT_cam_to_ee:\n{T_cam_to_ee}")


if __name__ == "__main__":
    main()
