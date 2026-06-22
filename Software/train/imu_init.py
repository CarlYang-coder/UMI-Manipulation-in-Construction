"""
IMU initialization helper.

Before running any xArm + iPhone USB stream task, move the arm to a
fixed IMU-calibration pose (joints: [0, 0, 0, 0, 0, -90, 0] degrees),
show the live camera feed, and wait for user confirmation that the
view is correct. Then proceed to home.

Shared by verify_pose.py and run_umi_ft.py.
"""

import time
import numpy as np
import cv2


# Fixed IMU init pose (joint angles in degrees)
IMU_INIT_JOINTS_DEG = [0, 0, 0, 0, 0, -90, 0]


def imu_init_sequence(arm, streamer, home_joints_deg,
                      show_display: bool = True):
    """
    Run IMU initialization:
      1. Move arm to IMU_INIT_JOINTS_DEG
      2. Wait for user to press red button on iPhone Record3D
         and confirm view (press Enter in terminal)
      3. Move arm to home_joints_deg

    Args:
        arm: XArmAPI instance (already motion_enabled, mode=0, state=0)
        streamer: Record3DStreamer instance (already connected)
        home_joints_deg: list of 7 joint angles (deg) for home pose
        show_display: unused (kept for API compatibility)

    Returns:
        True if user confirmed, False if cancelled.
    """
    # Ensure mode 0 for joint moves
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    # Move to IMU init pose
    print(f"\n=== IMU Init Pose: {IMU_INIT_JOINTS_DEG} ===")
    print("Moving to IMU init pose ...")
    ret = arm.set_servo_angle(angle=IMU_INIT_JOINTS_DEG, speed=20, wait=True)
    if ret != 0:
        print(f"  [ERROR] set_servo_angle returned {ret}")
        return False
    time.sleep(0.5)
    print("  Reached IMU init pose.")

    # Wait for user to press Enter in terminal (after pressing red button on iPhone)
    print("\n[IMU Init] Arm is at init pose.")
    print("           >>> Press the red button on iPhone Record3D now.")

    # Show live iPhone RGB so the user can verify the view matches training data.
    # Focus the OpenCV window and press Q / Enter to continue, Esc to abort.
    if streamer is not None and show_display:
        print("           >>> Live RGB preview open. Focus it and press Q/Enter "
              "to continue (Esc to abort).")
        win = "IMU init preview  [Q/Enter = continue, Esc = abort]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        aborted = False
        while True:
            if not streamer.wait_for_frame(timeout=0.5):
                continue
            rgb_np, _, _ = streamer.get_frame()
            if rgb_np is None:
                continue
            bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, "Q/Enter to continue, Esc to abort",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, bgr)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q'), 13):  # Q or Enter
                break
            if k == 27:  # Esc
                aborted = True
                break
        cv2.destroyWindow(win)
        if aborted:
            print("[INFO] Aborted by user")
            return False
    else:
        try:
            input("           >>> Then press Enter in this terminal to continue ... ")
        except KeyboardInterrupt:
            print("\n[INFO] Aborted by user")
            return False

    # Move to home
    print("\n=== Moving to home ===")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    ret = arm.set_servo_angle(angle=home_joints_deg, speed=20, wait=True)
    if ret != 0:
        print(f"  [ERROR] set_servo_angle to home returned {ret}")
        return False
    time.sleep(0.5)
    print("  Reached home.")

    return True
