# Hand-Eye Calibration Guide

This guide explains how to calibrate the rigid transform between the iPhone camera
(ARKit frame) and the xArm7 end-effector (TCP frame). This calibration is required
so that trajectory replay correctly maps iPhone motion to robot motion.

## Why Calibration Is Needed

When recording demonstrations, the iPhone is rigidly mounted on the xArm gripper.
iPhone ARKit provides camera poses in its own world frame, but the robot operates
in the robot base frame. These two coordinate systems are different:

| Direction | ARKit (landscape, camera forward) | xArm |
|-----------|-----------------------------------|------|
| Forward   | -Z                                | +X   |
| Left      | -X                                | +Y   |
| Up        | +Y                                | +Z   |

Without calibration, replayed trajectories will move in the wrong direction.

## Prerequisites

1. **Hardware setup**: iPhone mounted rigidly on xArm EE (same as demonstration setup)
2. **iPhone**: Record3D app installed, USB Streaming mode enabled
3. **Connection**: iPhone connected to PC via USB cable
4. **xArm**: Powered on, connected via Ethernet
5. **Python packages**:
   ```
   pip install record3d opencv-python opencv-contrib-python numpy scipy
   ```

## Calibration Steps

### Step 1: Collect Paired Pose Data

This script moves the xArm to 15 predefined poses. At each pose, it simultaneously
records the xArm EE pose and the iPhone ARKit pose.

```bash
# Before running:
# 1. Mount iPhone on xArm EE
# 2. Connect iPhone via USB
# 3. Open Record3D on iPhone, enable USB Streaming
# 4. Make sure xArm has enough space to move safely

python calibrate_hand_eye_collect.py --ip 192.168.1.224
```

Options:
- `--ip` : xArm IP address (default: 192.168.1.224)
- `--output` : Output file path (default: calibration/hand_eye_data.json)
- `--speed` : Joint move speed in deg/s (default: 15, slow for safety)
- `--settle_time` : Wait time at each pose in seconds (default: 2.0)

The robot will:
1. Move slowly to each calibration pose
2. Wait for the pose to settle
3. Read xArm EE position and ARKit camera position
4. Save all paired data to JSON

**Safety**: The predefined poses are conservative. The robot moves at 15 deg/s.
Keep the emergency stop button ready. Press Ctrl+C to abort.

### Step 2: Solve Calibration

This script reads the paired data and computes the camera-to-EE transform using
OpenCV's hand-eye calibration (tries 5 different methods, picks the best one).

```bash
python calibrate_hand_eye_solve.py
```

Options:
- `--input` : Input file from Step 1 (default: calibration/hand_eye_data.json)
- `--output` : Output calibration file (default: calibration/hand_eye_result.json)

The script will:
1. Convert all poses to SE(3) matrices
2. Run 5 calibration methods (Tsai, Park, Horaud, Andreff, Daniilidis)
3. Compute reprojection error for each method
4. Save the best result

### Step 3: Verify

Check the output:
- Translation should be roughly the physical offset between iPhone camera lens and
  the xArm TCP point (typically tens of mm)
- Rotation should reflect the iPhone mounting orientation
- Reprojection error should be < 5mm for translation and < 2 deg for rotation

### Step 4: Use in Replay

After calibration, `sim_replay.py` and `replay_trajectory.py` will automatically
load `calibration/hand_eye_result.json` and apply the coordinate transform.

(This integration is the next step after calibration is verified.)

## Output Files

```
calibration/
  hand_eye_data.json      <- Paired pose data (Step 1)
  hand_eye_result.json    <- Calibration result (Step 2)
```

### hand_eye_result.json format

```json
{
  "method": "PARK",
  "tx_cam_to_ee": [[...], [...], [...], [...]],
  "tx_ee_to_cam": [[...], [...], [...], [...]],
  "num_poses": 15,
  "avg_translation_error_mm": 2.34
}
```

- `tx_cam_to_ee`: 4x4 transform from iPhone camera frame to robot EE frame
- `tx_ee_to_cam`: 4x4 inverse transform

## Troubleshooting

**"No devices found"**
- Make sure Record3D is open on iPhone with USB Streaming enabled
- Try unplugging and replugging the USB cable
- On Windows, iTunes must be installed (for USB driver)

**"connect socket failed"**
- xArm is not reachable. Check IP address and network connection
- Try: `ping 192.168.1.224`

**Large reprojection error (> 10mm)**
- iPhone may not be rigidly mounted (loose mounting = inconsistent transform)
- Too few poses with enough variation. Edit CALIBRATION_POSES_DEG to add more
- ARKit may have drifted. Restart Record3D and try again

**Robot moves to unexpected positions**
- Check joint limits. Edit CALIBRATION_POSES_DEG if poses are out of range
- Reduce speed: `--speed 10`

## How It Works (Technical Details)

The hand-eye calibration solves the classic AX = XB problem:

Given N pairs of (T_base_to_ee, T_arkit_world_to_cam), we know:
- The relative motion of the EE between pose i and j: A = inv(T_ee_i) @ T_ee_j
- The relative motion of the camera between pose i and j: B = inv(T_cam_i) @ T_cam_j
- These must satisfy: X @ B = A @ X, where X = T_cam_to_ee

OpenCV's `cv2.calibrateHandEye()` solves for X using various closed-form and
iterative methods. We run all 5 available methods and pick the one with the
lowest reprojection error.

Reference: UMI official uses the same approach in
`scripts/calibrate_robot_world_hand_eye.py` with `cv2.calibrateRobotWorldHandEye()`.
We adapted it for xArm7 + iPhone ARKit (instead of UR5 + GoPro + ArUco tag).
