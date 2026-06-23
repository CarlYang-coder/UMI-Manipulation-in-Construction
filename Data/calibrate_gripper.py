"""
Standalone gripper calibration entry point.

Connects to iPhone via Record3D and runs ArUco-based gripper
range calibration. Saves result to gripper_range.json.

Usage:
    python calibrate_gripper.py
    python calibrate_gripper.py --output my_calibration.json
"""

import argparse
from Demonstration import DataCollector


def main():
    parser = argparse.ArgumentParser(description="UMI-FT Gripper Calibration")
    parser.add_argument("--dev_idx", type=int, default=0,
                        help="Record3D device index")
    parser.add_argument("--aruco_config", type=str, default="config/aruco_config.yaml",
                        help="Path to ArUco configuration YAML")
    parser.add_argument("--output", type=str, default="gripper_range.json",
                        help="Output path for calibration JSON")
    args = parser.parse_args()

    collector = DataCollector(
        dev_idx=args.dev_idx,
        aruco_config_path=args.aruco_config,
        calibration_path=args.output,
    )
    collector.run_calibration()


if __name__ == "__main__":
    main()
