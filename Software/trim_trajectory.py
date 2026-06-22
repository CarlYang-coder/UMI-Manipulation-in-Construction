"""
Trim trajectory CSV: remove IMU adjustment frames at the beginning.

Detection logic: During IMU adjustment, the gripper is closed (gripper_width ~ 0
or very small). When the user opens the gripper, that's the start of the real
demonstration. This script finds the first frame where gripper opens and trims
everything before it.

Usage:
    python trim_trajectory.py --csv Data/episode_xxxx/trajectory.csv
    python trim_trajectory.py --csv Data/episode_xxxx/trajectory.csv --threshold 0.5 --preview
"""

import csv
import argparse
import shutil
import os


def find_gripper_open_frame(rows, threshold, min_closed_frames=30):
    """
    Find the first frame where gripper transitions from closed to open.

    Args:
        rows: list of CSV row dicts
        threshold: gripper_width above this = open
        min_closed_frames: require at least this many closed frames before the open

    Returns:
        frame index of first open, or 0 if not found
    """
    closed_count = 0
    for i, row in enumerate(rows):
        width = float(row['gripper_width'])
        if width < threshold:
            closed_count += 1
        else:
            if closed_count >= min_closed_frames:
                return i
            closed_count = 0
    return 0


def main():
    parser = argparse.ArgumentParser(description="Trim IMU adjustment frames from trajectory CSV")
    parser.add_argument("--csv", required=True, help="Path to trajectory.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Gripper width threshold: below=closed, above=open (default: 0.5)")
    parser.add_argument("--preview", action="store_true",
                        help="Only show where the cut would be, don't modify the file")
    parser.add_argument("--no_backup", action="store_true",
                        help="Don't create a backup file")
    args = parser.parse_args()

    # Load CSV
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} frames from {args.csv}")

    # Show gripper width progression
    print("\nGripper width at key frames:")
    for i in range(0, min(len(rows), 500), 30):
        w = float(rows[i]['gripper_width'])
        status = "CLOSED" if w < args.threshold else "OPEN"
        print(f"  [{i:4d}] gripper_width={w:.4f}  {status}")

    # Find trim point
    trim_idx = find_gripper_open_frame(rows, args.threshold)

    if trim_idx == 0:
        print("\nNo closed->open transition found. No trimming needed.")
        return

    print(f"\nFound gripper open at frame {trim_idx}")
    print(f"  Before: gripper_width={float(rows[trim_idx-1]['gripper_width']):.4f} (closed)")
    print(f"  After:  gripper_width={float(rows[trim_idx]['gripper_width']):.4f} (open)")
    print(f"  Will remove {trim_idx} frames, keeping {len(rows) - trim_idx} frames")

    if args.preview:
        print("\n[Preview mode] No changes made.")
        return

    # Backup
    if not args.no_backup:
        backup_path = args.csv + '.pretrim'
        shutil.copy(args.csv, backup_path)
        print(f"  Backup saved to {backup_path}")

    # Trim and save
    trimmed_rows = rows[trim_idx:]
    with open(args.csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trimmed_rows)

    print(f"  Saved {len(trimmed_rows)} frames to {args.csv}")


if __name__ == "__main__":
    main()
