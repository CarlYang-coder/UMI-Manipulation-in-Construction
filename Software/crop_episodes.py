"""
Crop episodes: remove leading frames where gripper is closed (gripper_width == 0).
For each episode in Data_Raw/data/, finds the first frame where gripper opens,
then copies everything from that frame onward into Data_Raw/data_crop/.
Generates new trajectory.csv, rgb/, depth/ images, and rgb.mp4/depth.mp4 videos.
"""

import os
import csv
import shutil
import cv2
import glob


SRC_DIR = os.path.join("Data_Raw", "data")
DST_DIR = os.path.join("Data_Raw", "data_crop")


def find_first_open(csv_path):
    """Return the 0-based row index of the first row where gripper_width > 0.85."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if float(row["gripper_width"]) > 0.85:
                return i
    return None


def images_to_video(image_dir, video_path, fps=30.0):
    """Create mp4 video from a directory of png images."""
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not images:
        return
    frame = cv2.imread(images[0])
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for img_path in images:
        writer.write(cv2.imread(img_path))
    writer.release()


def process_episode(episode_name):
    src_episode = os.path.join(SRC_DIR, episode_name)
    csv_path = os.path.join(src_episode, "trajectory.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] No trajectory.csv in {episode_name}")
        return

    start_idx = find_first_open(csv_path)
    if start_idx is None:
        print(f"  [SKIP] Gripper never opens in {episode_name}")
        return

    # Read all rows
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    kept_rows = all_rows[start_idx:]
    total = len(all_rows)
    kept = len(kept_rows)
    print(f"  Cropping {episode_name}: removing first {start_idx} frames, keeping {kept}/{total}")

    # Create output dirs (clean old results first)
    dst_episode = os.path.join(DST_DIR, episode_name)
    dst_rgb = os.path.join(dst_episode, "rgb")
    dst_depth = os.path.join(dst_episode, "depth")
    if os.path.exists(dst_episode):
        shutil.rmtree(dst_episode)
    os.makedirs(dst_rgb)
    os.makedirs(dst_depth)

    # Copy images with new sequential numbering and write new csv
    new_csv_path = os.path.join(dst_episode, "trajectory.csv")
    with open(new_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for new_idx, row in enumerate(kept_rows):
            new_name = f"{new_idx:06d}.png"
            # Copy rgb
            src_rgb = os.path.join(src_episode, row["rgb_path"])
            if os.path.isfile(src_rgb):
                shutil.copy2(src_rgb, os.path.join(dst_rgb, new_name))
            # Copy depth
            src_depth = os.path.join(src_episode, row["depth_path"])
            if os.path.isfile(src_depth):
                shutil.copy2(src_depth, os.path.join(dst_depth, new_name))
            # Update paths in csv
            row["rgb_path"] = f"rgb/{new_name}"
            row["depth_path"] = f"depth/{new_name}"
            writer.writerow(row)

    # Generate videos
    print(f"  Generating rgb.mp4 ...")
    images_to_video(dst_rgb, os.path.join(dst_episode, "rgb.mp4"))
    print(f"  Generating depth.mp4 ...")
    images_to_video(dst_depth, os.path.join(dst_episode, "depth.mp4"))


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"Source directory not found: {SRC_DIR}")
        return

    os.makedirs(DST_DIR, exist_ok=True)
    episodes = sorted([d for d in os.listdir(SRC_DIR)
                        if os.path.isdir(os.path.join(SRC_DIR, d)) and not d.startswith(".")])

    print(f"Found {len(episodes)} episodes in {SRC_DIR}")
    for ep in episodes:
        process_episode(ep)

    print("\nDone! Cropped data saved to", DST_DIR)


if __name__ == "__main__":
    main()
