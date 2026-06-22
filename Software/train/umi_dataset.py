"""
UMI-FT Dataset for Diffusion Policy Training.

Loads episodes from Data_Raw/data_crop/ with:
  - RGB images (iPhone camera, 1920x1440 -> resized to 224x224)
  - Depth images (iPhone LiDAR, 256x192 -> resized to 224x224, clipped at 0.5m)
  - EE poses computed from camera poses via hand-eye calibration
  - Gripper width

Following UMI-FT paper:
  - Observation: 2 previous timesteps of (RGB, Depth, EE_pose)
  - Action: EE_pose trajectory over horizon
  - EE_pose = [tx, ty, tz, rx, ry, rz, gripper_width] (7D)
    where (tx,ty,tz) in meters, (rx,ry,rz) rotation vector in radians,
    gripper_width normalized [0,1]
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Image transforms ───────────────────────────────────────────────
# Official UMI-FT pipeline:
#   - Dataset outputs images in [0, 1] range (no CLIP/ImageNet normalize)
#   - Augmentation (RandomCrop + ColorJitter) is applied INSIDE the obs_encoder
#   - The identity normalizer in LinearNormalizer passes images through unchanged
# So the dataset transform is just: Resize + ToTensor.

def get_rgb_transform(img_size: int = 224):
    """Deterministic RGB transform — outputs [0, 1] tensor (no normalization)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # -> [0, 1]
    ])


def get_depth_transform(img_size: int = 224):
    """Deterministic depth transform — outputs [0, 1] tensor."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # -> [0, 1]
    ])


# Backwards-compat aliases
get_rgb_transform_train = get_rgb_transform
get_depth_transform_train = get_depth_transform


# ─── Pose utilities ─────────────────────────────────────────────────

def pose_to_matrix(tx, ty, tz, rx, ry, rz):
    """Convert translation + rotation vector to 4x4 SE(3) matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_pose_7d(T, gripper_width):
    """Convert 4x4 SE(3) matrix + gripper to 7D pose vector.
    Returns [tx, ty, tz, rx, ry, rz, gripper_width]."""
    pos = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.array([pos[0], pos[1], pos[2],
                     rotvec[0], rotvec[1], rotvec[2],
                     gripper_width], dtype=np.float32)


def load_hand_eye_calibration(cal_path: str) -> np.ndarray:
    """Load T_cam_to_ee from hand-eye calibration result."""
    with open(cal_path) as f:
        cal = json.load(f)
    return np.array(cal['tx_cam_to_ee'])


# ─── Dataset ────────────────────────────────────────────────────────

class UMIDataset(Dataset):
    """
    UMI-FT dataset for Diffusion Policy training.

    Each sample returns:
        obs: {
            "rgb":   (To, 3, 224, 224)   -- CLIP-normalized RGB
            "depth": (To, 3, 224, 224)   -- depth clipped at 0.5m, 3-channel
            "ee_pose": (To, 7)           -- [tx,ty,tz,rx,ry,rz,gripper]
        }
        action: (T, 7)                   -- EE pose trajectory
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        calibration_path: Union[str, Path],
        n_obs_steps: int = 2,
        horizon: int = 20,
        img_size: int = 224,
        depth_clip_m: float = 0.5,
        stride: int = 1,
        max_episodes: Optional[int] = None,
        gripper_min: float = 0.015,
        gripper_max: float = 0.085,
        rgb_transform=None,
        depth_transform=None,
        # Official UMI-FT down_sample_steps (task yaml)
        rgb_down_sample_steps: int = 10,
        lowdim_down_sample_steps: int = 5,
        action_down_sample_steps: int = 10,
        # Episode-based train/val split (official UMI-FT)
        val_ratio: float = 0.05,
        split_seed: int = 42,
        is_validation: bool = False,
        dtype=torch.float32,
        # kept for backward-compat but ignored (augmentation is done in encoder)
        augment: bool = False,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.img_size = img_size
        self.depth_clip_m = depth_clip_m
        self.stride = stride
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.rgb_down_sample_steps = rgb_down_sample_steps
        self.lowdim_down_sample_steps = lowdim_down_sample_steps
        self.action_down_sample_steps = action_down_sample_steps
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.is_validation = is_validation
        self.dtype = dtype

        if rgb_transform is None:
            rgb_transform = get_rgb_transform(img_size)
        if depth_transform is None:
            depth_transform = get_depth_transform(img_size)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        # Load hand-eye calibration
        self.calibration_path = calibration_path
        self.T_cam_to_ee = load_hand_eye_calibration(str(calibration_path))
        self.T_ee_to_cam = np.linalg.inv(self.T_cam_to_ee)

        # Discover episodes
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()
                               and d.name.startswith("episode_")])
        if max_episodes is not None:
            episode_dirs = episode_dirs[:max_episodes]

        # Episode-based train/val split (official UMI-FT)
        all_indices = np.arange(len(episode_dirs))
        rng = np.random.RandomState(split_seed)
        shuffled = rng.permutation(all_indices)
        n_val = max(1, int(round(len(episode_dirs) * val_ratio))) if len(episode_dirs) > 1 else 0
        val_set = set(shuffled[:n_val].tolist())
        train_set = set(shuffled[n_val:].tolist())
        keep = val_set if is_validation else train_set
        episode_dirs = [ep for i, ep in enumerate(episode_dirs) if i in keep]

        self.episodes = []
        self.index = []  # (episode_idx, start_frame)

        missing_ee_csv = []
        for ei, ep_dir in enumerate(episode_dirs):
            ee_csv_path = ep_dir / "ee_pose_trajectory.csv"
            csv_path = ep_dir / "trajectory.csv"
            if not csv_path.exists():
                continue
            if not ee_csv_path.exists():
                missing_ee_csv.append(ep_dir.name)
                continue

            # Load trajectory (for image paths)
            frames = self._load_trajectory(csv_path, ep_dir)
            if len(frames) < n_obs_steps + horizon:
                continue

            # Load pre-computed EE poses from ee_pose_trajectory.csv
            # (produced offline by generate_ee_pose_csv.py)
            ee_poses = self._load_ee_poses(ee_csv_path)  # (N, 7)
            if len(ee_poses) != len(frames):
                raise RuntimeError(
                    f"{ep_dir.name}: trajectory.csv has {len(frames)} frames "
                    f"but ee_pose_trajectory.csv has {len(ee_poses)}. "
                    f"Re-run `python generate_ee_pose_csv.py`.")

            episode = {
                "dir": ep_dir,
                "frames": frames,
                "ee_poses": ee_poses,
                "N": len(frames),
            }
            self.episodes.append(episode)

            # Build sample index with down_sample_steps (official UMI-FT)
            # Observation indices (relative to anchor s): [s - (To-1)*lowdim_dss, ..., s]
            #   for lowdim; [s - (To-1)*rgb_dss, ..., s] for rgb (same To).
            # Action indices (relative to s): [s, s+act_dss, s+2*act_dss, ..., s+(H-1)*act_dss]
            rgb_dss = rgb_down_sample_steps
            ld_dss = lowdim_down_sample_steps
            act_dss = action_down_sample_steps

            obs_back = max((n_obs_steps - 1) * rgb_dss,
                           (n_obs_steps - 1) * ld_dss)
            action_fwd = (horizon - 1) * act_dss

            N = len(frames)
            for s in range(obs_back, N - action_fwd, stride):
                self.index.append((len(self.episodes) - 1, s))

        if missing_ee_csv:
            print(f"[UMIDataset] WARNING: {len(missing_ee_csv)} episode(s) missing "
                  f"ee_pose_trajectory.csv — run `python generate_ee_pose_csv.py` first. "
                  f"Skipped: {missing_ee_csv[:5]}{'...' if len(missing_ee_csv) > 5 else ''}")

        assert len(self.index) > 0, (
            f"No valid training windows found in {data_dir}. "
            f"Did you run `python generate_ee_pose_csv.py`?")
        split_tag = "val" if is_validation else "train"
        print(f"[UMIDataset/{split_tag}] {len(self.episodes)} episodes, "
              f"{len(self.index)} samples (To={n_obs_steps}, T={horizon})")

    def get_validation_dataset(self):
        """Return a sibling dataset with the held-out validation episodes
        (official UMI-FT pattern)."""
        return UMIDataset(
            data_dir=self.data_dir,
            calibration_path=self.calibration_path,
            n_obs_steps=self.n_obs_steps,
            horizon=self.horizon,
            img_size=self.img_size,
            depth_clip_m=self.depth_clip_m,
            stride=self.stride,
            gripper_min=self.gripper_min,
            gripper_max=self.gripper_max,
            rgb_down_sample_steps=self.rgb_down_sample_steps,
            lowdim_down_sample_steps=self.lowdim_down_sample_steps,
            action_down_sample_steps=self.action_down_sample_steps,
            val_ratio=self.val_ratio,
            split_seed=self.split_seed,
            is_validation=True,
            dtype=self.dtype,
        )

    def _load_trajectory(self, csv_path: Path, ep_dir: Path) -> list:
        """Load trajectory CSV, return list of frame dicts."""
        frames = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append({
                    'timestamp': float(row['timestamp']),
                    'tx': float(row['tx']),
                    'ty': float(row['ty']),
                    'tz': float(row['tz']),
                    'rx': float(row['qx']),  # rotation vector, not quaternion
                    'ry': float(row['qy']),
                    'rz': float(row['qz']),
                    'gripper_width': float(row['gripper_width']),
                    'rgb_path': str(ep_dir / row['rgb_path']),
                    'depth_path': str(ep_dir / row['depth_path']),
                })
        return frames

    def _load_ee_poses(self, ee_csv_path: Path) -> np.ndarray:
        """Load pre-computed EE poses from ee_pose_trajectory.csv.

        The CSV is produced offline by generate_ee_pose_csv.py, with columns:
          timestamp, tx, ty, tz, qx, qy, qz, gripper_width, rgb_path, depth_path
        where tx..qz are body-frame accumulated EE pose (from identity) and
        qx,qy,qz are a rotation vector (radians). gripper_width is already
        normalized to [0, 1].
        """
        poses = []
        with open(ee_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                poses.append([
                    float(row['tx']), float(row['ty']), float(row['tz']),
                    float(row['qx']), float(row['qy']), float(row['qz']),
                    float(row['gripper_width']),
                ])
        return np.array(poses, dtype=np.float32)

    def _compute_ee_poses(self, frames: list) -> np.ndarray:
        """Convert camera trajectory to EE poses in a consistent frame.

        We use the first frame's EE pose as origin, then accumulate
        body-frame deltas to get EE poses for all subsequent frames.
        This produces smooth, robot-centric pose trajectories.
        """
        N = len(frames)

        # Camera poses as SE(3)
        cam_matrices = [pose_to_matrix(f['tx'], f['ty'], f['tz'],
                                       f['rx'], f['ry'], f['rz'])
                        for f in frames]

        # Gripper width in CSV is already normalized to [0, 1] range
        # (done during data collection via ArUco calibration).
        # If values are > 1, they need re-normalization from raw meters;
        # otherwise use directly.
        sample_gw = [f['gripper_width'] for f in frames[:10]]
        gw_already_normalized = all(0 <= g <= 1.1 for g in sample_gw if g > 0)

        if not gw_already_normalized:
            gripper_range = self.gripper_max - self.gripper_min
            if gripper_range <= 0:
                gripper_range = 1.0
        else:
            gripper_range = None  # flag: already normalized

        # Start from identity for EE (relative trajectory)
        T_ee = np.eye(4)
        ee_poses = np.zeros((N, 7), dtype=np.float32)

        def _normalize_gw(raw):
            if gripper_range is not None:
                return np.clip((raw - self.gripper_min) / gripper_range, 0, 1)
            else:
                return np.clip(raw, 0, 1)

        ee_poses[0] = matrix_to_pose_7d(T_ee, _normalize_gw(frames[0]['gripper_width']))

        for i in range(1, N):
            # Body-frame delta in camera frame
            delta_cam = np.linalg.inv(cam_matrices[i - 1]) @ cam_matrices[i]
            # Convert to EE body-frame delta
            delta_ee = self.T_cam_to_ee @ delta_cam @ self.T_ee_to_cam
            # Accumulate
            T_ee = T_ee @ delta_ee

            ee_poses[i] = matrix_to_pose_7d(T_ee, _normalize_gw(frames[i]['gripper_width']))

        return ee_poses

    def __len__(self) -> int:
        return len(self.index)

    def _load_depth(self, path):
        """Load depth image, convert to [0,1] float clipped at depth_clip_m."""
        depth_pil = Image.open(path)
        depth_np = np.array(depth_pil).astype(np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np.mean(axis=2)
        # Convert to meters if needed
        if depth_np.max() > 1.0:
            depth_np = depth_np / 1000.0
        if depth_np.max() > 10.0:
            depth_np = depth_np / 255.0
        # Clip and normalize
        depth_np = np.clip(depth_np, 0, self.depth_clip_m) / self.depth_clip_m
        return depth_np

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ei, s = self.index[idx]
        ep = self.episodes[ei]
        To = self.n_obs_steps
        T = self.horizon
        rgb_dss = self.rgb_down_sample_steps
        ld_dss = self.lowdim_down_sample_steps
        act_dss = self.action_down_sample_steps

        # Observation indices (down-sampled): anchor at s, step back by dss
        # rgb_obs_indices:    [s - (To-1)*rgb_dss, ..., s - rgb_dss, s]
        # lowdim_obs_indices: [s - (To-1)*ld_dss,  ..., s - ld_dss,  s]
        rgb_obs_indices = [s - (To - 1 - t) * rgb_dss for t in range(To)]
        ld_obs_indices = [s - (To - 1 - t) * ld_dss for t in range(To)]

        # Action indices: [s, s + act_dss, ..., s + (T-1)*act_dss]
        act_indices = [s + t * act_dss for t in range(T)]

        rgb_imgs = []
        depth_imgs = []

        for i in rgb_obs_indices:
            # ─── RGB ───
            rgb_pil = Image.open(ep['frames'][i]['rgb_path']).convert('RGB')
            rgb_tensor = self.rgb_transform(rgb_pil)
            rgb_imgs.append(rgb_tensor)

            # ─── Depth ───
            depth_np = self._load_depth(ep['frames'][i]['depth_path'])
            depth_pil_gray = Image.fromarray(
                (depth_np * 255).astype(np.uint8), mode='L')
            depth_tensor = self.depth_transform(depth_pil_gray)
            # Expand to 3 channels in [0, 1] (official: no normalize in dataset)
            depth_3ch = depth_tensor.repeat(3, 1, 1)
            depth_imgs.append(depth_3ch)

        rgb = torch.stack(rgb_imgs, dim=0).to(self.dtype)
        depth = torch.stack(depth_imgs, dim=0).to(self.dtype)

        # EE pose observations (lowdim down-sampling)
        ee_obs = torch.from_numpy(
            ep['ee_poses'][ld_obs_indices]).to(self.dtype)

        # Action: EE pose trajectory (action down-sampling)
        action = torch.from_numpy(
            ep['ee_poses'][act_indices]).to(self.dtype)

        return {
            "obs": {
                "rgb": rgb,
                "depth": depth,
                "ee_pose": ee_obs,
            },
            "action": action,
        }

    def get_normalizer(self, mode="limits"):
        """
        Compute normalizer for ee_pose and action fields.
        RGB and depth use identity (already CLIP-normalized).
        """
        # Import from local diffusion_policy copy
        dp_path = Path(__file__).resolve().parent / "diffusion_policy"
        if not dp_path.exists():
            # Fallback to Image_DP
            dp_path = Path("D:/Image_DP/video_train")
        sys.path.insert(0, str(dp_path.parent))

        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer, SingleFieldLinearNormalizer,
        )

        all_ee_pose = []
        all_action = []
        for ep in self.episodes:
            all_ee_pose.append(torch.from_numpy(ep['ee_poses']))
            all_action.append(torch.from_numpy(ep['ee_poses']))  # same space

        all_ee_pose = torch.cat(all_ee_pose, dim=0)
        all_action = torch.cat(all_action, dim=0)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"ee_pose": all_ee_pose, "action": all_action},
            mode=mode, last_n_dims=1,
        )
        # Vision inputs are already normalized
        normalizer["rgb"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["depth"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
