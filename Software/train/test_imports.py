import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")

import numpy as np
import torch
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation

from umi_ft_policy import UMIFTDiffusionPolicy
from umi_dataset import get_rgb_transform, load_hand_eye_calibration, pose_to_matrix, matrix_to_pose_7d
# NOT importing: GripperWidthTracker, imu_init_sequence
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

print("A: torch.load", flush=True)
ckpt = torch.load(r"D:\UMI_Gripper\train\checkpoints\best.pt", map_location="cpu", weights_only=False)
config = ckpt["config"]

print("B: create policy", flush=True)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=config["num_train_timesteps"],
    beta_schedule="squaredcos_cap_v2",
    clip_sample=False,
    prediction_type="epsilon",
)
policy = UMIFTDiffusionPolicy(
    action_dim=config["action_dim"],
    ee_pose_dim=config["action_dim"],
    horizon=config["horizon"],
    n_action_steps=config["n_action_steps"],
    n_obs_steps=config["n_obs_steps"],
    clip_out_dim=config["clip_out_dim"],
    fusion_d_model=config["fusion_d_model"],
    fusion_nhead=config["fusion_nhead"],
    fusion_num_layers=config["fusion_num_layers"],
    fusion_dim_feedforward=config["fusion_dim_feedforward"],
    noise_scheduler=noise_scheduler,
    num_inference_steps=config["num_infer_steps"],
)

print("C: load_state_dict + eval", flush=True)
policy.load_state_dict(ckpt["policy"])
policy.eval()

print("ALL OK", flush=True)
