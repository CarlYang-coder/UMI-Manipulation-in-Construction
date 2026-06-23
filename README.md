# UMI-Manipulation in Construction

Building upon **UMI-FT** (https://github.com/real-stanford/UMI-FT) and its iPhone-based data collection framework, this project establishes an end-to-end workflow for **construction robotics**.

Looking ahead, we aim to extend the system to **mobile robotic platforms**, enabling more complex construction manipulation tasks in **dynamic** and **large-scale environments**.

## Hardware Design

For hardware-related files, designs, and supporting materials, please refer to the [Hardware](./Hardware) directory.

<p align="center">
  <img src="./Hardware/image/hardware-design-1.png" alt="Hardware design 1" height="320" />
  <img src="./Hardware/image/hardware-design-2.png" alt="Hardware design 2" height="320" />
</p>

The hardware system is designed to support construction-oriented robotic manipulation tasks in real-world environments. It provides an integrated platform for demonstration collection, policy depl[...]

## Data Collection

### RGBD Demonstration Data Collection

To collect 3D demonstration data, an **iPhone 15 Pro or newer** is required, as these devices provide depth sensing capabilities through the built-in **LiDAR sensor**.

#### Procedure

1. Install the [**Record3D**](https://record3d.app/) application on the iPhone.
2. Launch the application and use the **Record** function to capture demonstrations.
3. Each recording is saved as an `.r3d` file, containing synchronized **RGB images**, **depth data**, and **camera pose information** for downstream processing.

### Data Processing

All software (scripts, configs, MuJoCo models, calibration data, training and
inference code) lives in the [`Software/`](./Software) directory. Large
artifacts — raw demonstrations and trained checkpoints — are hosted on
Google Drive (see [Software/README.md](./Software/README.md) for download links).

After collecting `.r3d` recordings with Record3D, run the following from
`Software/`:

```bash
# 1. Convert .r3d to per-episode frames + trajectory.csv
python Demonstration.py            # writes Data_Raw/data/episode_XXXX/...

# 2. Crop episodes to start at first gripper-open frame
python crop_episodes.py            # writes Data_Raw/data_crop/

# 3. (Optional) Trim noisy start / end
python trim_trajectory.py

# 4. (Optional) Downsample very long episodes
python downsample_episodes.py

# 5. Generate normalized EE-pose CSV used by training
python generate_ee_pose_csv.py     # writes ee_pose_trajectory.csv per episode
```

Each cropped episode layout:
```
Data_Raw/data_crop/episode_XXXX_TIMESTAMP/
├── rgb/000000.png ...     # 1920×1440 RGB frames
├── depth/000000.png ...   # 256×192 depth frames
├── rgb.mp4, depth.mp4     # quick-look videos
├── trajectory.csv         # iPhone camera pose + image paths
└── ee_pose_trajectory.csv # body-frame accumulated EE pose (generated)
```

## Calibration

Three calibrations are needed before any replay or rollout. All scripts live
under `Software/`:

| Step | Script | Output |
|------|--------|--------|
| 1. Gripper width scale (ArUco → meters) | `calibrate_gripper.py` | `Software/calibration/aruco_config.yaml` (already provided) |
| 2. Hand-eye data collection (xArm + iPhone) | `calibrate_hand_eye_collect.py` | `calibration/hand_eye_data.json` |
| 3. Hand-eye solve (PARK method) | `calibrate_hand_eye_solve.py` | `calibration/hand_eye_result_umi.json` (`T_cam_to_ee`) |
| 4. UMI ↔ xArm pose alignment | `calibrate_umi_xarm.py` | `train/pose_calibration_data.npz`, `pose_calibration_data_transform.npz` |

`Software/calibration/hand_eye_result_umi.json` defines the transform between
the iPhone camera frame and the UMI gripper EE frame. It is used everywhere
camera-to-EE conversion happens (data prep, training, replay, rollout).

## Trajectory Replay

Replay scripts let you verify a recorded demonstration on the **xArm7** before
training or rollout (also see the entry point
[`Software/replay_trajectory.py`](./Software/replay_trajectory.py)):

```bash
# 1. Replay a raw camera-pose trajectory (MuJoCo preview + real arm)
python replay_trajectory.py \
       --csv "Data_Raw/data_crop/episode_0000_.../trajectory.csv" \
       --ip 192.168.1.224 --speed_scale 0.3

# 2. Replay the body-frame EE pose used by training
python replay_ee_pose.py            # edit DATA_DIR in the file

# 3. Replay + record live RGBD for verification
python replay_with_recording.py

# 4. MuJoCo-only simulation (no real robot)
python sim_replay.py
```

All replay scripts apply the same `T_target = T_home @ R_site_inv @ T_body @ R_site`
transform that the policy uses at inference time, so they are a great way to
sanity-check coordinate-frame fixes before training.

## Training

The diffusion-policy training code follows the official **UMI-FT** pattern
(`Software/train/train_umi_ft.py`):

- **Backbone**: timm ViT-B/32 CLIP (`vit_base_patch32_clip_224.openai`), not frozen
- **Encoder**: separate ViT for RGB and depth, fused via
  `TransformerEncoderLayer` (modality-attention) — see
  `Software/train/timm_obs_encoder_umi.py`
- **Diffusion**: DDIM scheduler, `ConditionalUnet1D` (down_dims [256,512,1024])
- **Action**: 7D `[tx, ty, tz, rx, ry, rz, gripper_width]` (3D rotvec)
- **Horizon**: 20 sparse steps (≈3.3 s); `n_obs_steps = 2`
- **Aug**: `RandomCrop(ratio=0.95) + ColorJitter(0.3, 0.4, 0.5, 0.08)` inside the encoder
- **Optimizer**: AdamW, `lr = 3e-4`, weight_decay 1e-6, betas `(0.95, 0.999)`,
  vision backbone uses `lr × 0.1` (grouped LR)
- **Schedule**: cosine with 2000-step warmup, 300 epochs (`num_epochs` in cfg)
- **EMA**: enabled, `power=0.75`, `max_value=0.9999`

Run:
```bash
cd Software/train
python train_umi_ft.py
```

Adjust `batch_size`, `gradient_accumulate_every`, `num_workers`, and
`num_epochs` inside `cfg = dict(...)` to match your GPU/RAM. Checkpoints land
in `train/checkpoints/`:
- `best.pt` — lowest validation `naction_mse`
- `latest.pt` / `latest.ckpt` — most recent state
- `checkpoints/` — TopK ckpts by `train_loss`
- `sparse_normalizer.pkl` — observation/action normalizer

A pre-trained `best.pt` is available on Google Drive (see
[Software/README.md](./Software/README.md#external-assets)).

## Inference (Real-Robot Rollout)

Live policy rollout on the xArm7 (`Software/train/run_umi_ft.py`):

```bash
cd Software/train
python run_umi_ft.py
```

Pipeline per step (≈15 Hz):

1. **Observation**: Record3D delivers RGB + depth + camera pose; gripper
   width is read from the xArm.
2. **EE pose**: body-frame camera-delta is accumulated and converted via the
   hand-eye matrix into UMI EE space — identical to the dataset pipeline.
3. **Policy**: ViT CLIP encodes RGB + depth; modality-attention fuses them
   with the 7-D proprioception; a DDIM-sampled UNet returns a 20-step action
   chunk.
4. **Receding-horizon execution**: the first `EXECUTION_HORIZON = 4` actions
   are sent (overlap ratio 20 %, matching official UMI-FT).
5. **xArm conversion**: each action is mapped from UMI body-frame to xArm
   world coordinates via `T_target = T_home @ R_site_inv @ T_body @ R_site`,
   EMA-smoothed, and dispatched with `arm.set_servo_cartesian(...)`.

Offline evaluation utilities live next to the runner:
- `train/eval_closedloop.py` — feed dataset frames to the policy and compare
  predictions against ground truth.
- `train/eval_offline.py` / `eval_replay.py` — replay-style debugging.

## Multi-Model

Please check the diffrent [Models](./Software/train) below.
<p align="center">
  <img src="./Hardware/image/multi-model-1.png" alt="Hardware design 1" height="320" />
</p>
The first model is a lightweight baseline trained on 30 demos for comparison. It does not incorporate depth information, and the training and inference results show that it is only capable of completing localized actions.
<p align="center">
  <img src="./Hardware/image/multi-model-2.png" alt="Hardware design 2" height="320" />
</p>
The second model incorporates depth information and partially follows the architecture proposed in the original paper, but does not include tactile sensing. The number of demos is increased to 100.

