"""
CLIP ViT-B/32 Vision Encoder for UMI-FT.

Following UMI-FT paper:
  "iPhone RGB images from the two previous timesteps, with each image
   processed by a CLIP-pretrained ViT-B/32 encoder. Images are resized
   to 224x224 pixels."

  "Depth input from the iPhone is processed using the same vision encoder
   architecture as RGB."

Each encoder takes (B, T, 3, H, W) video input and produces (B, D) features
by processing each frame independently through CLIP ViT-B/32 and then
pooling over time.
"""

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn


class CLIPVideoEncoder(nn.Module):
    """
    CLIP ViT-B/32 based video encoder.

    Input:  (B, T, 3, H, W) -- T frames of 224x224 images
    Output: (B, out_dim)     -- aggregated feature

    Architecture:
      1. CLIP ViT-B/32 visual encoder (frozen or fine-tunable)
      2. Trainable projection MLP: 512 -> mlp_hidden -> out_dim
      3. Temporal pooling: mean or last
    """

    def __init__(
        self,
        out_dim: int = 512,
        pool: Literal["mean", "last"] = "mean",
        mlp_hidden: int = 512,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.pool = pool
        self.freeze_backbone = freeze_backbone

        # Load CLIP ViT-B/32
        try:
            import clip
            model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
            self.clip_visual = model.visual
            self.backbone_feat_dim = 512  # CLIP ViT-B/32 output dim
        except ImportError:
            # Fallback: use OpenCLIP
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai')
            self.clip_visual = model.visual
            self.backbone_feat_dim = 512

        # Convert to float32 (CLIP loads as float16)
        self.clip_visual = self.clip_visual.float()

        if freeze_backbone:
            for p in self.clip_visual.parameters():
                p.requires_grad = False

        # Trainable projection MLP
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(self.backbone_feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, out_dim),
        ])
        self.mlp = nn.Sequential(*layers)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLIP visual features from (N, 3, H, W) images."""
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.clip_visual(x)  # (N, 512)
        else:
            feat = self.clip_visual(x)
        return feat.float()  # ensure float32

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, 3, H, W)
        returns: (B, out_dim)
        """
        assert video.ndim == 5, f"Expected (B,T,C,H,W), got {tuple(video.shape)}"
        B, T, C, H, W = video.shape

        x = video.reshape(B * T, C, H, W)

        # Extract CLIP features
        feat = self._extract_features(x)  # (B*T, 512)
        proj = self.mlp(feat)              # (B*T, out_dim)
        proj = proj.reshape(B, T, self.out_dim)  # (B, T, out_dim)

        if self.pool == "mean":
            out = proj.mean(dim=1)
        elif self.pool == "last":
            out = proj[:, -1]
        else:
            raise ValueError(f"Unknown pool={self.pool}")

        return out
