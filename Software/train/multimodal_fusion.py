"""
Multimodal Fusion Module for UMI-FT.

Following UMI-FT paper:
  "The resulting tokens from the RGB, depth are passed to a transformer
   encoder layer with self-attention to obtain a combined visual
   representation."

  "the output of this fusion layer is concatenated with low-dimensional
   proprioceptive data, including end-effector pose from the previous
   two timesteps."

Architecture:
  RGB token  (from CLIP ViT-B/32)  -> [CLS_rgb]  \
                                                    -> TransformerEncoder -> fused_visual
  Depth token (from CLIP ViT-B/32) -> [CLS_depth] /

  final_cond = concat(fused_visual, temporal_agg(ee_pose))
"""

import torch
import torch.nn as nn
import math


class MultimodalFusion(nn.Module):
    """
    Transformer-based fusion of RGB and Depth visual tokens.

    Input:
        rgb_feat:   (B, D_rgb)    -- from CLIP RGB encoder
        depth_feat: (B, D_depth)  -- from CLIP Depth encoder

    Output:
        fused: (B, D_fused)       -- combined visual representation

    The tokens are passed through a TransformerEncoder with self-attention.
    A learnable modality embedding is added to distinguish RGB vs Depth.
    """

    def __init__(
        self,
        rgb_dim: int = 512,
        depth_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model if needed
        self.rgb_proj = nn.Linear(rgb_dim, d_model) if rgb_dim != d_model else nn.Identity()
        self.depth_proj = nn.Linear(depth_dim, d_model) if depth_dim != d_model else nn.Identity()

        # Learnable modality embeddings
        self.rgb_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.depth_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection: concatenate all tokens -> project to output dim
        # 2 tokens (RGB + Depth) -> fused
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
        )

        self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        """
        rgb_feat:   (B, D_rgb)
        depth_feat: (B, D_depth)
        returns:    (B, D_fused)
        """
        B = rgb_feat.shape[0]

        # Project to d_model
        rgb_tok = self.rgb_proj(rgb_feat).unsqueeze(1)    # (B, 1, d_model)
        depth_tok = self.depth_proj(depth_feat).unsqueeze(1)  # (B, 1, d_model)

        # Add modality embeddings
        rgb_tok = rgb_tok + self.rgb_type_embed
        depth_tok = depth_tok + self.depth_type_embed

        # Concatenate tokens: [RGB, Depth]
        tokens = torch.cat([rgb_tok, depth_tok], dim=1)  # (B, 2, d_model)

        # Self-attention fusion
        fused = self.transformer(tokens)  # (B, 2, d_model)

        # Concatenate all tokens and project
        fused_flat = fused.reshape(B, -1)  # (B, 2*d_model)
        out = self.output_proj(fused_flat)  # (B, d_model)

        return out
