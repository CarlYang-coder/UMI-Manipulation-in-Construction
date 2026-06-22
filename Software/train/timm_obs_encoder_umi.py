"""
TimmObsEncoderUMI: aligned with official TimmObsEncoderWithForce (UMI-FT),
but without force/wrench encoder. Keeps:
  - timm-based ViT CLIP backbone (share_rgb_model=False -> separate RGB/Depth encoders)
  - attention_pool_2d feature aggregation (official default for non-ViT; for ViT,
    the official code falls back to CLS token when feature_aggregation is set)
  - Modality-attention fusion via TransformerEncoderLayer + learnable position embedding
  - Linear projection + lowdim concatenation
  - Internal data augmentation: RandomCrop(ratio) -> Resize + ColorJitter
"""

import copy
import logging
import math
from typing import Dict, List, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    """Same as official UMI-FT (CLIP-style attention pooling for ResNet/ConvNext)."""

    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)                  # (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)       # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None,
            add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training, need_weights=False,
        )
        return x.squeeze(0)


class TimmObsEncoderUMI(ModuleAttrMixin):
    """
    Observation encoder aligned with UMI-FT official TimmObsEncoderWithForce
    (without force encoder).

    Inputs are a dict:
        {rgb_key: (B, T, C, H, W), ..., lowdim_key: (B, T, D), ...}
    Output: (B, feature_dim) global conditioning vector.
    """

    def __init__(
        self,
        img_size: int = 224,
        rgb_keys: Sequence[str] = ("rgb", "depth"),
        lowdim_keys: Sequence[str] = ("ee_pose",),
        lowdim_dims: Dict[str, int] = None,
        lowdim_horizons: Dict[str, int] = None,
        rgb_horizons: Dict[str, int] = None,
        model_name: str = "vit_base_patch32_clip_224.openai",
        pretrained: bool = True,
        frozen: bool = False,
        share_rgb_model: bool = False,
        use_group_norm: bool = True,
        downsample_ratio: int = 32,
        feature_aggregation: str = "attention_pool_2d",
        position_encoding: str = "learnable",
        fuse_mode: str = "modality-attention",
        random_crop_ratio: float = 0.95,
        color_jitter=(0.3, 0.4, 0.5, 0.08),
        imagenet_norm: bool = False,  # official: flag exists but not wired; [0,1] fed to ViT directly
    ):
        super().__init__()

        assert lowdim_dims is not None
        assert lowdim_horizons is not None
        assert rgb_horizons is not None

        self.rgb_keys = list(rgb_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.lowdim_dims = dict(lowdim_dims)
        self.lowdim_horizons = dict(lowdim_horizons)
        self.rgb_horizons = dict(rgb_horizons)
        self.img_size = img_size
        self.share_rgb_model = share_rgb_model
        self.feature_aggregation = feature_aggregation
        self.position_encoding = position_encoding
        self.fuse_mode = fuse_mode
        self.model_name = model_name

        # ─── Build vision backbone ───
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool="",
            num_classes=0,
        )
        if frozen:
            assert pretrained
            for p in model.parameters():
                p.requires_grad = False

        # Feature dim per backbone
        if model_name.startswith("resnet"):
            if downsample_ratio == 32:
                model = nn.Sequential(*list(model.children())[:-2])
                feature_dim = 512
            elif downsample_ratio == 16:
                model = nn.Sequential(*list(model.children())[:-3])
                feature_dim = 256
            else:
                raise NotImplementedError(downsample_ratio)
        elif model_name.startswith("convnext"):
            if downsample_ratio == 32:
                model = nn.Sequential(*list(model.children())[:-2])
                feature_dim = 1024
            else:
                raise NotImplementedError(downsample_ratio)
        else:
            # ViT
            feature_dim = 768

        # BN -> GN (only for non-pretrained; same guard as official)
        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=((x.num_features // 16)
                                if x.num_features % 16 == 0
                                else (x.num_features // 8)),
                    num_channels=x.num_features,
                ),
            )

        self.feature_dim = feature_dim

        # ─── Per-key vision encoder (share or deepcopy) ───
        self.key_model_map = nn.ModuleDict()
        for key in self.rgb_keys:
            self.key_model_map[key] = model if share_rgb_model else copy.deepcopy(model)
        if not share_rgb_model:
            # free the template — it has been deep-copied
            del model

        # ─── Augmentation transform applied inside encoder ───
        transforms = []
        if random_crop_ratio is not None:
            transforms += [
                torchvision.transforms.RandomCrop(
                    size=int(img_size * random_crop_ratio)),
                torchvision.transforms.Resize(size=img_size, antialias=True),
            ]
        if color_jitter is not None:
            b, c, s, h = color_jitter
            transforms.append(torchvision.transforms.ColorJitter(
                brightness=b, contrast=c, saturation=s, hue=h))
        if imagenet_norm:
            transforms.append(torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))
        self.vision_transform = nn.Identity() if len(transforms) == 0 \
            else nn.Sequential(*transforms)

        # ─── Feature aggregation (for non-ViT) ───
        feature_map_shape = [img_size // downsample_ratio, img_size // downsample_ratio]
        if model_name.startswith("vit"):
            # ViT uses CLS token; aggregation is ignored (same as official)
            if feature_aggregation is not None:
                logger.warning(
                    f"vit uses CLS token; feature_aggregation ({feature_aggregation}) ignored")
                self.feature_aggregation = None
        else:
            if feature_aggregation == "attention_pool_2d":
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_shape[0],
                    embed_dim=feature_dim,
                    num_heads=feature_dim // 64,
                    output_dim=feature_dim,
                )

        # ─── Modality-attention fusion ───
        # Number of visual "tokens" = sum of each rgb key's horizon
        n_v_features = sum(self.rgb_horizons[k] for k in self.rgb_keys)
        n_features = n_v_features  # no force features

        if fuse_mode == "modality-attention":
            self.transformer_encoder = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            self.linear_projection = nn.Linear(feature_dim * n_features, feature_dim)
            if position_encoding == "learnable":
                self.position_embedding = nn.Parameter(
                    torch.randn(n_features, feature_dim))
            elif position_encoding == "sinusoidal":
                pe = torch.zeros(n_features, feature_dim)
                pos = torch.arange(0, n_features, dtype=torch.float).unsqueeze(1)
                div = torch.exp(torch.arange(0, feature_dim, 2).float()
                                * (-math.log(2 * n_features) / feature_dim))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer("position_embedding", pe)
            else:
                self.position_embedding = None
        else:
            # concat mode: no transformer
            self.transformer_encoder = None
            self.linear_projection = None
            self.position_embedding = None

        # ─── Precompute output feature dim ───
        lowdim_flat = sum(self.lowdim_dims[k] * self.lowdim_horizons[k]
                          for k in self.lowdim_keys)
        if fuse_mode == "modality-attention":
            self._output_dim = feature_dim + lowdim_flat
        else:
            self._output_dim = feature_dim * n_features + lowdim_flat

    # ───────────────────────────────────────────────────────────────

    def output_dim(self) -> int:
        return self._output_dim

    def _aggregate(self, raw_feature):
        """Aggregate per-frame feature map to a single vector per frame."""
        if self.model_name.startswith("vit"):
            # ViT: CLS token
            return raw_feature[:, 0, :]

        # ResNet/ConvNext: (B, C, H, W)
        assert raw_feature.dim() == 4
        if self.feature_aggregation == "attention_pool_2d":
            return self.attention_pool_2d(raw_feature)
        # Fallback: global average pool
        return raw_feature.flatten(2).mean(-1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(obs.values())).shape[0]

        modality_features = []  # list of (B, T, D)
        lowdim_features = []    # list of (B, T*D)

        # ─── Process RGB-like keys ───
        for key in self.rgb_keys:
            img = obs[key]                           # (B, T, C, H, W)
            B, T = img.shape[:2]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.vision_transform(img)
            raw = self.key_model_map[key](img)       # ViT: (B*T, N_tokens, D); CNN: (B*T, D, h, w)
            feat = self._aggregate(raw)              # (B*T, D)
            modality_features.append(feat.reshape(B, T, -1))

        # ─── Process low-dim keys ───
        for key in self.lowdim_keys:
            data = obs[key]                          # (B, T, D)
            lowdim_features.append(data.reshape(batch_size, -1))

        # ─── Fuse ───
        if self.fuse_mode == "modality-attention":
            in_embeds = torch.cat(modality_features, dim=1)  # (B, n_features, D)
            if self.position_embedding is not None:
                in_embeds = in_embeds + self.position_embedding
            out_embeds = self.transformer_encoder(in_embeds) # (B, n_features, D)
            flat = torch.cat(
                [out_embeds[:, i] for i in range(out_embeds.shape[1])], dim=1)
            visual_flat = self.linear_projection(flat)        # (B, D)
        else:
            # concat mode
            visual_flat = torch.cat(
                [m.reshape(batch_size, -1) for m in modality_features], dim=-1)

        # Concat lowdim
        if lowdim_features:
            out = torch.cat([visual_flat] + lowdim_features, dim=-1)
        else:
            out = visual_flat

        return out
