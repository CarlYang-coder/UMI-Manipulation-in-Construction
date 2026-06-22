"""Verify normalizer is saved and loaded correctly from checkpoint."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")

import torch

ckpt = torch.load(r"D:\UMI_Gripper\train\checkpoints\best.pt",
                  map_location="cpu", weights_only=False)

# Find all normalizer-related keys
norm_keys = [k for k in ckpt["policy"].keys() if "normalizer" in k]
print(f"Found {len(norm_keys)} normalizer keys in checkpoint:")
for k in norm_keys[:20]:
    v = ckpt["policy"][k]
    print(f"  {k}: shape={tuple(v.shape)}, "
          f"min={v.min().item():.4f}, max={v.max().item():.4f}")

# Show action normalizer specifically
action_keys = [k for k in norm_keys if "action" in k]
print(f"\nAction normalizer keys ({len(action_keys)}):")
for k in action_keys:
    v = ckpt["policy"][k]
    print(f"  {k}:\n    {v.flatten()[:7]}")
