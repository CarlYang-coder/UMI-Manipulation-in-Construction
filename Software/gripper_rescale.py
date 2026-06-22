"""Map ArUco-measured gripper_width ([~0.29, ~0.97]) to a full-travel [0, 1]
control value, then scale to xArm gripper range [0, 850].

Same rescale used by both replay_ee_pose.py and run_umi_ft_RGBonly.py so
the "closed" and "open" endpoints are consistent between replay and rollout.

Endpoint constants come from empirical inspection of training CSVs. Tune
GW_CLOSED_IN / GW_OPEN_IN here and both scripts pick up the change.
"""
import numpy as np

# Empirical endpoints from training CSVs (tune if your data differs)
GW_CLOSED_IN = 0.30
GW_OPEN_IN = 0.90

# xArm gripper servo range (closed -> open)
XARM_GRIPPER_MAX = 850.0


def gw_to_xarm_pos(gw_raw: float) -> float:
    """Rescale a raw gripper_width in [GW_CLOSED_IN, GW_OPEN_IN] to xArm
    gripper position [0, 850]. Values outside the range are clipped."""
    norm = (gw_raw - GW_CLOSED_IN) / (GW_OPEN_IN - GW_CLOSED_IN)
    norm = float(np.clip(norm, 0.0, 1.0))
    return norm * XARM_GRIPPER_MAX


def xarm_pos_to_gw(pos: float) -> float:
    """Inverse mapping: xArm gripper feedback [0, 850] -> normalized gw in
    [GW_CLOSED_IN, GW_OPEN_IN] -- matches what the policy saw in training."""
    norm = float(np.clip(pos / XARM_GRIPPER_MAX, 0.0, 1.0))
    return GW_CLOSED_IN + norm * (GW_OPEN_IN - GW_CLOSED_IN)
