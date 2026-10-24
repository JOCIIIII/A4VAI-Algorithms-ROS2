from __future__ import annotations

import math
import numpy as np
from .. import settings as CFG


ALT_TARGET = CFG.ALT_TARGET


KP_ALT_VEL = CFG.KP_ALT_VEL


VZ_CMD_MAX = CFG.VZ_CMD_MAX


KP_YAW_RATE = CFG.KP_YAW_RATE


YAW_RATE_MAX = CFG.YAW_RATE_MAX


def altitude_velocity(z_now: float, target: float = ALT_TARGET) -> float:
    return float(np.clip(KP_ALT_VEL * (target - z_now), -VZ_CMD_MAX, VZ_CMD_MAX))


YAW_MODE_PATH = CFG.YAW_MODE_PATH


YAW_MODE_FIXED = CFG.YAW_MODE_FIXED


def select_yaw_target(path_yaw: float, velocity_enu, avoidance_active: bool,
                      mode: str = CFG.AVOIDANCE_YAW_MODE, *,
                      min_speed_mps: float = CFG.YAW_COMMAND_MIN_SPEED_MPS) -> float:
    """Choose path yaw or active avoidance-command yaw without changing velocity."""
    if mode == CFG.YAW_MODE_PATH:
        return float(path_yaw)
    if mode != CFG.YAW_MODE_COMMAND_VELOCITY:
        raise ValueError(
            f"unknown avoidance yaw mode: {mode!r}; use "
            f"{CFG.YAW_MODE_PATH!r} or {CFG.YAW_MODE_COMMAND_VELOCITY!r}"
        )
    vx, vy = float(velocity_enu[0]), float(velocity_enu[1])
    if not avoidance_active or math.hypot(vx, vy) <= min_speed_mps:
        return float(path_yaw)
    return float(math.atan2(vy, vx))


def yaw_rate_command(yaw_now: float, path_yaw: float,
                     mode: str = YAW_MODE_PATH) -> float:
    if mode == YAW_MODE_FIXED:
        target = 0.0
    elif mode == YAW_MODE_PATH:
        target = path_yaw
    else:
        raise ValueError(
            f"unknown yaw mode: {mode!r}; use {YAW_MODE_PATH!r} or {YAW_MODE_FIXED!r}"
        )
    err = (target - yaw_now + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(KP_YAW_RATE * err, -YAW_RATE_MAX, YAW_RATE_MAX))
