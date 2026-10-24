from __future__ import annotations

import math

import numpy as np

from .. import settings as CFG

K_Y = CFG.NMPC_K_Y
K_VY = CFG.NMPC_K_VY
VX_TARGET = CFG.VX_TARGET
MODEL_FORWARD_ACTION_SCALE_MPS = CFG.MODEL_FORWARD_ACTION_SCALE_MPS
MODEL_LATERAL_ACTION_SCALE_MPS = CFG.MODEL_LATERAL_ACTION_SCALE_MPS
VX_MAX = MODEL_FORWARD_ACTION_SCALE_MPS
VY_MAX = MODEL_LATERAL_ACTION_SCALE_MPS
NMPC_VY_REF_MAX = CFG.NMPC_VY_REF_MAX
BLEND_STEEPNESS = CFG.BLEND_STEEPNESS
LIDAR_CLEAR_THRESHOLD = CFG.LIDAR_CLEAR_THRESHOLD

WIND_FF_ENABLED = CFG.WIND_FF_ENABLED

SATURATION_ABS = MODEL_LATERAL_ACTION_SCALE_MPS / K_Y


def clear_gate(lidar_norm: np.ndarray) -> float:
    a = np.asarray(lidar_norm, float).ravel()
    valid = a[a < 0.99]
    lidar_min = float(valid.min()) if valid.size else 1.0
    return 1.0 / (1.0 + math.exp(-BLEND_STEEPNESS * (lidar_min - LIDAR_CLEAR_THRESHOLD)))


def solve(
    lat_err: float,
    lat_rate: float,
    lidar_norm: np.ndarray,
    wind_xy: tuple[float, float] = (0.0, 0.0),
    *,
    vx_target: float | None = None,
) -> tuple[float, float]:
    ff_x, ff_y = (wind_xy if WIND_FF_ENABLED else (0.0, 0.0))

    vy = -(K_Y * lat_err + K_VY * lat_rate)
    vy = float(np.clip(vy + ff_y, -MODEL_LATERAL_ACTION_SCALE_MPS, MODEL_LATERAL_ACTION_SCALE_MPS))
    vy = float(np.clip(vy, -NMPC_VY_REF_MAX, NMPC_VY_REF_MAX))
    vy *= clear_gate(lidar_norm)
    target = VX_TARGET if vx_target is None else vx_target
    vx = float(np.clip(target + ff_x, -MODEL_FORWARD_ACTION_SCALE_MPS, MODEL_FORWARD_ACTION_SCALE_MPS))
    return vx, vy


def to_obs(vx_ref: float, vy_ref: float) -> tuple[float, float]:
    return (float(np.clip(vx_ref / MODEL_FORWARD_ACTION_SCALE_MPS, -1.0, 1.0)),
            float(np.clip(vy_ref / MODEL_LATERAL_ACTION_SCALE_MPS, -1.0, 1.0)))
