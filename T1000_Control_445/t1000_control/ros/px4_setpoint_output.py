from __future__ import annotations

import math


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("active command fields must be finite")


def _horizontal_caps(max_xy: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(max_xy, tuple):
        return float(max_xy[0]), float(max_xy[1])
    cap = float(max_xy)
    return cap, cap


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def command_to_ned(
    vx_enu: float,
    vy_enu: float,
    vz_enu: float,
    *,
    isaac_sim: bool,
    max_xy: float | tuple[float, float],
    max_vz: float,
) -> tuple[float, float, float]:
    _require_finite(vx_enu, vy_enu, vz_enu)
    if not isaac_sim:
        return float(vx_enu), -float(vy_enu), -float(vz_enu)

    vx_cap, vy_cap = _horizontal_caps(max_xy)
    return (
        _clip(vy_enu, vy_cap),
        _clip(vx_enu, vx_cap),
        _clip(-float(vz_enu), float(max_vz)),
    )


def yaw_rate_to_ned(yaw_rate_enu: float) -> float:
    return -float(yaw_rate_enu)


def trajectory_fields(
    vx: float,
    vy: float,
    vz: float,
    yawspeed: float,
) -> dict[str, object]:
    _require_finite(vx, vy, vz, yawspeed)
    nan = float("nan")
    return {
        "velocity": [float(vx), float(vy), float(vz)],
        "yawspeed": float(yawspeed),
        "position": [nan, nan, nan],
        "yaw": nan,
    }
