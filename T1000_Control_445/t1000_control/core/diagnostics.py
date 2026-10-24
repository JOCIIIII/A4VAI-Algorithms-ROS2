from __future__ import annotations

import numpy as np
from .. import settings as CFG


H1_NEAR = 5.0 / CFG.LIDAR_MAX


H1_FRONT_DEG = 60.0


H1_MIN_ACT = 0.05


def h1_judge(bins, action):
    bins = np.asarray(bins, float).ravel()
    i = int(np.argmin(bins))
    d = float(bins[i])
    az = (i % CFG.LIDAR_H_BINS) * (360.0 / CFG.LIDAR_H_BINS)
    az_signed = az - 360.0 if az > 180.0 else az
    if d >= H1_NEAR or abs(az_signed) > H1_FRONT_DEG:
        return "skip", f"no target (range {d*CFG.LIDAR_MAX:.1f}m, angle {az_signed:+.0f} deg)"
    vy = float(action[1])
    if abs(vy) < H1_MIN_ACT:
        return "skip", f"side action is small ({vy:+.3f})"
    if az_signed == 0.0:
        return "skip", f"target is centered at {d*CFG.LIDAR_MAX:.1f}m"
    good = (az_signed > 0 and vy < 0) or (az_signed < 0 and vy > 0)
    side = "left" if az_signed > 0 else "right"
    away = "right" if vy < 0 else "left"
    return ("OK" if good else "wrong way",
            f"target {side} {d*CFG.LIDAR_MAX:.1f}m; command {away} ({vy:+.3f})")


def observation_lag(latency_s: float, lat_min: float | None):
    if lat_min is None or latency_s < lat_min:
        lat_min = latency_s
    return lat_min, max(0.0, latency_s - lat_min)
