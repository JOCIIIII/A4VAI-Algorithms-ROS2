"""Validate path handoff in the current world-ENU frame."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PathStartResult:
    accepted: bool
    gap_m: float


def validate_path_start(points, current_xy, tolerance_m) -> PathStartResult:
    points = np.asarray(points, dtype=float)
    current_xy = np.asarray(current_xy, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all()):
        raise ValueError('path must contain at least two finite XY points')
    if current_xy.shape != (2,) or not np.isfinite(current_xy).all():
        raise ValueError('current world-ENU pose must contain two finite values')
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('path start tolerance must be finite and positive')
    gap_m = float(np.linalg.norm(points[0] - current_xy))
    return PathStartResult(accepted=gap_m <= tolerance_m, gap_m=gap_m)
