from __future__ import annotations

import math
import numpy as np


def quat_to_yaw(quat_wxyz) -> float:
    w, x, y, z = np.asarray(quat_wxyz, float)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_rotate_inverse(quat_wxyz, v) -> np.ndarray:
    q = np.asarray(quat_wxyz, float)
    w, u = q[0], q[1:]
    v = np.asarray(v, float)
    return v * (2.0 * w * w - 1.0) - np.cross(u, v) * (2.0 * w) + u * (2.0 * (u @ v))


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
