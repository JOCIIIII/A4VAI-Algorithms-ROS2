from __future__ import annotations

import math
import numpy as np


def ned_to_enu_vec(v):
    v = np.asarray(v, float)
    return np.array([v[0], -v[1], -v[2]])


def enu_to_ned_vec(v):
    return ned_to_enu_vec(v)


def ned_to_enu_quat(q_ned):
    q = np.asarray(q_ned, float)
    r = math.sqrt(0.5)
    q_a = np.array([0.0, r, r, 0.0])
    q_b = np.array([0.0, 1.0, 0.0, 0.0])
    return _quat_mul(_quat_mul(q_a, q), q_b)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
