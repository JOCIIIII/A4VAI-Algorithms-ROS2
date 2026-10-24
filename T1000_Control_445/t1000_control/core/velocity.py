from __future__ import annotations

import numpy as np
from .. import settings as CFG


MODEL_FORWARD_ACTION_SCALE_MPS = CFG.MODEL_FORWARD_ACTION_SCALE_MPS
MODEL_LATERAL_ACTION_SCALE_MPS = CFG.MODEL_LATERAL_ACTION_SCALE_MPS


VX_MAX = MODEL_FORWARD_ACTION_SCALE_MPS
VY_MAX = MODEL_LATERAL_ACTION_SCALE_MPS


VEL_LOWPASS_TAU = CFG.VEL_LOWPASS_TAU


def compute_velocity_command(
    action, aux, *, speed_scale: float = 1.0,
    forward_cap_mps: float = MODEL_FORWARD_ACTION_SCALE_MPS,
    lateral_cap_mps: float = MODEL_LATERAL_ACTION_SCALE_MPS,
    policy_authority: float = 1.0,
    lateral_command_mps: float | None = None,
):
    a = np.asarray(action, float).ravel()
    if a.size != 2:
        raise ValueError(
            f"action has {a.size} values; this policy needs 2. "
            "A 4-value action is not the world445 policy."
        )

    vx_path = aux["vx_ref"] + float(a[0]) * MODEL_FORWARD_ACTION_SCALE_MPS * policy_authority
    vy_path = aux["vy_ref"] + float(a[1]) * MODEL_LATERAL_ACTION_SCALE_MPS * policy_authority
    if lateral_command_mps is not None:
        vy_path = lateral_command_mps

    vx_path = float(np.clip(vx_path, -forward_cap_mps, forward_cap_mps))
    vy_path = float(np.clip(vy_path, -lateral_cap_mps, lateral_cap_mps))

    d, n = aux["d_hat"], aux["n_hat"]
    vx_enu = vx_path * d[0] + vy_path * n[0]
    vy_enu = vx_path * d[1] + vy_path * n[1]

    return (vx_enu * speed_scale, vy_enu * speed_scale)


class VelocityRateLimiter:

    def __init__(self, accel_mps2: float, decel_mps2: float):
        self.accel_mps2 = float(accel_mps2)
        self.decel_mps2 = float(decel_mps2)
        self._y = None

    def __call__(self, vx: float, vy: float, dt: float) -> tuple[float, float]:
        target = np.array([vx, vy], float)
        if self._y is None:
            self._y = target
        elif dt > 0.:
            delta = target - self._y
            distance = float(np.linalg.norm(delta))
            rate = self.decel_mps2 if np.linalg.norm(target) < np.linalg.norm(self._y) else self.accel_mps2
            if distance > 0.:
                self._y = self._y + delta * min(1., rate * dt / distance)
        return float(self._y[0]), float(self._y[1])

    def reset(self):
        self._y = None

    def seed(self, vx: float, vy: float):
        self._y = np.array([vx, vy], float)


class VelocityLowpass:

    def __init__(self, tau: float = VEL_LOWPASS_TAU):
        self.tau = float(tau)
        self._y = None

    def __call__(self, vx: float, vy: float, dt: float) -> tuple[float, float]:
        if self.tau <= 0.0 or dt <= 0.0:
            return (vx, vy)
        if self._y is None:
            self._y = (vx, vy)
            return self._y
        a = dt / (self.tau + dt)
        self._y = (a * vx + (1 - a) * self._y[0],
                   a * vy + (1 - a) * self._y[1])
        return self._y

    def reset(self):
        self._y = None

    def seed(self, vx: float, vy: float):
        self._y = (float(vx), float(vy))
