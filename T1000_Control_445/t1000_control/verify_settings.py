import json
import math
from pathlib import Path

import numpy as np

from . import settings


def verify_settings():
    errors = settings.validate_settings()
    report = {
        "offline_only": True,
        "settings_file": str(Path(settings.__file__).resolve()),
        "checks": {"settings": not errors}, "errors": errors,
    }
    if errors:
        return {**report, "ok": False}
    report["effective_settings"] = {name: getattr(settings, name) for name in (
        "FLIGHT_ALTITUDE_M", "PATH_FORWARD_COMMAND_CAP_MPS",
        "PATH_LATERAL_COMMAND_CAP_MPS", "SPEED_SCALE",
    )}

    from .core.controller import ControlPipeline

    class SaturatingPolicy:
        def __init__(self, sign):
            self.sign = sign

        def predict(self, observation):
            return np.full(2, 10.0 * self.sign)

    cap_samples = []
    cap_checks = []
    forward = settings.PATH_FORWARD_COMMAND_CAP_MPS * settings.SPEED_SCALE
    lateral = settings.PATH_LATERAL_COMMAND_CAP_MPS * settings.SPEED_SCALE
    for goal, expected_xy in (((3.0, 0.0), (forward, lateral)),
                              ((0.0, 3.0), (-lateral, forward))):
        for sign in (1, -1):
            controller = ControlPipeline(
                SaturatingPolicy(sign),
                flight_altitude_m=settings.FLIGHT_ALTITUDE_M,
                yaw_mode=settings.POLICY_REGISTRY[settings.DEFAULT_POLICY]["yaw_mode"],
                policy_authority_mode="always",
                goal_stop=False,
            )
            controller.set_path([(0.0, 0.0), goal])
            result = controller.step(
                lidar_bins=np.ones(settings.LIDAR_TOTAL_BINS),
                pos_ned=[0.0, 0.0, -(settings.FLIGHT_ALTITUDE_M - 0.25)],
                vel_ned=[0.0, 0.0, 0.0], quat_ned=[1.0, 0.0, 0.0, 0.0],
            )
            cap_samples.append(list(result.velocity_enu[:2]))
            cap_checks.append(np.allclose(result.velocity_enu[:2],
                                           [sign * v for v in expected_xy], rtol=1e-7, atol=1e-7))

    expected_vz = max(-settings.VZ_CMD_MAX, min(settings.VZ_CMD_MAX, settings.KP_ALT_VEL * 0.25))
    expected_distance = min(1.0, math.hypot(3.0, 0.25) / settings.PATH_NORM)
    report["checks"].update(
        altitude_command=math.isclose(result.velocity_enu[2], expected_vz, rel_tol=1e-7, abs_tol=1e-7),
        goal_geometry=math.isclose(float(result.observation[438]), expected_distance, rel_tol=1e-7, abs_tol=1e-7),
        path_caps=all(cap_checks),
    )
    report["samples"] = {
        "altitude_velocity_mps": result.velocity_enu[2],
        "goal_distance_norm": float(result.observation[438]),
        "cap_velocity_enu_mps": cap_samples,
    }
    errors.extend(f"pipeline check failed: {name}" for name, passed in report["checks"].items() if not passed)
    return {**report, "ok": not errors}


def main():
    report = verify_settings()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
