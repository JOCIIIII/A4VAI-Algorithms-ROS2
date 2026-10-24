from __future__ import annotations

from types import MappingProxyType
from collections.abc import Mapping
import hashlib
import json
import math

OBS_DIM = 445
ACT_DIM = 2

FLIGHT_ALTITUDE_M = 2.0
PATH_FORWARD_COMMAND_CAP_MPS = 5.0
PATH_LATERAL_COMMAND_CAP_MPS = 3.0
SITL_ENU_X_CAP_MPS = 3.0
SITL_ENU_Y_CAP_MPS = 2.0
SITL_DURATION_S = 20.0
PATH_PUBLISH_HZ = 2.0
PATH_PUBLISH_MIN_HZ = 1.0
PATH_START_MODE = "require_near"
PATH_START_TOLERANCE_M = 2.0
CORNER_MODE = "sharp"
CORNER_MAX_DEVIATION_M = 1.5
CORNER_SPEED_MPS = 1.5
CORNER_DECEL_MPS2 = 1.5
CORNER_SAMPLE_SPACING_M = 0.25
ORDERED_WAYPOINTS_ENABLED = False
WAYPOINT_ACCEPTANCE_RADIUS_M = 2.0
WAYPOINT_HOLD_S = 0.3
POLICY_AUTHORITY_MODE = "always"
AVOID_ACTIVATE_DISTANCE_M = 10.0
AVOID_FULL_DISTANCE_M = 6.0
AVOID_FORWARD_HALF_ANGLE_DEG = 45.0
AVOID_MAX_AUTHORITY = 0.6
AVOID_FORWARD_CAP_MPS = 1.5
AVOID_PASS_MARGIN_M = 2.125
RECOVERY_START_CTE_M = 2.0
RECOVERY_FULL_CTE_M = 4.0
RECOVERY_FORWARD_CAP_MPS = 1.5
RECOVERY_LATERAL_CAP_MPS = 2.0
RECOVERY_CLEAR_HYSTERESIS_M = 1.0
RECOVERY_CORRIDOR_HALF_WIDTH_M = 1.0
RECOVERY_CORRIDOR_MARGIN_M = 0.75
GOAL_STOP_ENABLED = False
GOAL_STOP_RADIUS_M = 0.5
GOAL_DECEL_MPS2 = 1.0
GOAL_DWELL_MAX_SPEED_MPS = 0.2
GOAL_DWELL_S = 2.0

MODEL_FORWARD_ACTION_SCALE_MPS = 5.0
MODEL_LATERAL_ACTION_SCALE_MPS = 3.0

VX_MAX = MODEL_FORWARD_ACTION_SCALE_MPS
VY_MAX = MODEL_LATERAL_ACTION_SCALE_MPS
VZ_MAX = 2.0
VX_TARGET = 3.0

ALT_TARGET = FLIGHT_ALTITUDE_M
CTE_NORM = 10.0
PATH_NORM = 90.0
GOAL_ALT = ALT_TARGET


LIDAR_H_BINS = 72
LIDAR_V_LAYERS = 6
LIDAR_V_BOUNDARIES = [-7, -2, 8, 20, 32, 44, 52]
LIDAR_MAX = 40.0
LIDAR_MIN = 0.5
LIDAR_TOTAL_BINS = LIDAR_H_BINS * LIDAR_V_LAYERS
LIDAR_CLEAR_VALUE = 1.0


NMPC_K_Y = 4.017070
NMPC_K_VY = 1.264431
NMPC_VY_REF_MAX = 1.5
BLEND_STEEPNESS = 50.0
LIDAR_CLEAR_THRESHOLD = 0.075
WIND_FF_ENABLED = False


SENSOR_OFFSET = (0.0, 0.0, 0.1713)

SELF_REFLECTION_CUT_M = LIDAR_MIN


GROUND_FILTER = False
GROUND_Z_CUTOFF = -1.0
GROUND_EL_MAX_DEG = -5.0
GROUND_RANGE_TOL = 0.25


CONTROL_HZ = 20.0
LIDAR_HZ = 10.0
LIDAR_HZ_TOL = 1.0

VEL_LOWPASS_TAU = 0.18

KP_ALT_VEL = 1.0
VZ_CMD_MAX = 1.0
KP_YAW_RATE = 1.5
YAW_RATE_MAX = 1.0
YAW_COMMAND_MIN_SPEED_MPS = 0.3

SPEED_SCALE = 1.0

SPEED_SCALE_COMPENSATE = False


TOPIC_LIDAR = "/livox/lidar"

LIDAR_QOS_RELIABLE = True

LIDAR_QOS_DEPTH = 2
TOPIC_IMU = "/livox/imu"
TOPIC_PATH = "/t1000/path"
TOPIC_LIO_ODOM = "/Odometry"
TOPIC_PX4_VISUAL_ODOM = "/fmu/in/vehicle_visual_odometry"
TOPIC_PX4_OFFBOARD = "/fmu/in/offboard_control_mode"
TOPIC_PX4_SETPOINT = "/fmu/in/trajectory_setpoint"
TOPIC_PX4_COMMAND = "/fmu/in/vehicle_command"
TOPIC_PX4_LOCAL_POS = "/fmu/out/vehicle_local_position"
TOPIC_PX4_ATTITUDE = "/fmu/out/vehicle_attitude"
TOPIC_PX4_STATUS = "/fmu/out/vehicle_status_v1"
TOPIC_RC_MANUAL_CONTROL = "/fmu/out/manual_control_setpoint"
TOPIC_RC_CHANNELS = "/fmu/out/rc_channels"
TOPIC_RC_INPUT = "/fmu/out/input_rc"
TOPIC_CMD_PREVIEW = "/t1000/cmd_preview"
TOPIC_ISAAC_POSE = "/t1000/pose"
TOPIC_ISAAC_VELOCITY = "/t1000/velocity"
TOPIC_ISAAC_LIDAR_BINS = "/t1000/lidar_bins"

TOPIC_DIAG_BINS = "/t1000/lidar_bins"
TOPIC_DIAG_OBS = "/t1000/obs"
TOPIC_DIAG_ACTION = "/t1000/action"

PX4_TARGET_SYSTEM_ID = 1
PX4_TARGET_COMPONENT_ID = 1
PX4_SOURCE_SYSTEM_ID = 255
PX4_SOURCE_COMPONENT_ID = 0

VIO_MAX_RATE_HZ = 50.0

VIO_LAG_WARN_S = 0.30
VIO_MAX_JUMP_M = 2.0
VIO_MAX_SPEED_MPS = 20.0
VIO_REJECT_STREAK_RESET = 10
VIO_POS_VAR = 0.10
VIO_ORI_VAR = 0.05
VIO_VEL_VAR = 0.20

LIO_FRAME_ID = "camera_init"
LIO_CHILD_FRAME_ID = "body"
LIO_VELOCITY_WINDOW_S = 0.5
LIO_MAX_SAMPLE_AGE_S = 0.5
LIO_MAX_FUTURE_SAMPLE_S = 0.1
LIO_MIN_RATE_HZ = 8.0
LIO_PX4_RECOMMENDED_RATE_HZ = 30.0

PATH_HOLD_S = 1.0
PATH_SLOW_S = 3.0

RC_ABORT_SOURCE = "px4_mode_switch"

RC_ABORT_AUX = None

RC_ABORT_VERIFIED = False


RC_ABORT_CHANNEL = None

RC_ABORT_UNIT = "norm"
RC_ABORT_THRESHOLD = 0.5
RC_UNIT_SANITY_BOUND = 10.0
RC_LINK_TIMEOUT_S = 1.0

LIDAR_TIMEOUT_HOLD_S = 0.5
LIDAR_TIMEOUT_ABORT_S = 2.0

ABORT_ACTION_LAND = True

ODOM_TIMEOUT_HOLD_S = 0.5
ODOM_TIMEOUT_ABORT_S = 2.0
STATUS_TIMEOUT_HOLD_S = 5.0
STATUS_TIMEOUT_ABORT_S = 10.0

ODOM_GAP_WARN_S = ODOM_TIMEOUT_HOLD_S / 2.0

PREFLIGHT_MAX_SPEED_SCALE = 1.0


YAW_MODE_PATH = "path"
YAW_MODE_FIXED = "fixed"
YAW_MODE_COMMAND_VELOCITY = "command_velocity"
AVOIDANCE_YAW_MODE = YAW_MODE_PATH

POLICY_REGISTRY = {
    "policy_445.onnx": {
        "yaw_mode": YAW_MODE_PATH,
        "obs_dim": 445,
        "act_dim": 2,
        "training_run": "ppo_world445_20260828_122754",
        "checkpoint": "checkpoint_445.pt",
        "checkpoint_sha256": "d4d12f6d1fcf330fa503db2ef0c29d5c187f074d04f812243ad185af07715be5",
        "onnx_sha256": "8c82faed956f4a4fb5b101114e1d86dc67c817685a2da386727c2765e6630dd6",
        "note": (
            "Final world445 policy for simulation work. "
            "This package does not prove safe real flight."
        ),
        "mission_ok": False,
    },
}


DEFAULT_POLICY = "policy_445.onnx"


MODEL_CONTRACT = MappingProxyType(
    {
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "action_scales": (MODEL_FORWARD_ACTION_SCALE_MPS, MODEL_LATERAL_ACTION_SCALE_MPS),
        "vx_target": VX_TARGET,
        "vz_max": VZ_MAX,
        "lidar_shape": (LIDAR_H_BINS, LIDAR_V_LAYERS),
        "lidar_boundaries": tuple(LIDAR_V_BOUNDARIES),
        "lidar_range_m": (LIDAR_MIN, LIDAR_MAX),
        "path_norm": PATH_NORM,
        "cte_norm": CTE_NORM,
        "default_policy": DEFAULT_POLICY,
        "policy": MappingProxyType(
            {
                "yaw_mode": POLICY_REGISTRY[DEFAULT_POLICY]["yaw_mode"],
                "checkpoint_sha256": POLICY_REGISTRY[DEFAULT_POLICY]["checkpoint_sha256"],
                "onnx_sha256": POLICY_REGISTRY[DEFAULT_POLICY]["onnx_sha256"],
            }
        ),
    }
)


MODEL_CONTRACT_FINGERPRINT = "a5a6650398566be1e44c74826f7e5b30583ea5812641149af750c0741f36f125"


def model_contract_fingerprint(contract: Mapping[str, object] | None = None) -> str:
    payload = json.dumps(
        MODEL_CONTRACT if contract is None else contract,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=lambda value: dict(value),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_settings(*, corner_mode=None, policy_authority_mode=None, goal_stop=None,
                      ordered_waypoints=None) -> list[str]:
    from .safety.failsafe import _rc_config_problems

    errors: list[str] = []
    values = globals()
    errors.extend(_rc_config_problems(values))

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    for name, value in values.items():
        if name.isupper() and isinstance(value, (int, float)) and not math.isfinite(value):
            errors.append(f"{name} must be finite")
    if len(SENSOR_OFFSET) != 3 or not all(finite(v) for v in SENSOR_OFFSET):
        errors.append("SENSOR_OFFSET must contain three finite values")

    positive = (
        "FLIGHT_ALTITUDE_M", "PATH_FORWARD_COMMAND_CAP_MPS", "PATH_LATERAL_COMMAND_CAP_MPS",
        "SITL_ENU_X_CAP_MPS", "SITL_ENU_Y_CAP_MPS", "SITL_DURATION_S",
        "PATH_PUBLISH_HZ", "PATH_PUBLISH_MIN_HZ", "CONTROL_HZ", "LIDAR_HZ",
        "VZ_MAX", "VZ_CMD_MAX", "YAW_RATE_MAX", "YAW_COMMAND_MIN_SPEED_MPS",
        "SPEED_SCALE", "NMPC_VY_REF_MAX",
        "VIO_MAX_RATE_HZ", "VIO_LAG_WARN_S", "VIO_MAX_JUMP_M", "VIO_MAX_SPEED_MPS",
        "LIO_VELOCITY_WINDOW_S", "LIO_MIN_RATE_HZ", "LIO_PX4_RECOMMENDED_RATE_HZ",
        "LIO_MAX_SAMPLE_AGE_S",
        "LIDAR_TIMEOUT_HOLD_S", "LIDAR_TIMEOUT_ABORT_S",
        "ODOM_TIMEOUT_HOLD_S", "ODOM_TIMEOUT_ABORT_S", "RC_LINK_TIMEOUT_S",
        "STATUS_TIMEOUT_HOLD_S", "STATUS_TIMEOUT_ABORT_S",
        "PATH_HOLD_S", "PATH_SLOW_S", "PATH_START_TOLERANCE_M",
        "CORNER_MAX_DEVIATION_M", "CORNER_SPEED_MPS", "CORNER_DECEL_MPS2",
        "CORNER_SAMPLE_SPACING_M",
        "WAYPOINT_ACCEPTANCE_RADIUS_M", "WAYPOINT_HOLD_S",
        "AVOID_ACTIVATE_DISTANCE_M", "AVOID_FULL_DISTANCE_M",
        "AVOID_FORWARD_CAP_MPS", "AVOID_PASS_MARGIN_M",
        "RECOVERY_START_CTE_M", "RECOVERY_FULL_CTE_M",
        "RECOVERY_FORWARD_CAP_MPS", "RECOVERY_LATERAL_CAP_MPS",
        "RECOVERY_CLEAR_HYSTERESIS_M", "RECOVERY_CORRIDOR_HALF_WIDTH_M",
        "RECOVERY_CORRIDOR_MARGIN_M",
        "AVOID_FORWARD_HALF_ANGLE_DEG", "GOAL_STOP_RADIUS_M", "GOAL_DECEL_MPS2",
        "GOAL_DWELL_MAX_SPEED_MPS", "GOAL_DWELL_S",
    )
    for name in positive:
        if not finite(values[name]) or values[name] <= 0:
            errors.append(f"{name} must be finite and positive")
    mode = CORNER_MODE if corner_mode is None else corner_mode
    if PATH_START_MODE not in ("require_near", "prepend_current"):
        errors.append("PATH_START_MODE must be require_near or prepend_current")
    if mode not in ("sharp", "rounded"):
        errors.append("CORNER_MODE must be sharp or rounded")
    authority_mode = POLICY_AUTHORITY_MODE if policy_authority_mode is None else policy_authority_mode
    if authority_mode not in ("always", "obstacle"):
        errors.append("POLICY_AUTHORITY_MODE must be always or obstacle")
    if AVOIDANCE_YAW_MODE not in (YAW_MODE_PATH, YAW_MODE_COMMAND_VELOCITY):
        errors.append("AVOIDANCE_YAW_MODE must be path or command_velocity")
    if not isinstance(GOAL_STOP_ENABLED if goal_stop is None else goal_stop, bool):
        errors.append("GOAL_STOP_ENABLED must be boolean")
    if not isinstance(ORDERED_WAYPOINTS_ENABLED if ordered_waypoints is None else ordered_waypoints, bool):
        errors.append("ORDERED_WAYPOINTS_ENABLED must be boolean")
    if not finite(AVOID_MAX_AUTHORITY) or not 0. <= AVOID_MAX_AUTHORITY <= 1.:
        errors.append("AVOID_MAX_AUTHORITY must be in [0, 1]")
    if finite(AVOID_FORWARD_HALF_ANGLE_DEG) and AVOID_FORWARD_HALF_ANGLE_DEG > 180.:
        errors.append("AVOID_FORWARD_HALF_ANGLE_DEG must be at most 180")
    corner_limits = ("VX_TARGET", "PATH_FORWARD_COMMAND_CAP_MPS") if mode == "rounded" else ("VX_TARGET",)
    for upper in corner_limits:
        if finite(CORNER_SPEED_MPS) and finite(values[upper]) and CORNER_SPEED_MPS > values[upper]:
            errors.append(f"CORNER_SPEED_MPS must be at most {upper}")
    for name in ("VEL_LOWPASS_TAU", "LIDAR_HZ_TOL", "ODOM_GAP_WARN_S", "LIO_MAX_FUTURE_SAMPLE_S"):
        if not finite(values[name]) or values[name] < 0:
            errors.append(f"{name} must be finite and nonnegative")
    for name, minimum in (
        ("PX4_TARGET_SYSTEM_ID", 1), ("PX4_SOURCE_SYSTEM_ID", 1),
        ("PX4_TARGET_COMPONENT_ID", 0), ("PX4_SOURCE_COMPONENT_ID", 0),
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 255:
            errors.append(f"{name} must be an integer in [{minimum}, 255]")
    for lower, upper, strict in (
        ("LIDAR_TIMEOUT_HOLD_S", "LIDAR_TIMEOUT_ABORT_S", True),
        ("ODOM_TIMEOUT_HOLD_S", "ODOM_TIMEOUT_ABORT_S", True),
        ("STATUS_TIMEOUT_HOLD_S", "STATUS_TIMEOUT_ABORT_S", True),
        ("PATH_HOLD_S", "PATH_SLOW_S", True),
        ("PATH_PUBLISH_MIN_HZ", "PATH_PUBLISH_HZ", False),
        ("LIO_MIN_RATE_HZ", "LIO_PX4_RECOMMENDED_RATE_HZ", False),
        ("AVOID_FULL_DISTANCE_M", "AVOID_ACTIVATE_DISTANCE_M", True),
        ("AVOID_ACTIVATE_DISTANCE_M", "LIDAR_MAX", False),
        ("RECOVERY_START_CTE_M", "RECOVERY_FULL_CTE_M", True),
        ("RECOVERY_CLEAR_HYSTERESIS_M", "RECOVERY_START_CTE_M", True),
    ):
        a, b = values[lower], values[upper]
        if finite(a) and finite(b) and (a >= b if strict else a > b):
            errors.append(f"{lower} must be {'less than' if strict else 'at most'} {upper}")

    locked = {
        "OBS_DIM": MODEL_CONTRACT["obs_dim"], "ACT_DIM": MODEL_CONTRACT["act_dim"],
        "LIDAR_TOTAL_BINS": 432,
        "LIDAR_H_BINS": MODEL_CONTRACT["lidar_shape"][0],
        "LIDAR_V_LAYERS": MODEL_CONTRACT["lidar_shape"][1],
        "LIDAR_MIN": MODEL_CONTRACT["lidar_range_m"][0],
        "LIDAR_MAX": MODEL_CONTRACT["lidar_range_m"][1],
        "VX_TARGET": MODEL_CONTRACT["vx_target"],
        "VZ_MAX": MODEL_CONTRACT["vz_max"],
        "MODEL_FORWARD_ACTION_SCALE_MPS": MODEL_CONTRACT["action_scales"][0],
        "MODEL_LATERAL_ACTION_SCALE_MPS": MODEL_CONTRACT["action_scales"][1],
        "VX_MAX": MODEL_CONTRACT["action_scales"][0],
        "VY_MAX": MODEL_CONTRACT["action_scales"][1],
        "PATH_NORM": MODEL_CONTRACT["path_norm"], "CTE_NORM": MODEL_CONTRACT["cte_norm"],
        "DEFAULT_POLICY": MODEL_CONTRACT["default_policy"],
    }
    for name, expected in locked.items():
        if values[name] != expected:
            errors.append(f"{name} violates model contract: expected {expected!r}")
    if tuple(LIDAR_V_BOUNDARIES) != tuple(MODEL_CONTRACT["lidar_boundaries"]):
        errors.append("LIDAR_V_BOUNDARIES violates model contract (seven ordered boundaries required)")
    for name, scale in zip(
        ("PATH_FORWARD_COMMAND_CAP_MPS", "PATH_LATERAL_COMMAND_CAP_MPS"),
        MODEL_CONTRACT["action_scales"],
    ):
        if finite(values[name]) and values[name] > scale:
            errors.append(f"{name} exceeds model action scale {scale}")
    meta = POLICY_REGISTRY.get(DEFAULT_POLICY, {})
    for key, expected in {**MODEL_CONTRACT["policy"], "obs_dim": OBS_DIM, "act_dim": ACT_DIM}.items():
        if meta.get(key) != expected:
            errors.append(f"POLICY_REGISTRY[{DEFAULT_POLICY!r}].{key} violates model contract")
    try:
        if model_contract_fingerprint() != MODEL_CONTRACT_FINGERPRINT:
            errors.append("MODEL_CONTRACT fingerprint does not match the release baseline")
    except (TypeError, ValueError):
        errors.append("MODEL_CONTRACT fingerprint cannot be computed from valid finite JSON values")
    return errors
