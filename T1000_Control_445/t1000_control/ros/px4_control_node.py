import sys
import os
import math
import time
import hashlib
import json
from pathlib import Path as FilePath

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray

from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleAttitude, VehicleLocalPosition)

from ..inference.onnx_policy import OnnxPolicy
from ..inference.paths import default_model_path
from ..core.controller import ControlPipeline
from ..core.path_frame import path_from_poses
from .path_contract import validate_path_start

from ..safety import failsafe as FS
from ..core import velocity, attitude, coordinates, diagnostics
from ..sensors.binning import TOTAL_BINS, sensor_xyz_to_432bins
from ..sensors.parsing import check_pointcloud2_layout, pointcloud2_to_xyz
from .commander import PX4Commander, PX4_QOS
from .px4_setpoint_output import command_to_ned, trajectory_fields, yaw_rate_to_ned
from .sitl_lifecycle import (
    GoalDwellGate,
    LifecycleAction,
    LifecycleInputs,
    lifecycle_actions,
    timeout_actions,
)

from .. import settings as CFG
from ..__main__ import resolve_sitl_settings, validate_sitl_settings, validate_goal_lifecycle

POLICY_REGISTRY = CFG.POLICY_REGISTRY

POLICY_PATH = str(default_model_path())
CONTROL_HZ = CFG.CONTROL_HZ
PATH_TOPIC = CFG.TOPIC_PATH

LIDAR_TOPIC = CFG.TOPIC_LIDAR

LIDAR_SENSOR_OFFSET = CFG.SENSOR_OFFSET

GROUND_FILTER = CFG.GROUND_FILTER

SPEED_SCALE = CFG.SPEED_SCALE

FALLBACK_PATH = None

ALLOW_NO_RC = "--allow-no-rc" in sys.argv

DRY_RUN = "--dry-run" in sys.argv

ISAAC_SIM = os.environ.get("T1000_ISAAC_SIM", "0") == "1"
SIM_AUTO_START = os.environ.get("T1000_SIM_AUTO_START", "0") == "1"
SITL_CONFIRMED = os.environ.get("T1000_SITL_CONFIRMED", "0") == "1"
SIM_DURATION_S = CFG.SITL_DURATION_S
SIM_VX_CAP = CFG.SITL_ENU_X_CAP_MPS
SIM_VY_CAP = CFG.SITL_ENU_Y_CAP_MPS


def _sim_lifecycle_authorized():
    return (ISAAC_SIM and SIM_AUTO_START and ALLOW_NO_RC and
            SITL_CONFIRMED and not DRY_RUN)


def _validate_goal_options(until_goal, land_on_goal, ordered_waypoints, goal_stop):
    validate_goal_lifecycle(until_goal=until_goal, land_on_goal=land_on_goal,
        isaac_sim=ISAAC_SIM, sim_auto_start=SIM_AUTO_START, allow_no_rc=ALLOW_NO_RC,
        confirm_sitl=SITL_CONFIRMED, dry_run=DRY_RUN,
        ordered_waypoints=ordered_waypoints, goal_stop=goal_stop)


def _reset_goal(node):
    gate = getattr(node, '_goal_gate', None)
    if gate is not None:
        gate.reset()
    node._goal_ready = False


def _command_to_ned(vx_enu, vy_enu, vz_enu):
    return command_to_ned(
        vx_enu,
        vy_enu,
        vz_enu,
        isaac_sim=ISAAC_SIM,
        max_xy=(SIM_VX_CAP, SIM_VY_CAP),
        max_vz=float("inf"),
    )

SENSOR_QOS = QoSProfile(
    reliability=(ReliabilityPolicy.RELIABLE if CFG.LIDAR_QOS_RELIABLE
                 else ReliabilityPolicy.BEST_EFFORT),
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=CFG.LIDAR_QOS_DEPTH,
)


class RLInferenceNodePX4(Node):

    def __init__(self, sitl_settings=None, *, corner_mode=None, settings_snapshot=None,
                 policy_authority_mode=None, goal_stop=None, ordered_waypoints=None,
                 until_goal=False, land_on_goal=False):
        global SIM_DURATION_S, SIM_VX_CAP, SIM_VY_CAP
        _validate_goal_options(until_goal, land_on_goal, ordered_waypoints, goal_stop)
        settings_errors = CFG.validate_settings(corner_mode=corner_mode,
            policy_authority_mode=policy_authority_mode, goal_stop=goal_stop,
            ordered_waypoints=ordered_waypoints)
        if settings_errors:
            raise ValueError("Invalid control settings:\n- " + "\n- ".join(settings_errors))
        sitl_settings = resolve_sitl_settings() if sitl_settings is None else sitl_settings
        validate_sitl_settings(sitl_settings)
        corner_source = 'settings' if corner_mode is None else 'cli'
        corner_mode = CFG.CORNER_MODE if corner_mode is None else corner_mode
        SIM_DURATION_S = sitl_settings['duration'][0]
        SIM_VX_CAP = sitl_settings['vx_cap'][0]
        SIM_VY_CAP = sitl_settings['vy_cap'][0]
        super().__init__("rl_inference_node_px4")

        self.policy = OnnxPolicy(POLICY_PATH)
        self.sess = self.policy.session
        self.policy_meta = self._check_policy_io()
        self.yaw_mode = self.policy_meta["yaw_mode"]

        self.pos_ned = np.zeros(3, dtype=np.float64)
        self.vel_ned = np.zeros(3, dtype=np.float64)
        self.quat_ned = np.array([1.0, 0.0, 0.0, 0.0])
        self.lidar = np.ones(TOTAL_BINS)
        self.lidar_ready = False

        self._t_lidar = 0.0
        self._t_odom = 0.0
        self._t_position = 0.0
        self._t_velocity = 0.0
        self._t_attitude = 0.0
        self._position_seen = False
        self._attitude_seen = False
        self._velocity_seen = False
        self._odom_lat_min = None
        self._odom_lag = 0.0
        self._t_rc = 0.0
        self._xy_valid = False
        self._z_valid = False
        self._v_xy_valid = False
        self._v_z_valid = False
        self._rc_value = None
        self._rc_trusted = None
        self._fs_state = FS.NORMAL
        self._fs_last_log = ""

        self._preflight_ok, _why = FS.preflight_ok(allow_no_rc=ALLOW_NO_RC)
        for w in _why:
            self.get_logger().warn(f"[preflight] {w}")
        if not self._preflight_ok and not DRY_RUN:
            self.get_logger().error(
                "Preflight failed. Setpoints are off. "
                "Use --allow-no-rc only for a ground test."
            )

        self._dry_stats = {"n": 0, "vx": [], "vy": [], "states": {}}
        if DRY_RUN:
            self.get_logger().warn(
                "DRY-RUN: no /fmu/in/* output. "
                f"Commands go only to {CFG.TOPIC_CMD_PREVIEW}. "
                "Do not use this mode with motors connected."
            )
            self.cmd_preview_pub = self.create_publisher(
                Float32MultiArray, CFG.TOPIC_CMD_PREVIEW, 10)

        self.path_pts = None
        self.path_stamp = None
        self._received_path = None
        self.controller = ControlPipeline(
            self.policy, yaw_mode=self.yaw_mode,
            flight_altitude_m=CFG.FLIGHT_ALTITUDE_M,
            corner_mode=corner_mode,
            policy_authority_mode=policy_authority_mode, goal_stop=goal_stop,
            ordered_waypoints=ordered_waypoints,
        )
        if FALLBACK_PATH is not None:
            self.controller.set_path(FALLBACK_PATH)


        self._layout_checked = False

        self.commander = PX4Commander(self)
        self._offboard_seen = False
        self._landing = False

        self._sim_started_at = self._now()
        self._sim_duration_s = SIM_DURATION_S
        self._until_goal = until_goal
        self._land_on_goal = land_on_goal
        self._goal_gate = GoalDwellGate(
            CFG.GOAL_STOP_RADIUS_M,
            CFG.GOAL_DWELL_MAX_SPEED_MPS,
            CFG.GOAL_DWELL_S,
        )
        self._goal_ready = False
        self._last_diagnostics = None
        effective = {
            name: {'value': getattr(CFG, name), 'source': 'settings'}
            for name in ('FLIGHT_ALTITUDE_M', 'PATH_FORWARD_COMMAND_CAP_MPS',
                         'PATH_LATERAL_COMMAND_CAP_MPS', 'CORNER_MAX_DEVIATION_M',
                         'CORNER_SPEED_MPS', 'CORNER_DECEL_MPS2', 'CORNER_SAMPLE_SPACING_M',
                         'CONTROL_HZ', 'SPEED_SCALE', 'VEL_LOWPASS_TAU', 'VX_TARGET',
                         'KP_ALT_VEL', 'VZ_CMD_MAX', 'KP_YAW_RATE', 'YAW_RATE_MAX',
                         'AVOID_ACTIVATE_DISTANCE_M', 'AVOID_FULL_DISTANCE_M',
                         'AVOID_FORWARD_HALF_ANGLE_DEG', 'AVOID_MAX_AUTHORITY',
                         'AVOID_FORWARD_CAP_MPS', 'AVOID_PASS_MARGIN_M',
                         'AVOIDANCE_YAW_MODE',
                         'GOAL_STOP_RADIUS_M', 'GOAL_DECEL_MPS2',
                         'GOAL_DWELL_MAX_SPEED_MPS', 'GOAL_DWELL_S',
                         'YAW_COMMAND_MIN_SPEED_MPS',
                         'STATUS_TIMEOUT_HOLD_S', 'STATUS_TIMEOUT_ABORT_S',
                         'PATH_START_MODE', 'PATH_START_TOLERANCE_M')
        }
        effective['CORNER_MODE'] = {'value': self.controller.corner_mode, 'source': corner_source}
        effective['POLICY_AUTHORITY_MODE'] = {'value': self.controller.policy_authority_mode,
            'source': 'settings' if policy_authority_mode is None else 'cli'}
        effective['GOAL_STOP_ENABLED'] = {'value': self.controller.goal_stop,
            'source': 'settings' if goal_stop is None else 'cli'}
        effective['ORDERED_WAYPOINTS_ENABLED'] = {'value': self.controller.ordered_waypoints,
            'source': 'settings' if ordered_waypoints is None else 'cli'}
        effective['SITL_UNTIL_GOAL'] = {'value': until_goal, 'source': 'cli'}
        effective['SITL_LAND_ON_GOAL'] = {'value': land_on_goal, 'source': 'cli'}
        effective['YAW_MODE'] = {'value': self.yaw_mode, 'source': 'policy_registry'}
        for key, name in (('duration', 'SITL_DURATION_S'),
                          ('vx_cap', 'SITL_ENU_X_CAP_MPS'), ('vy_cap', 'SITL_ENU_Y_CAP_MPS')):
            value, source = sitl_settings[key]
            effective[name] = {'value': value, 'source': source}
        snapshot = {'settings_file': os.path.abspath(CFG.__file__), 'effective': effective,
                    'isaac_sim': ISAAC_SIM, 'dry_run': DRY_RUN}
        serialized = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        if settings_snapshot is not None:
            with open(settings_snapshot, 'x', encoding='utf-8') as stream:
                stream.write(serialized + '\n')
        self.get_logger().info('Effective settings: ' + serialized)
        self._sim_mode_sent = False
        self._sim_arm_sent = False
        self._sim_finished = False
        if _sim_lifecycle_authorized():
            self.create_timer(0.1, self._sim_timeout_tick)

        if ISAAC_SIM:
            self.create_subscription(PoseStamped, CFG.TOPIC_ISAAC_POSE,
                                     self._sim_pose_cb, 10)
            self.create_subscription(TwistStamped, CFG.TOPIC_ISAAC_VELOCITY,
                                     self._sim_velocity_cb, 10)
            self.create_subscription(Float32MultiArray, CFG.TOPIC_ISAAC_LIDAR_BINS,
                                     self._sim_lidar_cb, 10)
        else:
            self.create_subscription(PointCloud2, LIDAR_TOPIC,
                                     self._lidar_cb, SENSOR_QOS)
        self.create_subscription(Path, PATH_TOPIC, self._path_cb, 10)
        if not ISAAC_SIM:
            self.create_subscription(VehicleLocalPosition,
                                     CFG.TOPIC_PX4_LOCAL_POS,
                                     self._pos_cb, PX4_QOS)
            self.create_subscription(VehicleAttitude, CFG.TOPIC_PX4_ATTITUDE,
                                     self._att_cb, PX4_QOS)
        _src = getattr(CFG, "RC_ABORT_SOURCE", "input_rc")
        _slot = (getattr(CFG, "RC_ABORT_AUX", None) if _src == "manual_control"
                 else CFG.RC_ABORT_CHANNEL)
        if _src == FS.MODE_SWITCH:
            _slot = None
            self.get_logger().info(
                "RC abort uses the PX4 mode switch. RC topics are not read. "
                "Abort starts when nav_state leaves OFFBOARD."
            )
        if _slot is not None:
            _unit = getattr(CFG, "RC_ABORT_UNIT", "us")
            try:
                if _src == "manual_control":
                    from px4_msgs.msg import ManualControlSetpoint as _RcMsg
                    _topic = CFG.TOPIC_RC_MANUAL_CONTROL
                    _what = f"aux{_slot}"
                elif _unit == "norm":
                    from px4_msgs.msg import RcChannels as _RcMsg
                    _topic = CFG.TOPIC_RC_CHANNELS
                    _what = f"ch{_slot} (array index {_slot - 1})"
                else:
                    from px4_msgs.msg import InputRc as _RcMsg
                    _topic = CFG.TOPIC_RC_INPUT
                    _what = f"ch{_slot} (array index {_slot - 1})"
                self.create_subscription(_RcMsg, _topic, self._rc_cb, PX4_QOS)
                self.get_logger().info(
                    f"RC abort watch: {_what} >= {CFG.RC_ABORT_THRESHOLD} "
                    f"[{_unit}], {_topic}"
                )
                if _src == "manual_control":
                    self.get_logger().warn(
                        "manual_control_setpoint can come from a MAVLink joystick. "
                        "Each sample must have data_source == SOURCE_RC and valid set."
                    )
            except ImportError:
                self.get_logger().error(
                    f"px4_msgs has no RC message for {_src}. "
                    "The abort switch cannot be watched."
                )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, CFG.TOPIC_PX4_OFFBOARD, PX4_QOS)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, CFG.TOPIC_PX4_SETPOINT, PX4_QOS)

        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self.step = 0
        lidar_input = "/t1000/lidar_bins (Isaac pre-binned)" if ISAAC_SIM else LIDAR_TOPIC
        self.get_logger().info(
            f"Ready. Waiting for LiDAR({lidar_input}) and path({PATH_TOPIC}) "
            f"| SPEED_SCALE={SPEED_SCALE} | ground_filter={GROUND_FILTER} "
            f"| LPF tau={velocity.VEL_LOWPASS_TAU}s | sensor_offset={LIDAR_SENSOR_OFFSET}"
        )
        if SPEED_SCALE != 1.0:
            self.get_logger().warn(
                f"SPEED_SCALE={SPEED_SCALE} is not tested. "
                "It can move inputs outside the training range "
                f"(obs[435]={SPEED_SCALE*0.6:.3f} vs obs[444]=0.600). "
                "Only 1.0 is tested."
            )


    def _check_policy_io(self) -> dict:
        i = self.sess.get_inputs()[0]
        o = self.sess.get_outputs()[0]
        n_in, n_out = i.shape[-1], o.shape[-1]
        if i.name != "obs":
            raise SystemExit(
                f"ONNX input name is '{i.name}', but it must be 'obs'.\n"
                "Restore the bundled model and run the preflight check."
            )
        if n_in != CFG.OBS_DIM or n_out != CFG.ACT_DIM:
            raise SystemExit(
                f"ONNX shape is {n_in} inputs and {n_out} outputs; "
                f"need {CFG.OBS_DIM} and {CFG.ACT_DIM}.\n"
                "Restore the bundled model and run the preflight check."
            )

        name = os.path.basename(POLICY_PATH)
        meta = POLICY_REGISTRY.get(name)
        if meta is None:
            raise SystemExit(
                f"Policy '{name}' is not in POLICY_REGISTRY.\n"
                f"Known policies: {sorted(POLICY_REGISTRY)}"
            )
        expected_hash = meta.get("onnx_sha256")
        if expected_hash is not None:
            actual_hash = hashlib.sha256(FilePath(POLICY_PATH).read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise SystemExit(
                    f"Policy SHA-256 mismatch: {actual_hash} != {expected_hash}"
                )

        self.get_logger().info(
            f"Policy loaded: {name}  {i.name}{i.shape} -> {o.name}{o.shape}\n"
            f"Yaw mode: {meta['yaw_mode']} | {meta.get('note', '')}"
        )
        if not meta.get("mission_ok", False):
            self.get_logger().warn(
                "This policy is for an interface check, not a flight mission."
            )
        return meta


    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    _SOURCE_RC_FALLBACK = 1

    def _rc_cb(self, msg):
        _src = getattr(CFG, "RC_ABORT_SOURCE", "input_rc")
        if _src == "manual_control":
            _rc_id = getattr(msg, "SOURCE_RC", self._SOURCE_RC_FALLBACK)
            self._rc_trusted = (bool(getattr(msg, "valid", False))
                                and int(getattr(msg, "data_source", -1)) == _rc_id)
            aux = getattr(CFG, "RC_ABORT_AUX", None)
            try:
                self._rc_value = float(getattr(msg, f"aux{aux}"))
            except (TypeError, AttributeError, ValueError):
                self._rc_value = None
            self._t_rc = self._now()
            return

        self._rc_trusted = True
        i = CFG.RC_ABORT_CHANNEL - 1
        arr = getattr(msg, "channels", None)
        if arr is None:
            arr = getattr(msg, "values", None)
        try:
            self._rc_value = float(arr[i])
        except (IndexError, TypeError, AttributeError):
            self._rc_value = None
        self._t_rc = self._now()

    def _pos_cb(self, msg: VehicleLocalPosition):
        if ISAAC_SIM:
            return
        self.pos_ned = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
        self._position_seen = True
        self._velocity_seen = True
        self._t_odom = self._t_position = self._t_velocity = self._now()
        ts_us = getattr(msg, "timestamp_sample", 0) or getattr(msg, "timestamp", 0)
        if ts_us:
            lat = self._t_odom - ts_us * 1e-6
            self._odom_lat_min, self._odom_lag = diagnostics.observation_lag(
                lat, self._odom_lat_min)
        self._xy_valid = bool(getattr(msg, "xy_valid", False))
        self._z_valid = bool(getattr(msg, "z_valid", False))
        self._v_xy_valid = bool(getattr(msg, "v_xy_valid", False))
        self._v_z_valid = bool(getattr(msg, "v_z_valid", False))
        self.vel_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)

    def _att_cb(self, msg: VehicleAttitude):
        if ISAAC_SIM:
            return
        self.quat_ned = np.array(msg.q, dtype=np.float64)
        self._attitude_seen = True
        self._t_attitude = self._now()

    def _sim_pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.pos_ned = coordinates.enu_to_ned_vec([p.x, p.y, p.z])
        q_enu_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
        self.quat_ned = coordinates.ned_to_enu_quat(q_enu_wxyz)
        self._position_seen = self._attitude_seen = True
        self._xy_valid = True
        self._z_valid = self._v_xy_valid = self._v_z_valid = True
        self._t_odom = self._t_position = self._t_attitude = self._now()

    def _sim_velocity_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        self.vel_ned = coordinates.enu_to_ned_vec([v.x, v.y, v.z])
        self._velocity_seen = True
        self._t_velocity = self._now()

    def _sim_lidar_cb(self, msg: Float32MultiArray):
        if len(msg.data) != TOTAL_BINS:
            self.get_logger().error(
                f"Isaac LiDAR bin count {len(msg.data)} != {TOTAL_BINS}")
            return
        self.lidar = np.asarray(msg.data, dtype=np.float32)
        self.lidar_ready = True
        self._t_lidar = self._now()

    def _path_cb(self, msg: Path):
        try:
            xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            if not np.isfinite(xy).all():
                raise ValueError('path points must be finite')
            xy = path_from_poses(xy)
            if np.array_equal(xy, self._received_path):
                self.path_stamp = time.monotonic()
                return
            if (not self._position_seen or not self._xy_valid
                    or not 0. <= self._now() - self._t_position <= CFG.ODOM_TIMEOUT_HOLD_S):
                raise ValueError('current world-ENU pose is missing, invalid, or stale')
            current_xy = coordinates.ned_to_enu_vec(self.pos_ned)[:2]
            start = validate_path_start(xy, current_xy, CFG.PATH_START_TOLERANCE_M)
            if CFG.PATH_START_MODE == 'require_near':
                if not start.accepted:
                    raise ValueError(
                        f'first waypoint gap={start.gap_m!r} m exceeds '
                        f'tolerance={CFG.PATH_START_TOLERANCE_M!r} m')
                accepted = xy
            elif CFG.PATH_START_MODE == 'prepend_current':
                accepted = np.vstack((current_xy, xy))
            else:
                raise ValueError('PATH_START_MODE must be require_near or prepend_current')
            new_pts = self.controller.set_path(accepted)
            self.path_pts = new_pts
            _reset_goal(self)
            self._received_path = xy.copy()
            self.path_stamp = time.monotonic()
        except (ValueError, AttributeError) as e:
            self.get_logger().warn(f"Path rejected: {e}")

    def _lidar_cb(self, msg: PointCloud2):
        if not self._layout_checked:
            self._layout_checked = True
            if not check_pointcloud2_layout(msg):
                fields = [(f.name, f.offset, f.datatype) for f in msg.fields]
                raise SystemExit(
                    f"Bad PointCloud2 layout: {fields}\n"
                    "x, y, z must be float32 at offsets 0, 4, 8. Check the driver."
                )
            self.get_logger().info(f"Point cloud layout is valid ({msg.width} pts)")

        alt = float(-self.pos_ned[2]) if GROUND_FILTER else None
        self.lidar = sensor_xyz_to_432bins(
            pointcloud2_to_xyz(msg),
            sensor_offset=LIDAR_SENSOR_OFFSET,
            alt_agl=alt,
            filter_ground_points=GROUND_FILTER,
        )
        self.lidar_ready = True
        self._t_lidar = self._now()


    def _control_loop(self):
        if self._landing:
            _reset_goal(self)
            return
        self._track_offboard()
        if getattr(self, '_offboard_lost', lambda: False)():
            _reset_goal(self)
            self._abort_action()
            return
        if not (self._position_seen and self._attitude_seen and self._velocity_seen
                and self.commander.status_seen):
            _reset_goal(self)
            return

        if not self._preflight_ok and not DRY_RUN:
            _reset_goal(self)
            return
        self._publish_offboard_mode()
        if not self.lidar_ready:
            _reset_goal(self)
            return

        path, age = self._active_path()
        if path is None:
            _reset_goal(self)
            if self._failsafe_state(age) == FS.ABORT:
                self._abort_action()
                return
            self._publish_hold()
            return

        state = self._failsafe_state(age)
        if state == FS.ABORT:
            _reset_goal(self)
            self._abort_action()
            return
        try:
            result = self.controller.step(
                lidar_bins=self.lidar, pos_ned=self.pos_ned,
                vel_ned=self.vel_ned, quat_ned=self.quat_ned,
            )
        except ValueError as error:
            _reset_goal(self)
            self.get_logger().error(f"Control input rejected: {error}")
            self._abort_action()
            return
        pos_enu, aux, action = result.position_enu, result.aux, result.action
        vx_enu, vy_enu = result.velocity_enu[:2]
        vx_enu, vy_enu = FS.apply(vx_enu, vy_enu, state)
        health = state

        vz_enu = result.velocity_enu[2]
        yaw_rate = result.yaw_rate

        vx_n, vy_n, vz_n = _command_to_ned(vx_enu, vy_enu, vz_enu)
        self._publish_velocity_ned(vx_n, vy_n, vz_n, yaw_rate_to_ned(yaw_rate))
        if hasattr(self, '_goal_gate'):
            # Invert the confirmed Isaac output mapping after failsafe and caps.
            command_enu = ([vy_n, vx_n, -vz_n] if ISAAC_SIM
                           else [vx_n, -vy_n, -vz_n])
            if self.controller.policy_authority_mode == 'always':
                aux.setdefault('avoidance_phase', 'CLEAR')
            self._goal_ready = self._goal_gate.update(self._now(), aux, command_enu[:2],
                healthy=(state == FS.NORMAL and self.commander.is_offboard))
            aux.update(command_velocity_enu=command_enu,
                measured_velocity_enu=coordinates.ned_to_enu_vec(self.vel_ned).tolist(),
                waypoint_count=len(self.controller.original_path),
                goal_dwell_s=self._goal_gate.dwell_s)
            self._last_diagnostics = {key: aux.get(key) for key in (
                'active_waypoint_index', 'waypoint_count', 'waypoints_reached',
                'all_waypoints_reached', 'command_velocity_enu', 'measured_velocity_enu',
                'avoidance_phase', 'yaw_target', 'goal_distance_m', 'goal_dwell_s')}
            self._last_diagnostics.update(pos_enu=pos_enu.tolist(), cte_m=aux.get('lat'),
                elapsed_s=self._now() - self._sim_started_at, health=state)
            if self._goal_ready or self.step % max(1, int(CONTROL_HZ / 5)) == 0:
                self.get_logger().info('Controller diagnostics: ' +
                    json.dumps(self._last_diagnostics, allow_nan=False))
        if SIM_AUTO_START:
            self._sim_autostart()

        self.step += 1
        if self.step % int(CONTROL_HZ * 5) == 0:
            self._log(pos_enu, aux, action, health)

    def _sim_autostart(self):
        self._apply_sim_lifecycle_actions(
            lifecycle_actions(self._sim_lifecycle_inputs())
        )

    def _sim_timeout_tick(self):
        if getattr(self, '_offboard_lost', lambda: False)():
            _reset_goal(self)
            self._abort_action()
            return
        self._apply_sim_lifecycle_actions(
            timeout_actions(self._sim_lifecycle_inputs())
        )

    def _sim_lifecycle_inputs(self):
        return LifecycleInputs(
            sim_mode=ISAAC_SIM,
            auto_start=SIM_AUTO_START,
            allow_no_rc=ALLOW_NO_RC,
            sitl_confirmed=SITL_CONFIRMED,
            dry_run=DRY_RUN,
            elapsed_s=self._now() - self._sim_started_at,
            duration_s=self._sim_duration_s,
            offboard=self._sim_mode_sent,
            armed=self._sim_arm_sent,
            finished=self._sim_finished,
            until_goal=getattr(self, '_until_goal', False),
            land_on_goal=getattr(self, '_land_on_goal', False),
            ordered_waypoints=getattr(getattr(self, 'controller', None), 'ordered_waypoints', False),
            goal_stop=getattr(getattr(self, 'controller', None), 'goal_stop', False),
            goal_ready=getattr(self, '_goal_ready', False),
        )

    def _apply_sim_lifecycle_actions(self, actions):
        for action in actions:
            if action is LifecycleAction.REQUEST_OFFBOARD:
                self.commander.set_offboard()
                self._sim_mode_sent = True
            elif action is LifecycleAction.ARM:
                self.commander.arm()
                self._sim_arm_sent = True
            elif action is LifecycleAction.LAND:
                self.commander.land()
                self._landing = True
            elif action is LifecycleAction.FINISH:
                self._sim_finished = True

    def _track_offboard(self):
        if self.commander.status_seen and self.commander.is_offboard:
            self._offboard_seen = True

    def _offboard_lost(self):
        if not self.commander.status_seen or not self._offboard_seen:
            return None
        return not self.commander.is_offboard

    def _abort_action(self):
        if getattr(self, '_landing', False):
            return
        if getattr(self, "_offboard_lost", lambda: None)():
            self._landing = True
            self._sim_finished = True
            self.get_logger().error(
                "[abort] Pilot or PX4 left OFFBOARD. Setpoint output is stopped; "
                "the selected PX4 mode is not changed."
            )
            return
        if not getattr(CFG, "ABORT_ACTION_LAND", True):
            self._publish_hold()
            return
        self._landing = True
        if DRY_RUN:
            self.get_logger().error(
                "[abort] DRY-RUN: LAND is not sent. A live run would land and stop output."
            )
            return
        self.commander.land()
        if _sim_lifecycle_authorized():
            self._sim_finished = True
        self.get_logger().error(
            "[abort] LAND sent. Offboard output is now stopped. Take manual control."
        )

    def _failsafe_state(self, path_age_s: float) -> str:
        _t = self._now()
        receipt_times = (
            getattr(self, '_t_position', self._t_odom),
            getattr(self, '_t_velocity', self._t_odom),
            getattr(self, '_t_attitude', self._t_odom),
        )
        status_time = getattr(getattr(self, 'commander', None), '_t_status', self._t_odom)
        state, _scale, _why = FS.evaluate(
            lidar_age_s=_t - self._t_lidar,
            odom_age_s=max(_t - stamp for stamp in receipt_times),
            status_age_s=_t - status_time,
            path_age_s=path_age_s,
            xy_valid=self._xy_valid,
            z_valid=getattr(self, '_z_valid', True),
            v_xy_valid=getattr(self, '_v_xy_valid', True),
            v_z_valid=getattr(self, '_v_z_valid', True),
            rc_value=self._rc_value,
            rc_age_s=((_t - self._t_rc)
                      if (FS._rc_slot()[1] is not None
                          and CFG.RC_ABORT_SOURCE != FS.MODE_SWITCH) else None),
            rc_trusted=self._rc_trusted,
            offboard_lost=self._offboard_lost(),
        )
        if state != self._fs_state or "; ".join(_why) != self._fs_last_log:
            self._fs_state, self._fs_last_log = state, "; ".join(_why)
            self._log_failsafe(state, f"[failsafe] {state} {self._fs_last_log}")
        return state

    def _active_path(self):
        if self.path_pts is not None:
            return self.path_pts, time.monotonic() - self.path_stamp
        if FALLBACK_PATH is not None:
            return FALLBACK_PATH, 0.0
        return None, float("inf")

    def _log(self, pos_enu, aux, action, health):
        close = self.lidar[self.lidar < 1.0]
        min_d = close.min() * 40.0 if close.size else 40.0
        self.get_logger().info(
            f"step={self.step:5d} | ENU=({pos_enu[0]:.1f},{pos_enu[1]:.1f},{pos_enu[2]:.1f}) "
            f"| remain={aux['remain']:.1f}m seg={aux['seg']} "
            f"| cte={aux['lat']:+.2f}m | act=({action[0]:+.2f},{action[1]:+.2f}) "
            f"| nmpc=({aux['vx_ref']:.2f},{aux['vy_ref']:+.2f}) "
            f"| authority={aux['policy_authority']:.3f} forward_distance={aux['avoidance_distance_m']:.2f}m "
            f"avoid_progress={aux['avoidance_progress']:.3f} lqr_scale={aux['avoidance_lqr_scale']:.3f} "
            f"avoid_hold={aux['avoidance_holding']} obstacle_along={aux['avoidance_obstacle_along_m']} "
            f"avoid_vx_cap={aux['avoidance_forward_cap_mps']:.2f}m/s "
            f"| goal_distance={aux['goal_distance_m']:.2f}m goal_stop={aux['goal_stopped']} "
            f"| waypoint={aux.get('active_waypoint_index')}/{aux.get('waypoint_count')} "
            f"reached={aux.get('waypoints_reached')} all={aux.get('all_waypoints_reached')} "
            f"cmd_enu={aux.get('command_velocity_enu')} measured_enu={aux.get('measured_velocity_enu')} "
            f"phase={aux.get('avoidance_phase')} yaw_target={aux.get('yaw_target')} "
            f"goal_dwell={aux.get('goal_dwell_s')}s "
            f"| lidar_min={min_d:.1f}m | path={health}"
        )


    def _log_failsafe(self, state, msg):
        lg = self.get_logger()
        if state == FS.ABORT:
            lg.error(msg)
        elif state != FS.NORMAL:
            lg.warn(msg)
        else:
            lg.info(msg)

    def _publish_offboard_mode(self):
        if DRY_RUN:
            return
        m = OffboardControlMode()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.velocity = True
        m.position = m.acceleration = m.attitude = m.body_rate = False
        self.offboard_pub.publish(m)

    def _publish_velocity_ned(self, vx, vy, vz, yawspeed):
        fields = trajectory_fields(vx, vy, vz, yawspeed)
        if DRY_RUN:
            self._preview_command(*fields["velocity"], fields["yawspeed"])
            return
        m = TrajectorySetpoint()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.velocity = fields["velocity"]
        m.yawspeed = fields["yawspeed"]
        m.position = fields["position"]
        m.yaw = fields["yaw"]
        self.setpoint_pub.publish(m)

    def _preview_command(self, vx, vy, vz, yawspeed):
        m = Float32MultiArray()
        m.data = [float(vx), float(vy), float(vz), float(yawspeed)]
        self.cmd_preview_pub.publish(m)

        s = self._dry_stats
        s["n"] += 1
        s["vx"].append(float(vx))
        s["vy"].append(float(vy))
        s["states"][self._fs_state] = s["states"].get(self._fs_state, 0) + 1

        lim = []
        if not all(map(math.isfinite, (vx, vy, vz, yawspeed))):
            lim.append("value is not finite; PX4 may reject it")
        if lim:
            self.get_logger().error("[dry-run] " + "; ".join(lim))

        if s["n"] % int(CONTROL_HZ * 5) == 0:
            import statistics as _st
            vxs, vys = s["vx"][-int(CONTROL_HZ * 5):], s["vy"][-int(CONTROL_HZ * 5):]
            dist = " ".join(f"{k}:{v}" for k, v in sorted(s["states"].items()))
            self.get_logger().info(
                f"[dry-run] n={s['n']} | "
                f"NED vx {_st.mean(vxs):+.2f} (max {max(map(abs, vxs)):.2f}) | "
                f"NED vy {_st.mean(vys):+.2f} (max {max(map(abs, vys)):.2f}) | "
                f"failsafe [{dist}]")
            if not self._preflight_ok:
                self.get_logger().warn(
                    "Preflight still fails. Dry-run only blocks output; it does not pass preflight."
                )

    def _publish_hold(self):
        vz_enu = attitude.altitude_velocity(
            coordinates.ned_to_enu_vec(self.pos_ned)[2],
            target=self.controller.flight_altitude_m,
        )
        self._publish_velocity_ned(0.0, 0.0, -vz_enu, 0.0)
        if self.step % int(CONTROL_HZ * 2) == 0:
            self.get_logger().warn(f"No path. Holding still and waiting for {PATH_TOPIC}")
        self.step += 1


def main(sitl_settings=None, *, corner_mode=None, settings_snapshot=None,
         policy_authority_mode=None, goal_stop=None, ordered_waypoints=None,
         until_goal=False, land_on_goal=False):
    _validate_goal_options(until_goal, land_on_goal, ordered_waypoints, goal_stop)
    settings_errors = CFG.validate_settings(corner_mode=corner_mode,
        policy_authority_mode=policy_authority_mode, goal_stop=goal_stop,
        ordered_waypoints=ordered_waypoints)
    if settings_errors:
        raise ValueError("Invalid control settings:\n- " + "\n- ".join(settings_errors))
    sitl_settings = resolve_sitl_settings() if sitl_settings is None else sitl_settings
    validate_sitl_settings(sitl_settings)
    rclpy.init()
    node = RLInferenceNodePX4(sitl_settings=sitl_settings, corner_mode=corner_mode,
                            settings_snapshot=settings_snapshot,
                            policy_authority_mode=policy_authority_mode, goal_stop=goal_stop,
                            ordered_waypoints=ordered_waypoints, until_goal=until_goal,
                            land_on_goal=land_on_goal)
    try:
        while rclpy.ok() and not node._sim_finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping")
    except ExternalShutdownException:
        node.get_logger().info("External stop signal")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
