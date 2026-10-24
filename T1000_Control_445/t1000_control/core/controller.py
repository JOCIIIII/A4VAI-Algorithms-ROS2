from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .. import settings
from . import attitude, avoidance_authority, coordinates, observation, path_frame, quaternion, velocity
from .waypoint_progress import WaypointProgress


class Policy(Protocol):

    def predict(self, observation: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ControlResult:

    observation: np.ndarray
    action: np.ndarray
    aux: dict
    position_enu: np.ndarray
    velocity_enu: tuple[float, float, float]
    yaw_rate: float


class ControlPipeline:

    def __init__(self, policy: Policy, *, yaw_mode: str = settings.YAW_MODE_PATH,
                 flight_altitude_m: float = settings.ALT_TARGET,
                 corner_mode: str | None = None, policy_authority_mode: str | None = None,
                 goal_stop: bool | None = None, ordered_waypoints: bool | None = None,
                 avoidance_yaw_mode: str | None = None):
        self.policy = policy
        self.yaw_mode = yaw_mode
        self.avoidance_yaw_mode = (settings.AVOIDANCE_YAW_MODE
                                   if avoidance_yaw_mode is None else avoidance_yaw_mode)
        if self.avoidance_yaw_mode not in (settings.YAW_MODE_PATH,
                                           settings.YAW_MODE_COMMAND_VELOCITY):
            raise ValueError('invalid avoidance yaw mode; use path or command_velocity')
        self.flight_altitude_m = flight_altitude_m
        self.corner_mode = settings.CORNER_MODE if corner_mode is None else corner_mode
        self.policy_authority_mode = settings.POLICY_AUTHORITY_MODE if policy_authority_mode is None else policy_authority_mode
        if self.policy_authority_mode not in ('always', 'obstacle'):
            raise ValueError('policy authority mode must be always or obstacle')
        self.goal_stop = settings.GOAL_STOP_ENABLED if goal_stop is None else goal_stop
        self.ordered_waypoints = (settings.ORDERED_WAYPOINTS_ENABLED
                                  if ordered_waypoints is None else ordered_waypoints)
        if type(self.ordered_waypoints) is not bool:
            raise ValueError('ordered_waypoints must be boolean')
        self._waypoint_progress: WaypointProgress | None = None
        self._waypoint_recovery_index: int | None = None
        self._start_acquisition_checked = False
        self._goal_stopped = False
        self._avoidance_hold = avoidance_authority.AvoidanceHold()
        self.control_hz = settings.CONTROL_HZ
        self.speed_scale = settings.SPEED_SCALE
        self.forward_cap_mps = settings.PATH_FORWARD_COMMAND_CAP_MPS
        self.lateral_cap_mps = settings.PATH_LATERAL_COMMAND_CAP_MPS
        self._path: np.ndarray | None = None
        self._control_geometry: path_frame.RoundedPath | None = None
        self._projector = path_frame.MonotonicProjector()
        self._previous_reference: tuple[float, float] | None = None
        self._velocity_filter = velocity.VelocityLowpass()
        self._velocity_rate_limiter = velocity.VelocityRateLimiter(
            settings.CORNER_DECEL_MPS2, settings.CORNER_DECEL_MPS2,
        )

    @property
    def original_path(self) -> np.ndarray | None:
        if self._path is None:
            return None
        result = self._path.copy()
        result.setflags(write=False)
        return result

    @property
    def control_path(self) -> np.ndarray | None:
        if self._control_geometry is None:
            return None
        result = self._control_geometry.control_points.copy()
        result.setflags(write=False)
        return result

    def set_path(self, points) -> np.ndarray:
        if not np.isfinite(np.asarray(points, float)).all():
            raise ValueError('path points must be finite')
        path = path_frame.path_from_poses(points)
        if self._path is None or not np.array_equal(path, self._path):
            geometry = path_frame.build_control_path(
                points, mode=self.corner_mode,
                max_deviation_m=settings.CORNER_MAX_DEVIATION_M,
                sample_spacing_m=settings.CORNER_SAMPLE_SPACING_M,
                corner_speed_mps=settings.CORNER_SPEED_MPS,
                cruise_speed_mps=settings.VX_TARGET,
            )
            self._control_geometry = geometry
            self._path = geometry.original_points
            self._waypoint_progress = WaypointProgress(self._path) if self.ordered_waypoints else None
            self._waypoint_recovery_index = None
            self._start_acquisition_checked = False
            self._projector.reset()
            self._goal_stopped = False
            self._avoidance_hold = avoidance_authority.AvoidanceHold()
        return path

    def _forward_speed_cap(self, frame: dict) -> float:
        geometry = self._control_geometry
        seg = frame['seg']
        cruise = min(self.forward_cap_mps, settings.VX_TARGET)
        cap = min(cruise, float(geometry.segment_speed_caps[seg]))
        if geometry.corner_mask[seg]:
            return cap
        corners = np.flatnonzero(geometry.corner_mask[seg:])
        if corners.size:
            next_corner = seg + int(corners[0])
            distance = float(np.linalg.norm(np.diff(
                geometry.control_points[seg:next_corner + 1], axis=0), axis=1).sum()) - frame['s']
            delay = max(0., self._velocity_filter.tau) + 1.0 / self.control_hz
            braking_distance = max(0., distance - cruise * delay)
            corner_speed = float(geometry.segment_speed_caps[next_corner])
            cap = min(cap, float(np.sqrt(corner_speed**2 + 2. * settings.CORNER_DECEL_MPS2 * braking_distance)))
        return cap

    def _waypoint_recovery(self, position, progress, lidar_bins, current_yaw, frame):
        state = dict(
            waypoint_recovery_active=False,
            waypoint_recovery_blocked=False,
            waypoint_recovery_target_index=None,
            waypoint_recovery_distance_m=None,
            waypoint_recovery_scan_m=None,
            waypoint_recovery_required_clearance_m=None,
        )
        if progress is None:
            self._waypoint_recovery_index = None
            return state, None

        index = progress.active_index
        if self._waypoint_recovery_index != index:
            self._waypoint_recovery_index = None
        waypoint = self._path[index]
        offset = waypoint - position[:2]
        distance = float(np.linalg.norm(offset))
        final_homing = (progress.finished and self.goal_stop
                        and distance > settings.GOAL_STOP_RADIUS_M)
        if progress.finished and not final_homing:
            self._waypoint_recovery_index = None
            return state, None
        if final_homing:
            self._waypoint_recovery_index = index
        elif index == 0:
            if (not self._start_acquisition_checked
                    and self._waypoint_recovery_index is None
                    and 1e-9 < distance <= self._waypoint_progress.radius_m):
                self._waypoint_recovery_index = index
            self._start_acquisition_checked = True
        else:
            incoming = waypoint - self._path[index - 1]
            incoming /= np.linalg.norm(incoming)
            passed = float(np.dot(position[:2] - waypoint, incoming)) > 0.
            if self._waypoint_recovery_index is None and passed:
                self._waypoint_recovery_index = index
        if self._waypoint_recovery_index is None:
            return state, None

        target_radius = (settings.GOAL_STOP_RADIUS_M if final_homing
                         else self._waypoint_progress.radius_m)
        state.update(
            waypoint_recovery_active=True,
            waypoint_recovery_target_index=index,
            waypoint_recovery_distance_m=distance,
        )
        if distance <= target_radius:
            state.update(
                avoidance_phase='WAYPOINT_RECOVER',
                waypoint_recovery_scan_m=settings.LIDAR_MAX,
                waypoint_recovery_required_clearance_m=0.,
            )
            return state, (0., 0.)

        target_yaw = float(np.arctan2(offset[1], offset[0]))
        scan_distance = avoidance_authority.forward_distance(
            lidar_bins, current_yaw, target_yaw)
        travel = distance - target_radius
        required = min(
            settings.AVOID_FULL_DISTANCE_M,
            travel + settings.AVOID_PASS_MARGIN_M,
        )
        blocked = scan_distance <= required
        state.update(
            avoidance_phase='WAYPOINT_HOLD' if blocked else 'WAYPOINT_RECOVER',
            waypoint_recovery_blocked=blocked,
            waypoint_recovery_scan_m=scan_distance,
            waypoint_recovery_required_clearance_m=required,
        )
        if blocked:
            return state, (0., 0.)

        direction = offset / distance
        speed = min(
            settings.RECOVERY_FORWARD_CAP_MPS * self.speed_scale,
            max(0., distance - .5 * target_radius),
        )
        along = abs(float(np.dot(direction, frame['d_hat'])))
        lateral = abs(float(np.dot(direction, frame['n_hat'])))
        if along > 1e-12:
            path_cap = min(
                self.forward_cap_mps,
                frame.get('forward_cap_mps', self.forward_cap_mps),
            )
            speed = min(speed, path_cap * self.speed_scale / along)
        if lateral > 1e-12:
            speed = min(speed, self.lateral_cap_mps * self.speed_scale / lateral)
        command = direction * speed
        return state, (float(command[0]), float(command[1]))

    def step(self, *, lidar_bins, pos_ned, vel_ned, quat_ned) -> ControlResult:
        if self._path is None:
            raise ValueError('set_path must be called before a control step')

        lidar_bins = np.asarray(lidar_bins, float)
        pos_ned = np.asarray(pos_ned, float)
        vel_ned = np.asarray(vel_ned, float)
        quat_ned = np.asarray(quat_ned, float)
        for name, values in (
            ('LiDAR', lidar_bins), ('position', pos_ned),
            ('velocity', vel_ned), ('quaternion', quat_ned),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f'{name} must be finite')
        if quat_ned.shape != (4,) or np.linalg.norm(quat_ned) <= 1e-12:
            raise ValueError('quaternion must have four values and nonzero norm')

        position = coordinates.ned_to_enu_vec(pos_ned)
        measured_velocity = coordinates.ned_to_enu_vec(vel_ned)
        orientation = coordinates.ned_to_enu_quat(quat_ned)
        progress = None if self._waypoint_progress is None else self._waypoint_progress.update(position[:2])
        max_seg = (None if progress is None else
                   self._control_geometry.original_waypoint_max_segments[progress.active_index])
        obs, aux = observation.build_observation(
            lidar_bins=lidar_bins, pos_world=position, vel_world=measured_velocity,
            quat_wxyz=orientation, path_pts=self._control_geometry.control_points,
            nmpc_prev=self._previous_reference, projector=self._projector,
            advance=True, split_advance=True, return_aux=True,
            goal_altitude_m=self.flight_altitude_m,
            forward_speed_profile=self._forward_speed_cap if self.corner_mode == 'rounded' else None,
            projector_max_seg=max_seg,
        )
        aux.update(
            active_waypoint_index=None if progress is None else progress.active_index,
            active_waypoint_distance_m=None if progress is None else progress.distance_m,
            waypoints_reached=0 if progress is None else progress.reached,
            all_waypoints_reached=False if progress is None else progress.finished,
        )
        self._previous_reference = (aux['vx_ref'], aux['vy_ref'])
        if not np.isfinite(obs).all():
            raise ValueError('observation must be finite')
        action = np.asarray(self.policy.predict(obs), float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('policy action must contain two finite values')
        current_yaw = quaternion.quat_to_yaw(orientation)
        distance = avoidance_authority.forward_distance(lidar_bins, current_yaw, aux['path_yaw'])
        relevant_distance = distance
        obstacle_relevant = True
        if (self.ordered_waypoints and progress is not None
                and progress.active_index > 0 and not progress.finished):
            relevant_distance = avoidance_authority.waypoint_relevant_distance(
                distance, progress.distance_m, self._waypoint_progress.radius_m,
            )
            obstacle_relevant = relevant_distance == distance
        forward_cap = aux.get('forward_cap_mps', self.forward_cap_mps)
        aux.update(avoidance_distance_m=distance,
                   avoidance_relevant_distance_m=relevant_distance,
                   policy_authority=1.,
                   policy_authority_mode=self.policy_authority_mode,
                   avoidance_progress=0., avoidance_lqr_scale=1., avoidance_holding=False,
                   avoidance_obstacle_along_m=None, avoidance_forward_cap_mps=forward_cap,
                   avoidance_lateral_command_mps=None, avoidance_along_m=None)
        if self.policy_authority_mode == 'obstacle':
            points = self._control_geometry.control_points
            seg = aux['seg']
            along = float(np.linalg.norm(np.diff(points[:seg + 1], axis=0), axis=1).sum()
                          + np.dot(position[:2] - points[seg], aux['d_hat']))
            recovery_distance = None
            recovery_path_blocked = False
            lateral_rate = float(np.dot(measured_velocity[:2], aux['n_hat']))
            if self.ordered_waypoints and aux['lat']:
                recovery_lateral = avoidance_authority.recovery_lateral_command(
                    aux['lat'], lateral_rate, self.lateral_cap_mps,
                )
                remaining = max(
                    0., abs(aux['lat']) - settings.RECOVERY_CLEAR_HYSTERESIS_M,
                )
                recovery_time = remaining / max(abs(recovery_lateral), 1e-6)
                recovery_forward = min(
                    forward_cap, settings.VX_TARGET, settings.RECOVERY_FORWARD_CAP_MPS,
                )
                recovery_distance = avoidance_authority.recovery_corridor_distance(
                    lidar_bins, current_yaw, aux['path_yaw'], aux['lat'],
                    recovery_forward * recovery_time,
                )
                recovery_path_blocked = recovery_distance < settings.LIDAR_MAX
            aux.update(self._avoidance_hold.update(
                distance=relevant_distance,
                all_clear=bool(np.all(np.asarray(lidar_bins) == 1.)),
                along=along, lateral_reference=aux['vy_ref'], lateral_action=float(action[1]),
                forward_cap=forward_cap, lateral_cap=self.lateral_cap_mps,
                recovery=self.ordered_waypoints, cte=aux['lat'],
                lateral_rate=lateral_rate, recovery_distance=recovery_distance,
                recovery_path_blocked=recovery_path_blocked,
                obstacle_relevant=obstacle_relevant,
            ))
        waypoint_recovery, waypoint_command = self._waypoint_recovery(
            position, progress, lidar_bins, current_yaw, aux)
        aux.update(waypoint_recovery)
        target_vx, target_vy = velocity.compute_velocity_command(
            action, aux, speed_scale=self.speed_scale,
            forward_cap_mps=aux['avoidance_forward_cap_mps'],
            lateral_cap_mps=self.lateral_cap_mps,
            policy_authority=aux['policy_authority'],
            lateral_command_mps=aux['avoidance_lateral_command_mps'],
        )
        if (aux.get('avoidance_phase') == 'PASS'
                and aux['avoidance_distance_m'] > settings.AVOID_FULL_DISTANCE_M):
            direction = aux['d_hat']
            normal = aux['n_hat']
            pass_speed = min(
                aux['avoidance_forward_cap_mps'],
                settings.RECOVERY_FORWARD_CAP_MPS,
            ) * self.speed_scale
            along_speed = max(float(np.dot((target_vx, target_vy), direction)), pass_speed)
            lateral_speed = float(np.dot((target_vx, target_vy), normal))
            target_vx, target_vy = tuple(
                along_speed * direction + lateral_speed * normal
            )
        if waypoint_command is not None:
            vx, vy = waypoint_command
            if aux['waypoint_recovery_blocked']:
                self._velocity_filter.reset()
                self._velocity_rate_limiter.reset()
            else:
                self._velocity_filter.seed(vx, vy)
                self._velocity_rate_limiter.seed(vx, vy)
        elif aux.get('avoidance_phase') == 'HOLD':
            vx, vy = 0., 0.
            self._velocity_filter.reset()
            self._velocity_rate_limiter.reset()
        else:
            vx, vy = target_vx, target_vy
            if self.corner_mode == 'rounded':
                vx, vy = self._velocity_rate_limiter(vx, vy, 1.0 / self.control_hz)
            vx, vy = self._velocity_filter(vx, vy, 1.0 / self.control_hz)
            if aux.get('recovery_path_blocked'):
                along_velocity = float(np.dot((vx, vy), aux['d_hat']))
                along_velocity = float(np.clip(
                    along_velocity,
                    0.,
                    aux['avoidance_forward_cap_mps'] * self.speed_scale,
                ))
                vx, vy = tuple(along_velocity * aux['d_hat'])
                self._velocity_rate_limiter.seed(vx, vy)
                self._velocity_filter.seed(vx, vy)
            if (self.ordered_waypoints and self.policy_authority_mode == 'obstacle'
                    and aux['lat']):
                target_lateral = float(np.dot((target_vx, target_vy), aux['n_hat']))
                output_lateral = float(np.dot((vx, vy), aux['n_hat']))
                target_side = target_lateral * aux['lat']
                output_side = output_lateral * aux['lat']
                wrong_recovery_side = target_side <= 0. < output_side
                wrong_avoidance_side = (
                    aux.get('avoidance_phase') == 'AVOID'
                    and target_side > 0. >= output_side
                )
                if wrong_recovery_side or wrong_avoidance_side:
                    vx, vy = target_vx, target_vy
                    self._velocity_rate_limiter.seed(vx, vy)
                    self._velocity_filter.seed(vx, vy)
        goal_distance = float(np.linalg.norm(position[:2] - self._path[-1]))
        goal_cap = None
        progress_finished = progress is None or progress.finished
        if (self.goal_stop and progress_finished
                and aux['seg'] == len(self._control_geometry.control_points) - 2):
            self._goal_stopped |= goal_distance <= settings.GOAL_STOP_RADIUS_M
            goal_cap = (0. if self._goal_stopped else float(np.sqrt(
                2. * settings.GOAL_DECEL_MPS2 * max(0., goal_distance - settings.GOAL_STOP_RADIUS_M))))
            speed = float(np.hypot(vx, vy))
            if speed > goal_cap:
                vx, vy = vx * goal_cap / speed, vy * goal_cap / speed
        aux.update(goal_distance_m=goal_distance, goal_speed_cap_mps=goal_cap,
                   goal_stopped=self._goal_stopped)
        vz = attitude.altitude_velocity(position[2], target=self.flight_altitude_m)
        phase = aux.get('avoidance_phase')
        if phase is None:
            # Unordered legacy avoidance has no phase; progress/hold is its active marker.
            avoidance_active = (aux.get('avoidance_progress', 0.) > 0.
                                or bool(aux.get('avoidance_holding', False)))
        else:
            avoidance_active = phase in ('AVOID', 'PASS', 'RECOVER', 'WAYPOINT_RECOVER')
        selected_yaw_target = attitude.select_yaw_target(
            aux['path_yaw'], (vx, vy), avoidance_active, self.avoidance_yaw_mode,
            min_speed_mps=settings.YAW_COMMAND_MIN_SPEED_MPS,
        )
        yaw_target = (0.0 if self.yaw_mode == settings.YAW_MODE_FIXED
                      else selected_yaw_target)
        aux.update(yaw_target=yaw_target, avoidance_yaw_mode=self.avoidance_yaw_mode)
        yaw_rate = attitude.yaw_rate_command(
            current_yaw, yaw_target, mode=self.yaw_mode,
        )
        if not np.isfinite([vx, vy, vz, yaw_rate]).all():
            raise ValueError('control command must be finite')
        return ControlResult(obs, action, aux, position, (vx, vy, vz), yaw_rate)
