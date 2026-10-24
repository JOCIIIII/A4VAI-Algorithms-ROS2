import numpy as np

from .. import settings


def forward_distance(lidar_bins, current_yaw, path_yaw):
    lidar = np.asarray(lidar_bins, float)
    if (lidar.size != settings.LIDAR_TOTAL_BINS or not np.isfinite(lidar).all()
            or np.any(lidar < 0.) or np.any(lidar > 1.)):
        raise ValueError('LiDAR must contain 432 finite normalized bins in [0, 1]')
    count = settings.LIDAR_H_BINS
    shift = int(np.rint((current_yaw - path_yaw) * count / (2. * np.pi)))
    aligned = np.roll(lidar.reshape(settings.LIDAR_V_LAYERS, count), shift, axis=1)
    angles = (np.arange(count) * 360. / count + 180.) % 360. - 180.
    sector = np.abs(angles) <= settings.AVOID_FORWARD_HALF_ANGLE_DEG
    return float(aligned[:, sector].min() * settings.LIDAR_MAX)


def recovery_lateral_command(cte, lateral_rate, lateral_cap):
    cap = min(lateral_cap, settings.RECOVERY_LATERAL_CAP_MPS)
    return float(np.clip(
        -(settings.NMPC_K_Y * cte + settings.NMPC_K_VY * lateral_rate),
        -cap,
        cap,
    ))


def recovery_corridor_distance(lidar_bins, current_yaw, path_yaw, cte, forward_travel):
    lidar = np.asarray(lidar_bins, float)
    if (lidar.size != settings.LIDAR_TOTAL_BINS or not np.isfinite(lidar).all()
            or np.any(lidar < 0.) or np.any(lidar > 1.)):
        raise ValueError('LiDAR must contain 432 finite normalized bins in [0, 1]')
    count = settings.LIDAR_H_BINS
    shift = int(np.rint((current_yaw - path_yaw) * count / (2. * np.pi)))
    ranges = np.roll(
        lidar.reshape(settings.LIDAR_V_LAYERS, count), shift, axis=1,
    ) * settings.LIDAR_MAX
    angles = np.arange(count) * 2. * np.pi / count
    along = ranges * np.cos(angles)
    lateral = ranges * np.sin(angles)
    width = settings.RECOVERY_CORRIDOR_HALF_WIDTH_M
    margin = settings.RECOVERY_CORRIDOR_MARGIN_M
    recovery_progress = -np.copysign(1., cte) * lateral
    remaining = max(0., abs(cte) - settings.RECOVERY_CLEAR_HYSTERESIS_M)
    inside = (
        (ranges < settings.LIDAR_MAX)
        & (along >= -width)
        & (along <= forward_travel + width)
        & (recovery_progress >= -width)
        & (recovery_progress <= remaining + margin)
    )
    return float(ranges[inside].min()) if np.any(inside) else settings.LIDAR_MAX


def waypoint_relevant_distance(distance, waypoint_distance, acceptance_radius):
    travel = max(0., waypoint_distance - acceptance_radius)
    if distance > travel + settings.AVOID_PASS_MARGIN_M:
        return settings.LIDAR_MAX
    return distance


def residual_authority(distance_m):
    return settings.AVOID_MAX_AUTHORITY * avoidance_progress(distance_m)


def avoidance_progress(distance_m):
    fraction = np.clip(
        (settings.AVOID_ACTIVATE_DISTANCE_M - distance_m)
        / (settings.AVOID_ACTIVATE_DISTANCE_M - settings.AVOID_FULL_DISTANCE_M), 0., 1.,
    )
    return float(fraction * fraction * (3. - 2. * fraction))


class AvoidanceHold:
    def __init__(self):
        self.obstacle_along = None
        self._outward_command = None
        self._progress = 0.
        self._recovering = False
        self._close_reacquired = False

    def _clear(self):
        self.obstacle_along = None
        self._outward_command = None
        self._progress = 0.
        self._recovering = False
        self._close_reacquired = False

    def _legacy_update(self, *, distance, along, lateral_reference,
                       lateral_action, forward_cap, lateral_cap):
        """Keep the unordered avoidance behavior unchanged."""
        progress = avoidance_progress(distance)
        holding = False
        lateral = None
        if (self.obstacle_along is not None
                and along > self.obstacle_along + settings.AVOID_PASS_MARGIN_M):
            self._clear()
        if progress > 0.:
            self.obstacle_along = along + distance
            lateral = float(np.clip(
                (1. - progress) * lateral_reference
                + settings.AVOID_MAX_AUTHORITY * progress * lateral_action
                * settings.MODEL_LATERAL_ACTION_SCALE_MPS, -lateral_cap, lateral_cap))
            if (self._outward_command is not None
                    and (lateral_action * self._outward_command <= 0.
                         or lateral * self._outward_command <= 0.)):
                magnitude = max(abs(self._outward_command),
                                settings.AVOID_MAX_AUTHORITY * progress * abs(lateral_action)
                                * settings.MODEL_LATERAL_ACTION_SCALE_MPS)
                self._outward_command = float(np.copysign(min(magnitude, lateral_cap),
                                                         self._outward_command))
                self._progress = max(self._progress, progress)
                holding = True
            elif lateral * lateral_action > 0.:
                self._outward_command = lateral
                self._progress = progress
        elif self._outward_command is not None:
            holding = True
        if holding:
            progress = self._progress
            lateral = self._outward_command
        normal_cap = min(forward_cap, settings.VX_TARGET)
        cap = (forward_cap if progress == 0. else
               normal_cap + progress * (min(normal_cap, settings.AVOID_FORWARD_CAP_MPS) - normal_cap))
        return dict(policy_authority=settings.AVOID_MAX_AUTHORITY * progress,
                    avoidance_progress=progress, avoidance_lqr_scale=1. - progress,
                    avoidance_holding=holding, avoidance_obstacle_along_m=self.obstacle_along,
                    avoidance_forward_cap_mps=cap, avoidance_lateral_command_mps=lateral,
                    avoidance_along_m=along)

    def update(self, *, distance, all_clear, along, lateral_reference,
               lateral_action, forward_cap, lateral_cap, recovery=False,
               cte=0., lateral_rate=0., recovery_distance=None,
               recovery_path_blocked=False, obstacle_relevant=True):
        if not recovery:
            return self._legacy_update(
                distance=distance, along=along, lateral_reference=lateral_reference,
                lateral_action=lateral_action, forward_cap=forward_cap, lateral_cap=lateral_cap,
            )

        abs_cte = abs(cte)
        progress = avoidance_progress(distance)
        if obstacle_relevant and progress > 0. and self.obstacle_along is None:
            self.obstacle_along = along + distance
        active = self.obstacle_along is not None
        spatially_passed = (active
                            and along > self.obstacle_along + settings.AVOID_PASS_MARGIN_M)
        if spatially_passed and distance <= settings.AVOID_FULL_DISTANCE_M:
            outward_command = self._outward_command
            held_progress = self._progress
            if (outward_command is not None
                    and abs_cte >= settings.RECOVERY_START_CTE_M):
                outward_command = float(np.copysign(abs(outward_command), cte))
            self._clear()
            self.obstacle_along = along + distance
            self._outward_command = outward_command
            self._progress = held_progress
            active = True
            spatially_passed = False
            self._close_reacquired = True
        elif self._close_reacquired and distance > settings.AVOID_FULL_DISTANCE_M:
            self._close_reacquired = False
        release_ready = (distance <= settings.AVOID_FULL_DISTANCE_M or distance >= (
            settings.AVOID_ACTIVATE_DISTANCE_M + settings.RECOVERY_CLEAR_HYSTERESIS_M))

        recovering = abs_cte >= settings.RECOVERY_START_CTE_M
        if active:
            recovering = spatially_passed and (
                abs_cte > settings.RECOVERY_CLEAR_HYSTERESIS_M or not release_ready)
        elif not recovering and self._recovering:
            recovering = abs_cte > settings.RECOVERY_CLEAR_HYSTERESIS_M
        self._recovering = recovering
        passing = (active and not spatially_passed
                   and abs_cte >= settings.RECOVERY_START_CTE_M
                   and (not self._close_reacquired
                        or abs_cte >= settings.RECOVERY_FULL_CTE_M))
        if spatially_passed and not recovering:
            self._clear()
            if progress > 0.:
                self.obstacle_along = along + distance
            active = self.obstacle_along is not None

        lateral = None
        holding = False
        if progress > 0.:
            lateral = float(np.clip(
                (1. - progress) * lateral_reference
                + settings.AVOID_MAX_AUTHORITY * progress * lateral_action
                * settings.MODEL_LATERAL_ACTION_SCALE_MPS, -lateral_cap, lateral_cap))
            if (self._close_reacquired
                    and abs_cte >= settings.RECOVERY_START_CTE_M):
                magnitude = max(
                    abs(lateral),
                    0. if self._outward_command is None else abs(self._outward_command),
                )
                lateral = float(np.copysign(min(magnitude, lateral_cap), cte))
                self._outward_command = lateral
                self._progress = max(self._progress, progress)
                holding = True
            elif (active and self._outward_command is not None
                    and lateral * self._outward_command <= 0.):
                magnitude = max(abs(lateral), abs(self._outward_command))
                lateral = float(np.copysign(min(magnitude, lateral_cap),
                                            self._outward_command))
                self._outward_command = lateral
                progress = max(progress, self._progress)
                self._progress = progress
                holding = True
            elif lateral * lateral_action > 0.:
                self._outward_command = lateral
                self._progress = progress
        elif self._outward_command is not None and not recovering:
            if active and not spatially_passed:
                holding = True
                lateral = self._outward_command
                progress = self._progress

        normal_cap = min(forward_cap, settings.VX_TARGET)
        cap = (forward_cap if progress == 0. else
               normal_cap + progress * (min(normal_cap, settings.AVOID_FORWARD_CAP_MPS) - normal_cap))
        weight = 0.
        pathward_blocked = False
        forward_blocked = distance <= settings.AVOID_FULL_DISTANCE_M
        phase = 'AVOID' if self.obstacle_along is not None else 'CLEAR'
        if passing:
            lateral = 0.
            holding = True
            cap = min(cap, settings.AVOID_FORWARD_CAP_MPS)
            phase = 'PASS'
        elif recovering:
            self._outward_command = None
            self._progress = 0.
            weight = float(np.clip(
                (abs_cte - settings.RECOVERY_START_CTE_M)
                / (settings.RECOVERY_FULL_CTE_M - settings.RECOVERY_START_CTE_M), 0., 1.))
            lateral = recovery_lateral_command(cte, lateral_rate, lateral_cap)
            cap = min(cap, settings.RECOVERY_FORWARD_CAP_MPS)
            phase = 'RECOVER'
            pathward_blocked = recovery_path_blocked
            if pathward_blocked:
                lateral = 0.
                if forward_blocked:
                    phase = 'HOLD'
                    holding = True
                    cap = 0.
        authority = settings.AVOID_MAX_AUTHORITY * progress * (1. - weight)
        return dict(policy_authority=authority, avoidance_progress=progress,
                    avoidance_lqr_scale=1. - progress, avoidance_holding=holding,
                    avoidance_obstacle_along_m=self.obstacle_along,
                    avoidance_forward_cap_mps=cap, avoidance_lateral_command_mps=lateral,
                    avoidance_along_m=along, avoidance_phase=phase,
                    path_reference_weight=weight,
                    recovery_forward_cap_mps=settings.RECOVERY_FORWARD_CAP_MPS,
                    recovery_distance_m=recovery_distance,
                    recovery_path_blocked=pathward_blocked,
                    recovery_forward_blocked=forward_blocked,
                    recovery_lateral_command_mps=lateral)
