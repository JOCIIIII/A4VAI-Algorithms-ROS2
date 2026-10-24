from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping, Sequence
from numbers import Real
import math


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


class GoalDwellGate:
    """Accumulate only consecutive healthy, stopped, ordered-goal samples."""

    def __init__(self, radius_m=.5, max_speed_mps=.2, dwell_s=2.):
        self.radius_m = float(radius_m)
        self.max_speed_mps = float(max_speed_mps)
        self.required_dwell_s = float(dwell_s)
        self.reset()

    def reset(self):
        self.started_at = None
        self.previous_at = None
        self.dwell_s = 0.0

    def update(self, now, aux, command_xy, *, healthy):
        if not isinstance(aux, Mapping):
            self.reset()
            return False
        distance = aux.get('goal_distance_m')
        eligible = (healthy is True and aux.get('all_waypoints_reached') is True
                    and isinstance(aux.get('avoidance_phase'), str)
                    and aux.get('avoidance_phase') in ('CLEAR', 'AVOID', 'RECOVER')
                    and _finite_number(distance) and 0. <= distance <= self.radius_m
                    and _finite_number(now)
                    and isinstance(command_xy, Sequence)
                    and not isinstance(command_xy, (str, bytes, bytearray))
                    and len(command_xy) >= 2
                    and all(_finite_number(v) for v in command_xy[:2])
                    and math.hypot(*command_xy[:2]) <= self.max_speed_mps)
        if not eligible:
            self.reset()
            return False
        if self.previous_at is None or now < self.previous_at:
            self.started_at = now
        self.previous_at = now
        self.dwell_s = now - self.started_at
        return self.dwell_s >= self.required_dwell_s


class LifecycleAction(str, Enum):

    REQUEST_OFFBOARD = "request-offboard"
    ARM = "arm"
    LAND = "land"
    FINISH = "finish"


@dataclass(frozen=True)
class LifecycleInputs:

    sim_mode: bool
    auto_start: bool
    allow_no_rc: bool
    sitl_confirmed: bool
    dry_run: bool
    elapsed_s: float
    duration_s: float
    offboard: bool
    armed: bool
    finished: bool
    until_goal: bool = False
    land_on_goal: bool = False
    ordered_waypoints: bool = False
    goal_stop: bool = False
    goal_ready: bool = False


def lifecycle_actions(inputs: LifecycleInputs) -> tuple[LifecycleAction, ...]:

    if not (
        inputs.sim_mode
        and inputs.auto_start
        and inputs.allow_no_rc
        and inputs.sitl_confirmed
        and not inputs.dry_run
    ):
        return ()
    if inputs.finished:
        return ()
    if (inputs.until_goal or inputs.land_on_goal) and not (
            inputs.land_on_goal and inputs.ordered_waypoints and inputs.goal_stop):
        return ()

    if ((inputs.land_on_goal and inputs.goal_ready)
            or (not inputs.until_goal and inputs.elapsed_s >= inputs.duration_s)):
        if inputs.armed:
            return (LifecycleAction.LAND, LifecycleAction.FINISH)
        return (LifecycleAction.FINISH,)

    actions = []
    if inputs.elapsed_s >= 1.5 and not inputs.offboard:
        actions.append(LifecycleAction.REQUEST_OFFBOARD)
    if inputs.elapsed_s >= 2.0 and not inputs.armed:
        actions.append(LifecycleAction.ARM)
    return tuple(actions)


def timeout_actions(inputs: LifecycleInputs) -> tuple[LifecycleAction, ...]:

    return tuple(
        action for action in lifecycle_actions(inputs)
        if action in (LifecycleAction.LAND, LifecycleAction.FINISH)
    )
