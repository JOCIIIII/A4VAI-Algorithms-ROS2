from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .. import settings


@dataclass(frozen=True)
class WaypointProgressState:
    active_index: int
    distance_m: float
    reached: int
    finished: bool


class WaypointProgress:
    """Accept waypoints in order after a continuous in-radius dwell."""

    def __init__(self, points, *, acceptance_radius_m=None, hold_s=None, clock=None):
        self._points = np.array(points, dtype=float, copy=True)
        if (self._points.ndim != 2 or self._points.shape[1] != 2
                or len(self._points) == 0 or not np.isfinite(self._points).all()):
            raise ValueError('waypoints must be finite with shape (K>=1, 2)')
        self.radius_m = (settings.WAYPOINT_ACCEPTANCE_RADIUS_M
                         if acceptance_radius_m is None else acceptance_radius_m)
        self.hold_s = settings.WAYPOINT_HOLD_S if hold_s is None else hold_s
        for name, value in (('acceptance_radius_m', self.radius_m), ('hold_s', self.hold_s)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be a finite positive scalar')
        self._clock = time.monotonic if clock is None else clock
        self._reached = 0
        self._entered_at = None

    def update(self, position_xy) -> WaypointProgressState:
        position = np.asarray(position_xy, float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError('position must contain two finite values')
        active = min(self._reached, len(self._points) - 1)
        distance = float(np.linalg.norm(position - self._points[active]))
        if self._reached < len(self._points):
            if distance <= self.radius_m:
                now = self._clock()
                if self._entered_at is None:
                    self._entered_at = now
                if now - self._entered_at >= self.hold_s:
                    self._reached += 1
                    self._entered_at = None
                    active = min(self._reached, len(self._points) - 1)
                    distance = float(np.linalg.norm(position - self._points[active]))
            else:
                self._entered_at = None
        return WaypointProgressState(active, distance, self._reached,
                                     self._reached == len(self._points))
