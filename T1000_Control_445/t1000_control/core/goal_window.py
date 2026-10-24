from __future__ import annotations

import numpy as np
from . import path_frame as PF


class LocalGoalWindow:

    def __init__(self, path_pts, path_norm: float | None = None):
        pts = np.asarray(path_pts, float)
        assert pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2, \
            f"path must have shape (K>=2, 2): {pts.shape}"
        self._pts = pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(self._cum[-1])
        self.norm = float(PF.PATH_NORM if path_norm is None else path_norm)
        assert self.norm > 0, "path_norm must be positive"
        self.origin = 0.0

    def point_at(self, arclen: float) -> np.ndarray:
        a = float(np.clip(arclen, 0.0, self.total))
        k = int(np.searchsorted(self._cum, a, side="right") - 1)
        k = min(max(k, 0), len(self._pts) - 2)
        seg_len = max(self._cum[k + 1] - self._cum[k], 1e-9)
        t = (a - self._cum[k]) / seg_len
        return self._pts[k] + t * (self._pts[k + 1] - self._pts[k])

    def update(self, along: float, advance: bool = False) -> dict:
        is_final = (self.total - self.origin) <= self.norm
        if advance and not is_final and (along - self.origin) >= self.norm:
            self.origin = min(self.origin + self.norm, max(self.total - self.norm, 0.0))
            is_final = (self.total - self.origin) <= self.norm
        goal_arc = self.total if is_final else (self.origin + self.norm)
        return {
            "along_local": float(max(along - self.origin, 0.0)),
            "goal_xy": self.point_at(goal_arc),
            "origin": float(self.origin),
            "is_final": bool(is_final),
        }
