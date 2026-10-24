from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import settings as CFG
PATH_NORM = CFG.PATH_NORM


def segment_frame(pos_xy, a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    L = max(float(np.linalg.norm(d)), 1e-6)
    d_hat = d / L
    sp = np.asarray(pos_xy, float) - np.asarray(a, float)
    s = float(sp @ d_hat)
    e = float(sp[0] * d_hat[1] - sp[1] * d_hat[0])
    return s, e, d_hat


def project_polyline(pos_xy, pts):
    pts = np.asarray(pts, float)
    assert pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2, \
        f"path must have shape (K>=2, 2): {pts.shape}"
    pos = np.asarray(pos_xy, float)

    a, b = pts[:-1], pts[1:]
    d = b - a
    L = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    d_hat = d / L[:, None]
    sp = pos[None, :] - a
    s_all = (sp * d_hat).sum(1)
    e_all = sp[:, 0] * d_hat[:, 1] - sp[:, 1] * d_hat[:, 0]
    s_cl = np.clip(s_all, 0.0, L)
    foot = a + d_hat * s_cl[:, None]
    seg = int(np.argmin(np.linalg.norm(pos[None, :] - foot, axis=1)))

    cum = np.concatenate([[0.0], np.cumsum(L)])
    along = cum[seg] + s_cl[seg]
    return {
        "s": float(s_cl[seg]),
        "e": float(e_all[seg]),
        "d_hat": d_hat[seg],
        "seg": seg,
        "along": float(along),
        "remain": float(max(cum[-1] - along, 0.0)),
        "total": float(cum[-1]),
        "next_wp": pts[min(seg + 1, len(pts) - 1)],
    }


def perp(d_hat):
    return np.array([d_hat[1], -d_hat[0]], float)


def cross_track_velocity(vel_xy, d_hat):
    return float(np.asarray(vel_xy, float)[:2] @ perp(d_hat))


class MonotonicProjector:

    def __init__(self):
        self._seg = 0

    def reset(self, seg: int = 0):
        self._seg = int(seg)

    @property
    def seg(self) -> int:
        return self._seg

    def __call__(self, pos_xy, pts, advance: bool = False, max_seg=None) -> dict:
        pts = np.asarray(pts, float)
        assert pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2, \
            f"path must have shape (K>=2, 2): {pts.shape}"
        pos = np.asarray(pos_xy, float)
        K = pts.shape[0]
        if max_seg is not None and (
                isinstance(max_seg, (bool, np.bool_))
                or not isinstance(max_seg, (int, np.integer))
                or not 0 <= max_seg <= K - 2):
            raise ValueError('max_seg must be an existing nonnegative segment index')
        last_seg = K - 2 if max_seg is None else int(max_seg)

        seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        if self._seg > last_seg:
            self._seg = last_seg

        if advance:
            for _ in range(K - 1):
                s_cur, _, _ = segment_frame(pos, pts[self._seg], pts[self._seg + 1])
                if s_cur > seg_len[self._seg] and self._seg < last_seg:
                    self._seg += 1
                else:
                    break

        seg = self._seg
        s_cur, e_cur, d_hat = segment_frame(pos, pts[seg], pts[seg + 1])
        total = float(cum[-1])
        along = float(cum[seg] + s_cur)
        return {
            "s": float(s_cur),
            "e": float(e_cur),
            "d_hat": d_hat,
            "seg": seg,
            "along": along,
            "total": total,
            "remain": max(total - along, 0.0),
            "next_wp": pts[min(seg + 1, K - 1)],
        }


HOLD_S, SLOW_S, STOP_S = CFG.PATH_HOLD_S, CFG.PATH_SLOW_S, CFG.PATH_SLOW_S


def reference_health(age_s: float) -> tuple[str, float]:
    if age_s < HOLD_S:
        return "ok", 1.0
    if age_s < SLOW_S:
        return "stale", 0.5
    return "lost", 0.0


def path_from_poses(xy_list):
    pts = np.asarray(xy_list, float).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError(f"path has {pts.shape[0]} points; need at least 2")
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
            keep.append(i)
    if len(keep) < 2:
        raise ValueError("all path points are the same")
    return pts[keep]


@dataclass(frozen=True)
class RoundedPath:

    original_points: np.ndarray
    control_points: np.ndarray
    corner_mask: np.ndarray
    segment_speed_caps: np.ndarray
    original_waypoint_max_segments: np.ndarray | None = None


def build_control_path(points, *, mode, max_deviation_m, sample_spacing_m,
                       corner_speed_mps, cruise_speed_mps) -> RoundedPath:
    original = path_from_poses(points)
    if not np.isfinite(np.asarray(points, float)).all():
        raise ValueError("path points must be finite")
    if mode not in ("sharp", "rounded"):
        raise ValueError("mode must be sharp or rounded")
    for name, value in (("max_deviation_m", max_deviation_m),
                        ("sample_spacing_m", sample_spacing_m),
                        ("corner_speed_mps", corner_speed_mps),
                        ("cruise_speed_mps", cruise_speed_mps)):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if corner_speed_mps > cruise_speed_mps:
        raise ValueError("corner_speed_mps must be at most cruise_speed_mps")

    controls = [original[0]]
    mask = []
    waypoint_gates = [0]
    for i in range(1, len(original) - 1):
        waypoint = original[i]
        incoming = waypoint - original[i - 1]
        outgoing = original[i + 1] - waypoint
        left, right = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        u, v = incoming / left, outgoing / right
        alignment = float(u @ v)
        if mode == "sharp" or abs(alignment) >= 1. - 1e-6:
            controls.append(waypoint)
            mask.append(False)
            waypoint_gates.append(len(controls) - 2)
            continue
        half_angle_sine = np.sqrt((1. - alignment) / 2.)
        trim = min(max_deviation_m / half_angle_sine, left / 4., right / 4.)
        if trim < max(sample_spacing_m / 2., 1e-5):
            controls.append(waypoint)
            mask.append(False)
            waypoint_gates.append(len(controls) - 2)
            continue
        entry, exit_point = waypoint - trim * u, waypoint + trim * v
        count = max(2, 2 * int(np.ceil(trim / sample_spacing_m)))
        t = np.linspace(0., 1., count + 1)[:, None]
        curve = (1. - t)**2 * entry + 2. * (1. - t) * t * waypoint + t**2 * exit_point
        if np.any(np.linalg.norm(np.diff(curve, axis=0), axis=1) <= 1e-6):
            controls.append(waypoint)
            mask.append(False)
            waypoint_gates.append(len(controls) - 2)
            continue
        # The incoming half ends at the curve's closest approach to the vertex.
        # Keep the outgoing half locked until the original waypoint is accepted.
        waypoint_gates.append(len(controls) + count // 2 - 1)
        controls.extend(curve)
        mask.extend([False] + [True] * count)
    controls.append(original[-1])
    mask.append(False)
    waypoint_gates.append(len(controls) - 2)
    control_points = np.array(controls, dtype=float, copy=True)
    corner_mask = np.array(mask, dtype=bool)
    caps = np.where(corner_mask, corner_speed_mps, cruise_speed_mps)
    gates = np.array(waypoint_gates, dtype=int)
    for array in (original, control_points, corner_mask, caps, gates):
        array.setflags(write=False)
    return RoundedPath(original, control_points, corner_mask, caps, gates)
