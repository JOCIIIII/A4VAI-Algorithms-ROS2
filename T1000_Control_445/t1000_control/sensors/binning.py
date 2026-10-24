from __future__ import annotations

import numpy as np
from .. import settings as CFG


H_BINS       = CFG.LIDAR_H_BINS


V_BOUNDARIES = CFG.LIDAR_V_BOUNDARIES


V_LAYERS     = CFG.LIDAR_V_LAYERS


LIDAR_MAX    = CFG.LIDAR_MAX


LIDAR_MIN    = CFG.LIDAR_MIN


TOTAL_BINS   = CFG.LIDAR_TOTAL_BINS


GROUND_Z_CUTOFF = CFG.GROUND_Z_CUTOFF


GROUND_EL_MAX_DEG = CFG.GROUND_EL_MAX_DEG


GROUND_RANGE_TOL = CFG.GROUND_RANGE_TOL


def filter_ground(xyz_body: np.ndarray, alt_agl: float | None = None) -> np.ndarray:
    if xyz_body.shape[0] == 0:
        return xyz_body
    p = np.asarray(xyz_body, np.float32)
    keep = p[:, 2] > GROUND_Z_CUTOFF

    if alt_agl is not None and alt_agl > 0.1:
        d = np.linalg.norm(p, axis=1)
        d = np.maximum(d, 1e-6)
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / d, -1.0, 1.0)))
        looking_down = el < GROUND_EL_MAX_DEG
        sin_el = np.sin(np.radians(np.abs(el)))
        expect = alt_agl / np.maximum(sin_el, 1e-6)
        near_expected = np.abs(d - expect) <= GROUND_RANGE_TOL * expect
        keep &= ~(looking_down & near_expected)

    return p[keep]


def sensor_xyz_to_432bins(
    xyz_sensor: np.ndarray,
    sensor_offset=(0.0, 0.0, 0.0),
    alt_agl: float | None = None,
    filter_ground_points: bool = True,
    yaw: float | None = None,
) -> np.ndarray:
    bins = np.ones(TOTAL_BINS, dtype=np.float32)
    p = np.asarray(xyz_sensor, np.float32)
    if p.shape[0] == 0:
        return bins

    p = p + np.asarray(sensor_offset, np.float32)

    if filter_ground_points:
        p = filter_ground(p, alt_agl)
        if p.shape[0] == 0:
            return bins

    if yaw is not None:
        p = rotate_body_to_world(p, float(yaw))

    return _bin_body_frame(p)


def rotate_body_to_world(p_body: np.ndarray, yaw: float) -> np.ndarray:
    p = np.asarray(p_body, np.float32)
    c, s = np.float32(np.cos(yaw)), np.float32(np.sin(yaw))
    out = np.empty_like(p)
    out[:, 0] = c * p[:, 0] - s * p[:, 1]
    out[:, 1] = s * p[:, 0] + c * p[:, 1]
    out[:, 2] = p[:, 2]
    return out


def _bin_body_frame(xb_yb_zb: np.ndarray) -> np.ndarray:
    bins = np.ones(TOTAL_BINS, dtype=np.float32)
    xb, yb, zb = xb_yb_zb[:, 0], xb_yb_zb[:, 1], xb_yb_zb[:, 2]
    dist = np.sqrt(xb**2 + yb**2 + zb**2)

    m = (dist >= LIDAR_MIN) & (dist <= LIDAR_MAX)
    if not m.any():
        return bins
    xb, yb, zb, dist = xb[m], yb[m], zb[m], dist[m]

    az = np.degrees(np.arctan2(yb, xb)) % 360.0
    el = np.degrees(np.arcsin(np.clip(zb / np.maximum(dist, 1e-6), -1.0, 1.0)))

    h_idx = (az / 360.0 * H_BINS).astype(np.int64) % H_BINS

    v_idx = np.full(len(el), -1, dtype=np.int64)
    for v in range(V_LAYERS):
        in_layer = (el >= V_BOUNDARIES[v]) & (el < V_BOUNDARIES[v + 1]) & (v_idx < 0)
        v_idx[in_layer] = v
    ok = v_idx >= 0
    if not ok.any():
        return bins

    flat = (v_idx * H_BINS + h_idx).clip(0, TOTAL_BINS - 1)
    norm = (dist / LIDAR_MAX).clip(0.0, 1.0)
    np.minimum.at(bins, flat[ok], norm[ok])
    return bins


def xyz_to_432bins(
    xyz: np.ndarray,
    drone_pos_world: np.ndarray,
    drone_yaw: float,
) -> np.ndarray:
    bins = np.ones(TOTAL_BINS, dtype=np.float32)
    if len(xyz) == 0:
        return bins

    rel = xyz - drone_pos_world
    cy, sy = np.cos(-drone_yaw), np.sin(-drone_yaw)
    xb =  cy * rel[:, 0] + sy * rel[:, 1]
    yb = -sy * rel[:, 0] + cy * rel[:, 1]
    zb = rel[:, 2]
    dist = np.sqrt(xb**2 + yb**2 + zb**2)

    mask = (dist >= LIDAR_MIN) & (dist <= LIDAR_MAX)
    xb, yb, zb, dist = xb[mask], yb[mask], zb[mask], dist[mask]
    if len(dist) == 0:
        return bins

    az = np.degrees(np.arctan2(-yb, xb)) % 360.0
    el = np.degrees(np.arcsin(
        np.clip(zb / np.maximum(dist, 1e-6), -1.0, 1.0)
    ))

    h_idx = (az / 360.0 * H_BINS).astype(int) % H_BINS
    v_idx = np.searchsorted(V_BOUNDARIES[1:-1], el)
    flat  = (v_idx * H_BINS + h_idx).clip(0, TOTAL_BINS - 1)
    norm  = (dist / LIDAR_MAX).clip(0.0, 1.0)

    for i in range(len(flat)):
        bins[flat[i]] = min(bins[flat[i]], norm[i])
    return bins
