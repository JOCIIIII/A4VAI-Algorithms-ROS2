from __future__ import annotations

import math
import numpy as np
from .. import settings as CFG
from . import path_reference as NR
from . import path_frame as PF
from .quaternion import quat_to_yaw, quat_rotate_inverse, _wrap_to_pi


MODEL_FORWARD_ACTION_SCALE_MPS = CFG.MODEL_FORWARD_ACTION_SCALE_MPS
VX_MAX = MODEL_FORWARD_ACTION_SCALE_MPS


def _descale_velocity(v):
    if not getattr(CFG, "SPEED_SCALE_COMPENSATE", False):
        return v
    s = float(CFG.SPEED_SCALE)
    if s <= 0.0:
        raise ValueError(f"SPEED_SCALE={s} must be positive")
    return v / s


MODEL_LATERAL_ACTION_SCALE_MPS = CFG.MODEL_LATERAL_ACTION_SCALE_MPS
VY_MAX = MODEL_LATERAL_ACTION_SCALE_MPS


VZ_MAX = CFG.VZ_MAX


CTE_NORM = CFG.CTE_NORM


GOAL_ALT = CFG.GOAL_ALT


TOTAL_BINS = CFG.LIDAR_TOTAL_BINS


OBS_DIM = CFG.OBS_DIM


def _lateral_state(pos_loc, vel_w, pts, frame):
    d_hat = frame["d_hat"]
    n_hat = np.array([-d_hat[1], d_hat[0]])
    lat = float(pos_loc[:2] @ n_hat - pts[frame["seg"]] @ n_hat)
    lat_rate = float(vel_w[:2] @ n_hat)
    lat_rate = _descale_velocity(lat_rate)
    return d_hat, n_hat, lat, lat_rate


def build_observation(
    lidar_bins,
    pos_world,
    vel_world,
    quat_wxyz,
    path_pts=None,
    env_origin=np.zeros(3),
    nmpc_prev=None,
    return_aux=False,
    projector=None,
    advance=False,
    split_advance=True,
    local_goal=None,
    *,
    goal_altitude_m=GOAL_ALT,
    forward_speed_profile=None,
    projector_max_seg=None,
):
    pos_loc = np.asarray(pos_world, float) - np.asarray(env_origin, float)
    vel_w = np.asarray(vel_world, float)
    pts = straight_path() if path_pts is None else np.asarray(path_pts, float)

    _split = bool(split_advance) and projector is not None
    projector_options = {} if projector_max_seg is None else {'max_seg': projector_max_seg}
    if projector is None:
        pf = PF.project_polyline(pos_loc[:2], pts)
    else:
        pf = projector(pos_loc[:2], pts,
                       advance=(advance and not _split), **projector_options)
    d_hat, n_hat, lat, lat_rate = _lateral_state(pos_loc, vel_w, pts, pf)

    lidar = np.asarray(lidar_bins, float).ravel()
    assert lidar.size == TOTAL_BINS, f"lidar {lidar.size} != {TOTAL_BINS}"

    _lg = None if local_goal is None else local_goal.update(pf["along"], advance=advance)
    lat_along = pf["along"] if _lg is None else _lg["along_local"]
    pos_norm = np.clip(
        np.array([lat_along, lat, pos_loc[2]]) / PF.PATH_NORM, -1.0, 1.0)

    vb = quat_rotate_inverse(quat_wxyz, vel_w)
    vb = _descale_velocity(vb)
    vel_norm = np.array([
        np.clip(vb[0] / MODEL_FORWARD_ACTION_SCALE_MPS, -1.0, 1.0),
        np.clip(vb[1] / MODEL_LATERAL_ACTION_SCALE_MPS, -1.0, 1.0),
        np.clip(vb[2] / VZ_MAX, -1.0, 1.0),
    ])

    yaw = _wrap_to_pi(quat_to_yaw(quat_wxyz))

    _gxy = pts[-1] if _lg is None else _lg["goal_xy"]
    goal_loc = np.array([_gxy[0], _gxy[1], goal_altitude_m])
    diff_w = goal_loc - pos_loc
    goal_dist_norm = float(np.clip(np.linalg.norm(diff_w) / PF.PATH_NORM, 0.0, 1.0))
    diff_b = quat_rotate_inverse(quat_wxyz, diff_w)
    goal_ang = math.atan2(diff_b[1], diff_b[0])

    cte_norm = float(np.clip(pf["e"] / CTE_NORM, -1.0, 1.0))
    progress = float(np.clip(lat_along / PF.PATH_NORM, 0.0, 1.0))
    seg_yaw = math.atan2(d_hat[1], d_hat[0])
    heading_norm = float(np.clip(_wrap_to_pi(seg_yaw - yaw) / math.pi, -1.0, 1.0))

    if _split:
        pf_cmd = projector(pos_loc[:2], pts, advance=advance, **projector_options)
        d_hat, n_hat, lat, lat_rate = _lateral_state(pos_loc, vel_w, pts, pf_cmd)
    else:
        pf_cmd = pf
    cap = None if forward_speed_profile is None else forward_speed_profile(pf_cmd)
    vx_ref, vy_ref = NR.solve(lat, lat_rate, lidar, vx_target=cap)
    _obs_ref = (vx_ref, vy_ref) if nmpc_prev is None else nmpc_prev
    nmpc_obs = NR.to_obs(*_obs_ref)

    obs = np.concatenate([
        lidar,
        pos_norm,
        vel_norm,
        [goal_dist_norm],
        [goal_ang],
        [cte_norm],
        [progress],
        [heading_norm],
        list(nmpc_obs),
    ]).astype(np.float32)
    assert obs.shape == (OBS_DIM,), f"obs shape mismatch: {obs.shape}"
    if not return_aux:
        return obs
    return obs, {
        "d_hat": d_hat, "n_hat": n_hat,
        "vx_ref": vx_ref, "vy_ref": vy_ref,
        "lat": lat, "lat_rate": lat_rate,
        "seg": pf_cmd["seg"], "remain": pf_cmd["remain"],
        "path_yaw": math.atan2(d_hat[1], d_hat[0]),
        "obs_seg": pf["seg"], "obs_path_yaw": seg_yaw,
        **({"forward_cap_mps": cap} if cap is not None else {}),
    }


def straight_path(goal_x: float = 90.0) -> np.ndarray:
    return np.array([[0.0, 0.0], [goal_x, 0.0]], dtype=float)
