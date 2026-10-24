"""
상승 금지 회피 정책 참조 구현 (numpy + onnxruntime 만 필요, torch·isaaclab 불필요).
학습 환경 (drone_wp_rl/task) 의 관측 구성 · 출력 변환 · 속도 조절기를 그대로 옮긴 것. 규격 설명은 agent.md.

    pol = NoClimbPolicy("ca_noclimb_tau03-08.onnx", "cfg.json", path_xy, path_z)      # 전역 경로 (K,2), (K,)
    out = pol.step(pos, vel, R_wb, points=pts)          # 20 Hz. pts: 기체 기준, 월드 축 (ENU) 방향의 LiDAR 점 (N,3) [m]
    out["wp"], out["speed"]                             # WP [x,y,z] (월드 ENU), 수평 속도 명령 [m/s]

좌표: 월드 = ENU (x 동, y 북, z 위), 기체 = FLU (x 앞, y 왼쪽, z 위). R_wb = 기체 -> 월드 회전 행렬 (3,3).
자가 점검:  python ca_policy_ref.py            (cfg.json 의 시험 벡터로 ONNX 출력 확인)
"""
import json
import math
import pathlib

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Path:
    """전역 경로: 수평 꺾은선 xy (K,2) + 웨이포인트별 고도 z (K,). 고도는 호길이에 대해 구간 선형."""

    def __init__(self, xy, z):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.cum = np.r_[0.0, np.cumsum(seg)]
        self.total = float(self.cum[-1])
        self.seg = 0                                   # 지나온 구간 번호 (뒤로 가지 않는다)

    def project(self, p_xy):
        """현재 위치를 경로에 투영: (호길이 s, 부호 있는 횡오차 [+ = 경로 왼쪽], 경로까지 거리).
        구간은 [seg, seg+2] 창에서만 찾는다 (학습 환경과 같음)."""
        S = len(self.xy) - 1
        best = None
        for k in range(3):
            i = min(self.seg + k, S - 1)
            a, b = self.xy[i], self.xy[i + 1]
            ab = b - a
            L2 = max(float(ab @ ab), 1e-9)
            t = min(max(float((p_xy - a) @ ab) / L2, 0.0), 1.0)
            q = a + t * ab
            d = float(np.linalg.norm(p_xy - q))
            if best is None or d < best[0]:            # 동률이면 앞 구간
                best = (d, i, t, ab, q, math.sqrt(L2))
        d, i, t, ab, q, L = best
        self.seg = i
        tan = ab / L
        rel = p_xy - q
        return self.cum[i] + t * L, float(tan[0] * rel[1] - tan[1] * rel[0]), d

    def _seg_at(self, s):
        s = min(max(s, 0.0), self.total)
        i = int(np.searchsorted(self.cum, s, side="right")) - 1
        return s, min(max(i, 0), len(self.xy) - 2)

    def point_at(self, s):
        """호길이 s 의 점과 단위 접선."""
        s, i = self._seg_at(s)
        ab = self.xy[i + 1] - self.xy[i]
        tan = ab / max(float(np.linalg.norm(ab)), 1e-9)
        return self.xy[i] + tan * (s - self.cum[i]), tan

    def z_at(self, s):
        s, i = self._seg_at(s)
        t = min(max((s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9), 0.0), 1.0)
        return float(self.z[i] + t * (self.z[i + 1] - self.z[i]))


def pointcloud_to_image(points, ref_bearing, cfg):
    """LiDAR 점 (N,3; 기체 기준 상대 위치, 월드 축 방향 = 자세 보정 후) -> 거리 이미지 (n_el, n_az) [m].
    칸마다 가장 가까운 점의 거리, 빈 칸과 시야 밖은 d_sensor_max.
      방위 칸 j : 중심 = ref_bearing + j * (360 / n_az)°, 반시계 증가, 폭 ±(180 / n_az)°
      고도 칸 i : [el_min + i*w, el_min + (i+1)*w), w = (el_max - el_min) / n_el. i = 0 이 가장 아래."""
    n_az, n_el, d_max = int(cfg["lidar_n_az"]), int(cfg["lidar_n_el"]), float(cfg["d_sensor_max"])
    img = np.full((n_el, n_az), d_max, dtype=np.float32)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) == 0:
        return img
    rng = np.linalg.norm(p, axis=1)
    az = _wrap(np.arctan2(p[:, 1], p[:, 0]) - ref_bearing)
    ai = np.floor((az + math.pi / n_az) / (2 * math.pi / n_az)).astype(int) % n_az
    el = np.arctan2(p[:, 2], np.linalg.norm(p[:, :2], axis=1))
    w = math.radians(cfg["lidar_el_max_deg"] - cfg["lidar_el_min_deg"]) / n_el
    ei = np.floor((el - math.radians(cfg["lidar_el_min_deg"])) / w).astype(int)
    ok = (ei >= 0) & (ei < n_el) & (rng > 0)
    np.minimum.at(img, (ei[ok], ai[ok]), np.minimum(rng[ok], d_max).astype(np.float32))
    return img


class NoClimbPolicy:
    def __init__(self, onnx_path, cfg_path, path_xy, path_z, policy=None):
        import onnxruntime as ort
        meta = json.load(open(cfg_path))
        name = policy or pathlib.Path(onnx_path).stem
        self.c = dict(meta["cfg"])
        self.c.update(meta["policies"].get(name, {}).get("overrides", {}))
        self.obs_dim = int(meta["obs_dim"])
        self.dt = 1.0 / float(meta["policy_hz"])
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.reset(path_xy, path_z)

    def reset(self, path_xy, path_z):
        """새 전역 경로를 받을 때마다 호출."""
        self.path = Path(path_xy, path_z)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.stall_best = self.path.total              # 지금까지 가장 작았던 남은 거리
        self.stall_t = 0.0
        self._first = True

    def geometry(self, pos):
        """경로 관련 양. step 안에서 한 번 부른다 (지나온 구간 번호를 갱신하므로 한 스텝에 한 번만)."""
        c, pth = self.c, self.path
        pos = np.asarray(pos, dtype=np.float64)
        s, cross, dpath = pth.project(pos[:2])
        carrot, ctan = pth.point_at(s + c["path_lookahead"])
        to_ref = carrot - pos[:2]
        ref_bearing = math.atan2(to_ref[1], to_ref[0])
        return dict(s=s, cross=cross, dpath=dpath, to_ref=to_ref, ref_bearing=ref_bearing,
                    path_turn=_wrap(math.atan2(ctan[1], ctan[0]) - ref_bearing),
                    remaining=max(pth.total - s, 0.0), vdev=float(pos[2] - pth.z_at(s)),
                    z_carrot=pth.z_at(s + c["path_lookahead"]), z_wp=pth.z_at(s + c["wp_distance"]))

    def build_obs(self, g, pos, vel, R_wb, image, density):
        c = self.c
        pos, vel, R = np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64), np.asarray(R_wb, dtype=np.float64)
        v_max = c["v_max"]
        fwd = g["to_ref"] / max(float(np.linalg.norm(g["to_ref"])), 1e-6)
        left = np.array([-fwd[1], fwd[0]])
        vel_ref = np.array([vel[:2] @ fwd, vel[:2] @ left, vel[2]])
        ref3 = np.r_[g["to_ref"], g["z_carrot"] - pos[2]]
        ref3 = ref3 / max(float(np.linalg.norm(ref3)), 1e-6)
        obs = np.concatenate([
            [np.linalg.norm(vel) / v_max],                                   # 0      속력
            R.T @ vel / v_max,                                               # 1:4    기체 프레임 속도
            R.T @ ref3,                                                      # 4:7    경로 전방점 방향 (기체 프레임 단위벡터)
            vel_ref / v_max,                                                 # 7:10   기준 방위 프레임 속도 [전방, 좌측, 수직]
            [math.tanh(g["remaining"] / 10.0)],                              # 10     남은 경로 길이
            [np.clip(g["cross"] / c["max_path_dist"], -1, 1)],               # 11     횡오차 (+ = 왼쪽)
            [np.clip(g["vdev"] / 4.0, -1, 1)],                               # 12     경로 고도 대비 편차 (+ = 위)
            [g["path_turn"] / math.pi],                                      # 13     전방점에서의 경로 방향 - 기준 방위
            [np.clip(pos[2] / c["z_max"], 0, 1)],                            # 14     지면 고도
            np.clip(np.asarray(image, dtype=np.float64) / c["d_sensor_max"], 0, 1).reshape(-1),   # 15:207 거리 이미지 (고도 칸 우선)
            [float(np.clip(density, 0, 1))],                                 # 207    주변 장애물 밀집도
            R.T @ np.array([0.0, 0.0, -1.0]),                                # 208:211 기체 프레임 중력 방향
            self.prev_action,                                                # 211:214 직전 행동 (클립된 값)
        ])
        if c["obs_stall"]:
            tmr = min(max(self.stall_t / max(c["stall_time_s"], 1e-6), 0.0), 1.0)
            need = min(max((g["remaining"] - (self.stall_best - c["stall_min_progress"])) / c["stall_min_progress"], 0.0), 1.0)
            obs = np.r_[obs, tmr, need]                                      # 214:216 정체 타이머, 필요한 진행
        assert obs.shape[0] == self.obs_dim, obs.shape
        return obs.astype(np.float32)

    def step(self, pos, vel, R_wb, image=None, points=None, density=0.0):
        """한 정책 스텝 (20 Hz).
        image : (n_el, n_az) 거리 이미지 [m], 또는 f(ref_bearing) -> 이미지 인 함수. 방위 0 번 칸이 ref_bearing 방향이어야 한다.
        points: image 대신 LiDAR 점 (N,3; 기체 기준, 월드 축 방향). 둘 중 하나는 꼭 준다.
        density: 반경 4 m 안 장애물 수 / 4 를 [0,1] 로 자른 값 (agent.md 의 "밀집도" 참고)."""
        c = self.c
        g = self.geometry(pos)
        if not self._first and c["obs_stall"]:        # 정체 타이머: 남은 거리가 1 m 넘게 줄면 0 으로
            if g["remaining"] < self.stall_best - c["stall_min_progress"]:
                self.stall_best, self.stall_t = g["remaining"], 0.0
            else:
                self.stall_t += self.dt
        self._first = False
        if image is None:
            image = pointcloud_to_image(points, g["ref_bearing"], c)
        elif callable(image):
            image = image(g["ref_bearing"])
        image = np.minimum(np.asarray(image, dtype=np.float32), c["d_sensor_max"])
        obs = self.build_obs(g, pos, vel, R_wb, image, density)
        a = self.sess.run(None, {"obs": obs[None]})[0][0]                    # 이미 [-1,1] 로 클립됨
        self.prev_action = a.astype(np.float32).copy()
        # --- 출력 변환 (task/wp_tracker.py action_to_wp) ---
        azim = g["ref_bearing"] + float(a[0]) * math.radians(c["wp_azim_max_deg"])
        dz = float(a[1]) * (c["wp_alt_up"] if a[1] >= 0 else c["wp_alt_down"])      # 상승 금지: wp_alt_up = 0
        wp = np.array([pos[0] + c["wp_distance"] * math.cos(azim), pos[1] + c["wp_distance"] * math.sin(azim),
                       min(max(g["z_wp"] + dz, c["wp_z_min"]), c["wp_z_max"])])
        speed = min(max(0.5 * (float(a[2]) + 1.0), 0.0), 1.0) * c["v_max"]
        # --- 속도 조절기 (정책 밖, 학습 환경에 있던 규칙): 이미지의 최소 거리가 작으면 속도 상한을 낮춘다 ---
        d_img = float(image.min())
        if c["gov_k"] > 0:
            speed = min(speed, min(max(c["gov_k"] * (d_img - c["gov_d0"]) + c["gov_vmin"], c["gov_vmin"]), c["v_max"]))
        if c["speed_floor"] > 0:
            speed = max(speed, c["speed_floor"])
        return dict(wp=wp, speed=speed, action=a, obs=obs, image=image, d_img_min=d_img, **g)


if __name__ == "__main__":
    import onnxruntime as ort
    here = pathlib.Path(__file__).resolve().parent
    meta = json.load(open(here / "cfg.json"))
    for name, p in meta["policies"].items():
        sess = ort.InferenceSession(str(here / p["file"]), providers=["CPUExecutionProvider"])
        x = np.array(p["test_vectors"]["obs"], dtype=np.float32)
        y = sess.run(None, {"obs": x})[0]
        err = float(np.abs(y - np.array(p["test_vectors"]["action"])).max())
        print(f"{name}: 입력 {x.shape} -> 출력 {y.shape}, 기대값과 최대 차이 {err:.2e} {'OK' if err < 1e-4 else '불일치!'}")
