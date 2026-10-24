"""비행 기록에서 급제동 지표를 뽑는다 (시뮬레이션 판정표 analysis/brake2.py 와 같은 정의). numpy 만 쓴다.
    python brake_metrics.py <기록.csv> [기록2.csv ...]
  CSV: 첫 줄에 열 이름. 꼭 있어야 하는 열 = t [s], vx, vy [m/s] (수평 속도, 월드 좌표).
       있으면 쓰는 열 = cmd_speed [m/s] (CA 가 낸 속도 명령; ca_policy_ref.step() 의 speed), x, y [m] (거리 계산용. 없으면 속력을 적분).
  기록 주기는 아무래도 좋다 (20 Hz 로 다시 뽑는다). 이륙·착륙 구간은 빼고 넣는 것이 좋다.
  기록이 끊긴 구간 (0.5 s 넘게 빈 곳) 이 있으면 그 사이를 직선으로 잇게 되므로, 끊긴 곳에서 파일을 나눠 넣는다.
  지표:
    급감속 = 수평 속력이 0.3 s 동안 줄어든 평균 감속이 2 / 3 / 4 m/s² 를 넘는 순간 (1 s 안의 것은 하나로). cmd_speed 가 있으면 명령이 실제 속력보다
             0.3 m/s 넘게 낮을 때만 센다 (제동). 없으면 선회로 속력이 준 것도 함께 센다.
    깊은 제동 = 1 s 안에 1.5 m/s 넘게 줄고, 2 s 안에 2.5 m/s 넘게 줄어든 것.
    명령 낙폭 = cmd_speed 가 한 스텝 (0.05 s) 에 0.5 m/s 넘게 떨어진 횟수. 묶음 x8_30m_soft 에서는 '실제 속력 + 0.3' 까지 내려오는 것만 나와야 한다.
  시뮬레이션 (검증용 시드, 장애물 적은 장면) 의 값: 진로 보호막 묶음 3 m/s² 초과 0.89 회 / 100 m, 깊은 제동 0.44 → 완만한 제동 묶음 0.05, 0.02."""
import sys
import numpy as np
DT = 0.05


def load(path):
    with open(path) as f:
        cols = [c.strip() for c in f.readline().lstrip("#").strip().split(",")]
    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    d = {c: a[:, i] for i, c in enumerate(cols)}
    t = np.arange(d["t"][0], d["t"][-1], DT)
    return {k: np.interp(t, d["t"], v) for k, v in d.items() if k != "t"}, len(t)


def metrics(path):
    d, n = load(path)
    sp = np.hypot(d["vx"], d["vy"]); cs = d.get("cmd_speed")
    dist = float(np.hypot(np.diff(d["x"]), np.diff(d["y"])).sum()) if "x" in d and "y" in d else float(sp.sum() * DT)
    dec = np.zeros(n); dec[:n - 6] = (sp[:n - 6] - sp[6:]) / 0.3
    brk = np.ones(n, bool)
    if cs is not None:
        brk[:n - 6] = np.array([cs[k:k + 7].min() < sp[k] - 0.3 for k in range(n - 6)])
    pk = []
    for th in (2.0, 3.0, 4.0):
        k = m = 0
        while k < n - 6:
            if brk[k] and dec[k] > th: m += 1; k += 20
            else: k += 1
        pk.append(m)
    deep = k = 0
    while k < n - 20:
        if sp[k] - sp[k:k + 20].min() > 1.5:
            deep += int(sp[k] - sp[k:k + 40].min() > 2.5); k += 20
        else: k += 1
    h = max(dist, 1e-6) / 100.0
    out = (f"{path}: {n * DT:.0f} s, {dist:.0f} m, 평균 속력 {sp.mean():.2f} m/s, 최저 {sp.min():.2f} | 급감속 100 m 당 >2: {pk[0] / h:.2f}  >3: {pk[1] / h:.2f}  >4: {pk[2] / h:.2f} "
           f"(횟수 {pk[0]}·{pk[1]}·{pk[2]}) | 깊은 제동 {deep / h:.2f} ({deep}) | 가장 큰 감속 {dec.max():.1f} m/s² | 1 m/s 미만 시간 {(sp < 1.0).mean() * 100:.1f} %")
    if cs is not None:
        drop = cs[:-1] - cs[1:]
        out += f" | 명령 한 스텝 낙폭 > 0.5 m/s: {int((drop > 0.5).sum())} 회 (가장 큰 것 {drop.max():.2f})"
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(metrics(p))
