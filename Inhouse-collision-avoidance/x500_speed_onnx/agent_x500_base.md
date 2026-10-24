# X500급 회피 정책 묶음 (`x500_onnx`, 2026-10-15) — C-Track 미검증 후보

노드 사용법·입출력·관측 구성은 X8 묶음과 같습니다. 전체 설명은 같이 넣은 `agent_x8_base.md`(= `x8_30m_soft_onnx/agent.md`)에 있습니다.
이 문서에는 X8과 다른 점만 적습니다.

## 1. 파일
- `ca_x500.onnx`: 정책입니다. 입력 216, 출력 3(WP 방위, 고도, 속도)으로 X8과 같습니다.
- `cfg.json`: 노드가 읽는 설정입니다. **손으로 바꾸지 마세요.** 특히 `v_max`를 바꾸면 관측 정규화와 속도 출력 배율이 함께 바뀝니다. 속도별 판은 따로 학습해서 드립니다.
- `ca_policy_ref.py`: 참조 구현입니다. X8 묶음과 같은 파일입니다.
- `dynamics_x500.md`: 기체 값과 출처입니다.

## 2. X8 묶음과 다른 cfg.json 값

| 키 | X8 | X500 | 이유 |
|---|---|---|---|
| `d_crash` | 1.0 | 0.6 | 프로펠러 끝 0.39 m + 여유 0.21 m. 보호막 통로 반폭은 0.8 m(= `d_crash` + `guard_margin`) |
| `lidar_z_offset` | 0.3 | 0.1 | Mid-360이 기체 중심 위 0.10 m(가정) |
| `z_min` / `wp_z_min` | 0.94 / 1.14 | 0.74 / 0.94 | 다리 바닥이 중심 아래 0.24 m(가정) |
| `gov_d0` | 1.2 | 0.8 | 보호막이 켜져 있어 쓰이지 않음 |
| `tracker_*` | | | 학습 쪽 값입니다. 노드는 읽지 않습니다 |

보호막·속도 내림 제한·`v_max` 5 m/s·30 m 거리 이미지·상승 금지는 X8 묶음과 같습니다.

## 3. 노드에서 바꿀 것
- **점군 기준점:** 기체 중심입니다. LiDAR 장착 높이 0.10 m만큼 옮겨 주세요. 실제 장착 높이가 다르면 알려 주세요. 학습도 다시 맞춥니다.
- **안전 필터:** X8은 약 0.9 m입니다. X500에서는 충돌 반경에 맞춰 약 0.5 m를 권합니다. 이 값은 학습과 무관한 노드 설정입니다.

## 4. 성능 (시뮬레이션, X500 과제)
- 새 시드, 5 레벨에서 도달률 0.819, 충돌률 0.144입니다. 학습 없이 X8 정책을 쓰면 0.739 / 0.157입니다.
- 숲은 X500 크기에 맞춘 세트(`LEVELS_3DXB`)라 X8 수치와 비교할 수 없습니다.
- 추종 체인(PF MPPI + PX4)은 X8에서 잰 응답을 그대로 가정했습니다. X500으로 비행한 기록은 없습니다.
- 기울어지는 시야 조건(기울기 그대로 + 스캔 누락 + 노드 누적 0.4 s)에서는 도달률 0.789, 충돌률 0.174입니다(조정용 시드, L0·2·4).

## 5. 다시 만들기
```bash
G="guard_on=1,guard_vmin=0.6,guard_rise=2.5,guard_cmd=0,guard_fall=1.5,guard_fall_slack=0.3,guard_fall_cmd=1,guard_a=1.5,guard_th=0.6,guard_ah=0.0,guard_hmode=1"
python scripts/export_onnx.py --out deploy/x500_onnx --cfg_class Avoid3DForestWallX500EnvCfg --set "$G" \
  --ckpt "ca_x500|runs/wp3dx500_control/x8w_xa_s42/checkpoints/agent_28800.pt|"
python scripts/check_deploy_ref.py deploy/x500_onnx   # 관측 4.3e-6, WP 9.4e-6 m, 속도 4.2e-6 m/s, OK
```
