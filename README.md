# A4VAI 2세부 자율비행 알고리즘 (ROS 2 Humble)

[PX4-SITL-Runner](https://github.com/JOCIIIII/PX4-SITL-Runner)(A4VAI 2세부 SILS·PILS 시뮬레이터)에서 쓰는 ROS 2 패키지 모음입니다.
시뮬레이터의 `./scripts/install.sh`가 이 저장소를 받아 빌드하므로, 시뮬레이터를 쓸 때는 따로 받을 필요가 없습니다.
`main`과 `inhousesim-v1.0`은 같은 최신판입니다.

## 구성

| 폴더 | 내용 |
|---|---|
| `Inhouse-collision-avoidance/` | 학습 충돌회피 정책 묶음. 폴더마다 ONNX 모델, 참조 코드(`ca_policy_ref.py`), 설정(`cfg.json`), 설명(`agent.md`)이 있습니다 |
| `flight_driver/` | 비행 드라이버: 이륙 → OFFBOARD → 정책 비행 → 목표 근처에서 PX4 위치 제어 → 착륙. 여러 구간 임무와 일시정지를 지원합니다 |
| `flight_manager/` | 이전 비행 관리 노드 (SUPER·T1000 앱이 씁니다) |
| `realgazebo_fastplanner_adapters/` | PX4 위치와 라이다 → 오도메트리·점군, 목표점 발행 |
| `super_pf_bridge/` | SUPER 궤적 → 경로추종 웨이포인트 |
| `path_following_test/` | 고정 웨이포인트로 경로추종을 돌리는 시험 노드 |
| `T1000_Control_445/` | T1000 제어(전북대학교 제공): 경로와 라이다로 PX4 속도 명령을 냅니다. 설명서 포함 |
| `custom_msgs/` | 공용 메시지 |
| `A4VAI-PathFollowing/` | 경로추종 (서브모듈 `JOCIIIII/A4VAI-PathFollowing`, `inhousesim-v1.0`) |
| `super/` | SUPER 경로계획 (서브모듈 `JOCIIIII/SUPER`, `inhousesim-v1.0`) |
| `px4_msgs/` | PX4 v1.16 메시지 (서브모듈 `PX4/px4_msgs`, `release/1.16`) |

## 충돌회피 정책

학습 회피 노드(PX4-SITL-Runner `scripts/ros2/apps/noclimb_ca_node.py`)가 비행할 때 이 폴더에서 정책을 읽습니다(빌드할 필요 없음).
비행 프로필마다 어느 정책을 쓰는지와 실증 시나리오 시험맵에서의 결과는 PX4-SITL-Runner의
[비행 프로필 설명](https://github.com/JOCIIIII/PX4-SITL-Runner/blob/main/scripts/dev/flight_profiles/README.md)에 있습니다.

## 직접 받아 빌드할 때

```bash
mkdir -p ros2_ws && cd ros2_ws
git clone --recurse-submodules https://github.com/JOCIIIII/A4VAI-Algorithms-ROS2.git src
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

의존 패키지가 모두 들어 있는 도커 이미지는 시뮬레이터가 쓰는 `jociiiii/a4vai:inhousesim-v0.9`입니다.
시뮬레이터에 설치된 소스를 고친 뒤에는 PX4-SITL-Runner에서 `./scripts/run.sh ros2 build ros2_ws <패키지>`로 다시 빌드합니다.
