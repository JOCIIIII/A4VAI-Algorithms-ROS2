# Foxglove Package - CLAUDE.md

이 파일은 foxglove 패키지에 대한 상세 정보를 제공합니다.

## 패키지 개요

**foxglove**는 A4VAI 드론 시스템의 시각화 및 모니터링을 위한 ROS2 패키지입니다. Foxglove Studio와 같은 시각화 도구를 위해 드론의 상태, 경로, LiDAR 데이터, 장애물 정보를 퍼블리시합니다.

- **패키지 타입**: ament_python (ROS2 Python 패키지)
- **주요 노드**: `foxglove_node`
- **주요 역할**: 실시간 드론 상태 시각화 및 장애물 감지/분석

## 프로젝트 구조

```
foxglove/
├── foxglove/
│   ├── __init__.py              # 패키지 초기화 파일
│   └── foxglove.py              # 메인 노드 구현
├── resource/
│   └── foxglove                 # ROS2 리소스 마커
├── test/
│   ├── test_copyright.py        # 저작권 테스트
│   ├── test_flake8.py           # 코드 스타일 테스트
│   └── test_pep257.py           # Docstring 테스트
├── package.xml                  # 패키지 메타데이터 및 의존성
├── setup.cfg                    # 설정 파일
└── setup.py                     # 패키지 설치 스크립트
```

## 주요 기능

### 1. 드론 상태 시각화
- **드론 포즈(Pose) 퍼블리싱**: 실시간 위치 및 자세 정보
- **드론 속도 퍼블리싱**: 선속도 및 각속도
- **드론 경로 추적**: 비행 경로 히스토리 기록 및 시각화
- **FOV 원뿔 시각화**: 센서 시야각(Field of View) 표시

### 2. LiDAR 데이터 처리
- **포인트 클라우드 전처리**: 노이즈 제거 및 필터링
- **좌표계 변환**: Body frame(FLU) → World frame(ENU)
- **TF 브로드캐스팅**: `world` ↔ `SimpleFlight/RPLIDAR_A3` 변환

### 3. 장애물 감지 및 분석
- **DBSCAN 클러스터링**: 포인트 클라우드에서 장애물 그룹화
- **PCA 기반 OBB 생성**: Oriented Bounding Box로 장애물 표현
- **위협 레벨 계산**: 거리, 방위각, 속도 기반 위험도 평가
- **타겟 장애물 선정**: 회피가 필요한 장애물 식별

### 4. 좌표계 변환
- **NED → ENU 변환**: PX4의 NED 좌표계를 ROS2 표준 ENU로 변환
- **FRD → FLU 변환**: PX4 body frame을 ROS2 body frame으로 변환
- **쿼터니언 처리**: 자세 정보 변환 및 정규화

## 파일별 상세 설명

### foxglove.py

메인 노드 구현 파일로, `FoxgloveNode` 클래스와 `ObstacleCluster` 데이터 클래스를 포함합니다.

#### 클래스 구조

##### ObstacleCluster (dataclass)
장애물 클러스터 정보를 저장하는 데이터 클래스

**속성:**
- `cluster_id`: 클러스터 고유 ID
- `points`: (N, 3) 형태의 포인트 배열
- `obstacle_position`: 장애물 중심점 (3D)
- `obstacle_distance`: 드론으로부터의 거리
- `obstacle_rel_bearing`: 상대 방위각 (radians)
- `obstacle_rel_velocity`: 상대 속도
- `is_target_obstacle`: 회피 대상 여부
- `is_dangerous`: 위험 거리 이내 여부
- `is_in_path`: 비행 경로 상 존재 여부
- `threat_level`: 위협 레벨 (0~3)

##### FoxgloveNode (Node)
Foxglove 시각화를 위한 메인 ROS2 노드

**주요 변수:**
- `vehicle_pose`: 드론의 현재 위치 및 자세 (PoseStamped)
- `vehicle_velocity`: 드론의 속도 (Twist)
- `vehicle_path`: 드론의 비행 경로 히스토리 (Path)
- `heading_enu`: ENU 좌표계에서의 heading 각도
- `obstacle_info`: 감지된 장애물 정보 딕셔너리
- `latest_lidar_points_np`: 최신 LiDAR 포인트 클라우드

**주요 파라미터:**
- `safety_distance`: 안전 거리 (1.0m)
- `max_detection_range`: 최대 감지 거리 (20.0m)
- `danger_distance_threshold`: 위험 거리 임계값 (7.0m)
- `warning_distance_threshold`: 경고 거리 임계값 (10.0m)
- `path_angle_threshold`: 경로 상 판단 각도 임계값 (30°)
- `velocity_threshold`: 접근 속도 임계값 (5.0 m/s)

#### Subscriber

| Topic | Message Type | QoS | 설명 |
|-------|-------------|-----|------|
| `/fmu/out/vehicle_local_position` | VehicleLocalPosition | BEST_EFFORT | PX4 드론 위치 정보 |
| `/fmu/out/vehicle_attitude` | VehicleAttitude | BEST_EFFORT | PX4 드론 자세 정보 |
| `/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3` | PointCloud2 | BEST_EFFORT | LiDAR 포인트 클라우드 |

#### Publisher

| Topic | Message Type | QoS | 설명 |
|-------|-------------|-----|------|
| `/vehicle_path` | Path | BEST_EFFORT | 드론 비행 경로 |
| `/drone_pose` | PoseStamped | BEST_EFFORT | 드론 위치/자세 |
| `/drone_velocity` | Twist | BEST_EFFORT | 드론 속도 |
| `/fov_cone` | Marker | BEST_EFFORT | FOV 원뿔 시각화 |
| `/world_points` | PointCloud2 | BEST_EFFORT | World frame 포인트 클라우드 |
| `/filtered_point_cloud` | PointCloud2 | DEFAULT | 필터링된 포인트 클라우드 |
| `/cluster_point_cloud` | PointCloud2 | BEST_EFFORT | 클러스터별 포인트 클라우드 |
| `/obstacle_markers` | MarkerArray | BEST_EFFORT | 장애물 마커 (OBB) |
| `/obstacle_info` | MarkerArray | BEST_EFFORT | 장애물 상세 정보 |
| `/obstacle_flag` | Bool | DEFAULT | 장애물 감지 플래그 |

#### Timer

| Timer | Period | Callback | 설명 |
|-------|--------|----------|------|
| `tf_publish_timer_` | 0.05s (20Hz) | `publish_world_to_simpleflight_tf()` | TF 변환 브로드캐스트 |
| `foxglove_publish_timer_` | 0.05s (20Hz) | `publish_foxglove_data()` | Foxglove 데이터 퍼블리시 |
| `lidar_processing_timer_` | 0.05s (20Hz) | `process_lidar_points()` | LiDAR 데이터 처리 |

#### 주요 메서드

##### Subscriber 콜백
- `vehicle_local_position_callback(msg)`: PX4 위치 정보 수신 및 NED→ENU 변환
- `attitude_callback(msg)`: PX4 자세 정보 수신 및 FRD→FLU 변환
- `update_latest_lidar_msg(pc_msg)`: LiDAR 포인트 클라우드 수신

##### Publisher 메서드
- `publish_vehicle_pose()`: 드론 포즈 퍼블리시
- `publish_vehicle_path()`: 드론 경로 퍼블리시
- `publish_vehicle_velocity()`: 드론 속도 퍼블리시
- `publish_fov_cone()`: FOV 원뿔 마커 퍼블리시
- `publish_filtered_point_cloud()`: 필터링된 포인트 클라우드 퍼블리시
- `publish_cluster_points()`: 클러스터 포인트 클라우드 퍼블리시
- `publish_world_to_simpleflight_tf()`: TF 변환 브로드캐스트
- `publish_foxglove_data()`: 모든 Foxglove 데이터 통합 퍼블리시

##### LiDAR 처리 메서드
- `process_lidar_points()`: LiDAR 데이터 전체 처리 파이프라인
- `preprocess_points()`: 포인트 클라우드 전처리 (노이즈 제거, 필터링)
- `extract_obstacle_info(points_np)`: DBSCAN 클러스터링 및 장애물 정보 추출
- `transform_pc_body_to_world(body_points_np)`: Body frame → World frame 변환
- `transform_points_batch(points_np, transform)`: 벡터화된 좌표 변환

##### 장애물 분석 메서드
- `check_obstacle_flags()`: 모든 장애물의 플래그 및 위협 레벨 계산
- `_is_distance_dangerous(obstacle)`: 거리 기반 위험도 판단
- `_is_in_flight_path(obstacle)`: 비행 경로 상 존재 여부 판단
- `_is_target_for_avoidance(obstacle)`: 회피 대상 장애물 판단
- `_calculate_threat_level(obstacle)`: 종합 위협 레벨 계산 (0~3)
- `get_target_obstacles()`: 회피 대상 장애물 목록 반환
- `get_most_dangerous_obstacle()`: 가장 위험한 장애물 반환

##### 유틸리티 메서드
- `ned_to_enu(x_n, y_n, z_n)`: NED → ENU 좌표 변환
- `create_fov_cone_marker(position, yaw, pitch, ...)`: FOV 원뿔 마커 생성
- `distance_point_to_obb(point, center, R, size)`: 점과 OBB 간의 거리 계산
- `calculate_relative_bearing(obstacle_position, vehicle_position)`: 상대 방위각 계산
- `quaternion_to_rotation_matrix(quat)`: 쿼터니언 → 회전 행렬 변환

### package.xml

ROS2 패키지 메타데이터 및 의존성 정의

**의존성:**
- `rclpy`: ROS2 Python 클라이언트 라이브러리

**테스트 의존성:**
- `ament_copyright`, `ament_flake8`, `ament_pep257`, `python3-pytest`

### setup.py

패키지 설치 스크립트 및 엔트리 포인트 정의

**엔트리 포인트:**
- `foxglove = foxglove.foxglove:main`: `ros2 run foxglove foxglove` 명령으로 실행

## 알고리즘 상세

### 1. LiDAR 포인트 클라우드 처리 파이프라인

```
LiDAR Raw Data (PointCloud2)
    ↓
[preprocess_points]
  - 드론 중심 반경 필터링 (vehicle_radius = 0.01m)
  - 전방 필터링 (x > 0.0)
  - 지면 필터링 (z > 0.0)
    ↓
[transform_pc_body_to_world]
  - TF lookup: SimpleFlight/RPLIDAR_A3 → world
  - 쿼터니언 → 회전 행렬 변환
  - 벡터화 변환: R @ points.T + t
  - 고도 필터링 (z > 2.0m)
    ↓
[extract_obstacle_info]
  - DBSCAN 클러스터링 (eps=0.5, min_samples=3)
  - PCA로 주성분 추출 (3D OBB 방향)
  - 클러스터 중심점 계산
  - 거리, 방위각, 상대속도 계산
    ↓
[check_obstacle_flags]
  - 위험 거리 판단
  - 비행 경로 상 판단
  - 회피 대상 선정
  - 위협 레벨 계산 (0~3)
    ↓
Publish: obstacle markers, cluster points, obstacle info
```

### 2. 장애물 위협 레벨 시스템

**레벨 3 (매우 위험):**
- 조건: `distance < 2.0m` AND `is_in_path == True`
- 조치: 즉시 회피 필요

**레벨 2 (위험):**
- 조건: `distance < danger_threshold(7.0m)` AND `is_in_path == True`
- 조치: 회피 준비

**레벨 1 (주의):**
- 조건: `distance < warning_threshold(10.0m)` AND `is_in_path == True`
- 조치: 모니터링

**레벨 0 (안전):**
- 조건: 위 조건에 해당하지 않음
- 조치: 무시

### 3. 회피 대상 선정 로직

장애물이 회피 대상(`is_target_obstacle = True`)으로 선정되는 조건:

1. **(거리 & 경로 & 접근)**: `distance < 10.0m` AND `is_in_path` AND `relative_velocity < -5.0 m/s`
2. **OR (매우 가까움)**: `distance < 7.0m`

### 4. 좌표계 변환

#### NED → ENU 변환
```
NED: [North, East, Down]
ENU: [East, North, Up]

x_enu = y_ned
y_enu = x_ned
z_enu = -z_ned
```

#### FRD → FLU 변환 (자세)
```
PX4 quaternion (FRD → NED):
  q_px4 = [w, x, y, z]

변환 과정:
  1. q_frd_to_ned = [x, y, z, w]
  2. R_frd_to_ned = Rotation(q_frd_to_ned)
  3. R_ned_to_enu = Rotation.from_euler('ZYX', [π/2, 0, π])
  4. R_frd_to_enu = R_ned_to_enu * R_frd_to_ned
  5. R_frd_to_flu = Rotation.from_euler('XYZ', [π, 0, 0])
  6. R_flu_to_enu = R_frd_to_enu * R_frd_to_flu
  7. q_ros = R_flu_to_enu.as_quat()
```

## 실행 방법

### 패키지 빌드
```bash
cd /home/yonghajo/Documents/A4VAI-SITL/ROS2/ros2_ws
colcon build --packages-select foxglove
source install/setup.bash
```

### 노드 실행
```bash
ros2 run foxglove foxglove
```

### Launch 파일 (예시)
현재 launch 파일은 없지만, 다음과 같이 작성 가능:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='foxglove',
            executable='foxglove',
            name='foxglove_node',
            output='screen'
        )
    ])
```

## 의존성

### ROS2 패키지
- `rclpy`: ROS2 Python 클라이언트
- `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `visualization_msgs`, `std_msgs`: 표준 메시지 타입
- `px4_msgs`: PX4 통신용 메시지
- `tf2_ros`, `tf2_geometry_msgs`: TF 변환

### Python 라이브러리
- `numpy`: 수치 연산
- `scipy`: Rotation 변환
- `sklearn`: DBSCAN 클러스터링, PCA
- `sensor_msgs_py`: PointCloud2 처리

## 주의사항 및 알려진 이슈

### 1. QoS 설정
- PX4 통신은 `BEST_EFFORT` reliability 사용
- 일부 토픽은 메시지 손실 가능성 있음

### 2. LiDAR 토픽
- AirSim 환경 전용 토픽 사용: `/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3`
- 실제 하드웨어 사용 시 토픽 이름 변경 필요

### 3. 좌표계
- World frame: ENU
- Body frame: FLU
- PX4: NED, FRD 사용 → 변환 필수

### 4. 타이머 주기
- 모든 타이머가 20Hz(0.05s)로 동작
- 높은 주파수 필요 시 조정 가능하나 CPU 부하 고려

### 5. 장애물 감지 파라미터
- DBSCAN eps=0.5, min_samples=3은 실험적 값
- 환경에 따라 튜닝 필요
- 고도 필터 (z > 2.0m)는 지면 제거용이나 지형에 따라 조정 필요

## 향후 개선 사항

1. **장애물 추적**: 프레임 간 장애물 연관성 추적 (Kalman Filter 등)
2. **동적 파라미터**: `rclpy.parameter`를 통한 런타임 파라미터 조정
3. **Launch 파일**: 통합 실행을 위한 launch 파일 추가
4. **장애물 정보 퍼블리시**: 주석 처리된 `publish_obstacle_info()` 메서드 완성
5. **성능 최적화**: NumPy 벡터화 연산 더욱 활용
6. **에러 핸들링**: TF lookup 실패 시 재시도 로직
7. **로깅 레벨**: 상세한 디버그 로깅 추가

## 참고 자료

- [Foxglove Studio](https://foxglove.dev/)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [PX4 ROS2 Interface](https://docs.px4.io/main/en/ros/ros2_comm.html)
- [DBSCAN Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [PCA (Principal Component Analysis)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## 라이센스

TODO: License declaration

## 유지보수자

- admin (admin@todo.todo)
