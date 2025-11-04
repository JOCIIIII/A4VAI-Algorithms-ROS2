# LiDAR 기반 장애물 위협 판단 로직

## 개요

본 문서는 foxglove 패키지에서 LiDAR 데이터를 활용하여 장애물을 감지하고 위협 수준을 판단하는 전체 알고리즘을 설명합니다.

**파일**: `foxglove/foxglove/foxglove.py`

---

## 전체 처리 파이프라인

```
LiDAR Raw Data (PointCloud2)
    ↓
① 전처리 (preprocess_points)
    ↓
② 좌표 변환 (transform_pc_body_to_world)
    ↓
③ 클러스터링 (extract_obstacle_info)
    ↓
④ 장애물 정보 계산
    ↓
⑤ 위협 레벨 판단 (check_obstacle_flags)
    ↓
⑥ CA 상태 결정 (_update_collision_avoidance_state)
    ↓
/obstacle_flag 퍼블리시
```

---

## ① LiDAR 데이터 전처리

**메서드**: `preprocess_points()` (Line 678-716)

### 목적
LiDAR raw 포인트 클라우드에서 노이즈를 제거하고 유효한 장애물 포인트만 필터링

### 입력
- `self.latest_lidar_points_np`: generator (x, y, z)
- 좌표계: Body frame (FLU - Forward, Left, Up)

### 처리 과정

```python
# 1. Generator → NumPy array 변환
points_np = np.array(list(self.latest_lidar_points_np),
                     dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

# 2. 필터링 마스크 생성
vehicle_radius = 0.01  # m
distance_mask = np.sqrt(x**2 + y**2 + z**2) > vehicle_radius  # 드론 중심 제외
forward_mask = x > 0.0  # 전방만
ground_mask = z > 0.0   # 지면 제외 (FLU에서 z > 0 = 위쪽)

mask = distance_mask & forward_mask & ground_mask

# 3. 필터링된 포인트 반환
points = np.column_stack((x[mask], y[mask], z[mask]))
```

### 출력
- `filtered_points_np`: (N, 3) NumPy array (Body frame)

### 필터링 조건 요약

| 조건 | 설명 | 임계값 |
|-----|------|--------|
| 드론 중심 제외 | 드론 자체를 장애물로 감지 방지 | 반경 0.01m |
| 전방만 | 후방/측면 무시 (센서 FOV 고려) | x > 0 |
| 지면 제거 | 바닥을 장애물로 감지 방지 | z > 0 |

---

## ② 좌표 변환 (Body → World)

**메서드**: `transform_pc_body_to_world()` (Line 1121-1148)

### 목적
Body frame(FLU) 포인트를 World frame(ENU)으로 변환

### 처리 과정

```python
# 1. TF lookup (SimpleFlight/RPLIDAR_A3 → world)
transform = self.tf_buffer.lookup_transform(
    'world',
    'SimpleFlight/RPLIDAR_A3',
    rclpy.time.Time()
)

# 2. 쿼터니언 → 회전 행렬
R = quaternion_to_rotation_matrix(transform.rotation)
t = [transform.translation.x, transform.translation.y, transform.translation.z]

# 3. 벡터화 변환
world_points = (R @ body_points.T).T + t

# 4. 고도 필터링 (지면 제거)
altitude_filter = world_points[:, 2] > 2.0  # z > 2m
world_points = world_points[altitude_filter]
```

### 출력
- `world_points_np`: (N, 3) NumPy array (World frame ENU)

### 좌표계

| Frame | 축 정의 | 비고 |
|-------|---------|------|
| Body (FLU) | x=Forward, y=Left, z=Up | ROS2 표준 body frame |
| World (ENU) | x=East, y=North, z=Up | ROS2 표준 world frame |

---

## ③ DBSCAN 클러스터링

**메서드**: `extract_obstacle_info()` (Line 718-764)

### 목적
포인트 클라우드를 장애물 단위로 그룹화

### DBSCAN 파라미터

```python
clustering = DBSCAN(eps=0.5, min_samples=3).fit(points_np)
```

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `eps` | 0.5m | 이웃으로 간주할 최대 거리 |
| `min_samples` | 3 | 클러스터를 형성하는 최소 포인트 수 |

### 클러스터링 결과

```
labels = [-1, 0, 0, 1, 1, 1, -1, 2, 2, ...]
         ↑   ↑     ↑           ↑   ↑
       노이즈 클러스터0  클러스터1    클러스터2
```

- `label = -1`: 노이즈 (무시)
- `label >= 0`: 클러스터 ID

---

## ④ 장애물 정보 계산

**메서드**: `extract_obstacle_info()` (Line 730-764)

각 클러스터마다 다음 정보를 계산합니다.

### 4-1. PCA 기반 OBB (Oriented Bounding Box)

```python
# PCA로 주성분 추출
pca = PCA(n_components=3)
pca.fit(cluster_points)
R_pca = pca.components_  # (3, 3) 회전 행렬

# 오른손 좌표계 보장
if np.linalg.det(R_pca) < 0:
    R_pca[2, :] *= -1

# PCA 좌표계로 변환
transformed = np.dot(cluster_points - mean, R_pca.T)
min_pt, max_pt = np.min(transformed, axis=0), np.max(transformed, axis=0)
size = max_pt - min_pt  # OBB 크기
center_local = (max_pt + min_pt) / 2

# World frame으로 중심점 변환
obstacle_position = np.dot(center_local, R_pca) + mean
```

**결과:**
- `obstacle_position`: 장애물 중심점 (world frame)
- `obb_rotation`: OBB 회전 행렬 (R_pca)
- `obb_size`: OBB 크기 [길이, 너비, 높이]

### 4-2. 드론과의 거리 계산

**메서드**: `distance_point_to_obb()` (Line 1070-1087)

```python
# 드론 위치를 OBB 좌표계로 변환
p_local = np.dot(R.T, (drone_position - obstacle_center))

# OBB 절반 크기
half = size / 2.0

# OBB 외부 거리 계산
d = np.maximum(np.abs(p_local) - half, 0.0)

# 유클리드 거리
distance = np.linalg.norm(d)
```

**특징:**
- OBB 내부에 있으면 `distance = 0`
- OBB 표면까지의 최단 거리 반환

### 4-3. 상대 방위각 계산

**메서드**: `calculate_relative_bearing()` (Line 1089-1119)

```python
# 장애물 방향 벡터
dx = obstacle_position[0] - drone_position[0]
dy = obstacle_position[1] - drone_position[1]

# 절대 방위각 (world frame)
absolute_bearing = math.atan2(dy, dx)

# 상대 방위각 = 절대 - 드론 heading
relative_bearing = absolute_bearing - self.heading_enu

# -π ~ π 정규화
relative_bearing = math.atan2(sin(relative_bearing), cos(relative_bearing))
```

**출력 범위:**
- `0°`: 정면
- `+90°`: 왼쪽
- `-90°`: 오른쪽
- `±180°`: 후방

### 4-4. ObstacleCluster 데이터 클래스

```python
@dataclass
class ObstacleCluster:
    cluster_id: int                  # 클러스터 ID
    points: np.ndarray               # (N, 3) 포인트들
    obstacle_position: np.ndarray    # (3,) 중심점
    obstacle_distance: float         # 거리 (m)
    obstacle_rel_bearing: float      # 상대 방위각 (rad)
    obb_rotation: np.ndarray         # (3, 3) 회전 행렬
    obb_size: np.ndarray             # (3,) 크기 [x, y, z]
    is_target_obstacle: bool = False # 회피 대상 여부
    is_dangerous: bool = False       # 위험 거리 이내
    is_in_path: bool = False         # 경로 상 존재
    threat_level: int = 0            # 위협 레벨 (0~3)
```

---

## ⑤ 위협 레벨 판단

**메서드**: `check_obstacle_flags()` (Line 767-806)

각 장애물마다 4가지 플래그를 설정합니다.

### 5-1. is_dangerous (거리 기반 위험도)

**메서드**: `_is_distance_dangerous()` (Line 895-897)

```python
return obstacle.obstacle_distance < self.danger_distance_threshold  # 10.0m
```

**판단:**
- `True`: 거리 < 10m (위험)
- `False`: 거리 >= 10m (안전)

### 5-2. is_in_path (경로 상 존재 여부)

**메서드**: `_is_in_flight_path()` (Line 899-911)

```python
angle_diff = abs(obstacle.obstacle_rel_bearing)

# 전방 각도 범위 내
is_in_front = angle_diff < self.path_angle_threshold  # 24° (최근 수정됨)

# 충분히 가까운 거리
is_close_enough = obstacle.obstacle_distance < self.warning_distance_threshold  # 15.0m

return is_in_front and is_close_enough
```

**판단 조건:**
- 상대 방위각 < ±24° (전방)
- **AND** 거리 < 15m

**결과:**
- `True`: 비행 경로 상에 있음
- `False`: 경로 벗어남

### 5-3. is_target_obstacle (회피 대상)

**메서드**: `_is_target_for_avoidance()` (Line 913-923)

```python
# 조건 1: 경고 거리 이내 + 경로 상
distance_check = obstacle.obstacle_distance < 15.0  # warning_distance_threshold
path_check = obstacle.is_in_path

# 조건 2: 매우 가까움
very_close = obstacle.obstacle_distance < 10.0  # danger_distance_threshold

return (distance_check and path_check) or very_close
```

**판단 로직:**
- (거리 < 15m **AND** 경로 상) **OR** 거리 < 10m

**결과:**
- `True`: 회피 필요
- `False`: 회피 불필요

### 5-4. threat_level (종합 위협 레벨)

**메서드**: `_calculate_threat_level()` (Line 925-942)

```python
distance = obstacle.obstacle_distance

# 레벨 3: 매우 위험 (즉시 회피)
if distance < 2.0 and obstacle.is_in_path:
    return 3

# 레벨 2: 위험 (회피 준비)
if distance < 10.0 and obstacle.is_in_path:
    return 2

# 레벨 1: 주의 (모니터링)
if distance < 15.0 and obstacle.is_in_path:
    return 1

# 레벨 0: 안전
return 0
```

### 위협 레벨 요약표

| Level | 이름 | 거리 조건 | 경로 조건 | 색상 | 조치 |
|-------|------|-----------|-----------|------|------|
| 3 | DANGER | < 2m | 경로 상 | 빨강 | 즉시 회피 |
| 2 | WARNING | < 10m | 경로 상 | 주황 | 회피 준비 |
| 1 | CAUTION | < 15m | 경로 상 | 노랑 | 모니터링 |
| 0 | SAFE | >= 15m 또는 경로 밖 | - | 파랑 | 무시 |

---

## ⑥ CA 상태 결정

**메서드**: `_update_collision_avoidance_state()` (Line 808-893)

### 가장 위험한 장애물 선정

**메서드**: `get_most_dangerous_obstacle()` (Line 952-971)

```python
# threat_level > 0인 장애물만 필터링
dangerous_obstacles = [obs for obs in self.obstacle_info.values()
                       if obs.threat_level > 0]

if not dangerous_obstacles:
    return None

# (위협 레벨, 거리) 기준 정렬
most_dangerous = max(dangerous_obstacles,
                     key=lambda x: (x.threat_level, -x.obstacle_distance))
```

**우선순위:**
1. 위협 레벨 높은 순
2. 같은 레벨이면 거리 가까운 순

### CA 진입 조건 (Hysteresis Entry)

```python
if not self.avoidance_required:  # CA 중이 아닐 때
    if most_dangerous.threat_level >= self.ca_entry_threat_level:  # >= 2
        # CA 시작!
        self.avoidance_required = True
        self.obstacle_flag = True
```

**조건:**
- `threat_level >= 2` (WARNING 이상)
- 즉, 거리 < 10m **AND** 경로 상

### CA 종료 조건 (Hysteresis Exit)

```python
if self.avoidance_required:  # CA 진행 중
    if most_dangerous.threat_level <= self.ca_exit_threat_level:  # <= 1
        # 각도 조건
        angle_diff = abs(most_dangerous.obstacle_rel_bearing)
        is_obstacle_cleared = angle_diff > self.safe_angle_threshold  # 90°

        # 거리 조건 (현재: 25m)
        is_distance_safe = most_dangerous.obstacle_distance >= self.safe_distance_threshold  # 25.0m

        if is_distance_safe or is_obstacle_cleared:
            self.safe_distance_count += 1

            if self.safe_distance_count >= self.safe_count_required:  # 100회 (약 2초)
                # CA 종료!
                self.avoidance_required = False
                self.obstacle_flag = False
                self.safe_distance_count = 0
```

**조건 (AND 로직):**
1. `threat_level <= 1` (CAUTION 이하)
2. **AND** (`distance >= 25m` **OR** `|bearing| > 90°`)
3. **AND** 위 조건이 100회(약 2초) 지속

### Hysteresis 설계

```
CA 진입 ←──────────────────────→ CA 종료
threat_level >= 2              threat_level <= 1
(거리 < 10m)                   (거리 >= 25m OR 각도 > 90°)
```

**효과:**
- 떨림(chattering) 방지
- 10m ~ 25m 구간에서 안정적 동작

---

## ⑦ 최종 출력

### /obstacle_flag 퍼블리시

**토픽**: `/obstacle_flag` (std_msgs/Bool)

```python
flag_msg = Bool()
flag_msg.data = self.obstacle_flag  # True or False
self.obstacle_flag_publisher_.publish(flag_msg)
```

**의미:**
- `True`: 충돌회피 필요 (CA 모드)
- `False`: 안전 (PF 모드)

### 시각화 마커 퍼블리시

| 토픽 | 내용 |
|------|------|
| `/obstacle_markers` | OBB 박스 (MarkerArray) |
| `/obstacle_info` | 중심점, 거리선, 텍스트 (MarkerArray) |
| `/cluster_point_cloud` | 클러스터별 색상 포인트 (PointCloud2) |
| `/world_points` | World frame 전체 포인트 (PointCloud2) |

---

## 파라미터 요약

### 거리 임계값

```python
self.danger_distance_threshold = 10.0   # m (위험 거리)
self.warning_distance_threshold = 15.0  # m (경고 거리)
self.safe_distance_threshold = 25.0     # m (CA 종료 거리)
```

### 각도 임계값

```python
self.path_angle_threshold = np.deg2rad(24)   # 24° (경로 상 판단)
self.safe_angle_threshold = np.deg2rad(90)   # 90° (CA 종료 각도)
```

### CA 상태 임계값

```python
self.ca_entry_threat_level = 2   # CA 진입 (threat_level >= 2)
self.ca_exit_threat_level = 1    # CA 종료 (threat_level <= 1)
self.safe_count_required = 100   # 안전 상태 100회(약 2초) 유지
```

### 클러스터링 파라미터

```python
DBSCAN(eps=0.5, min_samples=3)
```

---

## 처리 주기 및 성능

### 타이머 주기
- LiDAR 콜백: 센서 주기에 따름 (약 50Hz)
- 처리 파이프라인: 실시간 (매 LiDAR 메시지마다)

### 연산 복잡도

| 단계 | 복잡도 | 비고 |
|------|--------|------|
| 전처리 | O(N) | N = 포인트 수 |
| 좌표 변환 | O(N) | 벡터화 연산 |
| DBSCAN | O(N log N) | sklearn 최적화 |
| PCA | O(M²) | M = 클러스터 포인트 수 |
| 위협 판단 | O(K) | K = 클러스터 수 |

---

## 알려진 제한사항

1. **전방만 감지**: Body frame에서 x > 0 조건으로 후방 무시
2. **고도 필터 고정**: z > 2.0m로 지면 제거 (지형에 따라 부정확할 수 있음)
3. **DBSCAN 파라미터**: eps=0.5는 실험적 값, 환경에 따라 조정 필요
4. **프레임 간 추적 없음**: 각 프레임 독립적으로 처리, 장애물 ID 불연속

---

## 튜닝 가이드

### 더 민감하게 (CA 자주 진입)
```python
self.danger_distance_threshold = 15.0  # 10 → 15
self.ca_entry_threat_level = 1         # 2 → 1
self.path_angle_threshold = np.deg2rad(30)  # 24 → 30
```

### 덜 민감하게 (CA 적게 진입)
```python
self.danger_distance_threshold = 7.0   # 10 → 7
self.ca_entry_threat_level = 3         # 2 → 3
self.path_angle_threshold = np.deg2rad(18)  # 24 → 18
```

### CA 빠른 종료
```python
self.safe_distance_threshold = 15.0    # 25 → 15
self.safe_count_required = 50          # 100 → 50 (1초)
```

### CA 신중한 종료
```python
self.safe_distance_threshold = 30.0    # 25 → 30
self.safe_count_required = 200         # 100 → 200 (4초)
```

---

## 디버깅 팁

### 로그 확인
```bash
# CA 진입/종료 로그
ros2 topic echo /obstacle_flag

# 장애물 정보 시각화 (Foxglove Studio)
/obstacle_markers
/obstacle_info
/cluster_point_cloud
```

### 파라미터 실시간 확인
코드에 로깅 추가:
```python
if most_dangerous:
    self.get_logger().info(
        f"Obstacle: dist={most_dangerous.obstacle_distance:.2f}m, "
        f"bearing={np.degrees(most_dangerous.obstacle_rel_bearing):.1f}°, "
        f"threat_level={most_dangerous.threat_level}"
    )
```

---

## 참고 자료

- **DBSCAN**: [scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- **PCA**: [scikit-learn documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- **ROS2 PointCloud2**: [sensor_msgs_py](https://github.com/ros2/common_interfaces/tree/rolling/sensor_msgs_py)

---

## 버전 히스토리

| 날짜 | 변경 사항 |
|------|-----------|
| 2025-11-03 | `path_angle_threshold` 30° → 24° 완화 |
| 2025-11-03 | `safe_distance_threshold` 25m → 10m 완화 (복원됨) |

---

**작성자**: Claude Code
**마지막 업데이트**: 2025-11-03
