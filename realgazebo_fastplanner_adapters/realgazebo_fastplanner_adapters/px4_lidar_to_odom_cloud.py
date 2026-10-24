#!/usr/bin/env python3
# PX4 상태(NED) + Gazebo LiDAR 를 ★시간정합★ 해 /odom(ENU) + /cloud(world ENU) 동시 합성.
#
# 기존 px4_odom_to_enu + lidar_to_cloud 2노드를 1노드로 통합한다. 두 노드 분리 구조는
# cloud 를 world 로 변환할 때 "최신 TF"(rclpy.time.Time())를, ROG-Map 은 "최신 odom"
# (odom_timeout 2초)을 써서 — 비행(이동) 중 cloud 점이 어긋난 pose 로 등록되는
# 시간 비동기 문제가 있었다(정지=회피, 이동=직진/충돌의 유력 원인).
#
# 이 노드는 FAST-LIO2 식으로 odom 을 시각별 버퍼에 쌓고, ★cloud.header.stamp 시각의
# pose 를 버퍼에서 보간(위치 lerp + 자세 slerp)★ 해 그 pose 로 cloud 를 world 변환한다.
# (gz LiDAR 는 per-point 타임스탬프가 없어 점별 de-skew 는 불가 — 프레임 단위 시각정합.)
#
#   in : /vehicle{N}/fmu/out/vehicle_local_position (px4_msgs, NED, timestamp us)
#        /vehicle{N}/fmu/out/vehicle_attitude        (px4_msgs, body->NED quat, timestamp us)
#        /vehicle{N}/scan/points                     (sensor_msgs/PointCloud2, lidar frame)
#   out: /odom   (nav_msgs/Odometry, world/ENU, child=base_link) — PX4 timestamp 보존
#        /cloud  (sensor_msgs/PointCloud2, world/ENU) — cloud stamp 시각 pose 로 변환
#        TF world->base_link (시각화/호환용)
#
# 좌표: NED->ENU (E=y, N=x, U=-z). 자세 q_enu = q_ned2enu * q_px4 * q_flu2frd.
# lidar 장착: base_link->lidar 정적 변환. ★상방시야 장착(pitch 0, z+0.1, 위 +52°).
import bisect
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformBroadcaster

from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude


# ---- quaternion helpers (w,x,y,z 또는 x,y,z,w 명시) ----
def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# NED->ENU world: 180deg about (1,1,0)/sqrt2 ; FLU->FRD body: 180deg about x
Q_NED2ENU = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
Q_FLU2FRD = (0.0, 1.0, 0.0, 0.0)


def px4_quat_to_enu_wxyz(q_px4_wxyz):
    return quat_mul_wxyz(quat_mul_wxyz(Q_NED2ENU, q_px4_wxyz), Q_FLU2FRD)


def quat_from_rpy_xyzw(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def quat_to_rot_xyzw(qx, qy, qz, qw):
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def slerp_wxyz(q0, q1, t):
    # q0,q1: (w,x,y,z). t in [0,1].
    a = np.array(q0, dtype=float)
    b = np.array(q1, dtype=float)
    d = float(np.dot(a, b))
    if d < 0.0:           # 최단경로
        b = -b
        d = -d
    if d > 0.9995:        # 거의 같으면 선형보간 후 정규화
        r = a + t * (b - a)
        return tuple(r / np.linalg.norm(r))
    th0 = math.acos(max(-1.0, min(1.0, d)))
    s0 = math.sin((1 - t) * th0) / math.sin(th0)
    s1 = math.sin(t * th0) / math.sin(th0)
    r = s0 * a + s1 * b
    return tuple(r / np.linalg.norm(r))


def body_z_wxyz(q):
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class Px4LidarToOdomCloud(Node):
    def __init__(self):
        super().__init__('px4_lidar_to_odom_cloud')
        ns = self.declare_parameter('px4_ns', '/vehicle1').value.rstrip('/')
        self.world_frame = self.declare_parameter('world_frame', 'world').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.publish_tf = bool(self.declare_parameter('publish_tf', True).value)
        self.buf_sec = float(self.declare_parameter('pose_buffer_sec', 1.0).value)
        # PX4 timestamp 와 lidar stamp 의 시간축이 다른 환경(AirSim: PX4=boot시각,
        # lidar=벽시계)에서는 수신 시각(ROS clock)으로 정렬한다.
        self.use_ros_time = bool(self.declare_parameter('use_ros_time', False).value)
        # free-ray 합성 (반환 없는 방향을 raycast_range_max 너머 endpoint 로 채워
        # rog_map 이 하늘/개활지를 known-free 로 마킹하게 함)
        self.synth_free_rays = bool(self.declare_parameter('synth_free_rays', False).value)
        self.synth_range = float(self.declare_parameter('synth_range', 150.0).value)
        self.synth_az_bins = int(self.declare_parameter('synth_az_bins', 90).value)
        self.synth_el_bins = int(self.declare_parameter('synth_el_bins', 16).value)
        self.synth_every_n = int(self.declare_parameter('synth_every_n', 4).value)
        self._cloud_cnt = 0
        # AirSim's ROS bridge polls the sensor faster than it produces scans and
        # republishes the latest scan unchanged. Registering every copy with the
        # pose at its own receive time smears one scan along the flight path
        # (v * poll gap per copy), so only the first copy of each stamp is used.
        self.drop_duplicate_scans = bool(self.declare_parameter('drop_duplicate_scans', True).value)
        # use_ros_time: time from scan capture to its first receipt [s]; the pose is
        # looked up that far back.
        self.cloud_latency = float(self.declare_parameter('cloud_latency_s', 0.0).value)
        # Returns closer than this to the sensor are dropped. AirSim reports a ray that
        # hit nothing as (0,0,0), i.e. a point at the sensor origin. rog_map without
        # raycasting only ignores points within raycast_range_min of the LATEST odom, so
        # at speed those origin points (registered at the scan pose, v * latency behind)
        # were inserted as obstacles trailing the vehicle - planning starts then sat in
        # phantom occupancy (CIRI 'min dis to obstacle', 'start point deeply occupied').
        self.min_range = float(self.declare_parameter('min_range', 0.5).value)
        # Motion de-skew (AirSim registration only). The AirSim lidar stores every point in
        # the sensor frame of the tick it was shot in, sweeps the azimuth 0 -> 360 deg (NED,
        # i.e. clockwise) over one rotation and stamps the cloud at the END of it. Registering
        # the whole sweep with the end pose shifts the oldest points (just right of straight
        # ahead) by the motion over a full rotation - 0.7 m at 7 m/s, plus yaw * range.
        # Here each point is registered with the pose at its own shot time
        # stamp - rot_period * (1 - azimuth/360). 0 = off.
        self.deskew_period = float(self.declare_parameter('deskew_rot_period_s', 0.0).value)
        self.deskew_bins = int(self.declare_parameter('deskew_bins', 24).value)
        # deskew_mode 'index': a pattern lidar (A4VAI-Unreal Mid-360 mode, like the real Livox) sends
        # every shot of the frame in firing order, misses as NaN, stamped at the last shot: shot i of
        # n left at stamp - (n - 1 - i) / deskew_pps, so the frame spans n / deskew_pps and its time
        # comes from the index, not the azimuth (which turns ~178 times per frame). 'azimuth' = the
        # rotating grid lidar above. deskew_rot_period_s > 0 switches de-skew on in both modes.
        self.deskew_mode = str(self.declare_parameter('deskew_mode', 'azimuth').value)
        self.deskew_pps = float(self.declare_parameter('deskew_pps', 200000.0).value)
        # deskew_mode 'field': the real lidar (livox_ros_driver2, xfer_format 0) gives every point its
        # own time (time_field, x time_field_scale -> s; ns for Livox). Its clock is the lidar's, so
        # only the offsets within the frame are used: the frame ends at receipt - cloud_latency_s
        # (use_ros_time), and each point is registered with the PX4 pose at its own time (no
        # simulator pose on the vehicle). The tilt gate (max_tilt_rate) uses the same PX4 poses.
        # De-skew is on with deskew_rot_period_s > 0, as in the other modes (its value is unused here).
        self.time_field = str(self.declare_parameter('time_field', 'timestamp').value)
        self.time_field_scale = float(self.declare_parameter('time_field_scale', 1e-9).value)
        self._warned_span = False
        # self_radius_m: a lidar on the vehicle also sees the vehicle (masts, antennas, cables, props): returns
        # within this horizontal distance of the vehicle centre and self_z_min_m .. self_z_max_m of its height
        # (body frame) are left out. 0 = off (simulator). The X500's Mid-360 saw ~2000 returns per frame within
        # 0.5 m on the bench and the policy then always had an obstacle at 0.51 m.
        self.self_r = float(self.declare_parameter('self_radius_m', 0.0).value)
        self.self_zmin = float(self.declare_parameter('self_z_min_m', -0.5).value)
        self.self_zmax = float(self.declare_parameter('self_z_max_m', 0.5).value)
        # Scans shot while the vehicle tilts faster than this [rad/s] (roll / pitch, over the
        # sweep; AirSim registration with de-skew only) are dropped. The mirrored AirSim pose
        # is a frame or two off the scan, so in a hard brake (1-2 rad/s) the ground 10-15 m
        # out was registered ~1 m up: phantom obstacles at 0.7-0.9 m AGL that SUPER parked
        # next to and could not plan out of for 190 s (ctrack_123_j). Normal flight exceeds
        # 0.8 rad/s 1-2 % of the time. 0 = off.
        self.max_tilt_rate = float(self.declare_parameter('max_tilt_rate', 0.0).value)
        self._n_tilt_drop = 0
        # gated_topic (optional): every scan still goes to /cloud, and only the scans that pass the
        # tilt gate also go here - a consumer that needs every scan (super_speed_guard.py tracks
        # moving objects; gaps restarted the RC cart's track, ctrack_s900_c) reads /cloud and
        # passes this one on to the planner. '' = tilting scans are dropped from /cloud itself.
        self.gated_topic = str(self.declare_parameter('gated_topic', '').value)
        # publish_empty_scans: a scan with no returns goes out as an empty cloud instead of being
        # skipped, so the consumer knows the lidar saw nothing rather than keeping its last scan
        # and, with a watchdog, taking the silence for a dead lidar. A 10 m lidar (the CA policy's
        # range) 1.5 m up sees no ground and, between obstacles, nothing at all: /cloud fell silent
        # for 2.7 s and the policy node held twice (ctrack_big_h). Off by default (SUPER as flown).
        self.publish_empty = bool(self.declare_parameter('publish_empty_scans', False).value)
        self._tilting = False
        self._last_scan_stamp = None
        # cloud stamp 가 버퍼 범위를 벗어나면(extrapolation) 허용 한계[s]. 넘으면 최신 pose 로 폴백.
        self.max_extrap = float(self.declare_parameter('max_extrapolation_sec', 0.1).value)
        # lidar 장착(base_link->lidar) 보정
        self.lidar_off = np.array([
            float(self.declare_parameter('lidar_offset_x', 0.0).value),
            float(self.declare_parameter('lidar_offset_y', 0.0).value),
            float(self.declare_parameter('lidar_offset_z', 0.1).value),
        ])
        roll = float(self.declare_parameter('lidar_roll', 0.0).value)
        pitch = float(self.declare_parameter('lidar_pitch', 0.0).value)
        yaw = float(self.declare_parameter('lidar_yaw', 0.0).value)
        qx, qy, qz, qw = quat_from_rpy_xyzw(roll, pitch, yaw)
        self.R_base_lidar = quat_to_rot_xyzw(qx, qy, qz, qw)   # lidar->base_link 회전

        # pose 버퍼: (t_sec, pos_enu[3], quat_enu_wxyz[4])
        self.buf = deque()
        self.q_enu = (1.0, 0.0, 0.0, 0.0)
        self.have_att = False

        self.create_subscription(VehicleAttitude, f'{ns}/fmu/out/vehicle_attitude',
                                 self.att_cb, qos_profile_sensor_data)
        self.create_subscription(VehicleLocalPosition, f'{ns}/fmu/out/vehicle_local_position',
                                 self.pos_cb, qos_profile_sensor_data)
        in_topic = self.declare_parameter('in_topic', f'{ns}/scan/points').value
        self.create_subscription(PointCloud2, in_topic, self.cloud_cb, qos_profile_sensor_data)

        # Register each scan with the simulator's own vehicle pose at the scan stamp
        # (same clock) instead of the PX4 estimate at receive time. In AirSim-SITL the
        # AirSim vehicle pose is relayed from Gazebo ~0.1 s late, so PX4-pose
        # registration shifts the map along the flight path by speed * delay. The
        # AirSim odom is in Gazebo ENU world axes; its offset to the PX4 local frame
        # is calibrated while the vehicle stands still before take-off.
        self.airsim_odom_topic = self.declare_parameter('airsim_odom_topic', '').value
        self.abuf = deque()
        self.T_air2px4 = None
        self._calib = []
        self._px4_pos = None
        self._px4_speed = None
        self._warned_air = False
        if self.airsim_odom_topic:
            self.create_subscription(Odometry, self.airsim_odom_topic, self.air_cb, qos_profile_sensor_data)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/cloud', qos_profile_sensor_data)
        self.gated_pub = (self.create_publisher(PointCloud2, self.gated_topic, qos_profile_sensor_data)
                          if self.gated_topic else None)
        self.tf_bc = TransformBroadcaster(self) if self.publish_tf else None
        self._logged_odom = False
        self._logged_cloud = False
        self._warned_buf = False
        self.get_logger().info(
            f'[px4_lidar_to_odom_cloud] up. ns={ns} in={in_topic} '
            f'buffer={self.buf_sec}s max_extrap={self.max_extrap}s')

    # ---- PX4 attitude (FRD->NED) -> ENU body quat ----
    def att_cb(self, msg: VehicleAttitude):
        q = (float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3]))
        self.q_enu = tuple(float(c) for c in px4_quat_to_enu_wxyz(q))
        self.have_att = True

    # ---- PX4 local position (NED) -> ENU odom + 버퍼 적재 ----
    def pos_cb(self, msg: VehicleLocalPosition):
        if not (msg.xy_valid and msg.z_valid) or not self.have_att:
            return
        # ★ PX4 timestamp(us, sim_time) 를 ROS stamp 로 — lidar stamp 와 같은 축.
        # use_ros_time 이면 수신 시각(ROS clock)을 사용 (AirSim: 시간축이 다름).
        if self.use_ros_time:
            now = self.get_clock().now()
            stamp = now.to_msg()
            t_sec = now.nanoseconds * 1e-9
        else:
            t_us = int(msg.timestamp)
            stamp = Time(nanoseconds=t_us * 1000).to_msg()
            t_sec = t_us * 1e-6

        e, n, u = float(msg.y), float(msg.x), float(-msg.z)
        ve, vn, vu = float(msg.vy), float(msg.vx), float(-msg.vz)
        pos = np.array([e, n, u])
        qw, qx, qy, qz = self.q_enu
        self._px4_pos = pos
        self._px4_speed = float(np.linalg.norm([ve, vn, vu]))

        # 버퍼 적재 + 오래된 항목 제거
        self.buf.append((t_sec, pos, (qw, qx, qy, qz)))
        while len(self.buf) > 2 and self.buf[-1][0] - self.buf[0][0] > self.buf_sec:
            self.buf.popleft()

        odom = Odometry()
        odom.header.stamp = stamp                 # ★ PX4 timestamp 보존
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = e
        odom.pose.pose.position.y = n
        odom.pose.pose.position.z = u
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.twist.twist.linear.x = ve
        odom.twist.twist.linear.y = vn
        odom.twist.twist.linear.z = vu
        self.odom_pub.publish(odom)

        if self.tf_bc is not None:
            tf = TransformStamped()
            tf.header = odom.header
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = e
            tf.transform.translation.y = n
            tf.transform.translation.z = u
            tf.transform.rotation = odom.pose.pose.orientation
            self.tf_bc.sendTransform(tf)

        if not self._logged_odom:
            self._logged_odom = True
            self.get_logger().info(f'[sync] first odom ENU=({e:.2f},{n:.2f},{u:.2f})')

    # ---- cloud.stamp 시각의 pose 를 버퍼에서 보간 ----
    def air_cb(self, msg: Odometry):
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        if self.abuf and st <= self.abuf[-1][0]:
            return
        self.abuf.append((st, pos, (q.w, q.x, q.y, q.z)))
        while len(self.abuf) > 2 and self.abuf[-1][0] - self.abuf[0][0] > self.buf_sec:
            self.abuf.popleft()
        # Calibrate on ~0.5 s of (near-)standstill. The app can arm < 2 s after start, so
        # a longer ground window was sometimes missed; the take-off hover then serves
        # (AirSim lags PX4 ~0.1 s, negligible at < 0.1 m/s).
        if self.T_air2px4 is None and self._px4_pos is not None and self._px4_speed < 0.1:
            self._calib.append(self._px4_pos - pos)
            if len(self._calib) >= 50:
                c = np.array(self._calib[-50:])
                if float(np.max(np.ptp(c, axis=0))) < 0.05:
                    self.T_air2px4 = np.median(c, axis=0)
                    d = quat_mul_wxyz(self.q_enu, (q.w, -q.x, -q.y, -q.z))
                    yaw_diff = math.degrees(2.0 * math.atan2(d[3], d[0]))
                    self.get_logger().info(
                        f'[airsim] AirSim->PX4 offset {np.round(self.T_air2px4, 3).tolist()} '
                        f'(yaw diff PX4 vs AirSim {yaw_diff:+.2f} deg); scans now use the AirSim pose')

    def _air_pose_at(self, st):
        if len(self.abuf) < 2:
            return None
        ts = [b[0] for b in self.abuf]
        if st < ts[0] - self.max_extrap or st > ts[-1] + self.max_extrap:
            return None
        if st <= ts[0]:
            return self.abuf[0][1], self.abuf[0][2]
        if st >= ts[-1]:
            return self.abuf[-1][1], self.abuf[-1][2]
        i = bisect.bisect_left(ts, st)
        t0, p0, q0 = self.abuf[i - 1]
        t1, p1, q1 = self.abuf[i]
        a = (st - t0) / (t1 - t0) if t1 > t0 else 0.0
        return p0 + a * (p1 - p0), slerp_wxyz(q0, q1, a)

    def _pose_at(self, t_query):
        if not self.buf:
            return None
        ts = [b[0] for b in self.buf]
        if t_query <= ts[0]:
            if ts[0] - t_query > self.max_extrap:
                return None
            return self.buf[0][1], self.buf[0][2]
        if t_query >= ts[-1]:
            if t_query - ts[-1] > self.max_extrap:
                return None
            return self.buf[-1][1], self.buf[-1][2]
        i = bisect.bisect_left(ts, t_query)
        t0, p0, q0 = self.buf[i - 1]
        t1, p1, q1 = self.buf[i]
        a = (t_query - t0) / (t1 - t0) if t1 > t0 else 0.0
        pos = p0 + a * (p1 - p0)
        quat = slerp_wxyz(q0, q1, a)
        return pos, quat

    def _frame_period(self, msg):
        """time one frame spans [s]: the rotation period, or n shots / pps for a pattern lidar"""
        if self.deskew_mode == 'index':
            return max(msg.width * msg.height, 1) / self.deskew_pps
        return self.deskew_period

    def cloud_cb(self, msg: PointCloud2):
        self._tilting = False
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if self.drop_duplicate_scans:
            if stamp == self._last_scan_stamp:
                return
            self._last_scan_stamp = stamp
        # use_ros_time: AirSim 브리지의 stamp 는 시뮬 지연이 누적되어 벽시계에서
        # 수십 초씩 밀릴 수 있다(데이터는 실시간, stamp 만 엉터리). 수신 시각으로 정렬.
        if self.use_ros_time:
            t_query = self.get_clock().now().nanoseconds * 1e-9 - self.cloud_latency
        else:
            t_query = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        res = self._pose_at(t_query)
        if res is None:
            if not self._warned_buf:
                self._warned_buf = True
                self.get_logger().warn(
                    '[sync] cloud stamp 가 odom 버퍼 범위 밖 (extrapolation 초과) — '
                    'odom 수신/시각 정렬 확인. 이 프레임 스킵.')
            return
        pos_enu, quat_enu_wxyz = res
        # until the offset is calibrated the PX4-pose registration above is used
        if self.airsim_odom_topic and self.T_air2px4 is not None:
            air = self._air_pose_at(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if air is None:
                if not self._warned_air:
                    self._warned_air = True
                    self.get_logger().warn('[airsim] scan stamp outside the AirSim pose buffer - frame skipped')
                return
            pos_enu, quat_enu_wxyz = air[0] + self.T_air2px4, air[1]
            if self.max_tilt_rate > 0.0 and self.deskew_period > 0.0:
                st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                start = self._air_pose_at(st - self._frame_period(msg))
                if start is not None:
                    c = float(np.clip(body_z_wxyz(start[1]) @ body_z_wxyz(air[1]), -1.0, 1.0))
                    rate = math.acos(c) / self._frame_period(msg)
                    self._tilting = rate > self.max_tilt_rate
                    if self._tilting:
                        self._n_tilt_drop += 1
                        if self._n_tilt_drop % 20 == 1:
                            self.get_logger().info(f'[airsim] tilting {rate:.2f} rad/s > {self.max_tilt_rate} - '
                                                   f'scan {"kept from " + self.gated_topic if self.gated_topic else "dropped"} '
                                                   f'({self._n_tilt_drop} so far)')
                        if not self.gated_topic:
                            return

        # AirSim lidar has only x,y,z; Gazebo adds intensity. Read what exists.
        names = {f.name for f in msg.fields}
        have_intensity = 'intensity' in names
        fields = ('x', 'y', 'z', 'intensity') if have_intensity else ('x', 'y', 'z')
        index_mode = self.deskew_mode == 'index'
        field_mode = self.deskew_mode == 'field' and not self.airsim_odom_topic
        ts_all = None
        if field_mode:
            # Livox mixes float32 x/y/z, uint8 tag/line and a float64 time: read_points_numpy refuses
            # mixed types (all of the cloud's fields), so read the structured points and stack them
            want = fields + ((self.time_field,) if self.time_field in names else ())
            arr = pc2.read_points(msg, field_names=want, skip_nans=False)
            pts = np.column_stack([np.asarray(arr[c], dtype=np.float64) for c in fields])
            if self.time_field in names:
                ts_all = np.asarray(arr[self.time_field], dtype=np.float64) * self.time_field_scale
        else:
            # index mode keeps the NaN shots until here: their position in the frame is their time
            pts = pc2.read_points_numpy(msg, field_names=fields, skip_nans=not index_mode)
        if pts.size == 0:
            return
        span = 0.0
        if field_mode:
            ok = np.isfinite(ts_all) if ts_all is not None else np.zeros(0, dtype=bool)
            span = float(ts_all[ok].max() - ts_all[ok].min()) if ok.any() else 0.0
            if not 0.0 < span < 0.5:
                if not self._warned_span:
                    self._warned_span = True
                    self.get_logger().warn(f'[field] frame time span {span:.3f} s from {self.time_field} - '
                                           f'registered without de-skew (check time_field / time_field_scale)')
                span = 0.0
            if self.max_tilt_rate > 0.0 and span > 0.0:
                start = self._pose_at(t_query - span)
                if start is not None:
                    c = float(np.clip(body_z_wxyz(start[1]) @ body_z_wxyz(quat_enu_wxyz), -1.0, 1.0))
                    rate = math.acos(c) / span
                    self._tilting = rate > self.max_tilt_rate
                    if self._tilting:
                        self._n_tilt_drop += 1
                        if self._n_tilt_drop % 20 == 1:
                            self.get_logger().info(f'[field] tilting {rate:.2f} rad/s > {self.max_tilt_rate} - '
                                                   f'scan {"kept from " + self.gated_topic if self.gated_topic else "dropped"} '
                                                   f'({self._n_tilt_drop} so far)')
                        if not self.gated_topic:
                            return
        xyz = pts[:, :3].astype(np.float64)
        n_all = xyz.shape[0]
        if have_intensity:
            inten = pts[:, 3].astype(np.float32)
        else:
            inten = np.zeros(xyz.shape[0], dtype=np.float32)
        keep = np.isfinite(xyz).all(axis=1)
        if self.min_range > 0.0:
            keep &= np.einsum('ij,ij->i', np.nan_to_num(xyz), np.nan_to_num(xyz)) >= self.min_range ** 2
        xyz = xyz[keep]
        inten = inten[keep]
        shot = np.flatnonzero(keep)                  # index of each kept point in the frame
        if xyz.shape[0] == 0 and not self.publish_empty:
            return

        # ★ free-ray 합성: 반환이 없는 방향(하늘 등)은 어떤 포인트도 없어 rog_map 에서
        #   영원히 UNKNOWN 으로 남는다. SUPER corridor 는 unknown 을 위험 취급하므로
        #   known-free 공간이 "지면에 맞은 레이 쐐기"뿐이 되어 궤적이 지면으로 끌려간다.
        #   해결: 스캔 패턴에서 비어 있는 (방위각,고도각) 빈에 raycast_range_max(100m)
        #   너머(150m) 가상 endpoint 를 추가 → rog_map 은 update_hit=false 로 경로만
        #   free 마킹 (prob_map.cpp:743-747), 팬텀 장애물 없음.
        if self.synth_free_rays and (self._cloud_cnt % self.synth_every_n == 0):
            r = np.linalg.norm(xyz, axis=1)
            valid = r > 1e-3
            az = np.arctan2(xyz[valid, 1], xyz[valid, 0])
            el = np.arcsin(np.clip(xyz[valid, 2] / r[valid], -1.0, 1.0))
            az_i = ((az + np.pi) / (2 * np.pi) * self.synth_az_bins).astype(int) % self.synth_az_bins
            el_lim = np.deg2rad(16.0)
            el_i = np.clip(((el + el_lim) / (2 * el_lim) * self.synth_el_bins).astype(int),
                           0, self.synth_el_bins - 1)
            occ = np.zeros((self.synth_az_bins, self.synth_el_bins), dtype=bool)
            occ[az_i, el_i] = True
            e_az, e_el = np.nonzero(~occ)
            if e_az.size:
                s_az = (e_az + 0.5) / self.synth_az_bins * 2 * np.pi - np.pi
                s_el = (e_el + 0.5) / self.synth_el_bins * 2 * el_lim - el_lim
                ce = np.cos(s_el)
                synth = self.synth_range * np.stack(
                    [ce * np.cos(s_az), ce * np.sin(s_az), np.sin(s_el)], axis=1)
                xyz = np.vstack([xyz, synth])
                inten = np.concatenate([inten, np.zeros(synth.shape[0], dtype=np.float32)])
                shot = np.concatenate([shot, np.full(synth.shape[0], n_all - 1)])
        self._cloud_cnt += 1

        # lidar 센서프레임 -> base_link : R_base_lidar · p + lidar_offset
        p_base = xyz @ self.R_base_lidar.T + self.lidar_off
        if self.self_r > 0.0 and p_base.shape[0]:
            own = ((np.einsum('ij,ij->i', p_base[:, :2], p_base[:, :2]) < self.self_r ** 2)
                   & (p_base[:, 2] > self.self_zmin) & (p_base[:, 2] < self.self_zmax))
            xyz, inten, shot, p_base = xyz[~own], inten[~own], shot[~own], p_base[~own]
        # base_link -> world(ENU) : R_world_base(보간된 자세) · p + pos(보간된 위치)
        qw, qx, qy, qz = quat_enu_wxyz
        R_wb = quat_to_rot_xyzw(qx, qy, qz, qw)
        p_world = (p_base @ R_wb.T + pos_enu).astype(np.float32)
        if (self.deskew_period > 0.0 and self.airsim_odom_topic and self.T_air2px4 is not None
                and xyz.shape[0]):
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            period = self._frame_period(msg)
            if index_mode:
                # fraction of the frame elapsed at each shot (the last one at the stamp)
                frac = (shot + 1.0) / max(n_all, 1)
            else:
                # published frame is FLU, the sweep angle is the NED (clockwise) azimuth
                frac = (np.arctan2(-xyz[:, 1], xyz[:, 0]) % (2.0 * np.pi)) / (2.0 * np.pi)
            b = np.minimum((frac * self.deskew_bins).astype(int), self.deskew_bins - 1)
            for k in np.unique(b):
                pose = self._air_pose_at(st - period * (1.0 - (k + 0.5) / self.deskew_bins))
                if pose is None:
                    continue
                qw, qx, qy, qz = pose[1]
                sel = b == k
                p_world[sel] = (p_base[sel] @ quat_to_rot_xyzw(qx, qy, qz, qw).T
                                + pose[0] + self.T_air2px4).astype(np.float32)
        if field_mode and self.deskew_period > 0.0 and span > 0.0 and xyz.shape[0]:
            ts = ts_all[shot]                        # synthetic rays carry the last point's time
            ts = np.where(np.isfinite(ts), ts, np.nanmax(ts_all))
            frac = (ts - (np.nanmax(ts_all) - span)) / span
            b = np.clip((frac * self.deskew_bins).astype(int), 0, self.deskew_bins - 1)
            for k in np.unique(b):
                pose = self._pose_at(t_query - span * (1.0 - (k + 0.5) / self.deskew_bins))
                if pose is None:
                    continue
                qw, qx, qy, qz = pose[1]
                sel = b == k
                p_world[sel] = (p_base[sel] @ quat_to_rot_xyzw(qx, qy, qz, qw).T
                                + pose[0]).astype(np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        structured = np.zeros(p_world.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32),
            ('z', np.float32), ('intensity', np.float32)])
        structured['x'] = p_world[:, 0]
        structured['y'] = p_world[:, 1]
        structured['z'] = p_world[:, 2]
        structured['intensity'] = inten
        out = pc2.create_cloud(msg.header, fields, structured)
        out.header.frame_id = self.world_frame
        if self.use_ros_time:
            # 밀린 원본 stamp 를 내보내면 rog_map/foxglove 쪽 시간 비교가 깨진다.
            out.header.stamp = self.get_clock().now().to_msg()
        else:
            out.header.stamp = msg.header.stamp   # ★ cloud 원본 시각 보존
        self.cloud_pub.publish(out)
        if self.gated_pub is not None and not self._tilting:
            self.gated_pub.publish(out)
        self._tilting = False

        if not self._logged_cloud:
            self._logged_cloud = True
            self.get_logger().info(
                f'[sync] first cloud: {p_world.shape[0]} pts @ pose '
                f'({pos_enu[0]:.1f},{pos_enu[1]:.1f},{pos_enu[2]:.1f}) '
                f'dt(cloud-pose)={abs(t_query - self.buf[-1][0])*1000:.0f}ms')


def main():
    rclpy.init()
    node = Px4LidarToOdomCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
