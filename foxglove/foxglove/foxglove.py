import math
import struct
import numpy as np
import random

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_geometry_msgs import do_transform_point

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped

from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from std_msgs.msg import Bool

from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude

from custom_msgs.msg import LocalWaypointSetpoint

from rclpy.duration import Duration


@dataclass
class ObstacleCluster:
    """단일 장애물 클러스터 정보"""
    cluster_id: int
    points: np.ndarray  # (N, 3) 형태의 포인트들
    obstacle_position: np.ndarray  # (3,) 중심점
    obstacle_distance: float
    obstacle_rel_bearing: float
    obb_rotation: np.ndarray = None  # (3, 3) OBB 회전 행렬
    obb_size: np.ndarray = None  # (3,) OBB 크기 [x, y, z]
    is_target_obstacle: bool = False
    is_dangerous: bool = False
    is_in_path: bool = False
    threat_level: int = 0


class FoxgloveNode(Node):
    def __init__(self):
        super().__init__('foxglove_node')
        self.get_logger().info("FoxgloveNode initialized")

        # region: Variable Declaration
        self.vehicle_pose = PoseStamped()
        self.vehicle_pose.header.frame_id = 'world'
        self.vehicle_pose.pose.position.x = 0.0
        self.vehicle_pose.pose.position.y = 0.0
        self.vehicle_pose.pose.position.z = 0.0
        self.vehicle_pose.pose.orientation.x = 0.0
        self.vehicle_pose.pose.orientation.y = 0.0
        self.vehicle_pose.pose.orientation.z = 0.0
        self.vehicle_pose.pose.orientation.w = 1.0

        self.vehicle_velocity = Twist()
        self.vehicle_velocity.linear.x = 0.0
        self.vehicle_velocity.linear.y = 0.0
        self.vehicle_velocity.linear.z = 0.0
        self.vehicle_velocity.angular.x = 0.0
        self.vehicle_velocity.angular.y = 0.0
        self.vehicle_velocity.angular.z = 0.0

        self.vehicle_position_np = np.array([0.0, 0.0, 0.0])
        self.vehicle_velocity_np = np.array([0.0, 0.0, 0.0])

        self.vehicle_path = Path()
        self.vehicle_path.header.frame_id = 'world'
        self.vehicle_path.poses = []

        self.heading_enu = 0.0

        # Waypoint visualization
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_z = []

        # for fov cone visualization
        self.global_points = []
        self.counter = 0

        self.safety_distance = 1.0
        self.max_detection_range = 20.0
        self.obstacle_flag = False

        self.latest_lidar_points_np = np.array([])
        self.filtered_points_np = np.array([])
        self.world_points_np = np.array([])

        self.obstacle_info: Dict[int, ObstacleCluster] = {}


        self.danger_distance_threshold = 10.0  # m
        self.warning_distance_threshold = 15.0  # m
        self.path_angle_threshold = np.deg2rad(30)  # 30도
        self.velocity_threshold = 5.0  # m/s (접근 속도)

        # Collision Avoidance State Variables
        self.avoidance_required = False  # 회피가 필요한 상태
        self.avoidance_completed = False  # 회피 완료 상태
        self.target_obstacle_id = None  # 회피 중인 타겟 장애물 ID
        self.previous_target_distance = None  # 이전 타겟 장애물 거리
        self.safe_distance_count = 0  # 안전 거리 유지 카운트 (회피 완료 판단용)
        self.safe_distance_threshold = 25.0  # m (회피 가능 거리 15m + 안전 마진 10m)
        self.safe_count_required = 100  # 안전 상태 유지 횟수 (라이다 주기 * 100 = 약 2초)
        self.safe_angle_threshold = np.deg2rad(90)  # 장애물이 측면/후방으로 벗어났는지 확인 (90도)

        # Hysteresis for obstacle flag (떨림 방지)
        self.ca_entry_threat_level = 2  # CA 진입 임계값
        self.ca_exit_threat_level = 1   # CA 종료 임계값 (히스테리시스)


        # endregion

        # region: QoS Profile Declaration
        self.qos_profile_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.qos_profile_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.qos_profile_default = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # endregion
        
        # region: Publisher Declaration
        self.vehicle_path_publisher_ = self.create_publisher(Path, '/vehicle_path', self.qos_profile_default)
        self.vehicle_pose_publisher_ = self.create_publisher(PoseStamped, '/drone_pose', self.qos_profile_default)
        self.vehicle_velocity_publisher_ = self.create_publisher(Twist, '/drone_velocity', self.qos_profile_default)
        self.vehicle_fov_publisher_ = self.create_publisher(Marker, '/fov_cone', self.qos_profile_default)
        
        self.world_points_publisher_ = self.create_publisher(PointCloud2, "/world_points", self.qos_profile_default)
        self.filtered_point_cloud_publisher_ = self.create_publisher(PointCloud2, "/filtered_point_cloud", 10)
        self.cluster_point_cloud_publisher_ = self.create_publisher(PointCloud2, "/cluster_point_cloud", self.qos_profile_default)
        self.obstacle_marker_publisher_ = self.create_publisher(MarkerArray, "/obstacle_markers", self.qos_profile_default)
        
        self.obstacle_info_publisher_ = self.create_publisher(MarkerArray, "/obstacle_info", self.qos_profile_default)
        self.obstacle_flag_publisher_ = self.create_publisher(Bool, "/obstacle_flag", 1)

        # Waypoint visualization
        self.waypoint_marker_publisher_ = self.create_publisher(MarkerArray, "/waypoint_markers", self.qos_profile_default)
        # endregion


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_br = TransformBroadcaster(self)

        self.reentrant_group = ReentrantCallbackGroup()


        # region: Subscriber Declaration
        self.px4_vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition, 
            '/fmu/out/vehicle_local_position', 
            self.vehicle_local_position_callback, 
            self.qos_profile_px4
        )

        self.px4_attitude_sub = self.create_subscription(
            VehicleAttitude, 
            '/fmu/out/vehicle_attitude', 
            self.attitude_callback, 
            self.qos_profile_px4
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            "/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3",
            self.update_latest_lidar_msg,
            self.qos_profile_lidar
        )

        # Waypoint subscription
        self.waypoint_sub = self.create_subscription(
            LocalWaypointSetpoint,
            "/local_waypoint_setpoint_to_plotter",
            self.waypoint_callback,
            1
        )
        # endregion

        # region: Timer Declaration
        self.tf_publish_timer_ = self.create_timer(0.01, self.publish_world_to_simpleflight_tf)
        # self.foxglove_publish_timer_ = self.create_timer(0.01, self.publish_foxglove_data)
        # endregion

    # region: Subscriber Callback Functions
    def vehicle_local_position_callback(self, msg):
        self.vehicle_pose.pose.position.x, self.vehicle_pose.pose.position.y, self.vehicle_pose.pose.position.z = self.ned_to_enu(msg.x, msg.y, msg.z)

        self.vehicle_velocity.linear.x, self.vehicle_velocity.linear.y, self.vehicle_velocity.linear.z = self.ned_to_enu(msg.vx, msg.vy, msg.vz)

        self.vehicle_position_np = np.array([self.vehicle_pose.pose.position.x, self.vehicle_pose.pose.position.y, self.vehicle_pose.pose.position.z])
        self.vehicle_velocity_np = np.array([self.vehicle_velocity.linear.x, self.vehicle_velocity.linear.y, self.vehicle_velocity.linear.z])

        self.heading_enu = (math.pi / 2) - msg.heading

        self.heading_enu = math.atan2(
            math.sin(self.heading_enu),
            math.cos(self.heading_enu)
        )

        current_pose = PoseStamped()
        current_pose.header.stamp = self.get_clock().now().to_msg()
        current_pose.header.frame_id = 'world'
        current_pose.pose.position.x = self.vehicle_pose.pose.position.x
        current_pose.pose.position.y = self.vehicle_pose.pose.position.y
        current_pose.pose.position.z = self.vehicle_pose.pose.position.z
        self.vehicle_path.poses.append(current_pose)

    def attitude_callback(self, msg):
        # PX4 quaternion: [w, x, y, z] 
        # Quaternion rotation from the FRD body frame to the NED earth frame

        q_px4 = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]

        q_frd_to_ned = [q_px4[1], q_px4[2], q_px4[3], q_px4[0]]
        q_ned_to_enu = R.from_euler('ZYX', [np.pi/2, 0, np.pi], degrees=False).as_quat()

        R_frd_to_ned = R.from_quat(q_frd_to_ned)
        R_ned_to_enu = R.from_quat(q_ned_to_enu)

        R_frd_to_enu = R_ned_to_enu * R_frd_to_ned
        R_frd_to_flu = R.from_euler('XYZ', [np.pi, 0, 0])

        R_flu_to_enu = R_frd_to_enu * R_frd_to_flu

        # euler_ros = R_flu_to_enu.as_euler('XYZ', degrees=True)

        
        # if abs(euler_ros[2]) > np.pi/4:
        #     euler_ros[1] *= -1

        # R_flu_to_enu_corrected = R.from_euler('XYZ', euler_ros, degrees=True)
        # q_ros = R_flu_to_enu_corrected.as_quat()

        q_ros = R_flu_to_enu.as_quat()

        self.vehicle_pose.pose.orientation.x = q_ros[0]
        self.vehicle_pose.pose.orientation.y = q_ros[1]
        self.vehicle_pose.pose.orientation.z = q_ros[2]
        self.vehicle_pose.pose.orientation.w = q_ros[3]

    def waypoint_callback(self, msg):
        """Receive waypoints from plotter topic and convert NED to ENU"""
        # Convert NED to ENU
        # NED: x=North, y=East, z=Down
        # ENU: x=East, y=North, z=Up
        self.waypoint_x = [y for y in msg.waypoint_y]  # East = y_ned
        self.waypoint_y = [x for x in msg.waypoint_x]  # North = x_ned
        self.waypoint_z = [z for z in msg.waypoint_z]  # Up = -z_ned

    def update_latest_lidar_msg(self, pc_msg):
        '''
        pc_msg : sensor_msgs/PointCloud2.msg
            header: std_msgs/Header
            height: uint32
            width: uint32
            fields: sensor_msgs/PointField[]
            is_bigendian: bool
            point_step: uint32
            row_step: uint32
            data: uint8[]
            is_dense: bool
        '''
        if pc_msg.is_dense:
            self.latest_lidar_points_np = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)

        self.publish_world_to_simpleflight_tf()
        self.process_lidar_points()
        self.publish_foxglove_data()
    # endregion

    # region: Publisher Functions
    def publish_vehicle_pose(self):
        self.vehicle_pose.header.stamp = self.get_clock().now().to_msg()
        self.vehicle_pose_publisher_.publish(self.vehicle_pose)

    def publish_vehicle_path(self):
        msg = Path()
        self.vehicle_path.header.stamp = self.get_clock().now().to_msg()
        self.vehicle_path_publisher_.publish(self.vehicle_path)

    def publish_vehicle_velocity(self):
        self.vehicle_velocity_publisher_.publish(self.vehicle_velocity)

    def publish_fov_cone(self):
        yaw = self.heading_enu
        pitch = 0.0
        position = (self.vehicle_pose.pose.position.x, self.vehicle_pose.pose.position.y, self.vehicle_pose.pose.position.z)
        fov_marker = self.create_fov_cone_marker(position, yaw, pitch)
        self.vehicle_fov_publisher_.publish(fov_marker)

    def publish_filtered_point_cloud(self):
        if self.filtered_points_np is None or len(self.filtered_points_np) == 0:
            return
        header = Header()
        header.frame_id = "SimpleFlight/RPLIDAR_A3"
        header.stamp = self.get_clock().now().to_msg()
        msg = pc2.create_cloud_xyz32(header, self.filtered_points_np)
        self.filtered_point_cloud_publisher_.publish(msg)
    def publish_world_points(self):
        if self.world_points_np is None or len(self.world_points_np) == 0:
            return
        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()
        msg = pc2.create_cloud_xyz32(header, self.world_points_np)
        self.world_points_publisher_.publish(msg)

    def publish_cluster_points(self):
        if not self.obstacle_info:
            return

        # 모든 클러스터 포인트를 하나의 리스트로 합치기 (RGB 색상 추가)
        all_points = []

        for obstacle_id, obstacle_info in self.obstacle_info.items():
            # 각 클러스터마다 고유한 색상 생성 (cluster_id 기반)
            # HSV를 사용해서 색상 분산
            hue = (obstacle_id * 137.5) % 360  # Golden angle for better distribution
            rgb = self._hsv_to_rgb(hue, 1.0, 1.0)

            # 각 포인트에 RGB 정보 추가
            for point in obstacle_info.points:
                all_points.append([
                    float(point[0]),  # x
                    float(point[1]),  # y
                    float(point[2]),  # z
                    rgb[0],           # r
                    rgb[1],           # g
                    rgb[2]            # b
                ])

        if len(all_points) == 0:
            return

        # Header 생성
        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        # XYZRGB PointCloud2 메시지 생성
        cluster_cloud_msg = self._create_cloud_xyzrgb(header, all_points)
        self.cluster_point_cloud_publisher_.publish(cluster_cloud_msg)
    
    def publish_world_to_simpleflight_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'SimpleFlight/RPLIDAR_A3'
        t.transform.translation.x = self.vehicle_pose.pose.position.x
        t.transform.translation.y = self.vehicle_pose.pose.position.y
        t.transform.translation.z = self.vehicle_pose.pose.position.z
        t.transform.rotation.w = self.vehicle_pose.pose.orientation.w
        t.transform.rotation.x = self.vehicle_pose.pose.orientation.x
        t.transform.rotation.y = self.vehicle_pose.pose.orientation.y
        t.transform.rotation.z = self.vehicle_pose.pose.orientation.z
        self.tf_br.sendTransform(t)
    
    def publish_obstacle_info(self):
        """장애물 정보를 시각화 마커로 퍼블리시"""
        if not self.obstacle_info:
            return

        obstacle_bb_marker_array = MarkerArray()
        obstacle_info_marker_array = MarkerArray()

        for obstacle_id, obstacle in self.obstacle_info.items():
            # 위협 레벨에 따른 색상 결정
            if obstacle.threat_level == 3:
                # 레벨 3: 빨강 (매우 위험)
                color_r, color_g, color_b = 1.0, 0.0, 0.0
            elif obstacle.threat_level == 2:
                # 레벨 2: 주황 (위험)
                color_r, color_g, color_b = 1.0, 0.5, 0.0
            elif obstacle.threat_level == 1:
                # 레벨 1: 노랑 (주의)
                color_r, color_g, color_b = 1.0, 1.0, 0.0
            else:
                # 레벨 0: 파랑 (안전)
                color_r, color_g, color_b = 0.0, 0.5, 1.0

            # =============================
            # ① OBB (Oriented Bounding Box) - CUBE
            # =============================
            obstacle_bb = Marker()
            obstacle_bb.header.frame_id = "world"
            obstacle_bb.header.stamp = self.get_clock().now().to_msg()
            obstacle_bb.ns = "obstacle_obb"
            obstacle_bb.id = int(obstacle_id)
            obstacle_bb.type = Marker.CUBE
            obstacle_bb.action = Marker.ADD

            # 중심 위치
            obstacle_bb.pose.position.x = float(obstacle.obstacle_position[0])
            obstacle_bb.pose.position.y = float(obstacle.obstacle_position[1])
            obstacle_bb.pose.position.z = float(obstacle.obstacle_position[2])

            # OBB의 방향을 쿼터니언으로 변환
            rot = R.from_matrix(obstacle.obb_rotation.T)
            qx, qy, qz, qw = rot.as_quat()
            obstacle_bb.pose.orientation.x = float(qx)
            obstacle_bb.pose.orientation.y = float(qy)
            obstacle_bb.pose.orientation.z = float(qz)
            obstacle_bb.pose.orientation.w = float(qw)

            # 크기 설정 (bbox edge length)
            obstacle_bb.scale.x = float(obstacle.obb_size[0])
            obstacle_bb.scale.y = float(obstacle.obb_size[1])
            obstacle_bb.scale.z = float(obstacle.obb_size[2])

            # 위협 레벨에 따른 색상
            obstacle_bb.color.a = 0.3
            obstacle_bb.color.r = color_r
            obstacle_bb.color.g = color_g
            obstacle_bb.color.b = color_b

            obstacle_bb.lifetime = Duration(seconds=0.2).to_msg()
            obstacle_bb_marker_array.markers.append(obstacle_bb)

            # =============================
            # ② 장애물 중심점 - SPHERE
            # =============================
            obstacle_center = Marker()
            obstacle_center.header.frame_id = "world"
            obstacle_center.header.stamp = self.get_clock().now().to_msg()
            obstacle_center.ns = "obstacle_center"
            obstacle_center.id = int(obstacle_id)
            obstacle_center.type = Marker.SPHERE
            obstacle_center.action = Marker.ADD

            obstacle_center.pose.position.x = float(obstacle.obstacle_position[0])
            obstacle_center.pose.position.y = float(obstacle.obstacle_position[1])
            obstacle_center.pose.position.z = float(obstacle.obstacle_position[2])

            obstacle_center.scale.x = 0.3
            obstacle_center.scale.y = 0.3
            obstacle_center.scale.z = 0.3

            obstacle_center.color.a = 0.8
            obstacle_center.color.r = color_r
            obstacle_center.color.g = color_g
            obstacle_center.color.b = color_b

            obstacle_center.lifetime = Duration(seconds=0.2).to_msg()
            obstacle_info_marker_array.markers.append(obstacle_center)

            # =============================
            # ③ 드론 → 장애물 선 - LINE_STRIP
            # =============================
            obstacle_line = Marker()
            obstacle_line.header.frame_id = "world"
            obstacle_line.header.stamp = self.get_clock().now().to_msg()
            obstacle_line.ns = "drone_to_obstacle"
            obstacle_line.id = 1000 + int(obstacle_id)
            obstacle_line.type = Marker.LINE_STRIP
            obstacle_line.action = Marker.ADD

            obstacle_line.scale.x = 0.05

            obstacle_line.color.a = 0.7
            obstacle_line.color.r = color_r
            obstacle_line.color.g = color_g
            obstacle_line.color.b = color_b

            obstacle_line.lifetime = Duration(seconds=0.2).to_msg()

            p1 = Point()
            p1.x = float(self.vehicle_pose.pose.position.x)
            p1.y = float(self.vehicle_pose.pose.position.y)
            p1.z = float(self.vehicle_pose.pose.position.z)

            p2 = Point()
            p2.x = float(obstacle.obstacle_position[0])
            p2.y = float(obstacle.obstacle_position[1])
            p2.z = float(obstacle.obstacle_position[2])

            obstacle_line.points = [p1, p2]
            obstacle_info_marker_array.markers.append(obstacle_line)

            # =============================
            # ④ 거리 텍스트 표시 - TEXT_VIEW_FACING
            # =============================
            distance_text = Marker()
            distance_text.header.frame_id = "world"
            distance_text.header.stamp = self.get_clock().now().to_msg()
            distance_text.ns = "distance_label"
            distance_text.id = 2000 + int(obstacle_id)
            distance_text.type = Marker.TEXT_VIEW_FACING
            distance_text.action = Marker.ADD

            # 드론과 장애물 중간 지점 위에 표시
            distance_text.pose.position.x = float((obstacle.obstacle_position[0] + self.vehicle_pose.pose.position.x) / 2)
            distance_text.pose.position.y = float((obstacle.obstacle_position[1] + self.vehicle_pose.pose.position.y) / 2)
            distance_text.pose.position.z = float((obstacle.obstacle_position[2] + self.vehicle_pose.pose.position.z) / 2) + 0.5

            distance_text.scale.z = 0.4

            distance_text.color.a = 1.0
            distance_text.color.r = 1.0
            distance_text.color.g = 1.0
            distance_text.color.b = 1.0

            distance_text.lifetime = Duration(seconds=0.2).to_msg()
            distance_text.text = f"{obstacle.obstacle_distance:.2f}m | L{obstacle.threat_level}"

            obstacle_info_marker_array.markers.append(distance_text)

            # =============================
            # ⑤ 위협 레벨 표시 - TEXT_VIEW_FACING (장애물 위)
            # =============================
            if obstacle.threat_level > 0:
                threat_text = Marker()
                threat_text.header.frame_id = "world"
                threat_text.header.stamp = self.get_clock().now().to_msg()
                threat_text.ns = "threat_label"
                threat_text.id = 3000 + int(obstacle_id)
                threat_text.type = Marker.TEXT_VIEW_FACING
                threat_text.action = Marker.ADD

                threat_text.pose.position.x = float(obstacle.obstacle_position[0])
                threat_text.pose.position.y = float(obstacle.obstacle_position[1])
                threat_text.pose.position.z = float(obstacle.obstacle_position[2]) + 1.0

                threat_text.scale.z = 0.5

                threat_text.color.a = 1.0
                threat_text.color.r = color_r
                threat_text.color.g = color_g
                threat_text.color.b = color_b

                threat_text.lifetime = Duration(seconds=0.2).to_msg()

                threat_labels = ["SAFE", "CAUTION", "WARNING", "DANGER"]
                threat_text.text = threat_labels[obstacle.threat_level]

                obstacle_info_marker_array.markers.append(threat_text)

        # Publish marker arrays
        self.obstacle_marker_publisher_.publish(obstacle_bb_marker_array)
        self.obstacle_info_publisher_.publish(obstacle_info_marker_array)

    def publish_waypoint_markers(self):
        """Publish waypoint visualization markers"""
        if not self.waypoint_x or len(self.waypoint_x) == 0:
            return

        marker_array = MarkerArray()

        # Publish waypoint spheres
        for i in range(len(self.waypoint_x)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position (NED to ENU conversion)
            marker.pose.position.x = float(self.waypoint_x[i])
            marker.pose.position.y = float(self.waypoint_y[i])
            marker.pose.position.z = float(self.waypoint_z[i])
            marker.pose.orientation.w = 1.0

            # Size
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5

            # Color: green for waypoints
            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)

        # Publish path line connecting waypoints
        if len(self.waypoint_x) > 1:
            line_marker = Marker()
            line_marker.header.frame_id = "world"
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.ns = "waypoint_path"
            line_marker.id = 1000
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD

            line_marker.scale.x = 0.1  # Line width

            # Color: cyan for path
            line_marker.color.a = 0.8
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 1.0

            line_marker.lifetime = Duration(seconds=0).to_msg()

            # Add all waypoints as line strip points
            for i in range(len(self.waypoint_x)):
                p = Point()
                p.x = float(self.waypoint_x[i])
                p.y = float(self.waypoint_y[i])
                p.z = float(self.waypoint_z[i])
                line_marker.points.append(p)

            marker_array.markers.append(line_marker)

        self.waypoint_marker_publisher_.publish(marker_array)
    # endregion


    # region: timer callback functions
    def publish_foxglove_data(self):
        self.publish_vehicle_pose()
        self.publish_vehicle_path()
        self.publish_fov_cone()
        self.publish_vehicle_velocity()
        self.publish_filtered_point_cloud()
        self.publish_world_points()
        self.publish_cluster_points()
        self.publish_obstacle_info()
        self.publish_waypoint_markers()

    def process_lidar_points(self):

        # preprocess points
        self.filtered_points_np = self.preprocess_points()

        # transform point cloud to world frame
        self.world_points_np = self.transform_pc_body_to_world(self.filtered_points_np)

        if len(self.world_points_np) > 0:

            self.extract_obstacle_info(self.world_points_np)

            self.check_obstacle_flags()
        
    # endregion


    # region: Lidar Preprocessing Functions
    def preprocess_points(self):
        """
        라이다 포인트 전처리

        Parameters
        ----------
        points_generator : generator
            pc2.read_points()가 반환한 generator

        Returns
        -------
        numpy.ndarray
            전처리된 포인트 배열 (N, 3)
        """

        # convert generator to list
        input_points_list = list(self.latest_lidar_points_np)
        # convert list to numpy array
        points_np = np.array(input_points_list, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        # sort by x, y, z
        x = points_np["x"]
        y = points_np["y"]
        z = points_np["z"]

        # Filter out points inside the vehicle
        vehicle_radius = 0.01
        distance_mask = np.sqrt((x)**2 + (y)**2 + (z)**2) > vehicle_radius
        forward_mask = x > 0.0
        ground_mask = z > 0.0

        mask = distance_mask & forward_mask & ground_mask

        x = x[mask]
        y = y[mask]
        z = z[mask]

        points = np.column_stack((x, y, z))

        return points

    def extract_obstacle_info(self, points_np):
        '''
        points_np: numpy.ndarray
            shape: (n, 3)
        '''
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(points_np)
        labels = clustering.labels_
        unique_labels = set(labels)

        self.obstacle_info.clear()

        for label in unique_labels:
            if label == -1:  # exclude noise points
                continue

            cluster_points_np = points_np[labels == label]

            # PCA for bounding box
            pca = PCA(n_components=3)
            pca.fit(cluster_points_np)
            R_pca = pca.components_ 

            if np.linalg.det(R_pca) < 0:
                R_pca[2, :] *= -1  # 마지막 축 반전

            transformed = np.dot(cluster_points_np - np.mean(cluster_points_np, axis=0), R_pca.T)
            min_pt, max_pt = np.min(transformed, axis=0), np.max(transformed, axis=0)
            size = max_pt - min_pt
            center_local = (max_pt + min_pt) / 2
            obstacle_position = np.dot(center_local, R_pca) + np.mean(cluster_points_np, axis=0)
    
            obstacle_distance = self.distance_point_to_obb(self.vehicle_position_np, obstacle_position, R_pca, size)

            obstacle_rel_bearing = self.calculate_relative_bearing(obstacle_position, self.vehicle_position_np)

            relative_pos = obstacle_position - self.vehicle_position_np

            self.obstacle_info[label] = ObstacleCluster(
                cluster_id=label,
                points=cluster_points_np,
                obstacle_position=obstacle_position,
                obstacle_distance=obstacle_distance,
                obstacle_rel_bearing=obstacle_rel_bearing,
                obb_rotation=R_pca,
                obb_size=size
            )


    def check_obstacle_flags(self):
        """모든 장애물의 플래그를 체크하고 위협 레벨 설정"""

        if not self.obstacle_info:
            # 장애물이 없으면 회피 불필요
            if self.obstacle_flag:
                self.obstacle_flag = False
                self.avoidance_required = False
                self.avoidance_completed = False
                self.target_obstacle_id = None
                self.previous_target_distance = None
                self.safe_distance_count = 0

            # Publish obstacle flag
            flag_msg = Bool()
            flag_msg.data = False
            self.obstacle_flag_publisher_.publish(flag_msg)
            return

        for obstacle_id, obstacle_info in self.obstacle_info.items():
            # 1. 거리 기반 위험도 판단
            obstacle_info.is_dangerous = self._is_distance_dangerous(obstacle_info)

            # 2. 경로 상에 있는지 판단
            obstacle_info.is_in_path = self._is_in_flight_path(obstacle_info)

            # 3. 타겟 장애물 판단 (회피 대상)
            obstacle_info.is_target_obstacle = self._is_target_for_avoidance(obstacle_info)

            # 4. 종합 위협 레벨 계산
            obstacle_info.threat_level = self._calculate_threat_level(obstacle_info)

        # Collision Avoidance 상태 결정 및 플래그 업데이트
        prev_flag = self.obstacle_flag
        self._update_collision_avoidance_state()

        # Publish obstacle flag
        flag_msg = Bool()
        flag_msg.data = self.obstacle_flag
        self.obstacle_flag_publisher_.publish(flag_msg)

    def _update_collision_avoidance_state(self):
        """충돌 회피 상태 업데이트 및 플래그 결정 (히스테리시스 적용)"""

        # 가장 위험한 장애물 찾기
        most_dangerous = self.get_most_dangerous_obstacle()

        if most_dangerous is None:
            # 위험한 장애물이 없음
            if self.avoidance_required:
                # 회피 중이었다면 안전 카운트 증가
                self.safe_distance_count += 1
                if self.safe_distance_count >= self.safe_count_required:
                    # 일정 시간 동안 장애물 없음 - 회피 완료
                    self.avoidance_completed = True
                    self.avoidance_required = False
                    self.obstacle_flag = False
                    self.target_obstacle_id = None
                    self.previous_target_distance = None
                    self.get_logger().info(
                        f"Collision avoidance COMPLETED - "
                        f"No dangerous obstacles for {self.safe_count_required} cycles"
                    )
            return

        # 위험한 장애물이 존재하는 경우
        if not self.avoidance_required:
            # 회피 중이 아닌 상태 → CA 진입 조건 확인
            if most_dangerous.threat_level >= self.ca_entry_threat_level:
                # 회피 시작
                self.avoidance_required = True
                self.avoidance_completed = False
                self.obstacle_flag = True
                self.target_obstacle_id = most_dangerous.cluster_id
                self.previous_target_distance = most_dangerous.obstacle_distance
                self.safe_distance_count = 0
        else:
            # 회피 진행 중 → CA 종료 조건 확인 (히스테리시스)
            if most_dangerous.threat_level <= self.ca_exit_threat_level:
                # 위협 레벨이 종료 임계값 이하로 낮아짐
                # 각도 조건: 장애물이 측면/후방으로 충분히 벗어났는지 확인
                angle_diff = abs(most_dangerous.obstacle_rel_bearing)
                is_obstacle_cleared = angle_diff > self.safe_angle_threshold  # 90도 이상 벗어남

                # 거리 조건: 안전 거리 이상
                is_distance_safe = most_dangerous.obstacle_distance >= self.safe_distance_threshold

                if is_distance_safe or is_obstacle_cleared:
                    # 안전 거리 도달 OR 장애물이 측면/후방으로 벗어남
                    self.safe_distance_count += 1

                    # 진행 상황 로깅 (10회마다)
                    if self.safe_distance_count % 10 == 0:
                        self.get_logger().info(
                            f"CA exit progress: {self.safe_distance_count}/{self.safe_count_required} - "
                            f"Dist: {most_dangerous.obstacle_distance:.1f}m (safe: {is_distance_safe}), "
                            f"Bearing: {np.degrees(angle_diff):.0f}° (cleared: {is_obstacle_cleared})"
                        )

                    if self.safe_distance_count >= self.safe_count_required:
                        # 안전 조건 + 위협 레벨 낮음 + 일정 시간 유지 - 회피 완료
                        self.avoidance_completed = True
                        self.avoidance_required = False
                        self.obstacle_flag = False
                        self.target_obstacle_id = None
                        self.previous_target_distance = None
                        self.get_logger().info(
                            f"✅ Collision avoidance COMPLETED - "
                            f"Distance: {most_dangerous.obstacle_distance:.2f}m, "
                            f"Bearing: {np.degrees(most_dangerous.obstacle_rel_bearing):.1f}°, "
                            f"Threat level: {most_dangerous.threat_level}, "
                            f"Safe count: {self.safe_distance_count}/{self.safe_count_required}"
                        )
                else:
                    # 거리도 부족하고 각도도 부족 (여전히 전방에 있음)
                    if self.safe_distance_count > 0:
                        # 안전 카운트가 리셋되는 경우 로깅
                        self.get_logger().info(
                            f"⚠️ CA exit condition NOT met - Safe count reset! "
                            f"Dist: {most_dangerous.obstacle_distance:.1f}m (need {self.safe_distance_threshold:.1f}m), "
                            f"Bearing: {np.degrees(angle_diff):.0f}° (need {np.degrees(self.safe_angle_threshold):.0f}°)"
                        )
                    self.safe_distance_count = 0
            else:
                # 여전히 위협적 - 카운트 리셋
                self.safe_distance_count = 0
                self.previous_target_distance = most_dangerous.obstacle_distance

    def _is_distance_dangerous(self, obstacle: ObstacleCluster) -> bool:
        """거리가 위험한 범위에 있는지 확인"""
        return obstacle.obstacle_distance < self.danger_distance_threshold
    
    def _is_in_flight_path(self, obstacle: ObstacleCluster) -> bool:
        """장애물이 비행 경로 상에 있는지 확인"""
        # obstacle.obstacle_rel_bearing은 이미 드론 정면 기준 상대 방위각
        # (calculate_relative_bearing에서 heading_enu를 이미 뺌)
        angle_diff = abs(obstacle.obstacle_rel_bearing)

        # 진행 방향 전방 일정 각도 내에 있는지 확인
        is_in_front = angle_diff < self.path_angle_threshold

        # 거리가 충분히 가까운지도 확인
        is_close_enough = obstacle.obstacle_distance < self.warning_distance_threshold

        return is_in_front and is_close_enough
    
    def _is_target_for_avoidance(self, obstacle: ObstacleCluster) -> bool:
        """회피 대상 장애물인지 판단"""
        # 조건 1: 위험 거리 이내
        distance_check = obstacle.obstacle_distance < self.warning_distance_threshold

        # 조건 2: 경로 상에 있음
        path_check = obstacle.is_in_path

        # 모든 조건을 만족하거나, 매우 가까운 경우
        return (distance_check and path_check) or \
               (obstacle.obstacle_distance < self.danger_distance_threshold)
    
    def _calculate_threat_level(self, obstacle: ObstacleCluster) -> int:
        """종합 위협 레벨 계산 (0~3)"""
        distance = obstacle.obstacle_distance
        
        # 레벨 3: 매우 위험 (즉시 회피 필요)
        if distance < 2.0 and obstacle.is_in_path:
            return 3
        
        # 레벨 2: 위험 (회피 준비)
        if distance < self.danger_distance_threshold and obstacle.is_in_path:
            return 2
        
        # 레벨 1: 주의 (모니터링)
        if distance < self.warning_distance_threshold and obstacle.is_in_path:
            return 1
        
        # 레벨 0: 안전
        return 0
    
    def get_target_obstacles(self) -> Dict[int, ObstacleCluster]:
        """회피가 필요한 장애물들만 반환"""
        return {
            obstacle_id: obstacle_info 
            for obstacle_id, obstacle_info in self.obstacle_info.items() 
            if obstacle_info.is_target_obstacle
        }
    
    def get_most_dangerous_obstacle(self) -> Optional[ObstacleCluster]:
        """가장 위험한 장애물 반환"""
        if not self.obstacle_info:
            return None
        
        dangerous_obstacles = [
            obs for obs in self.obstacle_info.values() 
            if obs.threat_level > 0
        ]
        
        if not dangerous_obstacles:
            return None
        
        # 위협 레벨이 높고, 거리가 가까운 순으로 정렬
        most_dangerous = max(
            dangerous_obstacles,
            key=lambda x: (x.threat_level, -x.obstacle_distance)
        )
        
        return most_dangerous

    # endregion


    # region: Utility Functions
    # convert NED to ENU position
    def ned_to_enu(self, x_n, y_n, z_n):
        """
        Position NED → ENU Frame Transformation
        NED: [x, y, z] = [North, East, Down]
        ENU: [x, y, z] = [East, North, Up]
        """
        # XY exchange + Z inversion
        x_e = y_n
        y_e = x_n
        z_e = -z_n
        return x_e, y_e, z_e

    def create_fov_cone_marker(self, position, yaw, pitch, fov_deg=20.0, range_m=7.0):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fov_cone"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST   # ✅ 면으로 표현
        marker.action = Marker.ADD

        # 색상 및 투명도
        marker.color.a = 0.4
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0

        # 중심점 (시야 원뿔의 시작점)
        origin = Point()
        origin.x, origin.y, origin.z = position

        half_fov = math.radians(fov_deg / 2.0)
        dirs = []
        for sign_y in [-1, 1]:
            for sign_p in [-1, 1]:
                dir_x = math.cos(pitch + sign_p * half_fov) * math.cos(yaw + sign_y * half_fov)
                dir_y = math.cos(pitch + sign_p * half_fov) * math.sin(yaw + sign_y * half_fov)
                dir_z = math.sin(pitch + sign_p * half_fov)
                dirs.append((dir_x, dir_y, dir_z))

        # 네 꼭짓점 계산
        corners = []
        for dx, dy, dz in dirs:
            p = Point()
            p.x = origin.x + range_m * dx
            p.y = origin.y + range_m * dy
            p.z = origin.z + range_m * dz
            corners.append(p)

        # ✅ 원뿔의 면 만들기 (중심과 각 꼭짓점 삼각형)
        # for i in range(4):
        #     p1 = origin
        #     p2 = corners[i]
        #     p3 = corners[(i + 1) % 4]
        #     marker.points.extend([p1, p2, p3])  # 한 삼각형 면

        p1 = origin
        p2 = corners[0]
        p3 = corners[1]
        marker.points.extend([p1, p2, p3])  

        p1 = origin
        p2 = corners[0]
        p3 = corners[2]
        marker.points.extend([p1, p2, p3])  


        p1 = origin
        p2 = corners[1]
        p3 = corners[3]
        marker.points.extend([p1, p2, p3])  

        p1 = origin
        p2 = corners[3]
        p3 = corners[2]
        marker.points.extend([p1, p2, p3])  
        
        
        # 한 삼각형 면
        # ✅ 바닥면도 추가 (FOV 끝의 네 점 연결)
        base_center = Point()
        base_center.x = sum([c.x for c in corners]) / 4.0
        base_center.y = sum([c.y for c in corners]) / 4.0
        base_center.z = sum([c.z for c in corners]) / 4.0
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = base_center
            marker.points.extend([p1, p2, p3])

        return marker
    
    def distance_point_to_obb(self, point, center, R, size):
        """
        point: [3] 내 위치 (x, y, z)
        center: [3] 박스 중심
        R: [3x3] 박스 회전행렬 (from PCA)
        size: [3] 박스 각 축 길이 (x, y, z)
        """
        # 내 위치를 박스 좌표계로 변환
        p_local = np.dot(R.T, (point - center))

        # 박스 절반 크기
        half = size / 2.0

        # 박스 내부면 거리는 0, 외부면은 초과분 계산
        d = np.maximum(np.abs(p_local) - half, 0.0)

        # 거리 = 초과된 부분의 유클리드 거리
        return np.linalg.norm(d)

    def calculate_relative_bearing(self, obstacle_position, vehicle_position):
        """
        상대 방위각 계산 (드론 정면 기준)

        Args:
            obstacle_position: [x, y, z] 장애물 위치 (world frame)
            vehicle_position: [x, y, z] 드론 위치 (world frame)

        Returns:
            bearing (radians): -π ~ π
                              0 = 정면, +π/2 = 왼쪽, -π/2 = 오른쪽
        """

        # 장애물까지의 벡터
        dx = obstacle_position[0] - vehicle_position[0]
        dy = obstacle_position[1] - vehicle_position[1]

        # 절대 방위각 (world frame)
        absolute_bearing = math.atan2(dy, dx)

        # 상대 방위각 = 절대 - 드론 heading
        obstacle_rel_bearing = absolute_bearing - self.heading_enu

        # -π ~ π 정규화
        obstacle_rel_bearing = math.atan2(
            math.sin(obstacle_rel_bearing),
            math.cos(obstacle_rel_bearing)
        )

        # Radian 반환
        return obstacle_rel_bearing
   
    def transform_pc_body_to_world(self, body_points_np):
        '''
        body_points_np: numpy.ndarray
            shape: (n, 3)
        
        Returns:
            world_points_np: numpy.ndarray
                shape: (n, 3)
        '''
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                'SimpleFlight/RPLIDAR_A3',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            world_points_np = self.transform_points_batch(body_points_np, transform)

            # Filter ground points (simple altitude filter)
            altitude_filter = world_points_np[:, 2] > 2.0
            world_points_np = world_points_np[altitude_filter]

            return world_points_np

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return np.array([])

    def transform_points_batch(self, points_np, transform):
        """
        Input:
            points_np: (N, 3) numpy array
            transform: geometry_msgs.msg.TransformStamped
        
        Returns:
            transformed_points_np: (N, 3) numpy array
        """
        # Quaternion → Rotation Matrix
        quat = transform.transform.rotation
        rotation_matrix = self.quaternion_to_rotation_matrix(quat)
        
        # Translation Vector
        trans = transform.transform.translation
        t = np.array([trans.x, trans.y, trans.z])
        
        # 벡터화 변환: R @ points.T + t
        # (3,3) @ (3,N) + (3,1) = (3,N) → transpose → (N,3)
        transformed_points_np = (rotation_matrix @ points_np.T).T + t

        return transformed_points_np

    def quaternion_to_rotation_matrix(self, quat):
        """Quaternion → Rotation Matrix"""
        if hasattr(quat, 'x'):
            quat_array = [quat.x, quat.y, quat.z, quat.w]
        else:
            quat_array = quat

        rotation = R.from_quat(quat_array)
        return rotation.as_matrix()

    def _hsv_to_rgb(self, h, s, v):
        """
        HSV를 RGB로 변환

        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            v: Value (0-1)

        Returns:
            (r, g, b): RGB 값 (0-255)
        """
        h = h / 60.0
        c = v * s
        x = c * (1 - abs((h % 2) - 1))
        m = v - c

        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

    def _create_cloud_xyzrgb(self, header, points):
        """
        XYZRGB PointCloud2 메시지 생성

        Args:
            header: std_msgs/Header
            points: List of [x, y, z, r, g, b] where rgb are 0-255

        Returns:
            sensor_msgs/PointCloud2
        """
        # PointField 정의
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        # 포인트 데이터를 바이트로 변환
        cloud_data = []
        for point in points:
            x, y, z, r, g, b = point
            # RGB를 uint32로 패킹 (0xRRGGBB 형식)
            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            cloud_data.append(struct.pack('fffI', x, y, z, rgb_int))

        # PointCloud2 메시지 생성
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16  # 4 bytes * 4 fields
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = b''.join(cloud_data)

        return cloud_msg
    # endregion




def main():
    rclpy.init()
    node = FoxgloveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
