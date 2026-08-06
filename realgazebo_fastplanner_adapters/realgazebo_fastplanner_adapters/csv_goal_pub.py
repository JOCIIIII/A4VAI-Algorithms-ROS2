#!/usr/bin/env python3
# CSV goal -> /waypoint_generator/waypoints (nav_msgs/Path) for Fast-Planner.
#
# Fast-Planner 의 KinoReplanFSM 은 /waypoint_generator/waypoints (nav_msgs/Path) 의
# 마지막 pose 를 목표(goal)로 삼아 replan 한다. 이 노드는 CSV 파일에서 goal 좌표를
# 읽어 그 토픽으로 발행한다. goal 은 ENU(world, z-up) 좌표다 — fast_planner 와 동일.
#
# CSV 형식 (헤더 1줄 스킵): x,y,z  (ENU, meters)
#   예)
#     x,y,z
#     10.0,0.0,1.5
#     8.0,-4.0,1.5
#
# 모드:
#   sequential=true(기본): odom 으로 현재 goal 도달을 판정해 다음 행으로 진행.
#                          odom 토픽(/odom, ENU)을 구독해 reach_radius 안에 들면 next.
#   sequential=false     : 첫 행만 1회 발행하고 종료(단일 goal).
#
# params: csv_path, sequential(bool), reach_radius(m), odom_topic, repeat(bool),
#         publish_period(s, odom 없을 때 강제 진행 간격; 0=비활성)
import csv
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       QoSDurabilityPolicy, QoSReliabilityPolicy)

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped


class CsvGoalPub(Node):
    def __init__(self):
        super().__init__('csv_goal_pub')
        self.csv_path = self.declare_parameter('csv_path', '').value
        self.sequential = bool(self.declare_parameter('sequential', True).value)
        self.reach_radius = float(self.declare_parameter('reach_radius', 1.0).value)
        self.repeat = bool(self.declare_parameter('repeat', False).value)
        self.publish_period = float(self.declare_parameter('publish_period', 0.0).value)
        self.world_frame = self.declare_parameter('world_frame', 'world').value
        odom_topic = self.declare_parameter('odom_topic', '/odom').value
        # ★ goal_pose_topic: 값이 있으면 그 토픽으로 geometry_msgs/PoseStamped 도 발행한다.
        #   SUPER fsm_node 는 goal 을 PoseStamped(/planning/click_goal)로 받기 때문(Path 아님).
        #   fast-planner 는 Path 만 쓰므로 기본 ""(비활성) → 기존 동작 무변경(플래너-중립).
        #   ENU 그대로(SUPER 도 ENU, z=고도). 변환 없음.
        self.goal_pose_topic = self.declare_parameter('goal_pose_topic', '').value

        self.goals = self._load_csv(self.csv_path)
        if not self.goals:
            self.get_logger().error(
                f'no goals loaded from csv_path="{self.csv_path}". '
                f'CSV must have header + rows of "x,y,z" (ENU).')
            raise SystemExit(1)

        self.idx = 0
        self.cur_xyz = None
        # ★ TRANSIENT_LOCAL(latch): goal 을 1회성으로 발행하므로 fast_planner 구독이
        #   늦게 떠 있으면 VOLATILE 로는 영영 놓쳐 INIT/"wait for goal" 에 멈춘다
        #   (통합 스크립트에서 fast_planner launch 가 csv_goal_pub 보다 늦음).
        #   latch 로 마지막 goal 을 보관해 늦게 붙는 구독자도 즉시 받게 한다.
        goal_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(Path, '/waypoint_generator/waypoints', goal_qos)
        # super 통합용 PoseStamped 발행자(goal_pose_topic 지정 시에만).
        self.pose_pub = None
        if self.goal_pose_topic:
            self.pose_pub = self.create_publisher(PoseStamped, self.goal_pose_topic, goal_qos)

        if self.sequential:
            self.create_subscription(Odometry, odom_topic, self.odom_cb,
                                     qos_profile_sensor_data)
        # 현재 goal 을 주기적으로 재발행한다. latch(TRANSIENT_LOCAL)만으로는 일부 DDS
        # 에서 늦게 뜨는 fast_planner 구독에 전달이 안 돼 "wait for goal" 에 멈추는
        # 사례가 있었다(통합 스크립트에서 fast_planner launch 가 csv_goal_pub 보다 늦음).
        # goal_repub_period 마다 현재 goal 을 다시 쏴서 확실히 전달한다. 같은 goal
        # 재발행은 fast_planner replan 만 유발하며 bridge throttle 이 wp 중복을 막는다.
        self.repub_period = float(self.declare_parameter('goal_repub_period', 2.0).value)
        self.kicked = False
        self.create_timer(self.repub_period, self._republish_current_goal)
        if self.publish_period > 0.0:
            self.create_timer(self.publish_period, self._advance_on_timer)

        self.get_logger().info(
            f'[csv_goal_pub] up. {len(self.goals)} goals, '
            f'sequential={self.sequential} reach_radius={self.reach_radius}m')

    def _load_csv(self, path):
        if not path or not os.path.exists(path):
            return []
        goals = []
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        for i, row in enumerate(rows):
            if i == 0:  # header
                continue
            vals = [c.strip() for c in row if c.strip() != '']
            if len(vals) < 3:
                continue
            try:
                goals.append((float(vals[0]), float(vals[1]), float(vals[2])))
            except ValueError:
                continue
        return goals

    def _make_path(self, xyz):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.world_frame
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
        return path

    def _publish_goal(self, xyz):
        # Path(/waypoint_generator/waypoints) + (옵션) PoseStamped(goal_pose_topic) 둘 다 발행.
        path = self._make_path(xyz)
        self.pub.publish(path)
        if self.pose_pub is not None:
            self.pose_pub.publish(path.poses[0])

    def _publish_current(self):
        xyz = self.goals[self.idx]
        self.cur_xyz = xyz
        self._publish_goal(xyz)
        self.get_logger().info(
            f'[csv_goal_pub] goal[{self.idx}] = ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})')

    def _republish_current_goal(self):
        # 현재 goal 을 주기적으로 재발행 (fast_planner 가 확실히 받게).
        # 모든 goal 도달 후엔 더 발행 안 함.
        if self.idx >= len(self.goals):
            return
        xyz = self.goals[self.idx]
        self.cur_xyz = xyz
        self._publish_goal(xyz)
        if not self.kicked:
            self.kicked = True
            self.get_logger().info(
                f'[csv_goal_pub] goal[{self.idx}] = ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}) '
                f'(republish every {self.repub_period}s)')

    def _next_goal(self):
        if self.idx + 1 < len(self.goals):
            self.idx += 1
            self._publish_current()
        elif self.repeat:
            self.idx = 0
            self._publish_current()
        else:
            self.get_logger().info('[csv_goal_pub] all goals reached.')
            # 범위 밖으로 올려 주기 재발행을 멈춘다.
            self.idx = len(self.goals)

    def odom_cb(self, msg: Odometry):
        if self.cur_xyz is None:
            return
        p = msg.pose.pose.position
        gx, gy, gz = self.cur_xyz
        d = ((p.x - gx) ** 2 + (p.y - gy) ** 2 + (p.z - gz) ** 2) ** 0.5
        if d <= self.reach_radius:
            self.get_logger().info(
                f'[csv_goal_pub] reached goal[{self.idx}] (d={d:.2f}m)')
            self._next_goal()

    def _advance_on_timer(self):
        # odom 없이도 강제로 다음 goal 로 (테스트/디버그용).
        if self.kicked and self.sequential:
            self._next_goal()


def main():
    rclpy.init()
    try:
        node = CsvGoalPub()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
