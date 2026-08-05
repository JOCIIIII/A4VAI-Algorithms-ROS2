#!/usr/bin/env python3
# Bridge: SUPER -> A4VAI PathFollowing.
#
# Subscribes the SUPER committed MINCO polynomial trajectory
# (/planning_cmd/poly_traj, mars_quadrotor_msgs/PolynomialTrajectory, ENU/world),
# evaluates it into a dense path, resamples to an evenly-spaced waypoint list,
# converts ENU->NED, and republishes as custom_msgs/LocalWaypointSetpoint on the
# topic the PathFollowing(MPPI) node consumes (wp_type=0). Also emits the two
# heartbeats the follower gates on (SUPER *is* the path-planning + collision-avoid
# stage). Each replan publishes a fresh waypoint list.
#
#   in : /planning_cmd/poly_traj   (mars_quadrotor_msgs/PolynomialTrajectory, ENU)
#   out: /local_waypoint_setpoint_to_PF  (custom_msgs/LocalWaypointSetpoint, NED)
#        /path_planning_heartbeat         (std_msgs/Bool)
#        /collision_avoidance_heartbeat   (std_msgs/Bool)
#
# ★ 설계: 이 브리지의 입력 측만 SUPER 고유(MINCO 다항식 평가)이고, 출력 측(호길이
#   재샘플 / ENU->NED / odom skip-ahead / throttle / heartbeat / 주기재발행)은
#   fastplanner_pf_bridge/traj_to_waypoints.py 와 동일한 플래너-중립 로직을 그대로
#   가져온다. fast-planner B-spline 과 SUPER MINCO 는 "전체 궤적 -> 재샘플 -> wp"
#   라는 같은 추상화이므로 출력 계약이 일치한다. (src/planners/README.md)
#
# MINCO 평가 규약 (SUPER piece.cpp::getPos, fsm_ros2.hpp 발행부에서 검증):
#   - piece 당 order_pos=7 -> 8 계수. 메시지 coef_pos_x[8*i : 8*i+8] = piece i 의 x 8계수.
#   - 계수는 ★내림차순★: pos(t) = Σ_{j=0..7} coef[8i+j] * t^(7-j)  (coef[8i]=t^7 계수).
#   - t 는 piece 로컬시간, 정규화 안 함: t ∈ [0, time_pos[i]] (time_pos[i]=piece duration).
#
# params: frame ('ned'|'enu'), waypoint_spacing_m, max_waypoints,
#         samples_per_piece, republish_*_eps, skip_ahead_m, wp_repub_period
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from std_msgs.msg import Bool

from mars_quadrotor_msgs.msg import PolynomialTrajectory
from custom_msgs.msg import LocalWaypointSetpoint
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class PolyToWaypoints(Node):
    def __init__(self):
        super().__init__("super_pf_bridge")
        self.frame = self.declare_parameter("frame", "ned").value          # 'ned' or 'enu'
        self.spacing = float(self.declare_parameter("waypoint_spacing_m", 0.5).value)
        self.max_wp = int(self.declare_parameter("max_waypoints", 60).value)
        # piece 당 다항식 샘플 수(평가 밀도). 호길이 재샘플 전 dense 곡선용.
        self.samples_per_piece = int(self.declare_parameter("samples_per_piece", 20).value)

        # ---- 아래 throttle/skip-ahead/QoS 는 traj_to_waypoints.py 와 동일 (플래너-중립) ----
        self.goal_eps = float(self.declare_parameter("republish_goal_eps", 10.0).value)
        self.start_eps = float(self.declare_parameter("republish_start_eps", 1.0).value)
        self.last_goal = None
        self.last_start = None

        self.skip_ahead = float(self.declare_parameter("skip_ahead_m", 1.5).value)
        self.drone_enu = None
        self.create_subscription(Odometry, "/odom", self._odom_cb,
                                 qos_profile_sensor_data)

        # ★ PF done 방지 (receding 비행): PF 는 wp 리스트 ★마지막 wp★ 에 final_wp_tolerance(0.5m)
        #   안으로 들면 done→착륙. SUPER 는 receding 이라 매 궤적 끝이 드론 앞 ~15m 일 뿐 최종
        #   goal 이 아닌데, 드론이 그 끝을 따라잡으면 PF 가 "도착"으로 오판해 중간 착륙한다.
        #   해결: 최종 goal(/planning/click_goal)에 아직 안 닿았으면 wp 끝점을 진행방향으로
        #   연장(extend_ahead_m)해 드론이 마지막 wp 에 못 닿게 → PF done 안 남. goal 근처
        #   (final_goal_tol_m 안)면 연장 안 함 → 정상 done→착륙.
        self.final_goal_tol = float(self.declare_parameter("final_goal_tol_m", 3.0).value)
        self.extend_ahead = float(self.declare_parameter("extend_ahead_m", 8.0).value)
        self.final_goal_enu = None
        goal_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, "/planning/click_goal",
                                 self._goal_cb, goal_qos)

        # ★ fsm_node 는 poly_traj 를 best_effort + keep_last(1) + volatile 로 발행한다
        #   (fsm_ros2.hpp:286-289). 구독을 RELIABLE(기본 depth=10)로 두면 QoS 비호환으로
        #   매칭이 안 돼 메시지를 한 건도 못 받는다 → best_effort 로 정확히 맞춘다.
        poly_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PolynomialTrajectory, "/planning_cmd/poly_traj",
                                 self.poly_cb, poly_qos)
        wp_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.wp_pub = self.create_publisher(LocalWaypointSetpoint,
                                            "/local_waypoint_setpoint_to_PF", wp_qos)
        self.pp_hb = self.create_publisher(Bool, "/path_planning_heartbeat", 10)
        self.ca_hb = self.create_publisher(Bool, "/collision_avoidance_heartbeat", 10)
        self.create_timer(0.1, self.heartbeat)  # 10 Hz
        self.have_path = False
        self.last_wp_msg = None
        self.repub_period = float(self.declare_parameter("wp_repub_period", 1.0).value)
        self.create_timer(self.repub_period, self._republish_wp)
        self.get_logger().info(
            f"[super_pf_bridge] up. frame={self.frame} spacing={self.spacing}m "
            f"max_wp={self.max_wp}")

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.drone_enu = np.array([p.x, p.y, p.z])

    def _goal_cb(self, msg: PoseStamped):
        q = msg.pose.position
        self.final_goal_enu = np.array([q.x, q.y, q.z])

    def _republish_wp(self):
        if self.last_wp_msg is not None:
            self.wp_pub.publish(self.last_wp_msg)

    def heartbeat(self):
        m = Bool(); m.data = True
        self.pp_hb.publish(m)
        self.ca_hb.publish(m)

    # ---- ENU (planner, z-up) -> PathFollowing external-waypoint convention. ----
    # (traj_to_waypoints.py 와 동일: PF build_waypoints(wp_type=0)가 z 를 내부에서
    #  부호반전하므로 altitude(+ENU.z)를 그대로 넘긴다.)
    @staticmethod
    def enu_to_ned(p):
        out = np.empty_like(p)
        out[:, 0] = p[:, 1]    # waypoint_x = North    = ENU y
        out[:, 1] = p[:, 0]    # waypoint_y = East     = ENU x
        out[:, 2] = p[:, 2]    # waypoint_z = ALTITUDE = +ENU z
        return out

    @staticmethod
    def resample_by_arclen(dense, spacing):
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total < 1e-6:
            return dense[:1]
        n = max(2, int(total / spacing) + 1)
        st = np.linspace(0.0, total, n)
        out = np.empty((n, 3))
        for i in range(3):
            out[:, i] = np.interp(st, s, dense[:, i])
        return out

    # ---- MINCO 다항식 평가 -> dense 곡선 (이 브리지의 SUPER 고유 입력 처리) ----
    def _eval_minco_dense(self, msg: PolynomialTrajectory):
        n_piece = int(msg.piece_num_pos)
        order = int(msg.order_pos)              # 7
        ncoef = order + 1                       # 8
        cx = np.array(msg.coef_pos_x, dtype=float)
        cy = np.array(msg.coef_pos_y, dtype=float)
        cz = np.array(msg.coef_pos_z, dtype=float)
        durs = np.array(msg.time_pos, dtype=float)
        if (n_piece <= 0 or len(durs) != n_piece
                or len(cx) != n_piece * ncoef
                or len(cy) != n_piece * ncoef or len(cz) != n_piece * ncoef):
            raise ValueError(
                f"coef/time size mismatch: pieces={n_piece} order={order} "
                f"len(cx)={len(cx)} len(t)={len(durs)}")

        # 계수 내림차순: pos(τ) = Σ_j coef[8i+j] * τ^(order-j).
        # np.polyval 이 정확히 이 규약(내림차순, 정규화X)이라 그대로 쓴다.
        pts = []
        for i in range(n_piece):
            d = float(durs[i])
            m = max(2, self.samples_per_piece)
            # 마지막 piece 만 끝점 포함, 중간은 끝점 중복 방지로 제외.
            taus = np.linspace(0.0, d, m, endpoint=(i == n_piece - 1))
            px = np.polyval(cx[ncoef * i: ncoef * i + ncoef], taus)
            py = np.polyval(cy[ncoef * i: ncoef * i + ncoef], taus)
            pz = np.polyval(cz[ncoef * i: ncoef * i + ncoef], taus)
            pts.append(np.stack([px, py, pz], axis=1))
        return np.concatenate(pts, axis=0) if pts else np.empty((0, 3))

    def poly_cb(self, msg: PolynomialTrajectory):
        # POSITION_TRAJ 비트가 없으면(HEART_BEAT 등) 위치궤적이 아니므로 무시.
        if not (int(msg.type) & PolynomialTrajectory.POSITION_TRAJ):
            return
        try:
            dense = self._eval_minco_dense(msg)
            if len(dense) < 2:
                return
            pts = self.resample_by_arclen(dense, self.spacing)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"minco parse failed: {e}")
            return

        if len(pts) < 2:
            return

        # ---- 이하 출력 측: traj_to_waypoints.py 와 동일 (플래너-중립) ----
        # odom 정렬: 드론에 가장 가까운 점부터 보존(곡선 평탄화 방지), wp[0]만 살짝 앞으로.
        if self.drone_enu is not None and len(pts) > 2:
            dists = np.linalg.norm(pts - self.drone_enu, axis=1)
            nearest = int(np.argmin(dists))
            start_i = nearest
            while start_i < len(pts) - 2 and \
                    float(np.linalg.norm(pts[start_i] - self.drone_enu)) < 0.5:
                start_i += 1
            pts = pts[start_i:]
            if len(pts) < 2:
                return

        if len(pts) > self.max_wp:
            idx = np.linspace(0, len(pts) - 1, self.max_wp).astype(int)
            pts = pts[idx]

        # throttle: goal·start 둘 다 거의 그대로면 무시(미세 replan 의 인덱스 리셋 방지).
        goal_pt = pts[-1]
        start_pt = pts[0]
        if self.last_goal is not None and self.last_start is not None:
            goal_same = float(np.linalg.norm(goal_pt - self.last_goal)) < self.goal_eps
            start_same = float(np.linalg.norm(start_pt - self.last_start)) < self.start_eps
            if goal_same and start_same:
                return
        self.last_goal = goal_pt
        self.last_start = start_pt

        # ★ PF done 방지: 최종 goal 에 아직 안 닿았으면(궤적 끝점이 goal 에서 멀면) wp 끝을
        #   진행방향으로 연장 → 드론이 마지막 wp 에 final_wp_tolerance(0.5m) 안으로 못 들어
        #   PF 가 done(중간 착륙) 안 함. goal 근처(final_goal_tol)면 연장 안 함 → 정상 도착·착륙.
        if self.final_goal_enu is not None and len(pts) >= 2:
            end_to_goal = float(np.linalg.norm(pts[-1] - self.final_goal_enu))
            if end_to_goal > self.final_goal_tol:
                d = pts[-1] - pts[-2]
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-6:
                    ext = pts[-1] + (d / nrm) * self.extend_ahead
                    pts = np.vstack([pts, ext])

        wp = self.enu_to_ned(pts) if self.frame == "ned" else pts
        out = LocalWaypointSetpoint()
        out.path_planning_complete = True
        out.waypoint_x = wp[:, 0].tolist()
        out.waypoint_y = wp[:, 1].tolist()
        out.waypoint_z = wp[:, 2].tolist()
        self.wp_pub.publish(out)
        self.last_wp_msg = out

        if not self.have_path:
            self.have_path = True
            self.get_logger().info(
                f"[super_pf_bridge] first path: {len(wp)} waypoints, frame={self.frame}")
        else:
            self.get_logger().info(
                f"[super_pf_bridge] wp updated -> {len(wp)} waypoints "
                f"(goal>={self.goal_eps}m or start>={self.start_eps}m moved)")


def main():
    rclpy.init()
    node = PolyToWaypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
