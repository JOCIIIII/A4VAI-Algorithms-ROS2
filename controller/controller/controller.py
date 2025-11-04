# Librarys

# Library for common
import numpy as np
import os
import math
import time

# ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
# Custom libraries
from .lib.common_fuctions import set_initial_variables, state_logger, publish_to_plotter, set_wp
from .lib.timer import HeartbeatTimer, MainTimer, CommandPubTimer
from .lib.subscriber import PX4Subscriber, FlagSubscriber, CmdSubscriber, HeartbeatSubscriber, EtcSubscriber
from .lib.publisher import PX4Publisher, HeartbeatPublisher, ModulePublisher, PlotterPublisher
from .lib.publisher import PubFuncHeartbeat, PubFuncPX4, PubFuncModule, PubFuncPlotter
from .lib.data_class import *

# custom message
from custom_msgs.msg import StateFlag
from custom_msgs.msg import GlobalWaypointSetpoint
from custom_msgs.msg import LocalWaypointSetpoint
from px4_msgs.msg import FusionWeight
# ----------------------------------------------------------------------------------------#

class Controller(Node):
    def __init__(self):
        super().__init__("controller")
        self.get_logger().info("Controller node initialized")

        self.weight = FusionWeight()
        self.current_target = 0.0  # 0 (ca) or 1 (pf)
        self.prev_target = 0.0
        self.transition_start_time = None
        self.transition_initial_weight = 0.0  # 전환 시작 시 weight 값
        self.transition_duration_to_pf = 5.0  # seconds (CA → PF: 천천히 전환하여 피치 급변 방지)
        self.transition_duration_to_ca = 1.0  # seconds (PF → CA: 적절한 회피 반응)
        self.ca_entry_delay = 0.0  # seconds (CA 진입 즉시 - 중앙 장애물 대응)
        self.initial_setting = False

        self.first_global_start_wp = [200.0, 800.0, 4.0]
        self.first_global_goal_wp = [750.0, 720.0, 4.0]

        self.second_global_start_wp = [750.0, 720.0, 4.0]
        self.second_global_goal_wp = [600.0, 500.0, 4.0]

        # Multi-waypoint mission management
        self.first_mission_completed = False
        self.second_mission_started = False
        self.second_mission_waypoint_received = False
        self.first_global_wp_published = False

        # CA exit transition management
        self.ca_exit_transition_active = False  # CA 종료 후 전환 중 플래그
        # ----------------------------------------------------------------------------------------#
        # region INITIALIZE
        dir = os.path.dirname(os.path.abspath(__file__))
        sim_name = "ca_pf_integ_test"
        set_initial_variables(self, dir, sim_name)

        self.guid_var.init_pos = np.array([0.0, 0.0, 4.0])
        self.offset = np.array([self.first_global_start_wp[0]*240/1024, self.first_global_start_wp[1]*240/1024, self.first_global_start_wp[2]])

        self.offboard_mode.attitude = True
        # endregion
        # -----------------------------------------------------------------------------------------#
        # region PUBLISHERS
        # PX4 publisher
        self.pub_px4 = PX4Publisher(self)
        self.pub_px4.declareVehicleCommandPublisher()
        self.pub_px4.declareOffboardControlModePublisher()
        self.pub_px4.declareAttitudeCommandPublisher()                  # Declare PX4 Attitude Command Publisher
        # self.pub_px4.declareFusionWeightPublisher()
        self.pub_px4.declareTrajectorySetpointPublisher()
        self.weight_publisher = self.create_publisher(FusionWeight, '/fmu/in/fusion_weight', self.qos_profile_px4)

        # module data publisher
        self.pub_module = ModulePublisher(self)
        self.pub_module.declareLocalWaypointPublisherToPF()
        self.pub_module.declareModeFlagPublisherToCC()
        self.pub_module.declareVehicleModePublisher()

        # Import custom message for global waypoint
        from custom_msgs.msg import GlobalWaypointSetpoint
        self.pub_global_waypoint = self.create_publisher(GlobalWaypointSetpoint, "/global_waypoint_setpoint", 1)

        self.sub_obstacle_flag = self.create_subscription(Bool, '/obstacle_flag', self.obstacle_flag_callback, 1)
        self.sub_local_waypoint = self.create_subscription(LocalWaypointSetpoint, "/local_waypoint_setpoint_from_PP", self.local_waypoint_callback, 1)
        # heartbeat publisher
        self.pub_heartbeat = HeartbeatPublisher(self)
        self.pub_heartbeat.declareControllerHeartbeatPublisher()
        self.pub_heartbeat.declarePathPlanningHeartbeatPublisher()
        self.pub_heartbeat.declareCollisionAvoidanceHeartbeatPublisher()

        # plotter publisher (for foxglove visualization)
        self.pub_plotter = PlotterPublisher(self)
        self.pub_plotter.declareGlobalWaypointPublisherToPlotter()
        self.pub_plotter.declareLocalWaypointPublisherToPlotter()
        self.pub_plotter.declareHeadingPublisherToPlotter()
        self.pub_plotter.declareStatePublisherToPlotter()
        self.pub_plotter.declareMinDistancePublisherToPlotter()
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region PUB FUNC
        self.pub_func_heartbeat = PubFuncHeartbeat(self)
        self.pub_func_px4       = PubFuncPX4(self)
        self.pub_func_module  = PubFuncModule(self)
        self.pub_func_plotter   = PubFuncPlotter(self)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region SUBSCRIBERS
        self.sub_px4 = PX4Subscriber(self)
        self.sub_px4.declareVehicleLocalPositionSubscriber()
        self.sub_px4.declareVehicleAttitudeSubscriber()

        self.sub_cmd = CmdSubscriber(self)
        self.sub_cmd.declarePFAttitudeSetpointSubscriber()
        self.sub_cmd.declareCAVelocitySetpointSubscriber()

        self.sub_flag = FlagSubscriber(self)
        self.sub_flag.declareConveyLocalWaypointCompleteSubscriber()
        self.sub_flag.declarePFCompleteSubscriber()

        self.rand_point_sub = self.create_subscription(
            Bool,
            "/ca_rand_point_flag",
            self.rand_point_callback,
            qos_profile_sensor_data,  # best-effort sensor QoS
        )

        self.sub_etc = EtcSubscriber(self)
        self.sub_etc.declareHeadingWPIdxSubscriber()

        self.sub_hearbeat = HeartbeatSubscriber(self)
        # Todo : Collision Avoidance Hearbeat need to be added
        # self.sub_hearbeat.declareCollisionAvoidanceHeartbeatSubscriber()
        self.sub_hearbeat.declarePathFollowingHeartbeatSubscriber()
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region TIMER
        self.timer_offboard_control = MainTimer(self)
        self.timer_offboard_control.declareOffboardControlTimer(self.offboard_control_main)

        self.timer_cmd = CommandPubTimer(self)
        self.timer_cmd.declareAttitudeCommandTimer()
        # self.timer_cmd.declareFusionWeightTimer()
        self.timer_cmd.declareOffboardVelocityControlTimer()

        self.timer_heartbeat = HeartbeatTimer(self)
        self.timer_heartbeat.declareControllerHeartbeatTimer()
        # self.timer_heartbeat.declarePathPlanningHeartbeatTimer()
        self.timer_heartbeat.declareCollisionAvoidanceHeartbeatTimer()


        self.timer_weight = self.create_timer(0.01, self.weight_callback)
        # endregion
    # --------------------------------------------------------------------------------------------#
    # region MAIN CODE
    def offboard_control_main(self):

        # send offboard mode and arm mode command to px4
        if self.mode_status.DISARM == True:
            self.mode_status.TAKEOFF = True
            self.mode_status.DISARM = False

        # send offboard mode and arm mode command to px4
        if self.offboard_var.counter == self.offboard_var.flight_start_time and self.mode_status.TAKEOFF == True:
            # arm cmd to px4
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_arm_mode)
            # offboard mode cmd to px4
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_takeoff_mode)

        # takeoff after a certain period of time
        elif self.offboard_var.counter <= self.offboard_var.flight_start_time:
            self.offboard_var.counter += 1

        # check if the vehicle is ready to initial position
        if self.mode_status.TAKEOFF == True and self.state_var.z > self.guid_var.init_pos[2]:
            self.mode_status.TAKEOFF = False
        
        if self.mode_status.TAKEOFF == False and self.flags.pf_get_local_waypoint == False and not self.first_mission_completed and not self.second_mission_started and not self.first_global_wp_published:
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_position_mode)
            # Publish first global waypoint only once
            self.global_waypoint_publish(self.first_global_start_wp, self.first_global_goal_wp)
            self.first_global_wp_published = True
            self.get_logger().info('First global waypoint published')
            
        # check if path following is recieved the local waypoint
        if self.flags.pf_get_local_waypoint == True and self.flags.pf_done == False:
            self.mode_status.OFFBOARD = True
            self.mode_status.PATH_FOLLOWING = True
            self.get_logger().info('Mode Status : OFFBOARD/Path Following')
            # publish_to_plotter(self)

        # check if path following is recieved the local waypoint
        if self.mode_status.OFFBOARD == True and self.flags.pf_done == False:
            if self.initial_setting == False:
                self.initial_setting = True
                self.offboard_mode.attitude = False
                self.offboard_mode.velocity = True
                self.current_target = 1.0  # 0 (ca) or 1 (pf)
                self.prev_target = 1.0
                self.weight.fusion_weight = float(1.0)
            self.pub_func_module.publish_flags()

            # Publish waypoint data to foxglove for visualization
            publish_to_plotter(self)

            if self.mode_status.PATH_FOLLOWING == True:
                self.current_target = 1.0

            if self.mode_status.COLLISION_AVOIDANCE == True:
                self.current_target = 0.0

            self.pub_func_px4.publish_offboard_control_mode(self.offboard_mode)
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_offboard_mode)

        # First mission completed - start second mission immediately
        if self.flags.pf_done == True and self.first_mission_completed == False and self.second_mission_started == False and self.mode_status.OFFBOARD == True:
            self.get_logger().info('First mission completed! Starting second mission...')
            self.first_mission_completed = True
            self.second_mission_started = True

            # Reset flags for second mission
            self.flags.pf_done = False
            self.flags.pf_get_local_waypoint = False
            self.initial_setting = False  # Reset initial setting for second mission

            # Clear waypoints for new path planning
            self.guid_var.waypoint_x = []
            self.guid_var.waypoint_y = []
            self.guid_var.waypoint_z = []
            self.guid_var.real_wp_x = []
            self.guid_var.real_wp_y = []
            self.guid_var.real_wp_z = []
            self.guid_var.cur_wp = 0

            # Update offset for second mission
            self.offset = np.array([self.second_global_start_wp[0]*240/1024,
                                   self.second_global_start_wp[1]*240/1024,
                                   self.second_global_start_wp[2]])

            # Publish second global waypoint immediately
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_position_mode)
            self.global_waypoint_publish(self.second_global_start_wp, self.second_global_goal_wp)
            self.get_logger().info('Second global waypoint published!')

        # Second mission completed - land (only if second mission waypoint was received and executed)
        if self.flags.pf_done == True and self.first_mission_completed == True and self.second_mission_started == True and self.second_mission_waypoint_received == True and self.mode_status.LANDING == False:
            self.get_logger().info('Second mission completed! Mode Status : LANDING')
            self.mode_status.OFFBOARD = False
            self.mode_status.PATH_FOLLOWING = False
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_land_mode)

            # check if the vehicle is landed
            if np.abs(self.state_var.vz_n) < 0.05 and np.abs(self.state_var.z < 0.05):
                self.mode_status.LANDING = True
                self.get_logger().info('Vehicle is landed')

        # if the vehicle is landed, disarm the vehicle
        if self.mode_status.LANDING == True and self.mode_status.is_disarmed == False:
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_disarm_mode)
            self.mode_status.is_disarmed = True
            self.get_logger().info('Vehicle is disarmed')

        # state_logger(self)
    # endregion

    def obstacle_flag_callback(self, msg):
        # self.get_logger().info(f"Flag received: PATH_FOLLOWING: {msg.PATH_FOLLOWING}, COLLISION_AVOIDANCE: {msg.COLLISION_AVOIDANCE}") 씨발 누가 쏘는거야
        self.flags.obstacle_flag = msg.data

        # update mode status
        if self.mode_status.OFFBOARD == True:

            if self.flags.obstacle_flag == False and self.mode_status.COLLISION_AVOIDANCE == True:
                # CA 종료 시 즉시 PF로 전환하지 않고, 전환 준비만 함
                if not self.ca_exit_transition_active:
                    self.ca_exit_transition_active = True
                    self.get_logger().info("🔄 CA→PF Transition: Resuming path following")

                    # Update waypoints: Insert current position as new waypoint
                    # Keep only waypoints from current heading waypoint onward
                    self.guid_var.waypoint_x = self.guid_var.waypoint_x[self.guid_var.cur_wp:]
                    self.guid_var.waypoint_y = self.guid_var.waypoint_y[self.guid_var.cur_wp:]
                    self.guid_var.waypoint_z = self.guid_var.waypoint_z[self.guid_var.cur_wp:]

                    # Insert current position as first waypoint (collision avoidance end point)
                    self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, self.state_var.x))
                    self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, self.state_var.y))
                    self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, self.state_var.z))

                    # Update real waypoint variables
                    self.guid_var.real_wp_x = self.guid_var.waypoint_x
                    self.guid_var.real_wp_y = self.guid_var.waypoint_y
                    self.guid_var.real_wp_z = self.guid_var.waypoint_z

                    # Publish updated waypoints to path following
                    self.pub_func_module.local_waypoint_publish(False)
                    self.get_logger().info(f"✅ Updated waypoints - New first WP: ({self.state_var.x:.2f}, {self.state_var.y:.2f}, {self.state_var.z:.2f})")

                # PathFollowing will handle trajectory and yaw control
                # No need for manual yaw transition - let PathFollowing do it naturally
            elif self.flags.obstacle_flag == True and (self.mode_status.PATH_FOLLOWING == True or self.ca_exit_transition_active):
                # 충돌회피 진입 (PF → CA 또는 CA exit transition 취소)

                # CA exit transition이 진행 중이었다면 취소
                if self.ca_exit_transition_active:
                    self.get_logger().info("⚠️ CA exit transition cancelled - obstacle detected again")
                    self.ca_exit_transition_active = False
                    # weight_callback이 자동으로 현재 weight에서 0.0으로 부드럽게 전환
                else:
                    self.get_logger().info("🚨 Mode Status: COLLISION_AVOIDANCE")

                # 충돌회피 진입 시 현재 전진 속도(vx_b) 저장
                captured_vx = self.state_var.vx_b
                self.veh_vel_set.ca_initial_vx = captured_vx

                # CA 진입 시간 기록 (velocity ramping 용)
                self.veh_vel_set.ca_start_time = time.time()

                # 초기 body velocity 설정 (CA callback이 호출되기 전까지 사용)
                from .lib.common_fuctions import BodytoNED
                self.veh_vel_set.body_velocity = np.array([captured_vx, 0.0, 0.0])
                self.veh_vel_set.ned_velocity = BodytoNED(self.veh_vel_set.body_velocity, self.state_var.dcm_b2n)
                self.veh_vel_set.ned_velocity[2] = 0.0  # 고도 유지

                # 모드 전환
                self.mode_status.PATH_FOLLOWING = False
                self.mode_status.COLLISION_AVOIDANCE = True
                self.pub_func_module.publish_vehicle_mode()







            # if self.mode_status.PATH_FOLLOWING == True:
            #     z = self.guid_var.waypoint_z[self.guid_var.cur_wp]
            #     self.guid_var.waypoint_x = self.guid_var.waypoint_x[self.guid_var.cur_wp:]
            #     self.guid_var.waypoint_y = self.guid_var.waypoint_y[self.guid_var.cur_wp:]
            #     self.guid_var.waypoint_z = self.guid_var.waypoint_z[self.guid_var.cur_wp:]

            #     # self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, msg.x))
            #     # self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, msg.y))
            #     # self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, z))

            #     self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, self.state_var.x))
            #     self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, self.state_var.x))
            #     self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, self.state_var.y))
            #     self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, self.state_var.y))
            #     self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, self.state_var.z))
            #     self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, self.state_var.z))

            #     self.guid_var.real_wp_x = self.guid_var.waypoint_x
            #     self.guid_var.real_wp_y = self.guid_var.waypoint_y
            #     self.guid_var.real_wp_z = self.guid_var.waypoint_z

            #     self.pub_func_module.local_waypoint_publish(False)

    def rand_point_callback(self, msg: Bool):
        self.flags.rand_point_flag = msg.data

    def global_waypoint_publish(self, start_point, goal_point):
        msg = GlobalWaypointSetpoint()
        msg.start_point = start_point
        msg.goal_point = goal_point
        self.pub_global_waypoint.publish(msg)

    def local_waypoint_publish(self):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = True
        msg.waypoint_x = self.guid_var.waypoint_x
        msg.waypoint_y = self.guid_var.waypoint_y
        msg.waypoint_z = self.guid_var.waypoint_z
        self.local_waypoint_publisher_to_pf.publish(msg)
        self.get_logger().info(f'waypoint_x: {self.guid_var.waypoint_x}')
        self.get_logger().info(f'waypoint_y: {self.guid_var.waypoint_y}')
        self.get_logger().info(f'waypoint_z: {self.guid_var.waypoint_z}')

    def weight_callback(self):
        self.weight.timestamp = int(Clock().now().nanoseconds / 1000)  # time in microseconds
        if not self.mode_status.OFFBOARD:
            self.weight.fusion_weight = float(0.0)
            self.weight_publisher.publish(self.weight)
            return

        # CA exit transition: Start weight transition even before mode switches to PF
        if self.ca_exit_transition_active:
            target_weight = 1.0
        else:
            target_weight = 1.0 if self.mode_status.PATH_FOLLOWING else 0.0

        # 전환 시작 감지: target_weight가 변경되면 새로운 전환 시작
        if target_weight != self.prev_target:
            self.transition_start_time = time.time()
            # 현재 weight 값을 전환 시작점으로 저장
            self.transition_initial_weight = self.weight.fusion_weight
            self.transition_direction = target_weight  # 목표 weight (0.0 or 1.0)
            self.prev_target = target_weight

        # 전환 중이 아니면 target_weight 그대로 사용
        if self.transition_start_time is None:
            self.weight.fusion_weight = float(target_weight)
            self.weight_publisher.publish(self.weight)
            return

        elapsed = time.time() - self.transition_start_time

        # 전환 duration 선택
        if self.transition_direction == 1.0:  # → PF (목표: 1.0)
            duration = self.transition_duration_to_pf
        else:  # → CA (목표: 0.0)
            # CA 진입 초기 딜레이 적용
            if elapsed < self.ca_entry_delay:
                self.weight.fusion_weight = self.transition_initial_weight
                self.weight_publisher.publish(self.weight)
                return
            elapsed = elapsed - self.ca_entry_delay
            duration = self.transition_duration_to_ca

        # Smootherstep 보간: initial_weight → target_weight
        if elapsed < duration:
            t = elapsed / duration
            alpha = self.smootherstep(t)
            # 현재 weight = 시작 weight + (목표 weight - 시작 weight) * alpha
            current_weight = self.transition_initial_weight + (self.transition_direction - self.transition_initial_weight) * alpha
            self.weight.fusion_weight = float(current_weight)

            # CA exit transition: Switch to PF mode when fusion_weight reaches 0.5
            if self.ca_exit_transition_active and current_weight >= 0.5 and self.mode_status.COLLISION_AVOIDANCE:
                self.mode_status.PATH_FOLLOWING = True
                self.mode_status.COLLISION_AVOIDANCE = False
                self.ca_exit_transition_active = False
                self.pub_func_module.publish_vehicle_mode()

        else:
            # 전환 완료: 목표 weight 도달
            self.weight.fusion_weight = float(self.transition_direction)
            self.transition_start_time = None
            self.ca_exit_transition_active = False  # Reset transition flag

        self.weight_publisher.publish(self.weight)

    def smoothstep(self, t):
        """
        Smoothstep interpolation (3차 S-curve)
        입력: t ∈ [0, 1]
        출력: smooth transition from 0 to 1
        - t=0에서 미분값 0 (부드러운 시작)
        - t=1에서 미분값 0 (부드러운 종료)
        """
        if t <= 0.0:
            return 0.0
        elif t >= 1.0:
            return 1.0
        else:
            # 3t^2 - 2t^3
            return t * t * (3.0 - 2.0 * t)

    def smootherstep(self, t):
        """
        Smootherstep interpolation (5차 S-curve, 더 부드러움)
        입력: t ∈ [0, 1]
        출력: 6t^5 - 15t^4 + 10t^3
        - 1차, 2차 미분값도 0 (가속도까지 부드러움)
        """
        if t <= 0.0:
            return 0.0
        elif t >= 1.0:
            return 1.0
        else:
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def local_waypoint_callback(self, msg):
        if self.flags.pf_get_local_waypoint == False:
            self.get_logger().info('Local waypoint received from path planning')

            xy = np.array([msg.waypoint_x, msg.waypoint_y])*240/1024 # [2 X N]

            # Additional offset to align with drone's local frame
            additional_offset = 140.625  # 600 * 100/1024

            # theta = np.pi/2

            # dcm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # new_xy = dcm @ xy

            # new_init_pos = dcm @ self.guid_var.init_pos[:2]

            # self.get_logger().info(f'xy: {new_xy.shape}')

            for i in range(len(msg.waypoint_x)):
                self.guid_var.waypoint_x.append(-(float(xy[0, i] - self.offset[0] - additional_offset)))
                self.guid_var.waypoint_y.append((float(xy[1, i] - self.offset[1] + additional_offset)))
                self.guid_var.waypoint_z.append(float(msg.waypoint_z[i]))

            self.guid_var.real_wp_x = self.guid_var.waypoint_x
            self.guid_var.real_wp_y = self.guid_var.waypoint_y
            self.guid_var.real_wp_z = self.guid_var.waypoint_z

            self.local_waypoint_publish()
            self.flags.pf_get_local_waypoint = True

            # Mark that second mission waypoint is received
            if self.second_mission_started and not self.second_mission_waypoint_received:
                self.second_mission_waypoint_received = True
                self.flags.pf_done = False  # Ensure pf_done is False for second mission
                self.get_logger().info('Second mission waypoint received and ready to execute')

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
