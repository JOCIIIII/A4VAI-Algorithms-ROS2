# Librarys

# Library for common
import numpy as np
import os

# ROS libraries
import rclpy
from rclpy.node import Node

from ..lib.common_fuctions import set_initial_variables, state_logger, publish_to_plotter, set_wp
from ..lib.timer import HeartbeatTimer, MainTimer, CommandPubTimer
from ..lib.subscriber import PX4Subscriber, FlagSubscriber, CmdSubscriber, EtcSubscriber
from ..lib.publisher import PX4Publisher, HeartbeatPublisher, ModulePublisher, PlotterPublisher
from ..lib.publisher import PubFuncHeartbeat, PubFuncPX4, PubFuncModule, PubFuncPlotter

from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint

class PathFollowingTest(Node):
    def __init__(self):
        super().__init__("give_global_waypoint")

        ############################################################################################################
        # input: start_point, goal_point
        # set start and goal point
        self.start_point = [50.0, 50.0, 5.0]
        self.goal_point = [950.0, 950.0, 5.0]
        ############################################################################################################
        
        # ----------------------------------------------------------------------------------------#
        # region INITIALIZE
        dir = os.path.dirname(os.path.abspath(__file__))
        sim_name = "pf_unit_test"
        set_initial_variables(self, dir, sim_name)
        
        self.offboard_mode.attitude = True
        self.guid_var.init_pos = np.array([self.start_point[0]*400/1024, self.start_point[1]*400/1024, self.start_point[2]])
        # test mode 
        # 1 : normal, 2 : wp change
        self.test_mode = 2
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region PUBLISHERS
        self.pub_px4        = PX4Publisher(self)
        self.pub_px4.declareVehicleCommandPublisher()
        self.pub_px4.declareOffboardControlModePublisher()
        self.pub_px4.declareVehicleAttitudeSetpointPublisher()

        self.pub_module   = ModulePublisher(self)
        self.pub_module.declareLocalWaypointPublisherToPF()

        self.pub_global_waypoint = self.create_publisher(GlobalWaypointSetpoint, "/global_waypoint_setpoint", 1)
        self.sub_local_waypoint = self.create_subscription(LocalWaypointSetpoint, "/local_waypoint_setpoint_from_PP", self.local_waypoint_callback, 1)

        self.pub_heartbeat  = HeartbeatPublisher(self)
        self.pub_heartbeat.declareControllerHeartbeatPublisher()
        self.pub_heartbeat.declareCollisionAvoidanceHeartbeatPublisher()
        self.pub_heartbeat.declarePathPlanningHeartbeatPublisher()

        self.pub_plotter    = PlotterPublisher(self)
        self.pub_plotter.declareGlobalWaypointPublisherToPlotter()
        self.pub_plotter.declareLocalWaypointPublisherToPlotter()
        self.pub_plotter.declareHeadingPublisherToPlotter()
        self.pub_plotter.declareStatePublisherToPlotter()
        self.pub_plotter.declareMinDistancePublisherToPlotter()
        # end region
        # ----------------------------------------------------------------------------------------#
        # region SUBSCRIBERS
        self.sub_px4 = PX4Subscriber(self)
        self.sub_px4.declareVehicleLocalPositionSubscriber(self.state_var)

        self.sub_cmd = CmdSubscriber(self)
        self.sub_cmd.declarePFAttitudeSetpointSubscriber(self.veh_att_set)

        self.sub_flag = FlagSubscriber(self)
        self.sub_flag.declareConveyLocalWaypointCompleteSubscriber(self.mode_flag)
        self.sub_flag.declarePFCompleteSubscriber(self.mode_flag)

        self.sub_etc = EtcSubscriber(self)
        self.sub_etc.declareHeadingWPIdxSubscriber(self.guid_var)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region PUB FUNC
        self.pub_func_heartbeat = PubFuncHeartbeat(self)
        self.pub_func_px4       = PubFuncPX4(self)
        self.pub_func_module  = PubFuncModule(self)
        self.pub_func_plotter   = PubFuncPlotter(self)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region TIMER
        self.timer_offboard_control = MainTimer(self, self.offboard_var)
        self.timer_offboard_control.declareOffboardControlTimer(self.offboard_control_main)

        self.timer_cmd = CommandPubTimer(self, self.offboard_var)
        self.timer_cmd.declareOffboardAttitudeControlTimer(self.mode_flag, self.veh_att_set, self.pub_func_px4)
        
        self.timer_heartbeat = HeartbeatTimer(self, self.offboard_var, self.pub_func_heartbeat)
        self.timer_heartbeat.declareControllerHeartbeatTimer()
        self.timer_heartbeat.declareCollisionAvoidanceHeartbeatTimer()
        # endregion
    # ----------------------------------------------------------------------------------------#
    # region MAIN CODE
    def offboard_control_main(self):

        # send offboard mode and arm mode command to px4
        if self.offboard_var.counter == self.offboard_var.flight_start_time and self.mode_flag.is_takeoff == False:
            # arm cmd to px4
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_arm_mode)
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_takeoff_mode)
            # offboard mode cmd to px4

        # takeoff after a certain period of time
        elif self.offboard_var.counter <= self.offboard_var.flight_start_time:
            self.offboard_var.counter += 1

        # check if the vehicle is ready to initial position
        if self.mode_flag.is_takeoff == False and self.state_var.z > self.guid_var.init_pos[2]:
            self.mode_flag.is_takeoff = True
            self.get_logger().info('Vehicle is reached to initial position')

        # if the vehicle was taken off send local waypoint to path following and wait in position mode
        if self.mode_flag.is_takeoff == True and self.mode_flag.pf_recieved_lw == False:
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_position_mode)
            self.global_waypoint_publish(self.start_point, self.goal_point)
            
        # check if path following is recieved the local waypoint
        if self.mode_flag.pf_recieved_lw == True and self.mode_flag.pf_done == False:
            self.mode_flag.is_pf = True
            publish_to_plotter(self)

            # offboard mode
            self.pub_func_px4.publish_offboard_control_mode(self.offboard_mode)
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_offboard_mode)
        
        if self.mode_flag.pf_done == True and self.mode_flag.is_landed == False:
            self.mode_flag.is_pf = False
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_land_mode)

            # check if the vehicle is landed
            if np.abs(self.state_var.vz_n) < 0.05 and np.abs(self.state_var.z < 0.05):
                self.mode_flag.is_landed = True
                self.get_logger().info('Vehicle is landed')

        # if the vehicle is landed, disarm the vehicle
        if self.mode_flag.is_landed == True and self.mode_flag.is_disarmed == False:
            self.pub_func_px4.publish_vehicle_command(self.modes.prm_disarm_mode)    
            self.mode_flag.is_disarmed = True
            self.get_logger().info('Vehicle is disarmed')        

        state_logger(self)
    # endregion
    # ----------------------------------------------------------------------------------------#


    def global_waypoint_publish(self, start_point, goal_point):
        msg = GlobalWaypointSetpoint()
        msg.start_point = start_point
        msg.goal_point = goal_point
        self.pub_global_waypoint.publish(msg)

    def local_waypoint_callback(self, msg):
        if self.mode_flag.pf_recieved_lw == False:

            xy = np.array([msg.waypoint_x, msg.waypoint_y]) # [2 X N]
            
            theta = np.pi/2

            dcm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            new_xy = dcm @ xy

            new_init_pos = dcm @ self.guid_var.init_pos[:2]

            self.get_logger().info(f'xy: {new_xy.shape}')

            for i in range(len(msg.waypoint_x)):
                self.guid_var.waypoint_x.append(float(new_xy[0, i]*400/1024 - new_init_pos[0]))
                self.guid_var.waypoint_y.append(float(new_xy[1, i]*400/1024 - new_init_pos[1]))
                self.guid_var.waypoint_z.append(float(msg.waypoint_z[i]) + 1) 
            self.guid_var.real_wp_x = self.guid_var.waypoint_x
            self.guid_var.real_wp_y = self.guid_var.waypoint_y
            self.guid_var.real_wp_z = self.guid_var.waypoint_z

            self.local_waypoint_publish()
            self.mode_flag.pf_recieved_lw = True

        self.get_logger().info('Local waypoint recieved from path planning')

    def local_waypoint_publish(self):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = True
        msg.waypoint_x = self.guid_var.waypoint_x
        msg.waypoint_y = self.guid_var.waypoint_y
        msg.waypoint_z = self.guid_var.waypoint_z
        self.local_waypoint_publisher_pf.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    path_planning_test = PathFollowingTest()
    rclpy.spin(path_planning_test)
    path_planning_test.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()