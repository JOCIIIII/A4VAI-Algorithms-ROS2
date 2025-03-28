# Librarys

# Library for common
import numpy as np
import os

# ROS libraries
import rclpy
from rclpy.node import Node

# Custom libraries
from ..lib.common_fuctions import set_initial_variables, state_logger, publish_to_plotter, set_wp
from ..lib.timer import HeartbeatTimer, MainTimer, CommandPubTimer
from ..lib.subscriber import PX4Subscriber, FlagSubscriber, CmdSubscriber, HeartbeatSubscriber, EtcSubscriber
from ..lib.publisher import PX4Publisher, HeartbeatPublisher, ModulePublisher, PlotterPublisher
from ..lib.publisher import PubFuncHeartbeat, PubFuncPX4, PubFuncModule, PubFuncPlotter
from ..lib.data_class import *

# custom message
from custom_msgs.msg import StateFlag
# ----------------------------------------------------------------------------------------#

class CAPFIntegrationTest(Node):
    def __init__(self):
        super().__init__("ca_pf_integ_test")
        # ----------------------------------------------------------------------------------------#
        # region INITIALIZE
        dir = os.path.dirname(os.path.abspath(__file__))
        sim_name = "ca_pf_integ_test"
        set_initial_variables(self, dir, sim_name)
        
        self.offboard_mode.attitude = True
        # endregion
        # -----------------------------------------------------------------------------------------#
        # region PUBLISHERS
        # PX4 publisher
        self.pub_px4 = PX4Publisher(self)
        self.pub_px4.declareVehicleCommandPublisher()
        self.pub_px4.declareOffboardControlModePublisher()
        self.pub_px4.declareVehicleAttitudeSetpointPublisher()
        self.pub_px4.declareTrajectorySetpointPublisher()
        # module data publisher
        self.pub_module = ModulePublisher(self)
        self.pub_module.declareLocalWaypointPublisherToPF()
        self.pub_module.declareModeFlagPublisherToCC()

        self.sub_flag = self.create_subscription(StateFlag, '/mode_flag2control', self.flag_callback, 1)
        # heartbeat publisher
        self.pub_heartbeat = HeartbeatPublisher(self)
        self.pub_heartbeat.declareControllerHeartbeatPublisher()
        self.pub_heartbeat.declarePathPlanningHeartbeatPublisher()
        # plotter publisher
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
        self.sub_px4.declareVehicleLocalPositionSubscriber(self.state_var)
        self.sub_px4.declareVehicleAttitudeSubscriber(self.state_var)

        self.sub_cmd = CmdSubscriber(self)
        self.sub_cmd.declarePFAttitudeSetpointSubscriber(self.veh_att_set)
        self.sub_cmd.declareCAVelocitySetpointSubscriber(self.veh_vel_set, self.state_var, self.ca_var)

        self.sub_flag = FlagSubscriber(self)
        self.sub_flag.declareConveyLocalWaypointCompleteSubscriber(self.mode_flag)
        self.sub_flag.declarePFCompleteSubscriber(self.mode_flag)

        self.sub_etc = EtcSubscriber(self)
        self.sub_etc.declareHeadingWPIdxSubscriber(self.guid_var)

        self.sub_hearbeat = HeartbeatSubscriber(self)
        self.sub_hearbeat.declareCollisionAvoidanceHeartbeatSubscriber(self.offboard_var)
        self.sub_hearbeat.declarePathFollowingHeartbeatSubscriber(self.offboard_var)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region TIMER
        self.timer_offboard_control = MainTimer(self, self.offboard_var)
        self.timer_offboard_control.declareOffboardControlTimer(self.offboard_control_main)

        self.timer_cmd = CommandPubTimer(self, self.offboard_var)
        self.timer_cmd.declareOffboardAttitudeControlTimer(self.mode_flag, self.veh_att_set, self.pub_func_px4)
        self.timer_cmd.declareOffboardVelocityControlTimer(self.mode_flag, self.veh_vel_set, self.pub_func_px4)

        self.timer_heartbeat = HeartbeatTimer(self, self.offboard_var, self.pub_func_heartbeat)
        self.timer_heartbeat.declareControllerHeartbeatTimer()
        self.timer_heartbeat.declarePathPlanningHeartbeatTimer()
        # endregion
    # --------------------------------------------------------------------------------------------#
    # region MAIN CODE
    def offboard_control_main(self):
        # self.get_logger().info(str(self.guid_var.waypoint_x))
        if self.offboard_var.ca_heartbeat == True and self.offboard_var.pf_heartbeat == True:
            
            # send offboard mode and arm mode command to px4
            if self.mode_flag.is_standby == True:
                self.mode_flag.is_takeoff = True
                self.mode_flag.is_standby = False
            
            if self.offboard_var.counter == self.offboard_var.flight_start_time and self.mode_flag.is_takeoff == True:
                # arm cmd to px4
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_arm_mode)
                # offboard mode cmd to px4
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_takeoff_mode)

            # takeoff after a certain period of time
            elif self.offboard_var.counter <= self.offboard_var.flight_start_time:
                self.offboard_var.counter += 1

            # check if the vehicle is ready to initial position
            if self.mode_flag.is_takeoff == True and self.state_var.z > self.guid_var.init_pos[2]:
                self.mode_flag.is_takeoff = False
                self.mode_flag.is_pp_mode = True
                self.get_logger().info('Vehicle is reached to initial position')
                set_wp(self)

            if self.mode_flag.is_pp_mode == True:
                self.pub_func_module.local_waypoint_publish(True)
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_position_mode)
            
            if self.mode_flag.pf_recieved_lw == True and self.mode_flag.is_offboard == False:
                self.mode_flag.is_pp_mode = False
                self.mode_flag.is_offboard = True
                self.mode_flag.is_pf = True
                self.get_logger().info('Vehicle is in offboard mode')

            # check if path following is recieved the local waypoint
            if self.mode_flag.is_offboard == True and self.mode_flag.pf_done == False:
                publish_to_plotter(self)
                self.pub_func_module.publish_flags()

                if self.mode_flag.is_pf == True:
                    self.offboard_mode.attitude = True
                    self.offboard_mode.velocity = False
                
                if self.mode_flag.is_ca == True:
                    self.offboard_mode.attitude = False
                    self.offboard_mode.velocity = True

                self.pub_func_px4.publish_offboard_control_mode(self.offboard_mode)
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_offboard_mode)
                
            if self.mode_flag.pf_done == True and self.mode_flag.is_landed == False:
                self.mode_flag.is_offboard = False
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
            # self.get_logger().info(str(self.ca_var.depth_min_distance))
            state_logger(self)
    # endregion

    def flag_callback(self, msg):
        # self.get_logger().info(f"Flag received: is_pf: {msg.is_pf}, is_ca: {msg.is_ca}") 씨발 누가 쏘는거야
        self.mode_flag.is_ca = msg.is_ca
        self.mode_flag.is_pf = msg.is_pf
        if self.mode_flag.is_pf == True:
            self.get_logger().info("is_pf is True")
            z = self.guid_var.waypoint_z[self.guid_var.cur_wp]
            self.guid_var.waypoint_x = self.guid_var.waypoint_x[self.guid_var.cur_wp:]
            self.guid_var.waypoint_y = self.guid_var.waypoint_y[self.guid_var.cur_wp:]
            self.guid_var.waypoint_z = self.guid_var.waypoint_z[self.guid_var.cur_wp:]

            # self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, msg.x))
            # self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, msg.y))
            # self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, z))

            self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, self.state_var.x))
            self.guid_var.waypoint_x = list(np.insert(self.guid_var.waypoint_x, 0, self.state_var.x))
            self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, self.state_var.y))
            self.guid_var.waypoint_y = list(np.insert(self.guid_var.waypoint_y, 0, self.state_var.y))
            self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, self.state_var.z))
            self.guid_var.waypoint_z = list(np.insert(self.guid_var.waypoint_z, 0, self.state_var.z))

            self.guid_var.real_wp_x = self.guid_var.waypoint_x
            self.guid_var.real_wp_y = self.guid_var.waypoint_y
            self.guid_var.real_wp_z = self.guid_var.waypoint_z

            self.pub_func_module.local_waypoint_publish(False)

def main(args=None):
    rclpy.init(args=args)
    ca_pf_integration_test = CAPFIntegrationTest()
    rclpy.spin(ca_pf_integration_test)
    ca_pf_integration_test.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
