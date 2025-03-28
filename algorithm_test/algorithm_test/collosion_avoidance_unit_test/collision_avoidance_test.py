# Librarys

# Library for common
import numpy as np
import os

# ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
# ----------------------------------------------------------------------------------------#
# PX4 msgs libraries
from px4_msgs.msg import TrajectorySetpoint

from ..lib.common_fuctions import set_initial_variables, state_logger, publish_to_plotter, BodytoNED, set_wp
from ..lib.publisher import PX4Publisher, HeartbeatPublisher, PlotterPublisher, ModulePublisher
from ..lib.publisher import PubFuncHeartbeat, PubFuncPX4, PubFuncModule, PubFuncPlotter
from ..lib.subscriber import PX4Subscriber, CmdSubscriber, HeartbeatSubscriber, MissionSubscriber
from ..lib.timer import HeartbeatTimer, MainTimer, CommandPubTimer
# ----------------------------------------------------------------------------------------#
class CollisionAvoidanceTest(Node):
    def __init__(self):
        super().__init__("collision_avoidance_test")
        # ----------------------------------------------------------------------------------------#
        # region INITIALIZE
        dir = os.path.dirname(os.path.abspath(__file__))
        sim_name = "ca_unit_test"
        set_initial_variables(self, dir, sim_name)

        self.offboard_mode.velocity = True

        self.yaw_cmd_rad = 0.0
        self.vel_ned_cmd_normal = np.zeros(3)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region PUBLISHERS
        self.pub_px4 = PX4Publisher(self)
        self.pub_px4.declareVehicleCommandPublisher()
        self.pub_px4.declareOffboardControlModePublisher()
        self.pub_px4.declareTrajectorySetpointPublisher()

        self.pub_module = ModulePublisher(self)
        self.pub_module.declareLocalWaypointPublisherToPF()

        self.pub_heartbeat = HeartbeatPublisher(self)
        self.pub_heartbeat.declareControllerHeartbeatPublisher()
        self.pub_heartbeat.declarePathFollowingHeartbeatPublisher()
        self.pub_heartbeat.declarePathPlanningHeartbeatPublisher()

        self.pub_plotter = PlotterPublisher(self)
        self.pub_plotter.declareGlobalWaypointPublisherToPlotter()
        self.pub_plotter.declareLocalWaypointPublisherToPlotter()
        self.pub_plotter.declareHeadingPublisherToPlotter()
        self.pub_plotter.declareStatePublisherToPlotter()
        self.pub_plotter.declareMinDistancePublisherToPlotter()
        # end region
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
        self.sub_cmd.declareCAVelocitySetpointSubscriber(self.veh_vel_set, self.state_var, self.ca_var)

        self.sub_mission = MissionSubscriber(self)
        self.sub_mission.declareLidarSubscriber(self.state_var, self.guid_var, self.mode_flag, self.ca_var, self.pub_func_module)
        self.sub_mission.declareDepthSubscriber(self.mode_flag, self.ca_var)

        self.sub_hearbeat = HeartbeatSubscriber(self)
        self.sub_hearbeat.declareCollisionAvoidanceHeartbeatSubscriber(self.offboard_var)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region TIMER
        self.timer_offboard_control = MainTimer(self, self.offboard_var)
        self.timer_offboard_control.declareOffboardControlTimer(self.offboard_control_main)

        self.timer_cmd = CommandPubTimer(self, self.offboard_var)
        self.timer_cmd.declareOffboardVelocityControlTimer(self.mode_flag, self.veh_vel_set, self.pub_func_px4)

        self.timer_heartbeat = HeartbeatTimer(self, self.offboard_var, self.pub_func_heartbeat)
        self.timer_heartbeat.declareControllerHeartbeatTimer()
        self.timer_heartbeat.declarePathPlanningHeartbeatTimer()
        self.timer_heartbeat.declarePathFollowingHeartbeatTimer()
        # endregion
        self.set_forward_cmd()
    # ----------------------------------------------------------------------------------------#
    # region MAIN CODE
    def offboard_control_main(self):
        if self.offboard_var.ca_heartbeat == True:

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

                self.mode_flag.is_offboard = True

                self.offboard_mode.attitude = False
                self.offboard_mode.velocity = True

                self.get_logger().info('Vehicle is reached to initial position')
                self.pub_func_px4.publish_offboard_control_mode(self.offboard_mode)
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_offboard_mode)
                set_wp(self)
    
            # if the vehicle was taken off send local waypoint to path following and wait in position mode
            if not self.mode_flag.is_takeoff:
                
                # publish_to_plotter(self)

                if not self.mode_flag.is_ca:
                    self.mode_flag.is_manual = True
                else:
                    self.mode_flag.is_manual = False

                if self.mode_flag.is_manual:
                    self.get_logger().info("Going to designated position")
                else:
                    self.get_logger().info("Collision avoidance mode")
                    
                self.get_logger().info(str(self.veh_vel_set.ned_velocity))

                self.pub_func_px4.publish_offboard_control_mode(self.offboard_mode)
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_offboard_mode)
                
        state_logger(self)
    # endregion
    # ----------------------------------------------------------------------------------------#
    # region CALCULATION FUNC
    def set_forward_cmd(self):
        self.veh_vel_set.body_velocity = np.array([3, 0, 0])
        self.veh_vel_set.ned_velocity = BodytoNED(self.veh_vel_set.body_velocity, self.state_var.dcm_b2n)
        self.veh_vel_set.yaw = 135.0 * np.pi / 180.0
    # endregion
    # ----------------------------------------------------------------------------------------#

def main(args=None):
    rclpy.init(args=args)
    path_planning_test = CollisionAvoidanceTest()
    rclpy.spin(path_planning_test)
    path_planning_test.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
