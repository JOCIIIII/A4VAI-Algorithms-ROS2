import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math
import numpy as np

from px4_msgs.msg import VehicleCommand, OffboardControlMode , TrajectorySetpoint, VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleLocalPosition , VehicleAttitude, VehicleAngularVelocity, VehicleStatus, VehicleGlobalPosition

from custom_msgs.msg import LocalWaypointSetpoint, ConveyLocalWaypointComplete
from custom_msgs.msg import ControllerHeartbeat, PathFollowingHeartbeat, PathPlanningHeartbeat

from .give_global_waypoint import GiveGlobalWaypoint

class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        ## initialize flag
        # flag of start
        self.take_off_flag                      =   False
        self.initial_position_flag              =   False

        # flag of conveying local waypoint to another node
        self.path_planning_complete             =   False       # flag whether path planning is complete 
        self.convey_local_waypoint_to_PF_start  =   False
        self.convey_local_waypoint_is_complete  =   False       # flag whether path planning convey to path following
        self.path_following_flag = False

        # heartbeat signal of another module node
        self.path_planning_heartbeat            =   False
        self.path_following_heartbeat           =   False


        ## initialize State Variable
        # NED Position 
        self.x      =   0       # [m]
        self.y      =   0       # [m]
        self.z      =   0       # [m]

        # NED Velocity
        self.v_x    =   0       # [m/s]
        self.v_y    =   0       # [m/s]
        self.v_z    =   0       # [m/s]

        # Euler Angle
        self.psi    =   0
        self.theta  =   0
        self.phi    =   0

        # Body frame Angular Velocity
        self.p      =   0       # [rad/s]
        self.q      =   0       # [rad/s]
        self.r      =   0       # [rad/s]

        # initial position
        self.initial_position = [0.0, 0.0, -11.0]


        ## initialize path planning parameter
        # path planning global waypoint [x, z, y]
        self.start_point        =   [50.0, 5.0, 50.0]
        self.goal_point         =   [950.0, 5.0, 950.0]

        # path planning waypoint list
        self.waypoint_x         =   []
        self.waypoint_y         =   []
        self.waypoint_z         =   []

        #.. parameter - offboard control mode
        class prm_msg_off_con_mod:
            def __init__(self):        
                self.position        =   False
                self.velocity        =   False
                self.acceleration    =   False
                self.attitude        =   False
                self.body_rate       =   False
                
        self.prm_off_con_mod            =   prm_msg_off_con_mod()
        self.prm_off_con_mod.position   =   True

        # offboard times
        self.offboard_setpoint_counter = 0
        self.offboard_start_flight_time = 10

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.agl = 0
        ###.. - Start - set variable of publisher msg for PX4 - ROS2  ..###
        #
        #.. parameter - vehicle command 
        class prm_msg_veh_com:
            def __init__(self):
                self.CMD_mode   =   np.NaN
                self.params     =   np.NaN * np.ones(2)
                # self.params     =   np.NaN * np.ones(8) # maximum
                
        # arm command in ref. [2, 3] 
        self.prm_arm_mode                 =   prm_msg_veh_com()
        self.prm_arm_mode.CMD_mode        =   VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        self.prm_arm_mode.params[0]       =   1
                
        # disarm command in ref. [2, 3]
        self.prm_disarm_mode              =   prm_msg_veh_com()
        self.prm_disarm_mode.CMD_mode     =   VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        self.prm_disarm_mode.params[0]    =   0
        
        # offboard mode command in ref. [3]
        self.prm_offboard_mode            =   prm_msg_veh_com()
        self.prm_offboard_mode.CMD_mode   =   VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        self.prm_offboard_mode.params[0]  =   1
        self.prm_offboard_mode.params[1]  =   6
        
        #.. parameter - offboard control mode
        class prm_msg_off_con_mod:
            def __init__(self):        
                self.position        =   False
                self.velocity        =   False
                self.acceleration    =   False
                self.attitude        =   False
                self.body_rate       =   False
                
        self.prm_off_con_mod            =   prm_msg_off_con_mod()
        self.prm_off_con_mod.position   =   True

        class msg_veh_trj_set:
            def __init__(self):
                self.pos_NED    =   np.zeros(3)
                self.yaw_rad    =   0.
        
        self.veh_trj_set    =   msg_veh_trj_set()

        #.. variable - vehicle attitude setpoint
        class var_msg_veh_att_set:
            def __init__(self):
                self.roll_body  =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.pitch_body =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.yaw_body   =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.q_d        =   [np.NaN, np.NaN, np.NaN, np.NaN]
                self.yaw_sp_move_rate   =   np.NaN      # rad/s (commanded by user)
                
                # For clarification: For multicopters thrust_body[0] and thrust[1] are usually 0 and thrust[2] is the negative throttle demand.
                # For fixed wings thrust_x is the throttle demand and thrust_y, thrust_z will usually be zero.
                self.thrust_body    =   np.NaN * np.ones(3) # Normalized thrust command in body NED frame [-1,1]
                
        self.veh_att_set    =   var_msg_veh_att_set()
        #
        ###.. -  End  - set variable of publisher msg for PX4 - ROS2  ..###

        ## publiser and subscriber
        # declare publisher from ROS2 to PX4
        self.vehicle_command_publisher              =   self.create_publisher(VehicleCommand,             '/fmu/in/vehicle_command',           qos_profile)
        self.offboard_control_mode_publisher        =   self.create_publisher(OffboardControlMode,        '/fmu/in/offboard_control_mode',     qos_profile)
        self.trajectory_setpoint_publisher          =   self.create_publisher(TrajectorySetpoint,         '/fmu/in/trajectory_setpoint',       qos_profile)
        self.vehicle_attitude_setpoint_publisher    =   self.create_publisher(VehicleAttitudeSetpoint,    '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.agl_subscriber                         =   self.create_subscription(VehicleGlobalPosition,   '/fmu/out/vehicle_global_position',   self.agl_callback,   qos_profile)
        # declare subscriber from PX4 to ROS2 
        self.vehicle_local_position_subscriber      =   self.create_subscription(VehicleLocalPosition,    '/fmu/out/vehicle_local_position',   self.vehicle_local_position_callback,   qos_profile)
        self.vehicle_attitude_subscriber            =   self.create_subscription(VehicleAttitude,         '/fmu/out/vehicle_attitude',         self.vehicle_attitude_callback,         qos_profile)
        self.vehicle_angular_velocity_subscriber    =   self.create_subscription(VehicleAngularVelocity , '/fmu/out/vehicle_angular_velocity', self.vehicle_angular_velocity_callback, qos_profile)
        self.vehicle_status_subscriber              =   self.create_subscription(VehicleStatus,           '/fmu/out/vehicle_status',           self.vehicle_status_callback,           qos_profile)

        # declare subscriber from path planning
        self.local_waypoint_subscriber                  =   self.create_subscription(LocalWaypointSetpoint,         '/local_waypoint_setpoint_from_PP', self.path_planning_call_back,                   10)
        self.path_planning_heartbeat_subscriber         =   self.create_subscription(PathPlanningHeartbeat,         '/path_planning_heartbeat',         self.path_planning_heartbeat_call_back,         10)
        
        # declare local waypoint publisher to path following
        self.local_waypoint_publisher                   =   self.create_publisher(LocalWaypointSetpoint,            '/local_waypoint_setpoint_to_PF', 10)
        self.heartbeat_publisher                        =   self.create_publisher(ControllerHeartbeat,              '/controller_heartbeat', 10)
        
        # declare subscriber from pathfollowing
        self.PF_attitude_setpoint_subscriber_           =   self.create_subscription(VehicleAttitudeSetpoint,       '/pf_att_2_control',                 self.PF_Att2Control_callback,                  10)
        self.convey_local_waypoint_complete_subscriber  =   self.create_subscription(ConveyLocalWaypointComplete,   '/convey_local_waypoint_complete',   self.convey_local_waypoint_complete_call_back, 10) 
        self.path_following_heartbeat_subscriber        =   self.create_subscription(PathFollowingHeartbeat,        '/path_following_heartbeat',         self.path_following_heartbeat_call_back,       10)

        # algorithm timer
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)

        period_offboard_control_mode =   0.2         # required about 5Hz for attitude control (proof that the external controller is healthy
        self.offboard_main_timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_main)

        period_offboard_att_ctrl    =   0.004           # required 250Hz at least for attitude control
        self.attitude_control_call_timer =  self.create_timer(period_offboard_att_ctrl, self.publisher_vehicle_attitude_setpoint)

    # main code
    def offboard_control_main(self):
        # check another module nodes alive
        if self.path_following_heartbeat == True and self.path_planning_heartbeat == True:
            # send offboard mode and arm mode command to px4
            if self.offboard_setpoint_counter == self.offboard_start_flight_time :
                # offboard mode cmd to px4
                self.publish_vehicle_command(self.prm_offboard_mode)
                # arm cmd to px4
                self.publish_vehicle_command(self.prm_arm_mode)
    
            # takeoff after a certain period of time
            elif self.offboard_setpoint_counter <= self.offboard_start_flight_time:
                self.offboard_setpoint_counter += 1
    
            # send offboard heartbeat signal to px4 
            self.publish_offboard_control_mode(self.prm_off_con_mod)
    
            # check initial position
            if self.initial_position_flag == True:
                
                # stay initial position untill transmit local waypoint to path following is complete
                if self.convey_local_waypoint_is_complete == False:
                    self.takeoff_and_go_initial_position()
    
                    # check path planning complete
                    if self.path_planning_complete == False:
                    
                        # give global waypoint to path planning and path planning start
                        give_global_waypoint = GiveGlobalWaypoint()
                        give_global_waypoint.global_waypoint_publish(self.start_point, self.goal_point)
                        give_global_waypoint.destroy_node()
                    else:
                        # send local waypoint to pathfollowing
                        self.local_waypoint_publish()
                    
                # do path following if local waypoint transmit to path following is complete 
                else:
                    self.prm_off_con_mod.position   =   False
                    self.prm_off_con_mod.attitude   =   True
                    self.publish_offboard_control_mode(self.prm_off_con_mod)
                    self.path_following_flag = True
    
            # go initial position if not in initial position 
            else:
                self.veh_trj_set.pos_NED    =   self.initial_position
                self.takeoff_and_go_initial_position()
        else:
            pass

    # quaternion to euler
    def Quaternion2Euler(self, w, x, y, z):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1) * 57.2958

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2) * 57.2958

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4) * 57.2958

        return roll, pitch, yaw

    # subscribe local waypoint and path planning complete flag and publish local waypoint to path following
    def path_planning_call_back(self, msg):
        self.path_planning_complete = msg.path_planning_complete 
        self.waypoint_x             = msg.waypoint_x 
        self.waypoint_y             = msg.waypoint_y
        self.waypoint_z             = msg.waypoint_z
        print("                                          ")
        print("=====   Path Planning Complete!!     =====")
        print("                                          ")

    # publish local waypoint to path following
    def local_waypoint_publish(self):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = self.path_planning_complete
        msg.waypoint_x             = self.waypoint_x
        msg.waypoint_y             = self.waypoint_y
        msg.waypoint_z             = (np.add(self.waypoint_z, float(self.agl-5))).tolist()
        self.local_waypoint_publisher.publish(msg)

    # subscribe convey local waypoint complete flag from path following
    def convey_local_waypoint_complete_call_back(self, msg):
        self.convey_local_waypoint_is_complete = msg.convey_local_waypoint_is_complete


    ## heartbeat signal for debug mode
    # publish controller heartbeat signal to another module's nodes
    def publish_heartbeat(self):
        msg = ControllerHeartbeat()
        msg.controller_heartbeat = True
        self.heartbeat_publisher.publish(msg)
    
    # subscribe path planning heartbeat signal
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.path_planning_heartbeat

    # subscribe path following heartbeat signal
    def path_following_heartbeat_call_back(self,msg):
        self.path_following_heartbeat = msg.path_following_heartbeat

    def agl_callback(self,msg):
        self.agl = msg.alt_ellipsoid

    ## publish to px4
    # publish_vehicle_command to px4
    def publish_vehicle_command(self, prm_veh_com):
        msg                 =   VehicleCommand()
        msg.param1          =   prm_veh_com.params[0]
        msg.param2          =   prm_veh_com.params[1]
        msg.command         =   prm_veh_com.CMD_mode
        # values below are in [3]
        msg.target_system   =   1
        msg.target_component=   1
        msg.source_system   =   1
        msg.source_component=   1
        msg.from_external   =   True
        self.vehicle_command_publisher.publish(msg)

    # publish offboard control mode to px4
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg                 =   OffboardControlMode()
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher.publish(msg)

    # publish position offboard command to px4
    def publish_trajectory_setpoint(self,veh_trj_set):
        msg                 =   TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   veh_trj_set.pos_NED
        msg.yaw             =   veh_trj_set.yaw_rad
        self.trajectory_setpoint_publisher.publish(msg)

    # publish attitude offboard command to px4
    def publisher_vehicle_attitude_setpoint(self):
        if self.path_following_flag == True:
            msg                     =   VehicleAttitudeSetpoint()
            msg.roll_body           =   self.veh_att_set.roll_body
            msg.pitch_body          =   self.veh_att_set.pitch_body
            msg.yaw_body            =   self.veh_att_set.yaw_body
            msg.yaw_sp_move_rate    =   self.veh_att_set.yaw_sp_move_rate
            msg.q_d[0]              =   self.veh_att_set.q_d[0]
            msg.q_d[1]              =   self.veh_att_set.q_d[1]
            msg.q_d[2]              =   self.veh_att_set.q_d[2]
            msg.q_d[3]              =   self.veh_att_set.q_d[3]
            msg.thrust_body[0]      =   0.
            msg.thrust_body[1]      =   0.
            msg.thrust_body[2]      =   self.veh_att_set.thrust_body[2]
            self.vehicle_attitude_setpoint_publisher.publish(msg)
        else:
            pass

    # offboard control toward initial point
    def takeoff_and_go_initial_position(self):
        self.publish_trajectory_setpoint(self.veh_trj_set)
        if abs(self.z - self.initial_position[2]) < 0.3:
            self.initial_position_flag = True


    ## subscribe from px4
    # update position and velocity from px4
    def vehicle_local_position_callback(self, msg):
        # update NED position 
        self.x      =   msg.x
        self.y      =   msg.y
        self.z      =   msg.z
        # update NED velocity
        self.v_x    =   msg.vx
        self.v_y    =   msg.vy
        self.v_z    =   msg.vz

    # update attitude from px4
    def vehicle_attitude_callback(self, msg):
        self.psi , self.theta, self.phi     =   self.Quaternion2Euler(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
     
    # update body angular velocity from px4
    def vehicle_angular_velocity_callback(self, msg):
        self.p    =   msg.xyz[0]
        self.q    =   msg.xyz[1]
        self.r    =   msg.xyz[2]
    # update vehicle status from px4
    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status


    # update attitude offboard command from path following
    def PF_Att2Control_callback(self, msg):
        self.veh_att_set.roll_body          =   msg.roll_body
        self.veh_att_set.pitch_body         =   msg.pitch_body
        self.veh_att_set.yaw_body           =   msg.yaw_body
        self.veh_att_set.yaw_sp_move_rate   =   msg.yaw_sp_move_rate
        self.veh_att_set.q_d[0]             =   msg.q_d[0]
        self.veh_att_set.q_d[1]             =   msg.q_d[1]
        self.veh_att_set.q_d[2]             =   msg.q_d[2]
        self.veh_att_set.q_d[3]             =   msg.q_d[3]
        self.veh_att_set.thrust_body[0]     =   msg.thrust_body[0]
        self.veh_att_set.thrust_body[1]     =   msg.thrust_body[1]
        self.veh_att_set.thrust_body[2]     =   msg.thrust_body[2]




def main(args=None):
    print("======================================================")
    print("------------- main() in controller.py ----------------")
    print("======================================================")
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()
