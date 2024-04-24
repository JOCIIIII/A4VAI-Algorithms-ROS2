import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math
import numpy as np

from px4_msgs.msg import VehicleCommand, OffboardControlMode , TrajectorySetpoint, VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleLocalPosition , VehicleAttitude, VehicleAngularVelocity, VehicleStatus, VehicleGlobalPosition

from custom_msgs.msg import LocalWaypointSetpoint, ConveyLocalWaypointComplete
from custom_msgs.msg import Heartbeat
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2

from .give_global_waypoint import GiveGlobalWaypoint
from sensor_msgs_py import point_cloud2

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
        self.path_following_flag                =   False
        self.collision_avoidance_flag           =   False
        self.obstacle_flag                      =   False

        # heartbeat signal of another module node
        self.path_planning_heartbeat            =   False
        self.path_following_heartbeat           =   False
        self.collision_avoidance_heartbeat      =   False

        ## initialize State Variable
        # NED Position 
        self.x      =   0       # [m]
        self.y      =   0       # [m]
        self.z      =   0       # [m]

        # NED Velocity
        self.v_x    =   0       # [m/s]
        self.v_y    =   0       # [m/s]
        self.v_z    =   0       # [m/s]
        self.u      =   0   
        self.v      =   0   
        self.w      =   0  
        # Body Velocity CMD
        self.vel_cmd_body = [0.0, 0.0, 0.0]     # [m/s]
        self.vel_cmd_ned  = [0.0, 0.0, 0.0]     # [m/s]
        self.DCM_bn = np.zeros((3,3))
        self.DCM_nb = np.zeros((3,3))
        self.vel_cmd_body_x = 0.0
        self.vel_cmd_body_y = 0.0
        self.vel_cmd_body_z = 0.0
        self.min_dist = 0
        # Euler Angle
        self.psi    =   0       # [rad]
        self.theta  =   0       # [rad]
        self.phi    =   0       # [rad]

        # Body frame Angular Velocity
        self.p      =   0       # [rad/s]
        self.q      =   0       # [rad/s]
        self.r      =   0       # [rad/s]

        # initial position
        self.initial_position = [0.0, 0.0, -11.0]

        #.. callback state_logger
        self.period_state_logger = 0.1
        self.timer  =   self.create_timer(self.period_state_logger, self.state_logger)
        self.flightlogFile = open("/home/user/ros_ws/log/flight_log.txt",'w')
        self.datalogFile = open("/home/user/ros_ws/log/data_log.txt",'w')

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
                self.pos_NED        =   np.zeros(3)     # meters
                self.vel_NED        =   np.zeros(3)     # meters/second
                self.yaw_rad        =   0.
                self.yaw_vel_rad    =   0.                    # radians/second
        
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
        self.vehicle_command_publisher              =   self.create_publisher(VehicleCommand,               '/fmu/in/vehicle_command',           qos_profile)
        self.offboard_control_mode_publisher        =   self.create_publisher(OffboardControlMode,          '/fmu/in/offboard_control_mode',     qos_profile)
        self.trajectory_setpoint_publisher          =   self.create_publisher(TrajectorySetpoint,           '/fmu/in/trajectory_setpoint',       qos_profile)
        self.vehicle_attitude_setpoint_publisher    =   self.create_publisher(VehicleAttitudeSetpoint,      '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.agl_subscriber                         =   self.create_subscription(VehicleGlobalPosition,     '/fmu/out/vehicle_global_position',   self.agl_callback,   qos_profile)
        # declare subscriber from PX4 to ROS2   
        self.vehicle_local_position_subscriber      =   self.create_subscription(VehicleLocalPosition,      '/fmu/out/vehicle_local_position',    self.vehicle_local_position_callback,   qos_profile)
        self.vehicle_attitude_subscriber            =   self.create_subscription(VehicleAttitude,           '/fmu/out/vehicle_attitude',          self.vehicle_attitude_callback,         qos_profile)
        self.vehicle_angular_velocity_subscriber    =   self.create_subscription(VehicleAngularVelocity ,   '/fmu/out/vehicle_angular_velocity',  self.vehicle_angular_velocity_callback, qos_profile)
        self.vehicle_status_subscriber              =   self.create_subscription(VehicleStatus,             '/fmu/out/vehicle_status',            self.vehicle_status_callback,           qos_profile)

        # declare subscriber from path planning
        self.local_waypoint_subscriber                  =   self.create_subscription(LocalWaypointSetpoint,         '/local_waypoint_setpoint_from_PP',     self.path_planning_call_back,                   10)
        self.path_planning_heartbeat_subscriber         =   self.create_subscription(Heartbeat,                     '/path_planning_heartbeat',             self.path_planning_heartbeat_call_back,         10)
        
        # declare local waypoint publisher to path following
        self.local_waypoint_publisher                   =   self.create_publisher(LocalWaypointSetpoint,            '/local_waypoint_setpoint_to_PF',   10)
        self.heartbeat_publisher                        =   self.create_publisher(Heartbeat,                        '/controller_heartbeat',            10)
        
        # declare subscriber from pathfollowing
        self.PF_attitude_setpoint_subscriber_           =   self.create_subscription(VehicleAttitudeSetpoint,       '/pf_att_2_control',                    self.PF_Att2Control_callback,                   10)
        self.convey_local_waypoint_complete_subscriber  =   self.create_subscription(ConveyLocalWaypointComplete,   '/convey_local_waypoint_complete',      self.convey_local_waypoint_complete_call_back,  10) 
        self.path_following_heartbeat_subscriber        =   self.create_subscription(Heartbeat,                     '/path_following_heartbeat',            self.path_following_heartbeat_call_back,        10)
        
        # declare subscriber from collision avoidance
        self.CA_velocity_setpoint_subscriber_           =   self.create_subscription(Twist,                         '/ca_vel_2_control',                    self.CA2Control_callback,                       10)
        self.collision_avoidance_heartbeat_subscriber   =   self.create_subscription(Heartbeat,                     '/collision_avoidance_heartbeat',       self.collision_avoidance_heartbeat_call_back,   10)

        self.LidarSubscriber_ = self.create_subscription(PointCloud2, '/airsim_node/Typhoon_1/lidar/RPLIDAR_A3', self.LidarCallback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        # algorithm timer
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)

        period_offboard_control_mode =   0.2         # required about 5Hz for attitude control (proof that the external controller is healthy
        self.offboard_main_timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_main)

        period_offboard_att_ctrl    =   0.004        # required 250Hz at least for attitude control
        self.attitude_control_call_timer =  self.create_timer(period_offboard_att_ctrl, self.publisher_vehicle_attitude_setpoint)

        period_offboard_vel_ctrl    =   0.02         # required 50Hz at least for velocity control
        self.velocity_control_call_timer =  self.create_timer(period_offboard_vel_ctrl, self.publish_vehicle_velocity_setpoint)

    # main code
    def offboard_control_main(self):
        # check another module nodes alive
        if self.path_following_heartbeat == True and self.path_planning_heartbeat == True and self.collision_avoidance_heartbeat == True:
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
                    if self.obstacle_flag == False:
                        self.collision_avoidance_flag   =   False
                        self.prm_off_con_mod.position   =   False
                        self.prm_off_con_mod.velocity   =   False
                        self.prm_off_con_mod.attitude   =   True
                        self.publish_offboard_control_mode(self.prm_off_con_mod)
                        self.path_following_flag        =   True
                    else:
                        self.path_following_flag        =   False
                        self.prm_off_con_mod.position   =   False
                        self.prm_off_con_mod.velocity   =   True
                        self.prm_off_con_mod.attitude   =   False
                        self.publish_offboard_control_mode(self.prm_off_con_mod)
                        self.collision_avoidance_flag  =    True
    
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
        Roll = math.atan2(t0, t1) * 57.2958

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Pitch = math.asin(t2) * 57.2958

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Yaw = math.atan2(t3, t4) * 57.2958

        return Roll, Pitch, Yaw

    def BodytoNED(self):
        vel_cmd_body = np.array([self.vel_cmd_body_x,self.vel_cmd_body_y,self.vel_cmd_body_z])
        
        self.veh_trj_set.vel_NED = np.array((self.DCM_bn @ vel_cmd_body).tolist())

    def NEDtoBody(self):
        vel_body = np.array([self.v_x,self.v_y,self.v_z])
        
        self.u, self.v, self.w = np.array((self.DCM_nb @ vel_body).tolist())

    def DCM(self, _phi, _theta, _psi):
        PHI = math.radians(_phi)  
        THETA = math.radians(_theta)
        PSI = math.radians(_psi)
        # print(PHI, THETA, PSI)

        mtx_DCM = np.array([[math.cos(PSI)*math.cos(THETA), math.sin(PSI)*math.cos(THETA), -math.sin(THETA)], 
                            [(-math.sin(PSI)*math.cos(PHI))+(math.cos(PSI)*math.sin(THETA)*math.sin(PHI)), (math.cos(PSI)*math.cos(PHI))+(math.sin(PSI)*math.sin(THETA)*math.sin(PHI)), math.cos(THETA)*math.sin(PHI)], 
                            [(math.sin(PSI)*math.sin(PHI))+(math.cos(PSI)*math.sin(THETA)*math.cos(PHI)), (-math.cos(PSI)*math.sin(PHI))+(math.sin(PSI)*math.sin(THETA)*math.cos(PHI)), math.cos(THETA)*math.cos(PHI)]])
       
        return mtx_DCM

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
        msg = Heartbeat()
        msg.heartbeat = True
        self.heartbeat_publisher.publish(msg)
    
    # subscribe path planning heartbeat signal
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.heartbeat

    # subscribe path following heartbeat signal
    def path_following_heartbeat_call_back(self,msg):
        self.path_following_heartbeat = msg.heartbeat

    # heartbeat subscribe from collision avoidance
    def collision_avoidance_heartbeat_call_back(self,msg):
        self.collision_avoidance_heartbeat = msg.heartbeat
    
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
    def publish_position_setpoint(self,veh_trj_set):
        msg                 =   TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   veh_trj_set.pos_NED
        msg.yaw             =   veh_trj_set.yaw_rad
        self.trajectory_setpoint_publisher.publish(msg)

    # publish velocity offboard command to px4
    def publish_vehicle_velocity_setpoint(self):
        if self.collision_avoidance_flag == True:
            msg                 =   TrajectorySetpoint()
            msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
            msg.velocity        =   np.float32(self.veh_trj_set.vel_NED)
            msg.yawspeed        =   self.veh_trj_set.yaw_vel_rad
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
        self.publish_position_setpoint(self.veh_trj_set)
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

    def CA2Control_callback(self, msg):
        # update Body velocity CMD
        # self.veh_trj_set.vel_NED[0]            =   msg.linear.x
        # self.veh_trj_set.vel_NED[1]            =   msg.linear.y
        # self.veh_trj_set.vel_NED[2]            =   msg.linear.z
        # self.veh_trj_set.yaw_vel_rad   =   msg.angular.z

        self.vel_cmd_body_x            =   msg.linear.x
        self.vel_cmd_body_y            =   msg.linear.y
        self.vel_cmd_body_z            =   msg.linear.z
        self.veh_trj_set.yaw_vel_rad   =   msg.angular.z

        self.DCM_nb = self.DCM(self.phi, self.theta, self.psi)
        self.DCM_bn = np.transpose(self.DCM_nb)
        self.BodytoNED()
        
        
    def LidarCallback(self, pc_msg):
        if pc_msg.is_dense is True :
            input =point_cloud2.read_points(pc_msg)
            point = np.array(input, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            x = point['x']
            y = point['y']
            dist = np.sqrt(x **2 + y ** 2)
            self.min_dist = np.min(dist)
            
            # Datalog = "%f %f\n" %(dist, min_dist)
            # self.datalogFile.write(Datalog)
            if self.z < -2.0:
                if self.min_dist < 5.0:
                    self.obstacle_flag = True
                elif self.obstacle_flag == True and self.min_dist < 15.0:
                    self.obstacle_flag = True
                else : 
                    # if (self.get_clock().now().nanoseconds - self.phase_time) > 3 * 10.0**9 :
                    #     self.obstacle_flag = False
                        # print("self.get_clock().now().nanoseconds - self.phase_time :",self.get_clock().now().nanoseconds - self.phase_time)
                    # else : 
                    #     self.obstacle_flag = True
                    self.obstacle_flag = False
            else :
                self.obstacle_flag = False
        else : 
            pass


    def state_logger(self):
        self.get_logger().info("-----------------")
        # self.get_logger().info("sim_time =" + str(self.sim_time) )
        self.get_logger().info("path_planning_heartbeat =" + str(self.path_planning_heartbeat) )
        self.get_logger().info("path_following_heartbeat =" + str(self.path_following_heartbeat) )
        self.get_logger().info("collision_avoidance_heartbeat =" + str(self.collision_avoidance_heartbeat) )
        self.get_logger().info("obstacle_flag =" + str(self.obstacle_flag) )
        self.get_logger().info("path_following_flag =" + str(self.path_following_flag) )
        self.get_logger().info("collision_avoidance_flag =" + str(self.collision_avoidance_flag) )
        self.get_logger().info("initial_position_flag =" + str(self.initial_position_flag) )
        self.get_logger().info("NED Position:   [x]=" + str(self.x) +", [e]=" + str(self.y) +", [d]=" + str(self.z))
        self.get_logger().info("NED Velocity:   [v_x]=" + str(self.v_x) +", [v_y]=" + str(self.v_y) +", [v_z]=" + str(self.v_z))
        self.get_logger().info("Body Velocity:   [u]=" + str(self.u) +", [v]=" + str(self.v) +", [w]=" + str(self.w))
        self.get_logger().info("Euler Angle:   [psi]=" + str(self.psi) +", [theta]=" + str(self.theta) +", [phi]=" + str(self.phi))
        self.get_logger().info("Angular Velocity:   [p]=" + str(self.p) +", [q]=" + str(self.q) +", [r]=" + str(self.r))
        self.get_logger().info("Body Velocity CMD:   [cmd_x]=" + str(self.vel_cmd_body_x) +", [cmd_y]=" + str(self.vel_cmd_body_y) +", [cmd_z]=" + str(self.vel_cmd_body_z))
        self.get_logger().info("min_dist =" + str(self.min_dist) )
        flightlog = "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" %(
            self.obstacle_flag, self.path_following_flag, self.collision_avoidance_flag, self.x, self.y, 
            self.z, self.v_x, self.v_y, self.v_z, self.u, self.v, self.w, self.psi, 
            self.theta, self.phi, self.p, self.q, self.r, 
            self.vel_cmd_body_x,  self.vel_cmd_body_y,  self.vel_cmd_body_z,
            self.min_dist)
        self.flightlogFile.write(flightlog)
        # self.datalogFile.close()




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
