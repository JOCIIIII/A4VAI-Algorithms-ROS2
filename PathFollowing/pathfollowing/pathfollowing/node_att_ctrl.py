#.. public libaries
import numpy as np
import math as m

#.. ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32MultiArray

#.. PX4 libararies - sub.

#############################################################################################################
# added by controller
# changed EstimatorStates - > VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity
# added TimesyncStatus
from px4_msgs.msg import VehicleLocalPosition, TimesyncStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
#############################################################################################################

from px4_msgs.msg import HoverThrustEstimate 
from px4_msgs.msg import VehicleAcceleration 
from px4_msgs.msg import ActuatorOutputs 
#.. PX4 libararies - pub.
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleAttitudeSetpoint

#############################################################################################################
# added by controller
from custom_msgs.msg import LocalWaypointSetpoint, ConveyLocalWaypointComplete
from custom_msgs.msg import Heartbeat
#############################################################################################################

#.. PF algorithms libararies
# from .testpy1 import testpypy1
from .PF_modules.quadrotor_6dof import Quadrotor_6DOF
from .PF_modules.virtual_target import Virtual_Target
from .PF_modules.set_parameter import MPPI_Guidance_Parameter, Way_Point
from .PF_modules.Funcs_PF_Base import kinematics, distance_from_Q6_to_path, check_waypoint, virtual_target_position, \
    guidance_modules, compensate_Aqi_cmd, thrust_cmd, att_cmd, NDO_Aqi
from .PF_modules.utility_functions import DCM_from_euler_angle

class NodeAttCtrl(Node):
    
    def __init__(self):
        super().__init__('node_attitude_control')
        
        #.. Reference
        #   [1]: https://docs.px4.io/main/en/msg_docs/vehicle_command.html
        #   [2]: https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
        #   [3]: https://github.com/PX4/px4_ros_com/blob/release/1.13/src/examples/offboard/offboard_control.cpp
        #   [4]: https://docs.px4.io/main/ko/advanced_config/tuning_the_ecl_ekf.html
        #   [5]: https://hostramus.tistory.com/category/ROS2
        #   [6]: https://docs.px4.io/v1.14/en/flight_stack/controller_diagrams.html
        #   [7]: https://docs.px4.io/v1.14/en/flight_modes/offboard.html
        
        ###.. - Start - set pub. sub. for PX4 - ROS2 msg  ..###
        #
        #.. mapping of ros2-px4 message name using in this code
        class msg_mapping_ros2_to_px4:
            VehicleCommand          =   '/fmu/in/vehicle_command'
            OffboardControlMode     =   '/fmu/in/offboard_control_mode'
            TrajectorySetpoint      =   '/fmu/in/trajectory_setpoint'
            HoverThrustEstimate     =   '/fmu/out/hover_thrust_estimate'
            VehicleAcceleration     =   '/fmu/out/vehicle_acceleration'
            ActuatorOutputs         =   '/fmu/out/actuator_outputs'
            TimesyncStatus          =   '/fmu/out/timesync_status'
        #.. publishers - from ROS2 msgs to px4 msgs
        self.vehicle_command_publisher_             =   self.create_publisher(VehicleCommand, msg_mapping_ros2_to_px4.VehicleCommand, 10)
        self.offboard_control_mode_publisher_       =   self.create_publisher(OffboardControlMode, msg_mapping_ros2_to_px4.OffboardControlMode , 10)
        #.. subscriptions - from px4 msgs to ROS2 msgs
#############################################################################################################
# added by controller
        # changed EstimatorStates - > VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity
        # added TimesyncStatus
        self.vehicle_local_position_subscriber      =   self.create_subscription(VehicleLocalPosition,    '/fmu/out/vehicle_local_position',   self.vehicle_local_position_callback,   qos_profile_sensor_data)
        self.vehicle_attitude_subscriber            =   self.create_subscription(VehicleAttitude,         '/fmu/out/vehicle_attitude',         self.vehicle_attitude_callback,         qos_profile_sensor_data)
        self.vehicle_angular_velocity_subscriber    =   self.create_subscription(VehicleAngularVelocity , '/fmu/out/vehicle_angular_velocity', self.vehicle_angular_velocity_callback, qos_profile_sensor_data)
#############################################################################################################
        self.hover_thrust_estimate_subscription     =   self.create_subscription(HoverThrustEstimate, msg_mapping_ros2_to_px4.HoverThrustEstimate, self.subscript_hover_thrust_estimate, qos_profile_sensor_data)
        self.vehicle_acceleration_subscription      =   self.create_subscription(VehicleAcceleration, msg_mapping_ros2_to_px4.VehicleAcceleration, self.subscript_vehicle_acceleration, qos_profile_sensor_data)
        self.actuator_outputs_subscription          =   self.create_subscription(ActuatorOutputs, msg_mapping_ros2_to_px4.ActuatorOutputs, self.subscript_actuator_outputs, qos_profile_sensor_data)
        #
        ###.. -  End  - set pub. sub. for PX4 - ROS2 msg  ..###
        
        
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
        self.prm_off_con_mod.attitude   =   True
        
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
        
        
        ###.. - Start - set variable of subscription msg for PX4 - ROS2  ..###
        #
        #.. variable - estimator_state
        class var_msg_est_state:
            def __init__(self):
                self.pos_NED        =   np.zeros(3)
                self.vel_NED        =   np.zeros(3)
                self.eul_ang_rad    =   np.zeros(3)
                self.windvel_NE     =   np.zeros(2)         # there are no values from PX
                
        self.est_state = var_msg_est_state()
        #     
        ###.. -  End  - set variable of subscription msg for PX4 - ROS2  ..###
        
        
        ###.. - Start - set callback function (#1) to publish msg from ROS2 to PX4  ..###
        #
        # callback main_attitude_control
        period_offboard_att_ctrl    =   0.004           # required 250Hz at least for attitude control in [6]
        self.timer  =   self.create_timer(period_offboard_att_ctrl, self.publisher_vehicle_attitude_setpoint)
#############################################################################################################
# added by controller
        # declare heartbeat_timer
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)
#############################################################################################################
#############################################################################################################
# deleted by controller
# offboard control implement in controller
        # # callback offboard_control_mode
        # # offboard counter in [3]
        # period_offboard_control_mode    =   0.2         # required about 5Hz for attitude control (proof that the external controller is healthy in [7])
        # self.timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_mode)
        # self.prm_offboard_setpoint_counter_start_flight  =   m.ceil(2./period_offboard_control_mode)    # about 2 sec (after receiving the signal for more than a second in [7])
        # self.offboard_setpoint_counter_     =   0
#############################################################################################################
        ###.. -  End - set callback function to publish msg from ROS2 to PX4  ..###
        
        
        ###.. - Start - Vars. for PF algorithm ..###
        #
        #.. declare variables/instances
        self.Q6     =   Quadrotor_6DOF()
        
        self.timer  =   self.create_timer(self.Q6.dt_GCU, self.main_attitude_control)
        
        self.VT     =   Virtual_Target()
        self.VT_psi_cmd     =   Virtual_Target()
#############################################################################################################
# added by controller
        #.. set waypoint
        self.wp_type_selection   =   3       # | 0: straight line | 1: ractangle | 2: circle | 3: designed
        # initialize waypoint parameter
        self.waypoint_index     =   0
        self.waypoint_x         =   []
        self.waypoint_y         =   []
        self.waypoint_z         =   []
#############################################################################################################
        #.. MPPI setting
        self.MP     =   MPPI_Guidance_Parameter(self.Q6.Guid_type)
        self.MPPI_ctrl_input = np.array([self.MP.u1_init, self.MP.u2_init, self.MP.u3_init])
        #
        ###.. -  End  - Vars. for PF algorithm ..###
        
        
        ###.. - Start - set callback function (#2) to publish msg from ROS2 to ROS2  ..###
        #
        # Guid_type = | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        if self.Q6.Guid_type >= 3:
            period_MPPI_input   =   0.5 * self.MP.dt       # 2 times faster than MPPI dt
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_MPPI_input_int_Q6)
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_MPPI_input_dbl_Q6)
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_MPPI_input_dbl_VT)
            self.timer  =   self.create_timer(period_MPPI_input * 10., self.publish_MPPI_input_dbl_WP)
            
            period_GPR_input    =   0.01    # GPR update dt
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_GPR_input_dbl_NDO)
            
        #
        ###.. -  End  - set callback function (#2) to publish msg from ROS2 to ROS2  ..###
                
                
        ###.. - Start - set pub. sub. for ROS2 - ROS2 msg  ..###
        #
        if self.Q6.Guid_type >= 3:
            #.. publishers - from ROS2 msgs to ROS2 msgs
            self.MPPI_input_int_Q6_publisher_   =   self.create_publisher(Int32MultiArray, 'MPPI/in/int_Q6', 10)
            self.MPPI_input_dbl_Q6_publisher_   =   self.create_publisher(Float64MultiArray, 'MPPI/in/dbl_Q6', 10)
            self.MPPI_input_dbl_VT_publisher_   =   self.create_publisher(Float64MultiArray, 'MPPI/in/dbl_VT', 10)
            self.MPPI_input_dbl_WP_publisher_   =   self.create_publisher(Float64MultiArray, 'MPPI/in/dbl_WP', 10)
            self.GPR_input_dbl_NDO_publisher_   =   self.create_publisher(Float64MultiArray, 'GPR/in/dbl_Q6', 10)
            
            #.. subscriptions - from ROS2 msgs to ROS2 msgs
            self.MPPI_output_subscription   =   self.create_subscription(Float64MultiArray, 'MPPI/out/dbl_MPPI', self.subscript_MPPI_output, qos_profile_sensor_data)
#############################################################################################################
# added by controller
        # declare local waypoint subscriber from controller
        self.local_waypoint_subscriber                  =   self.create_subscription(LocalWaypointSetpoint, '/local_waypoint_setpoint_to_PF',self.local_waypoint_setpoint_call_back, 10) 
        
        # declare local waypoint receive complete flag publisher to controller
        self.local_waypoint_receive_complete_publisher  =   self.create_publisher(ConveyLocalWaypointComplete, '/convey_local_waypoint_complete', 10) 
        
        # declare attitude setpoint publisher to controller
        self.vehicle_attitude_setpoint_publisher        =   self.create_publisher(VehicleAttitudeSetpoint,   '/pf_att_2_control', 10)   
        
        # declare heartbeat_publisher 
        self.heartbeat_publisher                        =   self.create_publisher(Heartbeat,    '/path_following_heartbeat', 10)

        # declare heartbeat_subscriber 
        self.controller_heartbeat_subscriber            =   self.create_subscription(Heartbeat, '/controller_heartbeat',            self.controller_heartbeat_call_back,            10)
        self.path_planning_heartbeat_subscriber         =   self.create_subscription(Heartbeat, '/path_planning_heartbeat',         self.path_planning_heartbeat_call_back,        10)
        self.collision_avoidance_heartbeat_subscriber   =   self.create_subscription(Heartbeat, '/collision_avoidance_heartbeat',   self.collision_avoidance_heartbeat_call_back,   10)
        
        self.timesync_status_subscribe = False
        
        self.path_planning_complete = False
        self.variable_setting_complete = False 

        # heartbeat signal of another module node
        self.controller_heartbeat           = False
        self.path_planning_heartbeat        = False
        self.collision_avoidance_heartbeat  = False
#############################################################################################################
        ###.. -  End  - set pub. sub. for ROS2 - ROS2 msg  ..###
        
        
        ###.. - Start - set callback function (#3) to calculate the PF evaluation  ..###
        #
        period_PF_evaluation    =   0.1
        self.timer  =   self.create_timer(period_PF_evaluation, self.PF_evaluation)
        
        self.Q      =   np.array([0.05, 0.02, 2.0, 0.0])
        self.R      =   np.array([0.001, 0.001, 0.001])
        self.R_mat      =   np.zeros((3,3))
        self.R_mat[0,0], self.R_mat[1,1], self.R_mat[2,2] = self.R[0], self.R[1], self.R[2]
        
        self.Q_lim  =   np.array([0.5])
        self.a_lim  =   1.0
        
        self.unit_Rw1w2 =   np.ones(3)
        self.mag_thrust_cmd = 0.
        self.dist_to_path = 0.
        
        self.cost   =   np.zeros(3)
        #
        ###.. -  End  - set callback function (#2) to publish msg from ROS2 to ROS2  ..###
        
        
        ###.. - Start - set other variables  ..###
        #
        #.. Aqi_grav
        self.Aqi_grav   =   np.array([0., 0., 9.81])
        
        #.. takeoff_time
        self.takeoff_start = False
        self.takeoff_time = 0
        
        #.. callback state_logger
        # self.period_state_logger = 0.1
        # self.timer  =   self.create_timer(self.period_state_logger, self.state_logger)
        self.datalogFile = open("/home/user/log/point_mass_6d/datalogfile/datalog.txt",'w')
        
        self.att_ang_cmd = np.zeros(3)
        self.Aqi_cmd = np.zeros(3)
        self.Aqi_expect = np.zeros(3)
        self.norm_thrust_cmd = 0.
        self.accel_xyz = np.zeros(3)
        self.LOS_azim = 0.
        
        #.. temp. parameter
        # self.hover_thrust = 0.7         # iris
        # self.hover_thrust = 0.41        # typhoon
        self.hover_thrust = 0.541
        
        self.max_del_Psi     =   10. * m.pi/180.
        # self.max_del_Psi     =   5. * m.pi/180.
        self.first_time_flag = False
        self.first_time = 0
        #.. etc.
        self.sim_time = 0.
        self.actuator_outputs = np.zeros(16)
        #
        ###.. -  End  - set other variables  ..###
#############################################################################################################
# deleted by controller
# offboard control implement in controller
    #### main function
    # def offboard_control_mode(self):


        # #.. offboard mode
        # if self.offboard_setpoint_counter_ == self.prm_offboard_setpoint_counter_start_flight:
        #     # print("----- debug point [1] -----")
            
        #     # offboard mode cmd
        #     self.publish_vehicle_command(self.prm_offboard_mode)
        #     # arm cmd
        #     self.publish_vehicle_command(self.prm_arm_mode)
            
        # # set offboard control mode 
        # self.publish_offboard_control_mode(self.prm_off_con_mod)
            
        # # count offboard_setpoint_counter_
        # if self.offboard_setpoint_counter_ < self.prm_offboard_setpoint_counter_start_flight:
        #     self.offboard_setpoint_counter_ = self.offboard_setpoint_counter_ + 1



    #     pass
#############################################################################################################
#############################################################################################################
# changed by controller
    #.. main_attitude_control 
    def main_attitude_control(self):
        if self.controller_heartbeat == True and self.path_planning_heartbeat ==True and self.collision_avoidance_heartbeat == True:
            if self.timesync_status_subscribe == False:
                self.timesync_status_subscription          =   self.create_subscription(TimesyncStatus, '/fmu/out/timesync_status', self.subscript_timesync_status, qos_profile_sensor_data)
                self.timesync_status_subscribe = True
            if self.path_planning_complete == True and self.variable_setting_complete == False :
                #.. variable setting
                print("                                          ")
                print("=====    revieved Path Planning      =====")
                print("                                          ")

                self.Q6.Ri  =   self.est_state.pos_NED
                self.Q6.Vi  =   self.est_state.vel_NED
                self.Q6.att_ang =   self.est_state.eul_ang_rad
                self.Q6.cI_B    =   DCM_from_euler_angle(self.Q6.att_ang)
                # self.Q6.throttle_hover = self.hover_thrust
                #.. initialization

                self.WP     =   Way_Point(self.wp_type_selection, self.waypoint_x, self.waypoint_y ,self.waypoint_z)

                self.WP.init_WP(self.Q6.Ri)
                self.VT.init_VT_Ri(self.WP.WPs, self.Q6.Ri, self.Q6.look_ahead_distance)

                self.VT_psi_cmd.init_VT_Ri(self.WP.WPs, self.Q6.Ri, self.Q6.look_ahead_distance_psi_cmd)
                self.variable_setting_complete = True 
            else :
                pass

            if self.variable_setting_complete == True :
                #.. sim_time
                # if self.takeoff_start:
                #     current_time = int(Clock().now().nanoseconds / 1000) # time in microseconds
                #     self.sim_time   =   (current_time - self.takeoff_time) / 1000000

                #.. variable setting
                self.Q6.Ri  =   self.est_state.pos_NED.copy()
                self.Q6.Vi  =   self.est_state.vel_NED.copy()
                self.Q6.att_ang =   self.est_state.eul_ang_rad
                self.Q6.cI_B    =   DCM_from_euler_angle(self.Q6.att_ang)
                self.Q6.throttle_hover = self.hover_thrust
                self.Q6.Ai      =   np.matmul(np.transpose(self.Q6.cI_B), self.accel_xyz)

                ###### - start - PF algorithm ######
                #.. kinematics
                mag_Vqi, LOS_azim, LOS_elev, tgo, FPA_azim, FPA_elev, self.Q6.cI_W = kinematics(self.VT.Ri, self.Q6.Ri, self.Q6.Vi)
                _, LOS_azim_cmd, _, _, _, _, _ = kinematics(self.VT_psi_cmd.Ri, self.Q6.Ri, self.Q6.Vi)

                # LA_azim     =   FPA_azim - LOS_azim
                # LA_elev     =   FPA_elev - LOS_elev

                #.. distance from quadrotor to ref. path  
                self.dist_to_path, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, self.unit_Rw1w2 = \
                    distance_from_Q6_to_path(self.WP.WPs, self.Q6.WP_idx_heading, self.Q6.Ri, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed)


                #.. virtual target modules
                #.. directly decide a position of the virtual target 
                # check waypoint - quadrotor
                self.Q6.WP_idx_heading = check_waypoint(self.WP.WPs, self.Q6.WP_idx_heading, self.Q6.Ri, self.Q6.distance_change_WP)
                # virtual target position
                self.VT.Ri = virtual_target_position(self.dist_to_path, self.Q6.look_ahead_distance, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, self.WP.WPs)

                self.VT_psi_cmd.Ri = virtual_target_position(self.dist_to_path, self.Q6.look_ahead_distance_psi_cmd, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, self.WP.WPs)

                #.. guidance modules
                Aqi_cmd, self.Q6.flag_guid_trans = guidance_modules(self.Q6.Guid_type, mag_Vqi, self.Q6.WP_idx_passed, self.Q6.flag_guid_trans, self.Q6.WP_idx_heading, self.WP.WPs.shape[0],
                            self.VT.Ri, self.Q6.Ri, self.Q6.Vi, self.Q6.Ai, self.Q6.desired_speed, self.Q6.Kp_vel, self.Q6.Kd_vel, self.Q6.Kp_speed, self.Q6.Kd_speed, self.Q6.guid_eta, self.Q6.cI_W, tgo, self.MPPI_ctrl_input)                

                if self.takeoff_start == True:
                    self.Q6.thr_unitvec, self.Q6.out_NDO, self.Q6.z_NDO = NDO_Aqi(
                        self.Aqi_grav, self.Q6.mag_Aqi_thru, self.Q6.cI_B, self.Q6.thr_unitvec, self.Q6.gain_NDO, self.Q6.z_NDO, self.Q6.Vi, self.Q6.dt_GCU)

                #.. compensate Aqi_cmd
                self.Q6.Ai_est_dstb  =   self.Q6.out_NDO.copy()
                # self.Q6.Ai_est_dstb  =   np.zeros(3)
                Aqi_cmd = compensate_Aqi_cmd(Aqi_cmd, self.Q6.Ai_est_dstb, self.Aqi_grav)

                #.. thrust command
                self.Q6.mag_Aqi_thru, self.mag_thrust_cmd, norm_thrust_cmd = thrust_cmd(Aqi_cmd, self.Q6.throttle_hover, self.MP.a_lim, self.Q6.mass)

                self.Aqi_cmd =  Aqi_cmd.copy()


                #.. Psi cmd limit
                Psi_des     =   LOS_azim_cmd

                del_Psi     =   Psi_des - self.Q6.att_ang[2]

                self.LOS_azim   =   Psi_des

                if abs(del_Psi) > m.pi:
                    if Psi_des > self.Q6.att_ang[2]:
                        del_Psi = del_Psi - 2.*m.pi
                    else:
                        del_Psi = del_Psi + 2.*m.pi
                    pass

                del_Psi_limited = max(min(del_Psi, self.max_del_Psi), -self.max_del_Psi)

                # gab_del_Psi     =   del_Psi - del_Psi_limited
                # weight_del_Psi  =   max( self.max_del_Psi / (self.max_del_Psi + m.sqrt(abs(gab_del_Psi))), 0.2)

                # max_LA_azim_cmd =   30 * m.pi/180.
                # LA_azim_cmd =   FPA_azim - LOS_azim_cmd
                # weight_del_Psi  =   max( max_LA_azim_cmd / (max_LA_azim_cmd + m.sqrt(abs(LA_azim_cmd))), 0.05)

                Psi_cmd     =   self.Q6.att_ang[2] + del_Psi_limited # * weight_del_Psi
                # Psi_cmd     =   0.

                att_ang_cmd = att_cmd(self.Aqi_cmd, Psi_cmd, Psi_cmd)

                # max_tmp     =   15. * m.pi /180.
                # del_Phi     =   att_ang_cmd[0] - self.Q6.att_ang[0]
                # del_The     =   att_ang_cmd[1] - self.Q6.att_ang[1]
                # del_Phi_limited = max(min(del_Phi, max_tmp), -max_tmp)
                # del_The_limited = max(min(del_The, max_tmp), -max_tmp)
                # att_ang_cmd[0] = self.Q6.att_ang[0] + del_Phi_limited
                # att_ang_cmd[1] = self.Q6.att_ang[1] + del_The_limited

                # yaw continuity
                if abs(att_ang_cmd[2] - self.Q6.att_ang[2]) > 1.5*m.pi:
                    if att_ang_cmd[2] > self.Q6.att_ang[2]:
                        att_ang_cmd[2] = att_ang_cmd[2] - 2.*m.pi
                    else:
                        att_ang_cmd[2] = att_ang_cmd[2] + 2.*m.pi

                self.att_ang_cmd    =   att_ang_cmd.copy()

                # takeoff
                # if self.sim_time < 0.5:
                #     norm_thrust_cmd = 0.5

                self.norm_thrust_cmd = norm_thrust_cmd

                ###### -  end  - PF algorithm ######

                w, x, y, z = self.Euler2Quaternion(self.att_ang_cmd[0], self.att_ang_cmd[1], self.att_ang_cmd[2])
                self.veh_att_set.thrust_body    =   [0., 0., -norm_thrust_cmd]
                self.veh_att_set.q_d            =   [w, x, y, z]
                pass
            else :
                pass
        else:
            pass
#############################################################################################################
    #.. state_logger 
    # def state_logger(self):
    #     if (self.Q6.WP_idx_heading < self.WP.WPs.shape[0]-1):
    #     # if (self.Q6.WP_idx_heading < self.WP.WPs.shape[0]):
    #         # self.get_logger().info("-----------------")
            
    #         self.get_logger().info("-----------------")
    #         self.get_logger().info("sim_time =" + str(self.sim_time) )
    #         self.get_logger().info("flag_guid_trans =" + str(self.Q6.flag_guid_trans) )
    #         self.get_logger().info("dist_to_path      =" + str(self.dist_to_path) )
    #         self.get_logger().info("norm_thrust_cmd     =" + str(self.norm_thrust_cmd) )
    #         self.get_logger().info("Vi:         [0]=" + str(self.Q6.Vi[0]) +", [1]=" + str(self.Q6.Vi[1]) +", [2]=" + str(self.Q6.Vi[2]))
    #         self.get_logger().info("out_NDO:    [0]=" + str(self.Q6.out_NDO[0]) +", [1]=" + str(self.Q6.out_NDO[1]) +", [2]=" + str(self.Q6.out_NDO[2]))
    #         self.get_logger().info("att_cmd_deg:[0]=" + str(self.att_ang_cmd[0]*180./m.pi) +", [1]=" + str(self.att_ang_cmd[1]*180./m.pi) +", [2]=" + str(self.att_ang_cmd[2]*180./m.pi))
    #         self.get_logger().info("att_ang    :[0]=" + str(self.Q6.att_ang[0]*180./m.pi) +", [1]=" + str(self.Q6.att_ang[1]*180./m.pi) +", [2]=" + str(self.Q6.att_ang[2]*180./m.pi))
    #         self.get_logger().info("Aqi_cmd    :[0]=" + str(self.Aqi_cmd[0]) +", [1]=" + str(self.Aqi_cmd[1]) +", [2]=" + str(self.Aqi_cmd[2]))
    #         self.get_logger().info("Ai         :[0]=" + str(self.Q6.Ai[0]) + "[1]=" + str(self.Q6.Ai[1]) + "[2]=" + str(self.Q6.Ai[2]) )
    #         self.get_logger().info("Att_psi =" + str(round(self.Q6.att_ang[2]*180./m.pi, 2)) +", LOS_azim = " + str(round(self.LOS_azim*180./m.pi, 2)) +", del_Psi =" + str(round((self.Q6.att_ang[2] - self.LOS_azim)*180./m.pi, 2)))
    #         self.get_logger().info("Act_Out    :[0]=" + str(round(self.actuator_outputs[0], 2))+ ", [1]=" + str(round(self.actuator_outputs[1], 2))+ ", [2]=" + str(round(self.actuator_outputs[2], 2))+ ", [3]=" + str(round(self.actuator_outputs[3], 2))
    #                                        + ", [4]=" + str(round(self.actuator_outputs[4], 2))+ ", [5]=" + str(round(self.actuator_outputs[5], 2)) )
    #         self.get_logger().info("windvel_NE :[0]=" + str(self.est_state.windvel_NE[0]) + "[1]=" + str(self.est_state.windvel_NE[1]) )
            
            
        #     # if self.takeoff_start == True and sim_time  < 45.:
        #     if self.takeoff_start:
        #         w, x, y, z = self.Euler2Quaternion(self.Q6.att_ang[0], self.Q6.att_ang[1], self.Q6.att_ang[2])
                
        #         Data = "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \
        #              %f %f %f %f %f %f %f %f %f %f\n" %(
        #             self.sim_time, self.VT.Ri[0], self.VT.Ri[1], self.VT.Ri[2], self.Q6.Ri[0], 
        #             self.Q6.Ri[1], self.Q6.Ri[2], self.Q6.Vi[0], self.Q6.Vi[1], self.Q6.Vi[2], 
        #             self.Q6.out_NDO[0], self.Q6.out_NDO[1], self.Q6.out_NDO[2], self.Q6.att_ang[0], self.Q6.att_ang[1], 
        #             self.Q6.att_ang[2],  self.att_ang_cmd[0],  self.att_ang_cmd[1], self.att_ang_cmd[2], self.Q6.desired_speed, 
                    
        #             self.Aqi_cmd[0], self.Aqi_cmd[1], self.Aqi_cmd[2], self.Q6.Ai[0], self.Q6.Ai[1],
        #             self.Q6.Ai[2], self.veh_att_set.q_d[0], self.veh_att_set.q_d[1], self.veh_att_set.q_d[2], self.veh_att_set.q_d[3], 
                    
        #             w, x, y, z, self.norm_thrust_cmd,
        #             self.actuator_outputs[0], self.actuator_outputs[1], self.actuator_outputs[2], self.actuator_outputs[3], self.actuator_outputs[4], 
        #             self.actuator_outputs[5], self.cost[0]*self.period_state_logger, self.cost[1]*self.period_state_logger, self.cost[2]*self.period_state_logger, self.dist_to_path
        #             )
        #         self.datalogFile.write(Data)
        # else:
        #     #.. datalogfile close
        #     self.datalogFile.close()
        
        # pass
    
    #.. PF_evaluation
    def PF_evaluation(self):
        Aqi_cmd_wo_grav     =   self.Aqi_cmd + self.Aqi_grav
        tmp_Ru      =   np.matmul(self.R_mat, Aqi_cmd_wo_grav)
        cost_uRu    =   np.dot(Aqi_cmd_wo_grav, tmp_Ru)
            
        mag_Vqi_alinged_path   =   max(np.dot(self.unit_Rw1w2, self.Q6.Vi), 0)
        cost_distance   =   self.Q[0] * (self.dist_to_path ** 2)
        # print("1 = cost_distance" + str(round(cost_distance,2)))
        if self.dist_to_path > self.Q_lim[0]:
            cost_distance = cost_distance + self.Q[2] * (self.dist_to_path * self.dist_to_path)
            # print("2 = cost_distance" + str(round(cost_distance,2)))
        
        energy_cost1    =   self.mag_thrust_cmd/max(mag_Vqi_alinged_path, 0.1)
        cost_a          =   self.Q[1] * energy_cost1
        # print("3 = cost_a" + str(round(cost_a,2)))
        if self.Q6.WP_idx_passed >= 1:
            self.cost   =   np.array([cost_distance, cost_a, cost_uRu])
        else:
            self.cost   =   np.zeros(3)
        
        pass
        
        
                    
    ### publushers
    #.. publish_vehicle_command
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
        self.vehicle_command_publisher_.publish(msg)
        
        pass
        
    #.. publish_offboard_control_mode
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg                 =   OffboardControlMode()
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher_.publish(msg)
        
        pass
#############################################################################################################
# changed by controller
    # publish to controller
    #.. publisher_vehicle_attitude_setpoint 
    def publisher_vehicle_attitude_setpoint(self):
        msg                 =   VehicleAttitudeSetpoint()
        msg.roll_body       =   self.veh_att_set.roll_body
        msg.pitch_body      =   self.veh_att_set.pitch_body
        msg.yaw_body        =   self.veh_att_set.yaw_body
        msg.yaw_sp_move_rate    =   self.veh_att_set.yaw_sp_move_rate
        msg.q_d[0]          =   self.veh_att_set.q_d[0]
        msg.q_d[1]          =   self.veh_att_set.q_d[1]
        msg.q_d[2]          =   self.veh_att_set.q_d[2]
        msg.q_d[3]          =   self.veh_att_set.q_d[3]
        msg.thrust_body[0]  =   0.
        msg.thrust_body[1]  =   0.
        msg.thrust_body[2]  =   self.veh_att_set.thrust_body[2]
        self.vehicle_attitude_setpoint_publisher.publish(msg)
        
        pass
#############################################################################################################
    #.. publish_MPPI_input_int_Q6
    def publish_MPPI_input_int_Q6(self):
        msg         =   Int32MultiArray()
        msg.data    =   [self.Q6.WP_idx_heading, self.Q6.WP_idx_passed, self.Q6.Guid_type, self.Q6.flag_guid_trans]
        self.MPPI_input_int_Q6_publisher_.publish(msg)
        # self.get_logger().info('pub msgs: {0}'.format(msg.data))
        pass
    
    #.. publish_MPPI_input_dbl_Q6
    def publish_MPPI_input_dbl_Q6(self):
        tmp_float   =   0.
        msg         =   Float64MultiArray()
        msg.data    =   [self.Q6.throttle_hover, tmp_float, self.Q6.desired_speed, self.Q6.look_ahead_distance, self.Q6.distance_change_WP, 
                                      self.Q6.Kp_vel, self.Q6.Kd_vel, self.Q6.Kp_speed, self.Q6.Kd_speed, self.Q6.guid_eta, 
                                      self.Q6.tau_phi, self.Q6.tau_the, self.Q6.tau_psi, 
                                      self.Q6.Ri[0], self.Q6.Ri[1], self.Q6.Ri[2], 
                                      self.Q6.Vi[0], self.Q6.Vi[1], self.Q6.Vi[2], 
                                      self.Q6.Ai[0], self.Q6.Ai[1], self.Q6.Ai[2], 
                                      self.Q6.thr_unitvec[0], self.Q6.thr_unitvec[1], self.Q6.thr_unitvec[2],
                                      self.Q6.Ai_est_dstb[0], self.Q6.Ai_est_dstb[1], self.Q6.Ai_est_dstb[2]]
        self.MPPI_input_dbl_Q6_publisher_.publish(msg)
        # self.get_logger().info('pub msgs: {0}'.format(msg.data))
        pass
    
    #.. publish_MPPI_input_dbl_VT
    def publish_MPPI_input_dbl_VT(self):
        msg         =   Float64MultiArray()
        msg.data    =   [self.VT.Ri[0], self.VT.Ri[1], self.VT.Ri[2]]
        self.MPPI_input_dbl_VT_publisher_.publish(msg)
        # self.get_logger().info('pub msgs: {0}'.format(msg.data))
        pass
    
    #.. publish_MPPI_input_dbl_WP
    def publish_MPPI_input_dbl_WP(self):
        # self.get_logger().info('publish_MPPI_input_dbl_WP: {0}'.format(msg.data))
        if self.variable_setting_complete == True :
            msg         =   Float64MultiArray()
            msg.data    =   self.WP.WPs.reshape(self.WP.WPs.shape[0]*3,).tolist()
            self.MPPI_input_dbl_WP_publisher_.publish(msg)
        # print(self.WP.WPs)
        else: 
            pass
    
    #.. publish_GPR_input_dbl_NDO
    def publish_GPR_input_dbl_NDO(self):
        msg         =   Float64MultiArray()
        msg.data    =   [self.sim_time,
                         self.Q6.out_NDO[0], self.Q6.out_NDO[0], self.Q6.out_NDO[0]]
        self.GPR_input_dbl_NDO_publisher_.publish(msg)
        # self.get_logger().info('publish_MPPI_input_dbl_WP: {0}'.format(msg.data))
        # print(self.WP.WPs)
        pass
#############################################################################################################
# added by controller
# changed EstimatorStates - > VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity
    # subscribe position, velocity 
    def vehicle_local_position_callback(self, msg):
        # update NED position 
        self.est_state.pos_NED[0]      =   msg.x
        self.est_state.pos_NED[1]      =   msg.y
        self.est_state.pos_NED[2]      =   msg.z
        # update NED velocity
        self.est_state.vel_NED[0]    =   msg.vx
        self.est_state.vel_NED[1]    =   msg.vy
        self.est_state.vel_NED[2]    =   msg.vz

    # subscribe attitude 
    def vehicle_attitude_callback(self, msg):
        self.est_state.eul_ang_rad[0], self.est_state.eul_ang_rad[1], self.est_state.eul_ang_rad[2] = \
            self.Quaternion2Euler(msg.q[0], msg.q[1], msg.q[2], msg.q[3])

    def vehicle_angular_velocity_callback(self, msg):
        self.p    =   msg.xyz[0]
        self.q    =   msg.xyz[1]
        self.r    =   msg.xyz[2]

# receive local waypoint from controller
    def local_waypoint_setpoint_call_back(self,msg):
        self.path_planning_complete = msg.path_planning_complete
        self.waypoint_x             = msg.waypoint_x 
        self.waypoint_y             = msg.waypoint_y
        self.waypoint_z             = msg.waypoint_z
        print("                                          ")
        print("== receiving local waypoint complete!   ==")
        print("                                          ")
        self.local_waypoint_receive_complete_publish()

    # send local waypoint receive complete flag
    def local_waypoint_receive_complete_publish(self):
        msg = ConveyLocalWaypointComplete()
        msg.convey_local_waypoint_is_complete = True
        self.local_waypoint_receive_complete_publisher.publish(msg)


# heartbeat check function
    # heartbeat publish
    def publish_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.heartbeat_publisher.publish(msg)

    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self,msg):
        self.controller_heartbeat = msg.heartbeat

    # heartbeat subscribe from path following
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.heartbeat

    # heartbeat subscribe from collision avoidance
    def collision_avoidance_heartbeat_call_back(self,msg):
        self.collision_avoidance_heartbeat = msg.heartbeat
#############################################################################################################

#############################################################################################################
# deleted by controller
# changed EstimatorStates - > VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity
    # ### subscriptions        
    # #.. subscript subscript_estimator_states
    # def subscript_estimator_states(self, msg):        
    #     self.est_state.pos_NED[0]     =   msg.states[7]
    #     self.est_state.pos_NED[1]     =   msg.states[8]
    #     self.est_state.pos_NED[2]     =   msg.states[9]
    #     self.est_state.vel_NED[0]     =   msg.states[4]
    #     self.est_state.vel_NED[1]     =   msg.states[5]
    #     self.est_state.vel_NED[2]     =   msg.states[6]
    #     # Attitude
    #     self.est_state.eul_ang_rad[0], self.est_state.eul_ang_rad[1], self.est_state.eul_ang_rad[2] = \
    #         self.Quaternion2Euler(msg.states[0], msg.states[1], msg.states[2], msg.states[3])
        
    #     # Wind Velocity NE
    #     self.est_state.windvel_NE[0]  =   msg.states[22]
    #     self.est_state.windvel_NE[1]  =   msg.states[23]
    #     pass
#############################################################################################################
    #.. subscript_hover_thrust_estimate
    def subscript_hover_thrust_estimate(self, msg):
        # self.hover_thrust   =   msg.hover_thrust          # Is this value of msg.hover_thrust correct ???
        if self.takeoff_start == False:
            self.takeoff_start  =   True
            self.takeoff_time   =   int(Clock().now().nanoseconds / 1000)
            self.get_logger().info("takeoff_time = " + str(self.takeoff_time))
            self.get_logger().info('subscript_hover_thrust_estimate msgs: {0}'.format(msg.hover_thrust))
        pass
    
    #.. subscript_vehicle_acceleration
    def subscript_vehicle_acceleration(self, msg):
        # self.hover_thrust   =   msg.hover_thrust          # is the value of msg.hover_thrust correct ???
        self.accel_xyz[0] = msg.xyz[0]
        self.accel_xyz[1] = msg.xyz[1]
        self.accel_xyz[2] = msg.xyz[2]
        # self.get_logger().info('subscript_vehicle_acceleration msgs: {0}'.format(msg.xyz))
        pass
    
    #.. subscript_actuator_outputs
    def subscript_actuator_outputs(self, msg):
        self.actuator_outputs[0] = msg.output[0]
        self.actuator_outputs[1] = msg.output[1]
        self.actuator_outputs[2] = msg.output[2]
        self.actuator_outputs[3] = msg.output[3]
        self.actuator_outputs[4] = msg.output[4]
        self.actuator_outputs[5] = msg.output[5]
        # self.actuator_outputs[6] = msg.output[6]
        # self.actuator_outputs[7] = msg.output[7]
        # self.actuator_outputs[8] = msg.output[8]
        # self.actuator_outputs[9] = msg.output[9]
        # self.actuator_outputs[10] = msg.output[10]
        # self.actuator_outputs[11] = msg.output[11]
        # self.actuator_outputs[12] = msg.output[12]
        # self.actuator_outputs[13] = msg.output[13]
        # self.actuator_outputs[14] = msg.output[14]
        # self.actuator_outputs[15] = msg.output[15]
        # self.get_logger().info('subscript_actuator_outputs msgs: {0}'.format(msg.output))
        pass
    
    #.. subscript_MPPI_output
    def subscript_MPPI_output(self, msg):
        self.MPPI_ctrl_input[0] =   msg.data[0]
        self.MPPI_ctrl_input[1] =   msg.data[1]
        self.MPPI_ctrl_input[2] =   msg.data[2]
        if self.Q6.flag_guid_trans == 0:
            self.Q6.look_ahead_distance = self.MPPI_ctrl_input[0]
            self.Q6.desired_speed       = self.MPPI_ctrl_input[1]
            self.Q6.guid_eta            = self.MPPI_ctrl_input[2]
        # self.get_logger().info('subscript_MPPI_output msgs: {0}'.format(msg.data))
        pass
    def subscript_timesync_status(self, msg):
        if self.first_time_flag == False:
            self.first_time =   msg.timestamp * 0.000001
            self.first_time_flag = True
            self.takeoff_start = True
        else :
            self.sim_time =   msg.timestamp * 0.000001 - self.first_time
    ### Mathmatics Functions 
    #.. Quaternion to Euler
    def Quaternion2Euler(self, w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        Roll = m.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Pitch = m.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Yaw = m.atan2(t3, t4)
    
        return Roll, Pitch, Yaw
    
    #.. Euler to Quaternion
    def Euler2Quaternion(self, Roll, Pitch, Yaw):
        CosYaw = m.cos(Yaw * 0.5)
        SinYaw = m.sin(Yaw * 0.5)
        CosPitch = m.cos(Pitch * 0.5)
        SinPitch = m.sin(Pitch * 0.5)
        CosRoll = m.cos(Roll * 0.5)
        SinRoll= m.sin(Roll * 0.5)
        
        w = CosRoll * CosPitch * CosYaw + SinRoll * SinPitch * SinYaw
        x = SinRoll * CosPitch * CosYaw - CosRoll * SinPitch * SinYaw
        y = CosRoll * SinPitch * CosYaw + SinRoll * CosPitch * SinYaw
        
        # z = CosRoll * CosPitch * SinYaw - SinRoll * CosPitch * CosYaw
        
        z = CosRoll * CosPitch * SinYaw - SinRoll * SinPitch * CosYaw
        
        return w, x, y, z
        
def main(args=None):
    print("======================================================")
    print("------------- main() in node_att_ctrl.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    AttCtrl = NodeAttCtrl()
    rclpy.spin(AttCtrl)
    AttCtrl.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()


