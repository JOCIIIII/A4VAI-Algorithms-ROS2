#.. public libaries
import numpy as np
import math as m
import os
import time

#===================================================================================================================
#.. ROS libraries
import rclpy
from rclpy.node   import Node
from rclpy.clock  import Clock
from rclpy.qos    import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import Int32

#===================================================================================================================
#.. PX4 libararies - sub.
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAcceleration
from px4_msgs.msg import TimesyncStatus

#.. PX4 libararies - pub.
from px4_msgs.msg import VehicleAttitudeSetpoint

#===================================================================================================================
#.. Custom msgs
from custom_msgs.msg import LocalWaypointSetpoint            # sub.
from custom_msgs.msg import ConveyLocalWaypointComplete      # pub. 

#===================================================================================================================
#.. private libs.
from .necessary_settings import waypoint, quadrotor_iris_parameters
from .models import quadrotor
from .flight_functions.utility_funcs import DCM_from_euler_angle, Quaternion2Euler

#===================================================================================================================
 
class NodeAttCtrl(Node):
    
    def __init__(self):
        super().__init__('node_attitude_control')

        #.. simulation settings
        self.guid_type_case      =   3       # | 0: Pos. Ctrl     | 1: GL-based  | 2: MPPI-direct | 3: MPPI-GL
        self.wp_type_selection   =   4       # | 0: straight line | 1: rectangle | 2: circle      | 3: designed | 4: path planning solution
        
        #.. model settings
        Iris_Param_Physical      = quadrotor_iris_parameters.Physical_Parameter()
        Iris_Param_GnC           = quadrotor_iris_parameters.GnC_Parameter(self.guid_type_case)
        MPPI_Param               = quadrotor_iris_parameters.MPPI_Parameter(Iris_Param_GnC.Guid_type)
        GPR_Param                = quadrotor_iris_parameters.GPR_Parameter(MPPI_Param.dt_MPPI, MPPI_Param.N)
        self.QR                  = quadrotor.Quadrotor_6DOF(Iris_Param_Physical, Iris_Param_GnC, MPPI_Param, GPR_Param)
        self.WP                  = waypoint.Waypoint(self.wp_type_selection)

        #===================================================================================================================
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
        
        #.. variable - estimator_state
        class var_msg_est_state:
            def __init__(self):
                self.pos_NED        =   np.zeros(3)
                self.vel_NED        =   np.zeros(3)
                self.eul_ang_rad    =   np.zeros(3)
                self.accel_xyz      =   np.zeros(3)
                
        self.est_state = var_msg_est_state()

        #.. time variables
        self.first_time_flag                = False
        self.takeoff_time                   = 0.
        self.sim_time                       = 0.
        
        #.. status signal of another module node
        self.timesync_status_flag           = False
        self.variable_setting_complete      = False 
        self.path_planning_complete         = False
        
        # heartbeat signal of another module node
        self.controller_heartbeat           = False
        self.path_planning_heartbeat        = False
        self.collision_avoidance_heartbeat  = False

        #===================================================================================================================
        ###.. Subscribers ..###

        # declare heartbeat_subscriber 
        self.controller_heartbeat_subscriber            =   self.create_subscription(Bool, '/controller_heartbeat',            self.controller_heartbeat_call_back,            10)
        self.path_planning_heartbeat_subscriber         =   self.create_subscription(Bool, '/path_planning_heartbeat',         self.path_planning_heartbeat_call_back,        10)
        self.collision_avoidance_heartbeat_subscriber   =   self.create_subscription(Bool, '/collision_avoidance_heartbeat',   self.collision_avoidance_heartbeat_call_back,   10)

        #.. subscriptions - from px4 msgs to ROS2 msgs
        self.local_waypoint_subscriber                  =   self.create_subscription(LocalWaypointSetpoint, '/local_waypoint_setpoint_to_PF',self.local_waypoint_setpoint_call_back, 1) 
        self.vehicle_local_position_subscriber          =   self.create_subscription(VehicleLocalPosition,    '/fmu/out/vehicle_local_position',   self.vehicle_local_position_callback,   qos_profile_sensor_data)
        self.vehicle_attitude_subscriber                =   self.create_subscription(VehicleAttitude,         '/fmu/out/vehicle_attitude',         self.vehicle_attitude_callback,         qos_profile_sensor_data)
        self.vehicle_acceleration_subscription          =   self.create_subscription(VehicleAcceleration, '/fmu/out/vehicle_acceleration', self.subscript_vehicle_acceleration, qos_profile_sensor_data)      

        if self.guid_type_case >= 3:
            self.MPPI_output_subscription   =   self.create_subscription(Float64MultiArray, 'MPPI/out/dbl_MPPI', self.subscript_MPPI_output, qos_profile_sensor_data)

        #===================================================================================================================
        ###.. Publishers ..###

        self.vehicle_attitude_setpoint_publisher        =   self.create_publisher(VehicleAttitudeSetpoint,   '/pf_att_2_control', 10)
        self.local_waypoint_receive_complete_publisher  =   self.create_publisher(ConveyLocalWaypointComplete, '/convey_local_waypoint_complete', 10) 
        self.heading_wp_idx_publisher                   =   self.create_publisher(Int32, '/heading_waypoint_index', 1)
        self.heartbeat_publisher                        =   self.create_publisher(Bool, '/path_following_heartbeat', 10)
        self.pf_complete_publisher                      =   self.create_publisher(Bool, '/path_following_complete', 1)

        if self.guid_type_case >= 3:
            #.. publishers - from ROS2 msgs to ROS2 msgs
            self.MPPI_input_int_Q6_publisher_   =   self.create_publisher(Int32MultiArray,   'MPPI/in/int_Q6', 10)
            self.MPPI_input_dbl_Q6_publisher_   =   self.create_publisher(Float64MultiArray, 'MPPI/in/dbl_Q6', 10)
            self.MPPI_input_dbl_VT_publisher_   =   self.create_publisher(Float64MultiArray, 'MPPI/in/dbl_VT', 10)
            self.MPPI_input_dbl_WP_publisher_   =   self.create_publisher(Float64MultiArray, 'MPPI/in/dbl_WP', 10)
            self.GPR_input_dbl_NDO_publisher_   =   self.create_publisher(Float64MultiArray, 'GPR/in/dbl_Q6', 10)
               
        #===================================================================================================================
        ###.. Timers ..###
        period_heartbeat_mode       =   1        
        period_offboard_att_ctrl    =   self.QR.GnC_param.dt_GCU       # required 250Hz at least for attitude control in [6]

        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)
        self.timer            =   self.create_timer(period_offboard_att_ctrl, self.publisher_vehicle_attitude_setpoint)

        # Guid_type = | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        if self.guid_type_case >= 3:
            period_MPPI_input   =   0.5 * self.QR.MPPI_param.dt_MPPI    # 2 times faster than MPPI dt
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_MPPI_input_int_Q6)
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_MPPI_input_dbl_Q6)
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_MPPI_input_dbl_VT)
            self.timer  =   self.create_timer(period_MPPI_input * 10., self.publish_MPPI_input_dbl_WP)
            self.timer  =   self.create_timer(period_MPPI_input, self.publish_GPR_input_dbl_NDO)
        
        self.timer  =   self.create_timer(self.QR.GnC_param.dt_GCU, self.main_attitude_control)
                     
    #===================================================================================================================
    # Subscriber Call Back Functions  
    #===================================================================================================================
    
    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self,msg):
        self.controller_heartbeat = msg.data

    # heartbeat subscribe from path following
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.data

    # heartbeat subscribe from collision avoidance
    def collision_avoidance_heartbeat_call_back(self,msg):
        self.collision_avoidance_heartbeat = msg.data

    # receive local waypoint from controller
    def local_waypoint_setpoint_call_back(self,msg):
        # self.get_logger().info("msg.path_planning_complete: {0}".format(msg.path_planning_complete))
        
        if msg.path_planning_complete == True:
            self.path_planning_complete = True
            self.WP.waypoint_x             = msg.waypoint_x 
            self.WP.waypoint_y             = msg.waypoint_y
            self.WP.waypoint_z             = msg.waypoint_z

            # self.get_logger().info("                                          ")
            # self.get_logger().info("=====    revieved Path Planning      =====")
            # self.get_logger().info("                                          ")
            self.local_waypoint_receive_complete_publish()
        elif msg.path_planning_complete == False:
            self.QR.PF_var.reWP_flag      = 1
            # 241223 diy
            self.QR.PF_var.reWP_flag2mppi = 1

            self.WP.set_values(self.wp_type_selection, msg.waypoint_x, msg.waypoint_y, msg.waypoint_z)
            # print("                                          ")
            # print("==             updated waypoints        ==")
            # print("                                          ")

    # subscribe position, velocity 
    def vehicle_local_position_callback(self, msg):
        # update NED position 
        self.est_state.pos_NED[0]    =   msg.x
        self.est_state.pos_NED[1]    =   msg.y
        self.est_state.pos_NED[2]    =   msg.z
        # update NED velocity
        self.est_state.vel_NED[0]    =   msg.vx
        self.est_state.vel_NED[1]    =   msg.vy
        self.est_state.vel_NED[2]    =   msg.vz

    # subscribe attitude 
    def vehicle_attitude_callback(self, msg):
        self.est_state.eul_ang_rad[0], self.est_state.eul_ang_rad[1], self.est_state.eul_ang_rad[2] = \
            Quaternion2Euler(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
    
    #.. subscript_vehicle_acceleration
    def subscript_vehicle_acceleration(self, msg):
        # self.hover_thrust   =   msg.hover_thrust          # is the value of msg.hover_thrust correct ???
        self.est_state.accel_xyz[0] = msg.xyz[0]
        self.est_state.accel_xyz[1] = msg.xyz[1]
        self.est_state.accel_xyz[2] = msg.xyz[2]
        # self.get_logger().info('subscript_vehicle_acceleration msgs: {0}'.format(msg.xyz))
        pass
        
    #.. subscript_MPPI_output
    def subscript_MPPI_output(self, msg):
        self.QR.guid_var.MPPI_ctrl_input[0] =   msg.data[0]
        self.QR.guid_var.MPPI_ctrl_input[1] =   msg.data[1]
        self.QR.guid_var.MPPI_ctrl_input[2] =   msg.data[2]

        # self.get_logger().info('subscript_MPPI_output msgs: {0}'.format(msg.data))
        pass

    # main_attiude_control (initialization)
    def subscript_timesync_status(self, msg):
        if self.first_time_flag == False:
            self.first_time =   msg.timestamp * 0.000001
            self.first_time_flag = True
        else :
            self.sim_time =   msg.timestamp * 0.000001 - self.first_time

    #===================================================================================================================
    # Publication Functions   
    #===================================================================================================================
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

    # send local waypoint receive complete flag
    def local_waypoint_receive_complete_publish(self):
        msg = ConveyLocalWaypointComplete()
        msg.convey_local_waypoint_is_complete = True
        self.local_waypoint_receive_complete_publisher.publish(msg)

    def heading_wp_idx_publish(self):
        msg = Int32()
        msg.data = self.QR.PF_var.WP_idx_heading
        # self.get_logger().info('heading_wp_index: {0}'.format(self.QR.PF_var.WP_idx_heading))
        # self.get_logger().info('heading_wp_position: {0}'.format(self.WP.WPs[self.QR.PF_var.WP_idx_heading]))
        # self.get_logger().info('heading_wp_position: {0}'.format(self.WP.WPs))
        self.heading_wp_idx_publisher.publish(msg)
        
    # heartbeat publish
    def publish_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.heartbeat_publisher.publish(msg)

    # send path following complete flag
    def publish_pf_complete(self):
        msg = Bool()
        msg.data = self.QR.PF_var.PF_done
        self.pf_complete_publisher.publish(msg)

    #.. publish_MPPI_input_int_Q6
    def publish_MPPI_input_int_Q6(self):
        msg         =   Int32MultiArray()
        msg.data    =   [self.QR.PF_var.WP_idx_heading, self.QR.PF_var.WP_idx_passed, self.QR.GnC_param.Guid_type, self.QR.PF_var.reWP_flag2mppi]
        self.MPPI_input_int_Q6_publisher_.publish(msg)
        # self.get_logger().info('pub msgs: {0}'.format(msg.data))
        self.QR.PF_var.reWP_flag2mppi = 0
        pass
    
    #.. publish_MPPI_input_dbl_Q6
    def publish_MPPI_input_dbl_Q6(self):
        tmp_float   =   0.
        msg         =   Float64MultiArray()
        msg.data    =   [self.QR.state_var.Ri[0], self.QR.state_var.Ri[1], self.QR.state_var.Ri[2],
                         self.QR.state_var.Vi[0], self.QR.state_var.Vi[1], self.QR.state_var.Vi[2],
                         self.QR.state_var.att_ang[0], self.QR.state_var.att_ang[1], self.QR.state_var.att_ang[2],
                         self.QR.guid_var.T_cmd]

        self.MPPI_input_dbl_Q6_publisher_.publish(msg)
        # self.get_logger().info('pub msgs: {0}'.format(msg.data))
        pass
    
    #.. publish_MPPI_input_dbl_VT
    def publish_MPPI_input_dbl_VT(self):
        msg         =   Float64MultiArray()
        msg.data    =   [self.QR.PF_var.VT_Ri[0], self.QR.PF_var.VT_Ri[1], self.QR.PF_var.VT_Ri[2]]
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
            # self.get_logger().info('subscript_MPPI_input_dbl_WP msgs: {0}'.format(msg.data))
        # print(self.WP.WPs)
        else: 
            pass
    
    #.. publish_GPR_input_dbl_NDO
    def publish_GPR_input_dbl_NDO(self):
        msg         =   Float64MultiArray()
        msg.data    =   [self.sim_time,
                         self.QR.guid_var.out_NDO[0], self.QR.guid_var.out_NDO[1], self.QR.guid_var.out_NDO[2]]
        self.GPR_input_dbl_NDO_publisher_.publish(msg)
        # self.get_logger().info('publish_MPPI_input_dbl_WP: {0}'.format(msg.data))
        # print(self.WP.WPs)
        pass

    #===================================================================================================================
    # Functions
    #===================================================================================================================
    #.. main_attitude_control 
    def main_attitude_control(self):
        if self.controller_heartbeat == True and self.path_planning_heartbeat ==True and self.collision_avoidance_heartbeat == True:
        
            if self.timesync_status_flag == False:
                self.timesync_status_subscription   =   self.create_subscription(TimesyncStatus, '/fmu/out/timesync_status', self.subscript_timesync_status, qos_profile_sensor_data)
                self.timesync_status_flag = True

            if self.path_planning_complete == True and self.variable_setting_complete == False :
                
                self.get_logger().info("=====    revieved Path Planning      =====")

                ### --- Start - simulation setting --- ###
                
                # state variable initialization
                self.QR.update_states(self.est_state.pos_NED, self.est_state.vel_NED, self.est_state.eul_ang_rad, self.est_state.accel_xyz)
                
                 # waypoint settings                     
                self.WP.set_values(self.wp_type_selection, self.WP.waypoint_x, self.WP.waypoint_y, self.WP.waypoint_z)

                # self.WP.insert_WP(0, self.QR.state_var.Ri)
                # var for waypoint regeneration test
                self.count_stop = 0

                ### ---  End  - need configuration of simulation setting --- ###

                self.variable_setting_complete = True
            else :
                pass

            if self.variable_setting_complete == True :

                # For test------------------------------------------                
                # 20240914 diy
                # self.count_stop = self.count_stop + 1
                # print(self.count_stop)
                # if (self.count_stop>15000 and self.count_stop<18000):
                #     self.QR.PF_var.stop_flag = 1

                # if (self.count_stop == 18000):
                #     self.QR.PF_var.reWP_flag      = 1
                #     self.QR.PF_var.stop_flag      = 0
                # ---------------------------------------------------

                #.. waypoint regeneration, initialization 241223 diy
                if (self.QR.PF_var.reWP_flag == 1):
                    self.QR.PF_var.reWP_flag      = 0    
                    # self.WP.WPs                   = self.WP.reWPs
                    self.QR.PF_var.WP_idx_heading = 1
                    self.QR.PF_var.WP_idx_passed  = 0
                    self.QR.guid_var.z_NDO        = np.array([0., 0., 0.])

                # self.get_logger().info('self.QR.PF_var.WP_idx_passed: {0}'.format(self.QR.PF_var.WP_idx_passed))
                # self.get_logger().info('heading_wp_z : {0}'.format(-self.WP.WPs[self.QR.PF_var.WP_idx_heading][2]))
                #.. state variables updates (from px4)
                self.QR.update_states(self.est_state.pos_NED, self.est_state.vel_NED, self.est_state.eul_ang_rad, self.est_state.accel_xyz)
                
                #.. path following required information
                self.QR.PF_required_info(self.WP.WPs, self.sim_time, self.QR.GnC_param.dt_GCU)
                self.heading_wp_idx_publish()
                if self.QR.PF_var.PF_done == True:
                    self.publish_pf_complete()

                #.. guidance                
                self.QR.guid_Ai_cmd(self.WP.WPs.shape[0], self.QR.guid_var.MPPI_ctrl_input)  # based on the geometry and VT     
                self.QR.guid_compensate_Ai_cmd()                                             # gravity, disturbance rejection
                self.QR.guid_NDO_for_Ai_cmd()                                                # NDO for disturbance estimation
                self.QR.guid_convert_Ai_cmd_to_thrust_and_att_ang_cmd(self.WP.WPs)
                self.QR.guid_convert_att_ang_cmd_to_qd_cmd()
                
                #.. guidance command
                self.veh_att_set.thrust_body    =   [0., 0., -self.QR.guid_var.norm_T_cmd]
                self.veh_att_set.q_d            =   self.QR.guid_var.qd_cmd

                # if self.QR.PF_var.WP_idx_heading != (self.WP.WPs.shape[0] - 1):
                    # self.get_logger().info("mppi_Ax " + str(self.QR.guid_var.MPPI_ctrl_input[1]) +", mppi_eta " + str(self.QR.guid_var.MPPI_ctrl_input[2]))

                # pass
            else :
                pass
        else:
            pass
    

        
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


