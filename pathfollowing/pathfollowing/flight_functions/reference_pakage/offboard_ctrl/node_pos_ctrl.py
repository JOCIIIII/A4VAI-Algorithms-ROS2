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
from px4_msgs.msg import EstimatorStates
from px4_msgs.msg import HoverThrustEstimate 
from px4_msgs.msg import VehicleAcceleration 
from px4_msgs.msg import ActuatorOutputs 
#.. PX4 libararies - pub.
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

#.. PF algorithms libararies
# from .testpy1 import testpypy1
from .PF_modules.quadrotor_6dof import Quadrotor_6DOF
from .PF_modules.virtual_target import Virtual_Target
from .PF_modules.set_parameter import Way_Point
from .PF_modules.Funcs_PF_Base import kinematics, distance_from_Q6_to_path, check_waypoint, virtual_target_position, \
    guidance_modules, compensate_Aqi_cmd, thrust_cmd, att_cmd, NDO_Aqi
from .PF_modules.utility_functions import DCM_from_euler_angle



class NodePosCtrl(Node):
    
    def __init__(self):
        super().__init__('node_position_control')
        
        #.. Reference        
        #   [1]: https://docs.px4.io/main/en/msg_docs/vehicle_command.html
        #   [2]: https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
        #   [3]: https://github.com/PX4/px4_ros_com/blob/release/1.13/src/examples/offboard/offboard_control.cpp
        #   [4]: https://docs.px4.io/main/ko/advanced_config/tuning_the_ecl_ekf.html
        #   [5]: https://hostramus.tistory.com/category/ROS2
        
        #.. mapping of ros2-px4 message name using in this code
        #   from ' [basedir]/ws_sensor_combined/src/px4_ros_com/templates/urtps_bridge_topics.yaml '
        #   to ' [basedir]/PX4-Autopilot/msg/tools/urtps_bridge_topics.yaml '
        class msg_mapping_ros2_to_px4:
            VehicleCommand          =   '/fmu/in/vehicle_command'
            OffboardControlMode     =   '/fmu/in/offboard_control_mode'
            TrajectorySetpoint      =   '/fmu/in/trajectory_setpoint'
            EstimatorStates         =   '/fmu/out/estimator_states'
            HoverThrustEstimate     =   '/fmu/out/hover_thrust_estimate'
            VehicleAcceleration     =   '/fmu/out/vehicle_acceleration'
            ActuatorOutputs         =   '/fmu/out/actuator_outputs'
                    
        #.. publishers - from ROS2 msgs to px4 msgs
        self.vehicle_command_publisher_             =   self.create_publisher(VehicleCommand, msg_mapping_ros2_to_px4.VehicleCommand, 10)
        self.offboard_control_mode_publisher_       =   self.create_publisher(OffboardControlMode, msg_mapping_ros2_to_px4.OffboardControlMode , 10)
        self.trajectory_setpoint_publisher_         =   self.create_publisher(TrajectorySetpoint, msg_mapping_ros2_to_px4.TrajectorySetpoint, 10)      
        #.. subscriptions - from px4 msgs to ROS2 msgs
        self.estimator_states_subscription          =   self.create_subscription(EstimatorStates, msg_mapping_ros2_to_px4.EstimatorStates, self.subscript_estimator_states, qos_profile_sensor_data)
        self.hover_thrust_estimate_subscription     =   self.create_subscription(HoverThrustEstimate, msg_mapping_ros2_to_px4.HoverThrustEstimate, self.subscript_hover_thrust_estimate, qos_profile_sensor_data)
        self.vehicle_acceleration_subscription      =   self.create_subscription(VehicleAcceleration, msg_mapping_ros2_to_px4.VehicleAcceleration, self.subscript_vehicle_acceleration, qos_profile_sensor_data)
        self.actuator_outputs_subscription          =   self.create_subscription(ActuatorOutputs, msg_mapping_ros2_to_px4.ActuatorOutputs, self.subscript_actuator_outputs, qos_profile_sensor_data)
        
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
        
        #.. variable - vehicle position setpoint
        class msg_veh_trj_set:
            def __init__(self):
                self.pos_NED    =   np.zeros(3)
                self.yaw_rad    =   0.
        
        self.veh_trj_set    =   msg_veh_trj_set()
        
        #.. other parameter & variable
        #.. takeoff_time
        self.takeoff_start = False
        self.takeoff_time = 0
        
        # vehicle state variable
        self.pos_NED    =   np.zeros(3)
        self.vel_NED    =   np.zeros(3)
        self.eul_ang_rad    =   np.zeros(3)
        self.windvel_NE     =   np.zeros(2)
                
        # callback main_position_control
        period_offboard_pos_ctrl    =   0.02
        self.timer  =   self.create_timer(period_offboard_pos_ctrl, self.main_position_control)
        
        # callback offboard_control_mode
        # offboard counter in [3]
        period_offboard_control_mode    =   0.2
        self.timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_mode)
        self.prm_offboard_setpoint_counter_start_flight  =   m.ceil(2./period_offboard_control_mode)
        self.offboard_setpoint_counter_     =   0
        
        ###### - start - Vars. for PF algorithm ######
        #.. declare variables/instances
        self.Q6     =   Quadrotor_6DOF()
        self.Q6.dt_GCU  =   period_offboard_pos_ctrl
        # self.Q6.Guid_type  =   1           # | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        
        self.VT     =   Virtual_Target()
        self.VT_psi_cmd     =   Virtual_Target()
        
        #.. set waypoint
        wp_type_selection   =   0       # | 0: straight line | 1: ractangle | 2: circle | 3: designed
        self.WP     =   Way_Point(wp_type_selection)

        #.. hover_thrust
        self.Aqi_grav   =   np.array([0., 0., 9.81])
        self.hover_thrust = 0.7         # iris
        # self.hover_thrust = 0.5
        # self.hover_thrust = 0.41        # typhoon
        
        #.. callback state_logger
        Hz_state_logger = 10
        self.timer  =   self.create_timer(1/Hz_state_logger, self.state_logger)
        self.datalogFile = open("/root/point_mass_6d/datalogfile/datalog.txt",'w')
        
        self.att_ang_cmd = np.zeros(3)
        self.Aqi_cmd = np.zeros(3)
        
        self.accel_xyz = np.zeros(3)
        self.actuator_outputs = np.zeros(16)
        ###### -  end  - Vars. for PF algorithm ######
        
        
    ### main function
    def offboard_control_mode(self):
        if self.offboard_setpoint_counter_ == self.prm_offboard_setpoint_counter_start_flight:
            # print("----- debug point [1] -----")
            
            # offboard mode cmd
            self.publish_vehicle_command(self.prm_offboard_mode)
            # arm cmd
            self.publish_vehicle_command(self.prm_arm_mode)
            
        # set offboard cntrol mode 
        self.publish_offboard_control_mode(self.prm_off_con_mod)
            
        # count offboard_setpoint_counter_
        if self.offboard_setpoint_counter_ < self.prm_offboard_setpoint_counter_start_flight:
            self.offboard_setpoint_counter_ = self.offboard_setpoint_counter_ + 1
            #.. variable setting
            self.Q6.Ri  =   self.pos_NED
            self.Q6.Vi  =   self.vel_NED
            self.Q6.att_ang =   self.eul_ang_rad
            self.Q6.cI_B    =   DCM_from_euler_angle(self.Q6.att_ang)
            # self.Q6.throttle_hover = self.hover_thrust
            #.. initialization
            self.WP.init_WP(self.Q6.Ri)
            self.VT.init_VT_Ri(self.WP.WPs, self.Q6.Ri, self.Q6.look_ahead_distance)
            
            self.VT_psi_cmd.init_VT_Ri(self.WP.WPs, self.Q6.Ri, self.Q6.look_ahead_distance_psi_cmd)
        pass
    
    
    #.. main_position_control 
    def main_position_control(self):
        
        #.. variable setting
        self.Q6.Ri  =   self.pos_NED.copy()
        self.Q6.Vi  =   self.vel_NED.copy()
        self.Q6.att_ang =   self.eul_ang_rad
        self.Q6.cI_B    =   DCM_from_euler_angle(self.Q6.att_ang)
        self.Q6.throttle_hover = self.hover_thrust
        self.Q6.Ai      =   np.matmul(np.transpose(self.Q6.cI_B), self.accel_xyz)
        
        ###### - start - PF algorithm ######
        #.. kinematics
        mag_Vqi, LOS_azim, LOS_elev, tgo, FPA_azim, FPA_elev, self.Q6.cI_W = kinematics(self.VT.Ri, self.Q6.Ri, self.Q6.Vi)
        # mag_Vqi, LOS_azim, LOS_elev, tgo, FPA_azim, FPA_elev, self.Q6.cI_W = kinematics(self.VT_psi_cmd.Ri, self.Q6.Ri, self.Q6.Vi)
        LA_azim     =   FPA_azim - LOS_azim
        LA_elev     =   FPA_elev - LOS_elev
        
        #.. distance from quadrotor to ref. path  
        dist_to_path, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, unit_Rw1w2 = \
            distance_from_Q6_to_path(self.WP.WPs, self.Q6.WP_idx_heading, self.Q6.Ri, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed)
            
        
        #.. virtual target modules
        #.. directly decide a position of the virtual target 
        # check waypoint - quadrotor
        self.Q6.WP_idx_heading = check_waypoint(self.WP.WPs, self.Q6.WP_idx_heading, self.Q6.Ri, self.Q6.distance_change_WP)
        # virtual target position
        self.VT.Ri = virtual_target_position(dist_to_path, self.Q6.look_ahead_distance, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, self.WP.WPs)
        
        self.VT_psi_cmd.Ri = virtual_target_position(dist_to_path, self.Q6.look_ahead_distance_psi_cmd, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, self.WP.WPs)
        ###### -  end  - PF algorithm ######
        
        
        self.veh_trj_set.pos_NED    =   self.VT.Ri.copy()
        self.veh_trj_set.yaw_rad    =   LOS_azim
        self.publish_trajectory_setpoint(self.veh_trj_set)
        
        pass
    
    #.. state_logger 
    def state_logger(self):
        if (self.Q6.WP_idx_heading < self.WP.WPs.shape[0]-1):
        # if (self.Q6.WP_idx_heading < self.WP.WPs.shape[0]):
            # self.get_logger().info("-----------------")
            current_time = int(Clock().now().nanoseconds / 1000) # time in microseconds
            sim_time   =   (current_time - self.takeoff_time) / 1000000
            
            self.get_logger().info("-----------------")
            self.get_logger().info("sim_time =" + str(sim_time) )
            self.get_logger().info("flag_guid_trans =" + str(self.Q6.flag_guid_trans) )
            self.get_logger().info("throttle_hover      =" + str(self.Q6.throttle_hover) )
            self.get_logger().info("Vi:         [0]=" + str(self.Q6.Vi[0]) +", [1]=" + str(self.Q6.Vi[1]) +", [2]=" + str(self.Q6.Vi[2]))
            self.get_logger().info("out_NDO:    [0]=" + str(self.Q6.out_NDO[0]) +", [1]=" + str(self.Q6.out_NDO[1]) +", [2]=" + str(self.Q6.out_NDO[2]))
            self.get_logger().info("att_cmd_deg:[0]=" + str(self.att_ang_cmd[0]*180./m.pi) +", [1]=" + str(self.att_ang_cmd[1]*180./m.pi) +", [2]=" + str(self.att_ang_cmd[2]*180./m.pi))
            self.get_logger().info("att_ang    :[0]=" + str(self.Q6.att_ang[0]*180./m.pi) +", [1]=" + str(self.Q6.att_ang[1]*180./m.pi) +", [2]=" + str(self.Q6.att_ang[2]*180./m.pi))
            self.get_logger().info("Aqi_cmd    :[0]=" + str(self.Aqi_cmd[0]) +", [1]=" + str(self.Aqi_cmd[1]) +", [2]=" + str(self.Aqi_cmd[2]))
            self.get_logger().info("Ai         :[0]=" + str(self.Q6.Ai[0]) + "[1]=" + str(self.Q6.Ai[1]) + "[2]=" + str(self.Q6.Ai[2]) )
            self.get_logger().info("Act_Out    :[0]=" + str(round(self.actuator_outputs[0], 2))+ ", [1]=" + str(round(self.actuator_outputs[1], 2))+ ", [2]=" + str(round(self.actuator_outputs[2], 2))+ ", [3]=" + str(round(self.actuator_outputs[3], 2))
                                           + ", [4]=" + str(round(self.actuator_outputs[4], 2))+ ", [5]=" + str(round(self.actuator_outputs[5], 2)) )
            
            
            # if self.takeoff_start == True and sim_time  < 45.:
            if self.takeoff_start:
                self.sim_time = sim_time
                w, x, y, z = self.Euler2Quaternion(self.Q6.att_ang[0], self.Q6.att_ang[1], self.Q6.att_ang[2])
                
                Data = "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \
                     %f %f %f %f %f %f %f %f %f %f\n" %(
                    sim_time, self.VT.Ri[0], self.VT.Ri[1], self.VT.Ri[2], self.Q6.Ri[0], 
                    self.Q6.Ri[1], self.Q6.Ri[2], self.Q6.Vi[0], self.Q6.Vi[1], self.Q6.Vi[2], 
                    self.Q6.out_NDO[0], self.Q6.out_NDO[1], self.Q6.out_NDO[2], self.Q6.att_ang[0], self.Q6.att_ang[1], 
                    self.Q6.att_ang[2],  self.att_ang_cmd[0],  self.att_ang_cmd[1], self.att_ang_cmd[2], self.Q6.desired_speed, 
                    
                    self.Aqi_cmd[0], self.Aqi_cmd[1], self.Aqi_cmd[2], self.Q6.Ai[0], self.Q6.Ai[1],
                    self.Q6.Ai[2], 0., 0., 0., 0.,
                    
                    w, x, y, z, 0.,
                    self.actuator_outputs[0], self.actuator_outputs[1], self.actuator_outputs[2], self.actuator_outputs[3], self.actuator_outputs[4], self.actuator_outputs[5], 
                    0., 0., 0., 0.
                    )
                self.datalogFile.write(Data)
        else:
            #.. datalogfile close
            self.datalogFile.close()
        
        pass
    
                    
    ### publushers
    #.. publish_vehicle_command
    def publish_vehicle_command(self, prm_veh_com):
        msg                 =   VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
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
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher_.publish(msg)
        
        pass
        
    #.. publish_trajectory_setpoint
    def publish_trajectory_setpoint(self, trj_set):
        msg                 =   TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   trj_set.pos_NED.tolist()
        msg.yaw             =   trj_set.yaw_rad
        self.trajectory_setpoint_publisher_.publish(msg)
        
        pass
        
    ### subscriptions        
    #.. subscript subscript_estimator_states
    def subscript_estimator_states(self, msg):        
        self.pos_NED[0]     =   msg.states[7]
        self.pos_NED[1]     =   msg.states[8]
        self.pos_NED[2]     =   msg.states[9]
        self.vel_NED[0]     =   msg.states[4]
        self.vel_NED[1]     =   msg.states[5]
        self.vel_NED[2]     =   msg.states[6]
        # Attitude
        self.eul_ang_rad[0], self.eul_ang_rad[1], self.eul_ang_rad[2] = \
            self.Quaternion2Euler(msg.states[0], msg.states[1], msg.states[2], msg.states[3])
        
        # Wind Velocity NE
        self.windvel_NE[0]  =   msg.states[22]
        self.windvel_NE[1]  =   msg.states[23]
        pass
    
    #.. subscript_hover_thrust_estimate
    def subscript_hover_thrust_estimate(self, msg):
        self.hover_thrust   =   msg.hover_thrust          # is the value of msg.hover_thrust correct ???
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
        z = CosRoll * CosPitch * SinYaw - SinRoll * CosPitch * CosYaw
        
        return w, x, y, z
        
def main(args=None):
    print("======================================================")
    print("------------- main() in node_pos_ctrl.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    PosCtrl = NodePosCtrl()
    rclpy.spin(PosCtrl)
    PosCtrl.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()


