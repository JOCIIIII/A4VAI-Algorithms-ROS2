#.. public libaries
import numpy as np
import math as m

#===================================================================================================================
#.. ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32MultiArray

#===================================================================================================================
#.. private libs.
from .necessary_settings import waypoint, quadrotor_iris_parameters
from .models import quadrotor
from .flight_functions import MPPI_guidance, GPR


class Node_MPPI_Output(Node):
    
    def __init__(self):
        super().__init__('node_MPPI_output')
        
        #.. model settings
        Iris_Param_Physical      = quadrotor_iris_parameters.Physical_Parameter()
        Iris_Param_GnC           = quadrotor_iris_parameters.GnC_Parameter(4)
        MPPI_Param               = quadrotor_iris_parameters.MPPI_Parameter(Iris_Param_GnC.Guid_type)
        GPR_Param                = quadrotor_iris_parameters.GPR_Parameter(MPPI_Param.dt_MPPI, MPPI_Param.N)
        self.QR                  = quadrotor.Quadrotor_6DOF(Iris_Param_Physical, Iris_Param_GnC, MPPI_Param, GPR_Param)
        self.WP                  = waypoint.Waypoint()

        #.. gpr and mppi settings
        self.GP                  = GPR.GPR_Modules(GPR_Param)
        # self.QR.guid_var.MPPI_ctrl_input = np.array([MPPI_Param.u1_init, MPPI_Param.u2_init, MPPI_Param.u3_init]) 
        
        #===================================================================================================================
        #.. variable initialization
        self.sim_time                   = 0.       
        self.MPPI_input_int_Q6_received = False
        self.MPPI_input_dbl_WP_received = False
        self.MPPI_setting_complete      = False
 
        # self.GPRlog_t = open("/home/user/log/point_mass_6d/datalogfile/GPRlog_t.txt",'w+')
        # self.GPRlog_X = open("/home/user/log/point_mass_6d/datalogfile/GPRlog_X.txt",'w+')
        # self.GPRlog_Y = open("/home/user/log/point_mass_6d/datalogfile/GPRlog_Y.txt",'w+')
        # self.GPRlog_Z = open("/home/user/log/point_mass_6d/datalogfile/GPRlog_Z.txt",'w+')
                
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Subscribers

        #.. subscriptions - from ROS2 msgs to ROS2 msgs
        self.MPPI_input_int_Q6_subscription =   self.create_subscription(Int32MultiArray, 'MPPI/in/int_Q6', self.subscript_MPPI_input_int_Q6, qos_profile_sensor_data)
        self.MPPI_input_dbl_Q6_subscription =   self.create_subscription(Float64MultiArray, 'MPPI/in/dbl_Q6', self.subscript_MPPI_input_dbl_Q6, qos_profile_sensor_data)
        self.MPPI_input_dbl_WP_subscription =   self.create_subscription(Float64MultiArray, 'MPPI/in/dbl_WP', self.subscript_MPPI_input_dbl_WP, qos_profile_sensor_data)
        self.GPR_input_dbl_NDO_subscription =   self.create_subscription(Float64MultiArray, 'GPR/in/dbl_Q6', self.subscript_GPR_input_dbl_NDO, qos_profile_sensor_data)
        
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Publishers

        #.. publishers - from ROS2 msgs to ROS2 msgs
        self.MPPI_output_publisher_         =   self.create_publisher(Float64MultiArray, 'MPPI/out/dbl_MPPI', 10)

    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Timers

        # callback PF_MPPI_param
        period_MPPI_param       =   self.QR.MPPI_param.dt_MPPI
        dt_GPR_opt              =   self.QR.GPR_param.dt_GPR_opt
        period_GPR_logger       =   1.

        self.timer  =   self.create_timer(period_MPPI_param, self.PF_MPPI_param)
        self.timer  =   self.create_timer(dt_GPR_opt, self.GPR_HyperParamsOpt)
        # self.timer  =   self.create_timer(period_GPR_logger, self.GPR_logger)

        pass
    
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Subscriber Call Back Functions  

    #.. subscript_MPPI_input_int_Q6
    def subscript_MPPI_input_int_Q6(self, msg):
        self.QR.PF_var.WP_idx_heading   =   msg.data[0]
        self.QR.PF_var.WP_idx_passed    =   msg.data[1]
        self.QR.GnC_param.Guid_type     =   msg.data[2]
        self.QR.PF_var.reset_flag2mppi  =   msg.data[3]

        self.MPPI_input_int_Q6_received =  True
    
    #.. subscript_MPPI_input_dbl_Q6
    def subscript_MPPI_input_dbl_Q6(self, msg):
        self.QR.state_var.Ri[0]        =   msg.data[0]
        self.QR.state_var.Ri[1]        =   msg.data[1]
        self.QR.state_var.Ri[2]        =   msg.data[2]
        self.QR.state_var.Vi[0]        =   msg.data[3]
        self.QR.state_var.Vi[1]        =   msg.data[4]
        self.QR.state_var.Vi[2]        =   msg.data[5]
        self.QR.state_var.att_ang[0]   =   msg.data[6]
        self.QR.state_var.att_ang[1]   =   msg.data[7]
        self.QR.state_var.att_ang[2]   =   msg.data[8]
        self.QR.guid_var.T_cmd         =   msg.data[9]
        # self.get_logger().info('subscript_MPPI_input_dbl_Q6 msgs: {0}'.format(msg.data))
        pass
            
    #.. subscript_MPPI_input_dbl_WP
    def subscript_MPPI_input_dbl_WP(self, msg):
        WPs_tmp         =   np.array(msg.data)
        self.WP.WPs     =   WPs_tmp.reshape(int(WPs_tmp.shape[0]/3),3)

        self.MPPI_input_dbl_WP_received =  True
        # self.get_logger().info('subscript_MPPI_input_dbl_WP msgs: {0}'.format(msg.data))
        pass
    
    #.. subscript_GPR_input_dbl_NDO
    def subscript_GPR_input_dbl_NDO(self, msg):
        self.sim_time                =   msg.data[0]
        self.QR.guid_var.out_NDO[0]  =   msg.data[1]
        self.QR.guid_var.out_NDO[1]  =   msg.data[2]
        self.QR.guid_var.out_NDO[2]  =   msg.data[3]
        # self.get_logger().info('subscript_GPR_input_dbl_NDO msgs: {0}'.format(msg.data))
        
        self.GP.GPR_update(self.QR.guid_var.out_NDO)
 
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Publication Functions   
    
    #.. publish_MPPI_output
    def publish_MPPI_output(self):
        msg                 =   Float64MultiArray()
        msg.data            =   [self.QR.guid_var.MPPI_ctrl_input[0], self.QR.guid_var.MPPI_ctrl_input[1], self.QR.guid_var.MPPI_calc_time]
        
        self.MPPI_output_publisher_.publish(msg)
        # self.get_logger().info('mppi: {0}'.format(np.linalg.norm(self.M)))
        # self.get_logger().info("subscript_MPPI_output: [0]=" + str(self.MPPI_ctrl_input[0]) +", [1]=" + str(self.MPPI_ctrl_input[1]) +", [2]=" + str(self.MPPI_ctrl_input[2]))
        pass

    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region functions

    ###.. MPPI functions ..###
    #.. PF_MPPI_param 
    def PF_MPPI_param(self):
    
        if self.MPPI_input_int_Q6_received == True and self.MPPI_input_dbl_WP_received == True and self.MPPI_setting_complete == False:
            
            self.MG  = MPPI_guidance.MPPI_Guidance_Modules(self.QR.MPPI_param)
            
            self.MG.set_total_MPPI_code(self.WP.WPs.shape[0])
            
            self.get_logger().info('MPPI SETTING COMPLETE')
            self.MPPI_setting_complete = True
            pass
        else:
            pass

        if self.MPPI_setting_complete == True:

            if self.QR.PF_var.reset_flag2mppi == 1:
                # self.get_logger().info('PATH FOLLOWING RESTART!!!')
                self.MG.u0  =  self.MG.MP.u0_init * np.ones(self.MG.MP.N)
                self.MG.u1  =  self.MG.MP.u1_init * np.ones(self.MG.MP.N)
                self.QR.guid_var.MPPI_ctrl_input = np.array([self.MG.MP.u0_init, self.MG.MP.u1_init]) 
                self.GP.GP_param.set_values(self.GP.GP_param.dt_GPR, self.GP.GP_param.ne_GPR)
                
                pass
            
            if self.QR.PF_var.WP_idx_passed >= 1:
                self.GP.GPR_forecasting(self.sim_time)

                #.. disturbance in MPPI
                self.MG.Ai_est_dstb[:,0] = self.GP.GP_param.me_x_array
                self.MG.Ai_est_dstb[:,1] = self.GP.GP_param.me_y_array
                # self.MG.Ai_est_dstb[:,2] = self.GP.GP_param.me_z_array
                self.MG.Ai_est_dstb[:,2] = self.QR.guid_var.out_NDO[2]*np.ones(self.MG.MP.N)

                self.MG.Ai_est_var[:,0] = np.minimum(self.GP.GP_param.var_x_array + 1e-2, 1e0)
        
            else:
                self.MG.Ai_est_dstb[:,0] = self.QR.guid_var.out_NDO[0]*np.ones(self.MG.MP.N)
                self.MG.Ai_est_dstb[:,1] = self.QR.guid_var.out_NDO[1]*np.ones(self.MG.MP.N)
                self.MG.Ai_est_dstb[:,2] = self.QR.guid_var.out_NDO[2]*np.ones(self.MG.MP.N)

            #.. MPPI algorithm
            if self.QR.PF_var.reset_flag2mppi == 0:
                self.QR.guid_var.MPPI_ctrl_input, self.QR.guid_var.MPPI_calc_time = self.MG.run_MPPI_Guidance(self.QR, self.WP.WPs)

            self.publish_MPPI_output()
        else:
            pass
        
        pass
    
    ###.. GPR functions ..###
    #.. GPR_HyperParamsOpt 
    def GPR_HyperParamsOpt(self):
        
        if self.QR.PF_var.WP_idx_passed >= 1:
            self.GP.GPR_hyperparams_opt()
        
        pass

                
    #.. GPR_logger
    def GPR_logger(self):
        if self.QR.PF_var.WP_idx_passed >= 1:
            # print(self.GP.GP_param.te_array[0:10])
            # print(self.GP.GP_param.me_x_array[0:10])
            # np.savetxt(self.GPRlog_t, self.GPR.te_array.reshape(1, self.GPR.ne_GPR), delimiter=' ')
            # np.savetxt(self.GPRlog_X, self.GPR.me_x_array.reshape(1, self.GPR.ne_GPR), delimiter=' ')
            # np.savetxt(self.GPRlog_Y, self.GPR.me_y_array.reshape(1, self.GPR.ne_GPR), delimiter=' ')
            # np.savetxt(self.GPRlog_Z, self.GPR.me_z_array.reshape(1, self.GPR.ne_GPR), delimiter=' ')
            pass
    
    
def main(args=None):
    print("======================================================")
    print("------------- main() in node_MPPI_output.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    MPPI_Output = Node_MPPI_Output()
    rclpy.spin(MPPI_Output)
    MPPI_Output.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()

