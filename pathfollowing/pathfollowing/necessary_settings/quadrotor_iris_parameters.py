############################################################
#
#   - Name : quadrotor_parameters.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m

# private libs.


#.. MPPI_Parameter
class MPPI_Parameter():
    #.. initialize an instance of the class
    def __init__(self, MPPI_type) -> None:
        self.set_values(MPPI_type)
        pass
    
    #.. >>>>>  set_values  <<<<<
    def set_values(self, MPPI_type):
        self.MPPI_type =  MPPI_type  # | 0: Ctrl-based | 1: GL-based | 2: Direct | 3: GL-based-MPPI | 4: Ctrl-based MPPI | 9: test
        
        # cost-related
        # self.Q      =   np.array([4.0, 0.])     # dist, V
        self.Q      =   np.array([8.0, 0.])     # dist, V
        # self.R      =   np.array([0.004])
        self.R      =   np.array([0.001])
        # self.P      =   np.array([40.0])
        self.P      =   np.array([10.0])
        # self.P      =   np.array([5.0])
        self.cost_min_V_aligned = 0.3

        self.dt_MPPI = 0.05
        
        # MPPI parameters
        if self.MPPI_type == 3:
            self.flag_cost_calc     =   0
            #.. Parameters - low performance && short computation
            self.dt_MPPI     =   0.05
            self.K      =   256
            self.N      =   70
            self.nu     =   1000. 
            #.. u1: VTD, u2: desired_speed, u3: guid_eta
            self.var1   =   0.
            self.var2   =   1.0 * 1.0
            self.var3   =   1.0 * 1.0
            self.lamb1  =   1.0 * 4.0
            self.lamb2  =   1.0 * 0.5
            self.lamb3  =   1.0 * 0.5
            self.u1_init    =   4. # 3.
            self.u2_init    =   1.
            self.u3_init    =   0.

            # #.. Parameters - good / stable for 6dof
            # self.dt_MPPI     =   0.04
            # self.N      =   100
            # self.nu     =   1000.
            # # #.. u1: LAD, u2: desired_speed, u3: eta
            # self.var1   =   1.0 * 0.5           # 0.2
            # self.var2   =   self.var1
            # self.var3   =   self.var1
            # self.lamb1  =   1.0 *1.0           # 1.0
            # self.lamb2  =   self.lamb1
            # self.lamb3  =   self.lamb1
            # self.u1_init    =   3.0
            # self.u2_init    =   2.0
            # self.u3_init    =   2.
            
            
            
        elif self.MPPI_type == 2:
            # # cost-related
            # self.Q      =   np.array([0.3, 0.6 * 0.5])
            # self.R      =   np.array([0.001 * 0.5] )
            # self.cost_min_V_aligned = 0.5
            
            self.flag_cost_calc     =   0
            #.. Parameters - low performance && short computation
            # self.dt     =   0.02
            # self.K      =   256 * 4
            # self.N      =   250
            # self.K      =   256
            # self.dt     =   0.05
            # self.N      =   70
            # self.K      =   256 * 2
            # self.dt     =   0.02
            # self.N      =   175
            # self.K      =   256
            # self.dt     =   0.03
            # self.N      =   100
            self.nu     =   1000.       # 2.
            #.. u1: VTD, u2: desired_speed, u3: guid_eta
            self.var1   =   1.0 * 0.2
            # self.var1   =   1.0 * 0.1
            # self.var1   =   1.0 * 0.5 * 6.0
            self.var2   =   self.var1
            self.var3   =   self.var2
            self.lamb1  =   1.0 * 1.0
            # self.lamb1  =   1.0 * 0.5
            self.lamb2  =   self.lamb1
            self.lamb3  =   self.lamb1
            self.u1_init    =   0.
            self.u2_init    =   0.
            self.u3_init    =   0.
            
            
        elif self.MPPI_type == 9:
            self.flag_cost_calc     =   0
            #.. Parameters - low performance && short computation
            self.dt_MPPI     =   0.05
            self.K      =   1
            self.N      =   1100
            self.nu     =   1000.       # 2.
            #.. u1: VTD, u2: desired_speed, u3: guid_eta
            self.var1   =   1.0 * 0.5
            self.var2   =   self.var1
            self.var3   =   self.var1
            self.lamb1  =   1.0 *1.0
            self.lamb2  =   self.lamb1
            self.lamb3  =   self.lamb1
            self.u1_init    =   3.
            self.u2_init    =   2.
            self.u3_init    =   2.
        else:
            self.flag_cost_calc     =   0
            #.. Parameters - low performance && short computation
            self.dt_MPPI     =   0.05
            self.K      =   1
            self.N      =   1
            self.nu     =   1000.       # 2.
            #.. u1: VTD, u2: desired_speed, u3: guid_eta
            self.var1   =   1.0 * 0.5
            self.var2   =   self.var1
            self.var3   =   self.var1
            self.lamb1  =   1.0 *1.0
            self.lamb2  =   self.lamb1
            self.lamb3  =   self.lamb1
            self.u1_init    =   1.
            self.u2_init    =   1.
            self.u3_init    =   1.
            
        
        pass
    
    pass

#.. GPR Parameter
class GPR_Parameter():
        #.. initialize an instance of the class
    def __init__(self, dt_GPR, ne_GPR) -> None:
        self.set_values(dt_GPR, ne_GPR)
        pass

    #.. >>>>>  set_values  <<<<<
    def set_values(self, dt_GPR, ne_GPR):
        self.dt_GPR     =   dt_GPR
        self.dt_GPR_opt =   dt_GPR * 15

        #.. gaussian process regression parameter
        self.ne_GPR     =   ne_GPR    # forecasting number (ne = 2000, te = 2[sec])

        self.H_GPR      =   np.array([1.0, 0.0]).reshape(1,2)
        self.R_GPR_x    =   pow(0.001, 2)
        self.R_GPR_y    =   pow(0.001, 2)
        self.R_GPR_z    =   pow(0.001, 2)
        
        self.hyp_l_GPR  =   1 * np.ones(3)
        self.hyp_q_GPR  =   1 * np.ones(3)
        self.hyp_n_GPR  =   1000

        hyp_l           =   self.hyp_l_GPR[0]

        self.F_x_GPR    =   np.array([[0.0, 1.0], [-pow(hyp_l,2), -2*hyp_l]])
        self.A_x_GPR    =   np.array([[1.0, self.dt_GPR], [-pow(hyp_l,2)*self.dt_GPR, 1-2*hyp_l*self.dt_GPR]])
        self.Q_x_GPR    =   self.hyp_q_GPR[0] * np.array([[1/3*pow(self.dt_GPR,3), 0.5*pow(self.dt_GPR,2)-2*hyp_l/3*pow(self.dt_GPR,3)],
                                                      [0.5*pow(self.dt_GPR,2)-2*hyp_l/3*pow(self.dt_GPR,3),
                                                       self.dt_GPR-2*hyp_l*pow(self.dt_GPR,2)+4/3*pow(hyp_l,2)*pow(self.dt_GPR,3)]])
        self.m_x_GPR    = np.zeros([2, 1]).reshape(2, 1)
        self.P_x_GPR    = np.zeros([2, 2])

        self.F_y_GPR    = self.F_x_GPR[:]
        self.A_y_GPR    = self.A_x_GPR[:]
        self.Q_y_GPR    = self.Q_x_GPR[:]
        self.m_y_GPR    = self.m_x_GPR[:]
        self.P_y_GPR    = self.P_x_GPR[:]

        self.F_z_GPR    = self.F_x_GPR[:]
        self.A_z_GPR    = self.A_x_GPR[:]
        self.Q_z_GPR    = self.Q_x_GPR[:]
        self.m_z_GPR    = self.m_x_GPR[:]
        self.P_z_GPR    = self.P_x_GPR[:]

        self.count_GPR  = 1

        # Training data save
        self.training_data_x = []
        self.training_data_y = []
        self.training_data_z = []

        # Save data for plotting forecasting results
        self.te_array     =  np.zeros(self.ne_GPR)
        self.me_x_array   =  np.zeros(self.ne_GPR)
        self.var_x_array  =  np.zeros(self.ne_GPR)
        self.me_y_array   =  np.zeros(self.ne_GPR)
        self.var_y_array  =  np.zeros(self.ne_GPR)
        self.me_z_array   =  np.zeros(self.ne_GPR)
        self.var_z_array  =  np.zeros(self.ne_GPR)
        pass
pass 

#.. GnC_Parameter
class GnC_Parameter():
    #.. initialize an instance of the class
    def __init__(self, Guid_type=1) -> None:
        self.set_values(Guid_type)
        pass
    
    #.. >>>>>  set_values  <<<<<
    def set_values(self, Guid_type):
        #.. PF guidance
        self.dt_GCU     =   0.004
        self.Guid_type  =   Guid_type       # | 0: Ctrl-based | 1: GL-based | 2: Direct | 3: GL-based-MPPI | 4: Ctrl-based MPPI | 9: test
        self.desired_speed      =   3.
        self.virtual_target_distance     =   4.5
        self.distance_change_WP    =   self.virtual_target_distance
        self.dist_change_first_WP = 0.3
        
        #.. param. of Guid_tpye = 0
        self.Kp_vel     =   1.
        self.Kd_vel     =   0.0 * self.Kp_vel
        
        #.. param. of Guid_tpye = 1
        self.Kp_speed   =   1.
        self.Kd_speed   =   self.Kp_speed * 0.
        self.guid_eta   =   3.
        
        #.. NDO parameter    
        self.gain_NDO   =   1.0 * np.array([1.0,1.0,1.0])

        #.. attitude control parameter
        self.tau_phi    =   0.3
        self.tau_the    =   self.tau_phi
        self.tau_psi    =   self.tau_phi * 2.
        self.del_psi_cmd_limit = 15. * m.pi/180.
        
        # self.del_psi_cmd_limit = 0. * m.pi/180.
        # self.del_psi_cmd_limit = 10. * m.pi/180.
        # self.del_psi_cmd_limit = 20. * m.pi/180.
        # self.del_psi_cmd_limit = 30. * m.pi/180.
        
        self.tau_Wb     =   0.05 # in [https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf]

        self.tau_p      =   0.1
        self.tau_q      =   0.1
        self.tau_r      =   0.2

        self.alpha_p    =   0.1  
        self.alpha_q    =   0.1   
        self.alpha_r    =   0.1  
        pass
    
    pass


#.. Physical_Parameter
class Physical_Parameter():
    # [1] https://github.com/PX4/PX4-SITL_gazebo-classic/blob/main/models/iris/iris.sdf.jinja
    # [2] https://github.com/PX4/PX4-SITL_gazebo-classic/blob/main/models/typhoon_h480/typhoon_h480.sdf.jinja
    # [3] https://github.com/mvernacc/gazebo_motor_model_docs/blob/master/notes.pdf
    # [4] https://discuss.px4.io/t/gazebo-sdf-file-motor-parameters-definitions/34747
    # [5] https://github.com/PX4/PX4-SITL_gazebo-classic/issues/110
    # [6] https://discuss.px4.io/t/trouble-checking-windspeeds-set-during-gazebo-simulation/25717/3
    # [7] https://github.com/PX4/PX4-SITL_gazebo-classic/blob/27298574ce33a79ba6cfc31ed4604974605e7257/src/gazebo_motor_model.cpp#L195
    # [8] https://px4.github.io/Firmware-Doxygen/de/d41/pwm__params__aux_8c.html#ac9bea3554f1e915406d9c5ba2330973b
    
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.set_values()
        pass
    
    #.. >>>>>  set_values  <<<<<
    def set_values(self):
        # PX4 specs (iris) in [1]
        timeConstantUp          =   0.0125
        timeConstantDown        =   0.025
        maxRotVelocity          =   1100
        maxRotVelocity_limitied =   956.5   # max 2200 but actual 2000 in [8]
        motorConstant           =   5.84e-06
        momentConstant          =   0.06
        rotorDragCoefficient    =   0.000175
        rollingMomentCoefficient    =   1e-06
        
        d                   =   0.4
        g                   =   9.81
        
        #.. physical model: iris in [1]
        self.Ixx            =   0.029125     
        self.Iyy            =   0.029125
        self.Izz            =   0.055225
        self.inertia        =   np.diag( [ self.Ixx,   self.Iyy,   self.Izz ] )  
        self.mass           =   1.535
        self.tau_throttle   =   0.5 * (timeConstantUp + timeConstantDown)
        # self.Lx_M           =   (d / m.sqrt(2))
        # self.Ly_M           =   (d / m.sqrt(2))
        self.Lx_M           =   0.13
        self.Ly_M           =   0.21
        self.Lz_M           =   0.023

        # hover throtle level
        self.n_Motor        =   4
        self.T_hover        =   self.mass * g
        # self.throttle_hover =   0.73       # experiment in PX4_sitl
        # self.T_max          =   self.T_hover / self.throttle_hover**2
        # self.T_max_M        =   self.T_max / self.n_Motor
        # self.motorConstant  =   motorConstant
        # self.momentConstant =   momentConstant
        # self.maxRotVel      =   m.sqrt(self.T_max_M / self.motorConstant)
        
        self.motorConstant  =   motorConstant
        self.momentConstant =   momentConstant
        self.maxRotVel      =   maxRotVelocity
        self.T_max_M        =   self.motorConstant * self.maxRotVel**2
        self.T_max          =   self.T_max_M * self.n_Motor
        self.throttle_hover =   m.sqrt(self.T_hover / self.T_max)
        self.rotor_turning_direction = np.array([-1., 1., -1., 1]) # CW(1) and CCW(-1) w.r.t. rotor axis(-z_B)
        
        # motor coefficient
        self.Kq_Motor       =   self.motorConstant/self.T_max_M
        self.Kt_Motor       =   self.Kq_Motor / self.momentConstant
        self.rotorDragCoefficient   =   rotorDragCoefficient
        self.rollingMomentCoefficient   =   rollingMomentCoefficient
        
        # simple - psuedo rotor drag coeff.        
        simple_motor_rot_vel_sum  =   4 * self.maxRotVel * self.throttle_hover
        self.psuedo_rotor_drag_coeff = simple_motor_rot_vel_sum * self.rotorDragCoefficient
        
        # collocation matrix
        self.eta_Fb, self.eta_Mb = self.eta_Xshape_quadrotors(self.T_max_M, self.Lx_M, self.Ly_M, self.Kq_Motor, self.Kt_Motor)
        self.Mat_CA = self.mat_CA_quadrotors(self.eta_Fb, self.eta_Mb)
    
        # rotor drag coefficient is: = D / (? * ? * R² * (? * R)²) in [5]
        
        # aerodynamic
        # self.CdA            =   0.107 * 4 * 10.       # 0.107
        self.CdA            =   0.
        self.ClA            =   0.          # small enough to ignore, compared to the Cd
        self.delXcp         =   0.
        pass
    
    #..calculate a matrix(eta) of rotors for quatdrotors
    # https://www.cantorsparadise.com/how-control-allocation-for-multirotor-systems-works-f87aff1794a2
    def eta_Xshape_quadrotors(self, max_thrust_per_rotor, Lx_M, Ly_M, Kq_Motor, Kt_Motor):
        # Throttle to Force and Moment Matrix For Quadrotor
        eta_Fb 	        =  - max_thrust_per_rotor * np.ones(4)
        eta_Mb        	=   np.zeros( (3, 4) )
        
        # Rolling & Ritching Effects, X-shape rotors
        eta_Mb[:,0]     =   np.cross( [  Lx_M,     Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[0]  ] )
        eta_Mb[:,1]   	=   np.cross( [  Lx_M,    -Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[1]  ] )
        eta_Mb[:,2]   	=   np.cross( [ -Lx_M,    -Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[2]  ] )
        eta_Mb[:,3]    	=   np.cross( [ -Lx_M,     Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[3]  ] )
        
        # Yawing Effects, ?? ? 3?? ?? ?? 0? ??? ?? ?? ???.
        eta_Mb[2,0]  	=   -(Kq_Motor / Kt_Motor) * eta_Fb[0]
        eta_Mb[2,1]    	=    (Kq_Motor / Kt_Motor) * eta_Fb[1]
        eta_Mb[2,2]  	=   -(Kq_Motor / Kt_Motor) * eta_Fb[2]
        eta_Mb[2,3]  	=    (Kq_Motor / Kt_Motor) * eta_Fb[3]
        
        return eta_Fb, eta_Mb

    #..calculate command allocation matrix of rotors for quatdrotors
    def mat_CA_quadrotors(self, eta_Fb, eta_Mb):
        eta_Mat         =   np.vstack( [eta_Fb, eta_Mb] )
        W_Mat           =   np.diag( 1.0 * np.ones(4) )
        inv_W_Mat       =   np.linalg.inv( W_Mat )
        Mat_CA          =	np.matmul( inv_W_Mat, np.linalg.pinv( np.matmul(eta_Mat, inv_W_Mat) ) )
        return Mat_CA
    
    pass
