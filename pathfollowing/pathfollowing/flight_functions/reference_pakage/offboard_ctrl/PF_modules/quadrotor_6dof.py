############################################################
#
#   - Name : model_quadrotor_3dof.py
#
#                   -   Created by E. T. Jeong, 2024.09.13
#
############################################################

#.. Library
# pulbic libs.
import math as m
import numpy as np

# private libs.

# A Q3rotor 6dof model integrated in a class which cosnsists of several modules 
class Quadrotor_6DOF():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        #.. Parameters for quadrotor
        # vehicle physical properties
        self.Ixx                =   0.042        # 0.011
        self.Iyy                =   0.042        # 0.015
        self.Izz                =   0.076        # 0.021
        self.inertia            =   np.diag( [ self.Ixx,   self.Iyy,   self.Izz ] )  
        self.mass               =   2.265 
        self.tau_throttle       =   0.0937 / 2              # Time Constant for Main Throttle Dynamics      [sec]
        self.d                  =   0.4          #0.2675  
        self.Lx_M               =   (self.d / m.sqrt(2))
        self.Ly_M               =   (self.d / m.sqrt(2))
        self.Kq_Motor           =   0.00000064          # 0.000000003216
        self.Kt_Motor           =   0.0000225           # 0.0000001871 

        # hover throtle level
        self.throttle_hover     =   0.7
        # self.throttle_hover     =   0.3
        
        # aerodynamic
        self.CdA    =   0.107       # 0.107
        self.ClA    =   0.          # small enough to ignore, compared to the Cd
        
        #.. # rotor throttle to force and moment matrix
        g0                      =   9.81
        num_rotor               =   4
        self.max_thrust_per_rotor   =   g0*self.mass/(self.throttle_hover*num_rotor)    #   maximum throttle per a rotor
        self.eta_Fb, self.eta_Mb    =   eta_Xshape_quadrotors(self.max_thrust_per_rotor, self.Lx_M, self.Ly_M, self.Kq_Motor, self.Kt_Motor)
        self.Mat_CA =   mat_CA_quadrotors(self.eta_Fb, self.eta_Mb)
        
        #.. variable
        self.Ri     =   np.array([0., 0., -10.])
        self.Vi     =   np.array([0., 0., 0.])
        self.Vi_prev    =   np.array([0., 0., 0.])
        self.Vb     =   np.array([0., 0., 0.])
        self.Ai     =   np.array([0., 0., 0.])
        self.thr_unitvec    =   np.array([0., 0., 0.])
        self.att_ang    =   np.array([0., 0., 0.])
        self.Wb     =   np.array([0., 0., 0.])
        self.throttle   =   np.zeros(4)
        self.cI_W   =   np.zeros((3,3))
        self.cI_B   =   np.zeros((3,3))
        
        self.WP_idx_heading     =   1
        self.WP_idx_passed      =   0
        self.p_closest_on_path  =   np.array([0., 0., 0.])
        self.cost_total     =   0.
        
        self.mag_Aqi_thru   =   9.81
        self.out_NDO    =   np.zeros(3)
        self.z_NDO      =   np.zeros(3)
        self.Ai_est_dstb    =   np.zeros(3)
        
        self.temp_Ai_Aero_delayed   =   np.zeros(3)
        
        self.int_err_Wb =   np.zeros(3)
        
        #.. parameter        
        # guidance
        self.dt_GCU     =   0.004
        # self.Guid_type  =   0       # | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        # self.Guid_type  =   1       # | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        self.Guid_type  =   3       # | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        self.flag_guid_trans    =   1
        self.desired_speed      =   3.0
        self.look_ahead_distance    =   self.desired_speed * 1.5
        self.distance_change_WP     =   self.look_ahead_distance
        self.Kp_vel     =   1.0     # 1.0
        self.Kd_vel     =   0. * self.Kp_vel
        self.Kp_speed   =   1.0
        self.Kd_speed   =   self.Kp_speed * 0.
        self.guid_eta   =   2. 
        
        # self.gain_NDO   =   0.8 * np.array([1.0,1.0,1.0])
        self.gain_NDO   =   1.0 * np.array([1.0,1.0,1.0])
        # self.gain_NDO   =   1.5 * np.array([1.0,1.0,1.0])
        
        self.look_ahead_distance_psi_cmd =   self.look_ahead_distance * 2.

        #.. gaussian process regression parameter
        self.ne_GPR     =   500    # forecasting number (ne = 2000, te = 2[sec])

        self.H_GPR      =   np.array([1.0, 0.0]).reshape(1,2)
        self.R_GPR_x    =   pow(0.001, 2)
        self.R_GPR_y    =   pow(0.01,  2)
        self.R_GPR_z    =   pow(0.1,   2)
        
        self.hyp_l_GPR  =   1 * np.ones(3)
        self.hyp_q_GPR  =   1 * np.ones(3)
        self.hyp_n_GPR  =   1000

        hyp_l           =   self.hyp_l_GPR[0]

        self.F_x_GPR    =   np.array([[0.0, 1.0], [-pow(hyp_l,2), -2*hyp_l]])
        self.A_x_GPR    =   np.array([[1.0, self.dt_GCU], [-pow(hyp_l,2)*self.dt_GCU, 1-2*hyp_l*self.dt_GCU]])
        self.Q_x_GPR    =   self.hyp_q_GPR[0] * np.array([[1/3*pow(self.dt_GCU,3), 0.5*pow(self.dt_GCU,2)-2*hyp_l/3*pow(self.dt_GCU,3)],
                                                      [0.5*pow(self.dt_GCU,2)-2*hyp_l/3*pow(self.dt_GCU,3),
                                                       self.dt_GCU-2*hyp_l*pow(self.dt_GCU,2)+4/3*pow(hyp_l,2)*pow(self.dt_GCU,3)]])
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
        

        #.. attitude control parameter
        # self.tau_phi    =   0.4
        # self.tau_phi    =   0.5
        self.tau_phi    =   0.6
        self.tau_the    =   self.tau_phi
        self.tau_psi    =   self.tau_phi * 2.
        
        pass
    pass


#..calculate a matrix(eta) of rotors for quatdrotors
# https://www.cantorsparadise.com/how-control-allocation-for-multirotor-systems-works-f87aff1794a2
def eta_Xshape_quadrotors(max_thrust_per_rotor, Lx_M, Ly_M, Kq_Motor, Kt_Motor):
    # Throttle to Force and Moment Matrix For Quadrotor
    eta_Fb 	        =  - max_thrust_per_rotor * np.ones(4)
    eta_Mb        	=   np.zeros( (3, 4) )
    
    # Rolling & Ritching Effects, X-shape rotors
    eta_Mb[:,0]     =   np.cross( [  Lx_M,     Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[0]  ] )
    eta_Mb[:,1]   	=   np.cross( [  Lx_M,    -Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[1]  ] )
    eta_Mb[:,2]   	=   np.cross( [ -Lx_M,    -Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[2]  ] )
    eta_Mb[:,3]    	=   np.cross( [ -Lx_M,     Ly_M,   0 ], [ 0.0,   0.0,   eta_Fb[3]  ] )
    
    # Yawing Effects, 계산 시 3번째 행이 모두 0이 되므로 그냥 덮어 씌운다.
    eta_Mb[2,0]  	=   -(Kq_Motor / Kt_Motor) * eta_Fb[0]
    eta_Mb[2,1]    	=    (Kq_Motor / Kt_Motor) * eta_Fb[1]
    eta_Mb[2,2]  	=   -(Kq_Motor / Kt_Motor) * eta_Fb[2]
    eta_Mb[2,3]  	=    (Kq_Motor / Kt_Motor) * eta_Fb[3]
    
    return eta_Fb, eta_Mb

#..calculate command allocation matrix of rotors for quatdrotors
def mat_CA_quadrotors(eta_Fb, eta_Mb):
    eta_Mat         =   np.vstack( [eta_Fb, eta_Mb] )
    W_Mat           =   np.diag( 1.0 * np.ones(4) )
    inv_W_Mat       =   np.linalg.inv( W_Mat )
    Mat_CA          =	np.matmul( inv_W_Mat, np.linalg.pinv( np.matmul(eta_Mat, inv_W_Mat) ) )
    return Mat_CA