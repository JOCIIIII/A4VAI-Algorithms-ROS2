############################################################
#
#   - Name : data_structure.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np


# private libs.


#.. Save_Data
class Save_Data():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.t                          =   []

        self.Rqi                        =   []
        self.Vqi                        =   []
        self.att_ang                    =   []
        self.throttle                   =   []
        self.Fi_thrust                  =   []
        self.Fi_aero                    =   []
        self.Fi_rej                     =   []
        self.Rti                        =   []
        
        self.virtual_target_distance    =   []
        self.desired_speed              =   []
        self.guid_eta                   =   []
        
        self.MPPI_u1                    =   []
        self.MPPI_u2                    =   []
        self.MPPI_u3                    =   []
        self.MPPI_calc_time             =   []
        
        self.att_ang_cmd                =   []
        self.throttle_cmd               =   []
        
        self.cost_arr                   =   []
        self.total_cost                 =   []
        self.terminal_cost              =   0.
        self.dist_to_path               =   []
        
        self.AeroForce_i                =   []

        self.te_array                   =   []
        self.me_x_array                 =   []
        self.NDO_x                      =   []
        pass
    pass

#.. State_Var
class State_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Ri         =   np.array([0., 0., 0.])
        self.Vi         =   np.array([0., 0., 0.])
        self.att_ang    =   np.array([0., 0., 0.])
        self.Ai         =   np.array([0., 0., 0.])
        self.cI_B       =   np.identity(3)
        pass
    pass

#.. Path_Following_Var
class Path_Following_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.VT_Ri                       =   np.array([0., 0., 0.])
        self.WP_idx_passed               =   0
        self.WP_idx_heading              =   1
        self.PF_goal                     =   False
        self.PF_done                     =   False
        self.WP_manual                   =   0
        self.reWP_flag                   =   0 # 20240914 diy
        self.reset_flag2mppi             =   0 # 241223 diy
        self.stop_flag                   =   0
        self.intr_flag                   =   0
        self.intr_prev                   =   0
        self.coli_flag                   =   0
        self.tran_flag                   =   0
        self.point_closest_on_path_i     =   np.array([0., 0., 0.])
        self.dist_to_path                =   9999.
        self.unit_Rw1w2                  =   np.array([1., 0., 0.])
        self.cost_arr                    =   np.array([0., 0., 0.])
        self.total_cost                  =   0.
        self.terminal_cost               =   0.
        self.init_WP_idx_passed          =   0
        self.final_WP_idx_passed         =   0
        self.init_point_closest_on_path  =   np.array([1., 0., 0.])
        self.final_point_closest_on_path =   np.array([1., 0., 0.])
        self.init_time                   =   0.
        self.final_time                  =   0.
        pass
    pass

#.. Env_Var
class Env_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.temperature =   0.
        self.pressure    =   0.
        self.rho         =   0.
        self.grav        =   0.
        self.Wind_Vi     =   np.array([0., 0., 0.])
        pass
    pass

#.. Aerodyn_Var
class Aerodyn_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Fb          =   np.array([0., 0., 0.])
        self.Mb          =   np.array([0., 0., 0.])
        pass
    pass

#.. Guid_Var
class Guid_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Ai_cmd             =   np.array([0., 0., 0.])
        self.Ai_cmd_compensated =   np.array([0., 0., 0.])
        self.Ai_disturbance     =   np.array([0., 0., 0.])
        self.T_cmd              =   9.81
        self.norm_T_cmd         =   0
        self.att_ang_cmd        =   np.array([0., 0., 0.])
        self.out_NDO            =   np.array([0., 0., 0.])
        self.z_NDO              =   np.array([0., 0., 0.])
        self.Ai_rotor_drag      =   np.array([0., 0., 0.])
        self.MPPI_ctrl_input    =   np.array([0.5, 2.])
        self.MPPI_calc_time     =   0.
        self.qd_cmd             =   np.array([0., 0., 0., 0.])
        self.guid_type_used     =   0
        pass
    pass

#.. Ctrl_Var
class Ctrl_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Wb_cmd        =   np.array([0., 0., 0.])
        self.Mb_cmd        =   np.array([0., 0., 0.])
        self.throttle_cmd  =   np.array([0., 0., 0., 0.])
        self.int_err_Wb    =   np.array([0., 0., 0.])
        pass
    pass

#.. Act_Var
class Act_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.throttle      =   np.array([0., 0., 0., 0.])
        self.dot_throttle  =   np.array([0., 0., 0., 0.])
        pass
    pass

#.. Dyn_Var
class Dyn_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Fb_thrust     =   np.array([0., 0., 0.])
        self.Mb_thrust     =   np.array([0., 0., 0.])
        self.dot_Vb        =   np.array([0., 0., 0.])
        self.dot_Wb        =   np.array([0., 0., 0.])
        self.dot_Ri        =   np.array([0., 0., 0.])
        self.dot_att_ang   =   np.array([0., 0., 0.])
        pass
    pass


