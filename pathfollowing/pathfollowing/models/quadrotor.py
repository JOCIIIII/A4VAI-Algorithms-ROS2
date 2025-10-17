############################################################
#
#   - Name : quadrotor_model.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m
import copy
import sys, os

# private libs.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from necessary_settings.quadrotor_iris_parameters import Physical_Parameter, GnC_Parameter, MPPI_Parameter, GPR_Parameter
from etc_class.data_structure import Path_Following_Var, Guid_Var, State_Var
from flight_functions import path_following_required_info, guidance_path_following, utility_funcs
    
#.. Quadrotor_6DOF
class Quadrotor_6DOF():
    #.. initialize an instance of the class
    def __init__(self,
                 init_physical_param:Physical_Parameter,
                 init_GnC_param:GnC_Parameter,
                 init_MPPI_Param:MPPI_Parameter,
                 init_GPR_Param:GPR_Parameter
                 ) -> None:
        self.initialize_variables(init_physical_param, init_GnC_param, init_MPPI_Param, init_GPR_Param)
        pass
    
    #.. initialize_variables
    def initialize_variables(self,
                 init_physical_param:Physical_Parameter,
                 init_GnC_param:GnC_Parameter,
                 init_MPPI_Param:MPPI_Parameter,
                 init_GPR_Param:GPR_Parameter
                 ):
        self.physical_param = copy.deepcopy(init_physical_param)
        self.GnC_param = copy.deepcopy(init_GnC_param)
        self.MPPI_param = copy.deepcopy(init_MPPI_Param)
        self.GPR_param = copy.deepcopy(init_GPR_Param)
        self.PF_var = Path_Following_Var()
        self.guid_var = Guid_Var()
        self.state_var = State_Var()

        # env vars.
        self.grav = 9.81

        pass

    #.. WP_manual
    def WP_manual_set(self, WP_WPs):
        
        WP_WPs[self.PF_var.WP_idx_heading]   =  WP_WPs[self.PF_var.WP_idx_heading + 1]
        WP_WPs[self.PF_var.WP_idx_passed+1]  =  self.PF_var.VT_Ri
        self.PF_var.WP_manual = 0
        
        return WP_WPs

    #.. PF_required_info 
    def PF_required_info(self, WP_WPs, dt):

        # 20240914 diy 
        # stop position of virtual target
        tmp_VT_Ri = self.PF_var.VT_Ri

        self.PF_var.dist_to_path, self.PF_var.point_closest_on_path_i, self.PF_var.WP_idx_passed = path_following_required_info.distance_to_path(
            WP_WPs, self.PF_var.WP_idx_heading, self.state_var.Ri, self.PF_var.point_closest_on_path_i, self.PF_var.WP_idx_passed)
        self.PF_var.WP_idx_heading, self.PF_var.PF_done= path_following_required_info.check_waypoint(WP_WPs, self.PF_var.WP_idx_heading, self.state_var.Ri, self.GnC_param.distance_change_WP)
        self.PF_var.VT_Ri = path_following_required_info.VTP_decision(self.PF_var.dist_to_path, self.GnC_param.virtual_target_distance, self.PF_var.point_closest_on_path_i, self.PF_var.WP_idx_passed, WP_WPs)
        
        # 20240914 diy 
        if (self.PF_var.stop_flag == 1):
            self.PF_var.VT_Ri = tmp_VT_Ri
        
        # calc. cost
        # u = self.guid_var.MPPI_ctrl_input #np.reshape(self.guid_var.MPPI_ctrl_input,(-1))        
        # self.PF_var.total_cost = self.PF_var.total_cost + np.sum(self.PF_var.cost_arr)
        pass
        
    #.. guid_Ai_cmd
    def guid_Ai_cmd(self, WP_WPs_shape0, MPPI_ctrl_input):
        self.guid_var.Ai_cmd, self.guid_var.guid_type_used, self.PF_var.intr_prev, self.PF_var.reset_flag2mppi = guidance_path_following.guidance_modules(
            self.GnC_param.Guid_type, self.PF_var.WP_idx_passed, self.PF_var.WP_idx_heading, WP_WPs_shape0,
            self.PF_var.VT_Ri, self.state_var.Ri, self.state_var.Vi, self.state_var.Ai, self.GnC_param.desired_speed, self.GnC_param.Kp_vel, 
            self.GnC_param.Kd_vel, self.GnC_param.guid_eta, MPPI_ctrl_input, self.PF_var.intr_flag, self.PF_var.intr_prev)
        pass
    
    #.. guid_compensate_Ai_cmd
    def guid_compensate_Ai_cmd(self):
        # calc. simple rotor drag model
        Fi_drag = guidance_path_following.simple_rotor_drag_model(
            self.state_var.Vi, self.physical_param.psuedo_rotor_drag_coeff, self.state_var.cI_B)
        self.guid_var.Ai_rotor_drag = Fi_drag / self.physical_param.mass
        # compensate disturbance
        self.guid_var.Ai_disturbance = self.guid_var.out_NDO + self.guid_var.Ai_rotor_drag
        self.guid_var.Ai_cmd_compensated = self.guid_var.Ai_cmd - self.guid_var.Ai_disturbance
        # compensate gravity
        self.guid_var.Ai_cmd_compensated[2] = self.guid_var.Ai_cmd_compensated[2] - self.grav
        pass
        
    #.. guid_NDO_for_Ai_cmd
    def guid_NDO_for_Ai_cmd(self):
        self.guid_var.out_NDO, self.guid_var.z_NDO = guidance_path_following.NDO_for_Ai_cmd(
            self.guid_var.T_cmd, self.physical_param.mass, self.grav, self.state_var.cI_B,
            self.GnC_param.gain_NDO, self.guid_var.z_NDO, self.state_var.Vi, self.GnC_param.dt_GCU,
            self.guid_var.Ai_rotor_drag)
        pass

    #.. prev_guid_convert_Ai_cmd_to_thrust_and_att_ang_cmd
    def guid_convert_Ai_cmd_to_thrust_and_att_ang_cmd(self, WP_WPs):
        self.guid_var.T_cmd, self.guid_var.norm_T_cmd, self.guid_var.att_ang_cmd = guidance_path_following.convert_Ai_cmd_to_thrust_and_att_ang_cmd(
            self.state_var.cI_B, self.guid_var.Ai_cmd_compensated, self.physical_param.mass, self.physical_param.T_max, 
            WP_WPs, self.PF_var.WP_idx_heading, self.state_var.Ri, self.state_var.att_ang, self.GnC_param.del_psi_cmd_limit)
        pass

    #.. guid_convert_att_ang_cmd_to_qd_cmd
    def guid_convert_att_ang_cmd_to_qd_cmd(self):
        w, x, y, z = utility_funcs.Euler2Quaternion(self.guid_var.att_ang_cmd)
        self.guid_var.qd_cmd = [w, x, y, z]    
        pass

    #.. update_states
    def update_states(self, Ri, Vi, att_ang, accel_xyz):
        self.state_var.Ri       = Ri
        self.state_var.Vi       = Vi
        self.state_var.att_ang  = att_ang
        self.state_var.cI_B     = utility_funcs.DCM_from_euler_angle(att_ang)
        self.state_var.Ai       = np.matmul(np.transpose(self.state_var.cI_B), accel_xyz)
    
    pass