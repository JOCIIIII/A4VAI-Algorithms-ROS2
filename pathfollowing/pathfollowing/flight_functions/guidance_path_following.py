############################################################
#
#   - Name : guidance_path_following.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m

# private libs.
from flight_functions.utility_funcs import azim_elev_from_vec3, DCM_from_euler_angle


#.. guidance modules
def guidance_modules(QR_Guid_type, QR_WP_idx_passed, QR_WP_idx_heading, WP_WPs_shape0, VT_Ri, QR_Ri, QR_Vi, QR_Ai,
                     QR_desired_speed, QR_Kp_vel, QR_Kd_vel, QR_guid_eta, MPPI_ctrl_input, QR_intr_flag, QR_intr_prev):
    
    QR_reset_mppi = 0
    QR_tran_flag  = 0

    #-------------------------------------------------------------
    # starting phase
    if QR_WP_idx_passed < 1:
        QR_tran_flag  = 1
        pass

    #-------------------------------------------------------------    
    # interrupt phase
    if (QR_intr_flag == 1):
        QR_Guid_type = 0
        QR_reset_mppi = 1

    if (QR_intr_flag == 0 and QR_intr_prev == 1):
        QR_tran_flag = 1

    QR_intr_prev = QR_intr_flag

    #-------------------------------------------------------------
    # transition phase (stop -> pursuit guidance)
    if (QR_tran_flag == 1):
        QR_Guid_type = 0
        QR_reset_mppi = 1
        if (np.linalg.norm(QR_Vi) > 1.0):
            QR_tran_flag = 0
   
    #-------------------------------------------------------------
    # terminal phase
    if (QR_WP_idx_heading >= (WP_WPs_shape0 - 1)):
        QR_Guid_type = 0

    #-------------------------------------------------------------
    # guidance command
    Aqi_cmd     =   np.zeros(3)

    if QR_Guid_type == 0:
        #.. guidance - position & velocity control
        # position control
        err_Ri               = VT_Ri - QR_Ri
        Kp_pos               = 0.5 #QR_desired_speed/max(np.linalg.norm(err_Ri),QR_desired_speed)     # (terminal WP, tgo < 1) --> decreasing speed
        derr_Ri              = 0. - QR_Vi
        Vqi_cmd              = Kp_pos * err_Ri
        dVqi_cmd             = Kp_pos * derr_Ri *0.0
        # velocity control
        err_Vi               = Vqi_cmd - QR_Vi
        derr_Vi              = dVqi_cmd - QR_Ai
        QR_Kp_vel            = Kp_pos * 3.0
        Aqi_cmd              = QR_Kp_vel * err_Vi + QR_Kd_vel * derr_Vi * 0.0
    
    elif (QR_Guid_type == 3) or (QR_Guid_type == 4): 
        
        #.. guidance - GL - parameters by MPPI
        QR_guid_eta          = np.maximum(MPPI_ctrl_input[1],0.5)
        
        # calc. variables
        QR_mag_Vi            = np.linalg.norm(QR_Vi)
        FPA_azim, FPA_elev   = azim_elev_from_vec3(QR_Vi)
        QR_cI_W              = DCM_from_euler_angle(np.array([0., FPA_elev, FPA_azim]))
        
        #.. a_x command
        Aqw_cmd              = np.zeros(3)
        Aqw_cmd[0]           = MPPI_ctrl_input[0]    # guid_type 3? a_cmd_x ???? MPPI ????? ?? ????? ?? **-??240827-**
        
        # pursuit guidance law
        Rqti                 = VT_Ri - QR_Ri
        Rqtw                 = np.matmul(QR_cI_W,Rqti)
        err_azim, err_elev   = azim_elev_from_vec3(Rqtw)
        Aqw_cmd[1]           = QR_guid_eta * QR_mag_Vi * m.sin(err_azim)       # GL ?? ??, a_cmd = eta*V*sin(lambda) **-??240827-**
        Aqw_cmd[2]           = -2.0 * QR_mag_Vi * m.sin(err_elev)      # GL ?? ??, a_cmd = eta*V*sin(lambda) **-??240827-**
        # command coordinate change
        cW_I                 = np.transpose(QR_cI_W)
        Aqi_cmd              = np.matmul(cW_I, Aqw_cmd)

    return Aqi_cmd, QR_Guid_type, QR_intr_prev, QR_reset_mppi

#.. prev_convert_Ai_cmd_to_thrust_and_att_ang_cmd
def convert_Ai_cmd_to_thrust_and_att_ang_cmd(cI_B, Ai_cmd, mass, T_max, WP_WPs, WP_idx_heading, Ri, att_ang, del_psi_cmd_limit):    
    # thrust cmd
    mag_Ai_cmd  = np.linalg.norm(Ai_cmd)
    Ab_cmd      = np.matmul(cI_B, Ai_cmd)
    T_cmd       = min(abs(Ab_cmd[2]) * mass, T_max)
    norm_T_cmd  = T_cmd / T_max

    # attitude angle cmd
    WP_heading  =   WP_WPs[WP_idx_heading]
    Rqwi        =   WP_heading - Ri
    if WP_idx_heading < WP_WPs.shape[0]-1:
        psi_des, _  =   azim_elev_from_vec3(Rqwi)       # toward to the heading waypoint
    else:
        WP_idx_passed = max(WP_idx_heading - 1, 0)
        WP_passed   =   WP_WPs[WP_idx_passed]
        WP12        =   WP_heading - WP_passed
        psi_des, _  =   azim_elev_from_vec3(WP12)
        
    # att_ang_cmd -  del_psi_cmd limitation
    del_psi     =   psi_des - att_ang[2]
    if abs(del_psi) > 1.0*m.pi:
        if psi_des > att_ang[2]:
            psi_des = psi_des - 2.*m.pi
        else:
            psi_des = psi_des + 2.*m.pi
    del_psi     =   max(min(psi_des - att_ang[2], del_psi_cmd_limit), -del_psi_cmd_limit)
    psi_des     =   att_ang[2] + del_psi
    
    euler_psi   =   np.array([0., 0., psi_des])
    mat_psi     =   DCM_from_euler_angle(euler_psi)
    Apsi_cmd    =   np.matmul(mat_psi , Ai_cmd)
    phi         =   min(max(m.asin(Apsi_cmd[1]/mag_Ai_cmd), -0.5236), 0.5236)
    sintheta    =   min(max(-Apsi_cmd[0]/m.cos(phi)/mag_Ai_cmd, -0.5236), 0.5236)
    theta       =   m.asin(sintheta)
    psi         =   psi_des
            
    att_ang_cmd = np.array([phi, theta, psi])
    return T_cmd, norm_T_cmd, att_ang_cmd

#.. NDO_for_Ai_cmd
def NDO_for_Ai_cmd(T_cmd, mass, grav, QR_cI_B, QR_gain_NDO, QR_z_NDO, QR_Vi, QR_dt_GCU, Ai_rotor_drag):
    # Aqi_thru_wo_grav for NDO
    Ab_thrust   =   np.array([0., 0., -T_cmd/mass])            # ?? thrust ??
    Ai_thrust   =   np.matmul(np.transpose(QR_cI_B), Ab_thrust)     # ?? ??
    
    # nonlinear disturbance observer
    dz_NDO      =   np.zeros(3)
    dz_NDO[0]   =   -QR_gain_NDO[0]*QR_z_NDO[0] - QR_gain_NDO[0] * (QR_gain_NDO[0]*QR_Vi[0] + Ai_thrust[0] + Ai_rotor_drag[0])
    dz_NDO[1]   =   -QR_gain_NDO[1]*QR_z_NDO[1] - QR_gain_NDO[1] * (QR_gain_NDO[1]*QR_Vi[1] + Ai_thrust[1] + Ai_rotor_drag[1])
    dz_NDO[2]   =   -QR_gain_NDO[2]*QR_z_NDO[2] - QR_gain_NDO[2] * (QR_gain_NDO[2]*QR_Vi[2] + Ai_thrust[2] + Ai_rotor_drag[2] + grav)
    
    QR_out_NDO     =   np.zeros(3)
    QR_out_NDO[0]  =   QR_z_NDO[0] + QR_gain_NDO[0]*QR_Vi[0]
    QR_out_NDO[1]  =   QR_z_NDO[1] + QR_gain_NDO[1]*QR_Vi[1]
    QR_out_NDO[2]  =   QR_z_NDO[2] + QR_gain_NDO[2]*QR_Vi[2]

    QR_z_NDO  =   QR_z_NDO + dz_NDO*QR_dt_GCU
    
    return QR_out_NDO, QR_z_NDO

def simple_rotor_drag_model(QR_Vi, psuedo_rotor_drag_coeff, cI_B):
    #.. drag model (ref: The True Role of Accelerometer Feedback in Quadrotor Control)
    #.. assumption: motor_rot_vel = hovering_motor_rot_vel --> set psuedo_rotor_drag_coeff
    cB_I                                 = np.transpose(cI_B)
    joint_axis_b                         = np.array([0., 0., -1.])
    joint_axis_i                         = np.matmul(cB_I, joint_axis_b)
    velocity_parallel_to_rotor_axis      = np.dot(QR_Vi, joint_axis_i) * joint_axis_i
    velocity_perpendicular_to_rotor_axis = QR_Vi - velocity_parallel_to_rotor_axis 
    Fi_drag                              = -psuedo_rotor_drag_coeff * velocity_perpendicular_to_rotor_axis
    
    return Fi_drag