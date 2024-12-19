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
import os
import time

# private libs.
from flight_functions.utility_funcs import azim_elev_from_vec3, DCM_from_euler_angle


#.. guidance modules
def guidance_modules(QR_Guid_type, QR_WP_idx_passed, QR_WP_idx_heading, WP_WPs_shape0,VT_Ri, QR_Ri, QR_Vi, QR_Ai, QR_virtual_target_distance,
                     QR_desired_speed, QR_Kp_vel, QR_Kd_vel, QR_Kp_speed, QR_Kd_speed, QR_guid_eta, MPPI_ctrl_input, QR_stop_flag):
    
    # starting phase
    if QR_WP_idx_passed < 1:
        QR_Guid_type     = 0
        QR_desired_speed = 3.0
        pass

    # terminal phase
    if (QR_WP_idx_heading == (WP_WPs_shape0 - 1)):
        QR_Guid_type = 0
        pass
    
    # stop phase
    if (QR_stop_flag == 1):
        QR_Guid_type = 0
        QR_desired_speed = 0.0

    # guidance command
    Aqi_cmd     =   np.zeros(3)
    if QR_Guid_type == 4:
        QR_desired_speed = MPPI_ctrl_input[1]
        QR_Kp_vel        = MPPI_ctrl_input[2]

    elif QR_Guid_type == 0:
        #.. guidance - position & velocity control
        # position control
        err_Ri               = VT_Ri - QR_Ri
        Kp_pos               = QR_desired_speed/max(np.linalg.norm(err_Ri),QR_desired_speed)     # (terminal WP, tgo < 1) --> decreasing speed
        derr_Ri              = 0. - QR_Vi
        Vqi_cmd              = Kp_pos * err_Ri
        dVqi_cmd             = Kp_pos * derr_Ri
        # velocity control
        err_Vi               = Vqi_cmd - QR_Vi
        derr_Vi              = dVqi_cmd - QR_Ai
        Aqi_cmd              = QR_Kp_vel * err_Vi + QR_Kd_vel * derr_Vi

    elif QR_Guid_type == 1:
        # calc. variables
        QR_mag_Vi            = np.linalg.norm(QR_Vi)
        FPA_azim, FPA_elev   = azim_elev_from_vec3(QR_Vi)
        QR_cI_W              = DCM_from_euler_angle(np.array([0., FPA_elev, FPA_azim]))
        #.. guidance - GL - parameters by MPPI
        Aqw_cmd              = np.zeros(3)

        # err_Ri
        err_Ri               = VT_Ri - QR_Ri
        w_des_V              = min(max(np.linalg.norm(err_Ri) / QR_virtual_target_distance, 0.5), 1.0)

        # a_x command
        err_mag_V            = w_des_V * QR_desired_speed - QR_mag_Vi

        # err_mag_V            = QR_desired_speed - QR_mag_Vi
        dQR_mag_Vi           = np.dot(QR_Vi, QR_Ai) / max(QR_mag_Vi, 0.1)
        derr_mag_V           = 0. - dQR_mag_Vi
        Aqw_cmd[0]           = QR_Kp_speed * err_mag_V + QR_Kd_speed * derr_mag_V

        # pursuit guidance law
        Rqti                 = VT_Ri - QR_Ri
        Rqtw                 = np.matmul(QR_cI_W,Rqti)
        err_azim, err_elev   = azim_elev_from_vec3(Rqtw)
        Aqw_cmd[1]           = QR_guid_eta * QR_mag_Vi * m.sin(err_azim)       # GL ?? ??, a_cmd = eta*V*sin(lambda) **-??240827-**
        Aqw_cmd[2]           = -QR_guid_eta * QR_mag_Vi * m.sin(err_elev)      # GL ?? ??, a_cmd = eta*V*sin(lambda) **-??240827-**
        
        # command coordinate change
        cW_I                 = np.transpose(QR_cI_W)
        Aqi_cmd              = np.matmul(cW_I, Aqw_cmd)
    
    elif QR_Guid_type == 3: 
        
        #.. guidance - GL - parameters by MPPI
        QR_guid_eta          = MPPI_ctrl_input[2]
        
        # calc. variables
        QR_mag_Vi            = np.linalg.norm(QR_Vi)
        FPA_azim, FPA_elev   = azim_elev_from_vec3(QR_Vi)
        QR_cI_W              = DCM_from_euler_angle(np.array([0., FPA_elev, FPA_azim]))
        
        #.. a_x command
        Aqw_cmd              = np.zeros(3)
        Aqw_cmd[0]           = MPPI_ctrl_input[1]    # guid_type 3? a_cmd_x ???? MPPI ????? ?? ????? ?? **-??240827-**
        
        # pursuit guidance law
        Rqti                 = VT_Ri - QR_Ri
        Rqtw                 = np.matmul(QR_cI_W,Rqti)
        err_azim, err_elev   = azim_elev_from_vec3(Rqtw)
        Aqw_cmd[1]           = QR_guid_eta * QR_mag_Vi * m.sin(err_azim)       # GL ?? ??, a_cmd = eta*V*sin(lambda) **-??240827-**
        Aqw_cmd[2]           = -QR_guid_eta * QR_mag_Vi * m.sin(err_elev)      # GL ?? ??, a_cmd = eta*V*sin(lambda) **-??240827-**
        # command coordinate change
        cW_I                 = np.transpose(QR_cI_W)
        Aqi_cmd              = np.matmul(cW_I, Aqw_cmd)

    elif QR_Guid_type == 2:
        Aqi_cmd              = MPPI_ctrl_input
    return Aqi_cmd, np.linalg.norm(QR_Vi)


#.. convert_Ai_cmd_to_thrust_and_att_ang_cmd
def convert_Ai_cmd_to_thrust_and_att_ang_cmd(cI_B, Ai_cmd, mass, T_max, WP_WPs, WP_idx_heading, VT_Ri, Ri, att_ang, del_psi_cmd_limit, Vi, flightlogFile):    

    # thrust cmd
    mag_Ai_cmd  = np.linalg.norm(Ai_cmd)
    Ab_cmd      = np.matmul(cI_B, Ai_cmd)
    T_cmd       = min(mag_Ai_cmd*mass, T_max)
    norm_T_cmd  = T_cmd / T_max

    if WP_idx_heading == 0:
        p1p2 = WP_WPs[WP_idx_heading + 1] - WP_WPs[WP_idx_heading]
        p1p2 = np.array([p1p2[0], p1p2[1], 0.])
    else:
        p1p2 = WP_WPs[WP_idx_heading] - WP_WPs[WP_idx_heading - 1]
        p1p2 = np.array([p1p2[0], p1p2[1], 0.])
    
    p1r1 = Ri - WP_WPs[WP_idx_heading]
    p1p2_norm = np.linalg.norm(p1p2)
    numerator = np.linalg.norm(np.cross(p1r1, p1p2))
    path_dist = numerator / p1p2_norm

    WP_heading = WP_WPs[WP_idx_heading]
    Vb      = np.matmul(cI_B, Vi)
    
    p        =   VT_Ri - Ri
    p_norm = np.linalg.norm(np.array([p[0], p[1], 0.]))

    psi_VT, _  =   azim_elev_from_vec3(p) 
    Rc = np.array([0., 0.])
    turn_radius = 0.
    if path_dist < 1 or abs(psi_VT - att_ang[2]) > np.deg2rad(60):
        psi = psi_VT
        is_guide = 0
    else:
        turn_radius = Vb[0]**2 / np.abs(Ai_cmd[1])
        turn_radius = max(turn_radius, 2*p_norm)
        d = 0.5*np.sqrt(((VT_Ri[0] - Ri[0])**2 + (VT_Ri[1] - Ri[1])**2))
        alpha = np.arccos(np.clip(d / turn_radius, -1, 1))
        if np.cross(p1p2, p1r1)[2] > 0:
            turn_direction = 1
        else:
            turn_direction = -1
        beta = psi_VT + turn_direction * alpha
        Rc = np.array([Ri[0] + turn_radius*np.cos(beta), Ri[1] + turn_radius*np.sin(beta), Ri[2]])
        p_c = Rc - Ri
        tangential = turn_direction*np.cross(p_c, np.array([0., 0., 1.]))
        psi_c, _ = azim_elev_from_vec3(tangential)

        del_psi     =   psi_c - att_ang[2]
   
        if abs(del_psi) > 1.0*m.pi:
            if psi_VT > att_ang[2]:
                psi_VT = psi_VT - 2.*m.pi
            else:
                psi_VT = psi_VT + 2.*m.pi
        del_psi     =   max(min(psi_VT - att_ang[2], del_psi_cmd_limit), -del_psi_cmd_limit)
        psi     =   att_ang[2] + del_psi
        
        is_guide = 1


        flightlog = "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n" % (
            time.time(),
            Ri[0],
            Ri[1],
            WP_WPs[WP_idx_heading-1][0],  
            WP_WPs[WP_idx_heading-1][1],
            WP_WPs[WP_idx_heading][0],  
            WP_WPs[WP_idx_heading][1],
            VT_Ri[0],
            VT_Ri[1],
            Vb[0],
            Ab_cmd[2],
            Rc[0],
            Rc[1],
            turn_radius,
            path_dist,
            is_guide,
            psi,
            VT_Ri[2],
            Ri[2]
        )
        flightlogFile.write(flightlog)

    euler_psi   =   np.array([0., 0., psi])
    mat_psi     =   DCM_from_euler_angle(euler_psi)
    Apsi_cmd    =   np.matmul(mat_psi , Ai_cmd)
    phi         =   m.asin(Apsi_cmd[1]/mag_Ai_cmd)
    phi         =   0.0
    sintheta    =   min(max(-Apsi_cmd[0]/m.cos(phi)/mag_Ai_cmd, -1.), 1.)
    theta       =   m.asin(sintheta)
    
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