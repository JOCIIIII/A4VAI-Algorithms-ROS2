############################################################
#
#   - Name : main_MPPI.py
#
#                   -   Created by E. T. Jeong, 2023.12.12
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m
from numpy.linalg import norm

# private libs.
from .utility_functions import azim_elev_from_vec3, DCM_from_euler_angle

#.. kinematics
def kinematics(VT_Ri, Q6_Ri, Q6_Vi):
    Rqti        =   VT_Ri - Q6_Ri
    mag_Rqti    =   norm(Rqti)
    mag_Vqi     =   norm(Q6_Vi)
    LOS_azim, LOS_elev    =   azim_elev_from_vec3(Rqti)
    tgo         =   mag_Rqti / max(mag_Vqi, 0.001)
    FPA_azim, FPA_elev   =   azim_elev_from_vec3(Q6_Vi)
    cI_W     =   DCM_from_euler_angle(np.array([0., FPA_elev, FPA_azim]))
    return mag_Vqi, LOS_azim, LOS_elev, tgo, FPA_azim, FPA_elev, cI_W


#.. distance from quadrotor to ref. path  
def distance_from_Q6_to_path(WP_WPs, Q6_WP_idx_heading, Q6_Ri, Q6_p_closest_on_path, Q6_WP_idx_passed):
    dist_to_path        =   999999.
    for i_WP in range(Q6_WP_idx_heading,0,-1):
        Rw1w2       =   WP_WPs[i_WP] - WP_WPs[i_WP-1]
        mag_Rw1w2   =   norm(Rw1w2)
        Rw1q        =   Q6_Ri - WP_WPs[i_WP-1]
        mag_w1p     =   min(max(np.dot(Rw1w2, Rw1q)/max(mag_Rw1w2,0.001), 0.), mag_Rw1w2)
        p_closest_on_path   =   WP_WPs[i_WP-1] + mag_w1p * Rw1w2/max(mag_Rw1w2,0.001)
        mag_Rqp     =   norm(p_closest_on_path - Q6_Ri)
        if dist_to_path < mag_Rqp:
            break
        else:
            unit_Rw1w2      =   Rw1w2/max(mag_Rw1w2,0.001)
            dist_to_path    =   mag_Rqp
            Q6_p_closest_on_path    =   p_closest_on_path
            Q6_WP_idx_passed        =   max(i_WP-1, 0)
            pass
        pass
    return dist_to_path, Q6_p_closest_on_path, Q6_WP_idx_passed, unit_Rw1w2
    
    
#.. check waypoint - quadrotor
def check_waypoint(WP_WPs, Q6_WP_idx_heading, Q6_Ri, Q6_distance_change_WP):
    Rqw2i       =   WP_WPs[Q6_WP_idx_heading] - Q6_Ri
    mag_Rqw2i   =   norm(Rqw2i)
    if mag_Rqw2i < Q6_distance_change_WP:
        Q6_WP_idx_heading = min(Q6_WP_idx_heading + 1, WP_WPs.shape[0] - 1)
    return Q6_WP_idx_heading


#.. virtual target position      
def virtual_target_position(dist_to_path, Q6_look_ahead_distance, Q6_p_closest_on_path, Q6_WP_idx_passed, WP_WPs):
    if dist_to_path >= Q6_look_ahead_distance:
        VT_Ri   =   Q6_p_closest_on_path
    else:
        total_len   = dist_to_path
        p1  =   Q6_p_closest_on_path
        for i_WP in range(Q6_WP_idx_passed+1, WP_WPs.shape[0]):
            # check segment whether Rti exist
            p2          =   WP_WPs[i_WP]
            Rp1p2       =   p2 - p1
            mag_Rp1p2   =   norm(Rp1p2)
            if total_len + mag_Rp1p2 > Q6_look_ahead_distance:
                mag_Rp1t    =   Q6_look_ahead_distance - total_len
                VT_Ri       =   p1 + mag_Rp1t * Rp1p2/max(mag_Rp1p2,0.001)
                break
            else:
                p1  =   p2
                total_len   =   total_len + mag_Rp1p2
                if i_WP == WP_WPs.shape[0] - 1:
                    VT_Ri   =   p2
                pass
            pass
        pass
    return VT_Ri

#.. guidance modules
def guidance_modules(Q6_Guid_type, mag_Vqi, Q6_WP_idx_passed, Q6_flag_guid_trans, Q6_WP_idx_heading, WP_WPs_shape0,
                     VT_Ri, Q6_Ri, Q6_Vi, Q6_Ai, Q6_desired_speed, Q6_Kp_vel, Q6_Kd_vel, Q6_Kp_speed, Q6_Kd_speed, Q6_guid_eta,
                     Q6_cI_W, tgo, MPPI_ctrl_input):
    # guidance transition
    ratio_guid_type     =   np.zeros(4)
    ratio_guid_type[Q6_Guid_type] = 1.
    if Q6_Guid_type != 0:
        if (mag_Vqi < 0.2 or Q6_WP_idx_passed < 1):
            # low speed --> generate FPA firstly for efficiently using guidance law
            Q6_flag_guid_trans = 1
            # if SP.t > 1.:
            #     print("t = " + str(round(SP.t,5)) + ", low speed --> guid. trans.")
        elif(mag_Vqi > 0.5):
            # high speed --> enough speed to avoid flag chattering
            Q6_flag_guid_trans = 0
        else:
            # gap between mag_Vqi (0.2, 0.5)
            pass
        if (Q6_WP_idx_heading == (WP_WPs_shape0 - 1)):
            # low speed approach to terminal WP
            Q6_flag_guid_trans = 1
            ratio_guid_type[0] =   1.
            ratio_guid_type[Q6_Guid_type] =   1. - ratio_guid_type[0]
            
        if (Q6_flag_guid_trans == 1):
            ratio_guid_type[0] =   1.
            ratio_guid_type[Q6_Guid_type] =   1. - ratio_guid_type[0]
            # ratio_guid_type[0] =   0.5
            # ratio_guid_type[Q6_Guid_type] =   1. - ratio_guid_type[0]
            
    # guidance command
    Aqi_cmd     =   np.zeros(3)
    if Q6_Guid_type == 0 or Q6_flag_guid_trans == 1:
        #.. guidance - position & velocity control
        # position control
        err_Ri      =   VT_Ri - Q6_Ri
        Kp_pos      =   Q6_desired_speed/max(norm(err_Ri),Q6_desired_speed)     # (terminal WP, tgo < 1) --> decreasing speed
        derr_Ri     =   0. - Q6_Vi
        Vqi_cmd     =   Kp_pos * err_Ri
        dVqi_cmd    =   Kp_pos * derr_Ri
        # velocity control
        err_Vi      =   Vqi_cmd - Q6_Vi
        derr_Vi     =   dVqi_cmd - Q6_Ai
        Aqi_cmd     =   Aqi_cmd + ratio_guid_type[0] * (Q6_Kp_vel * err_Vi + Q6_Kd_vel * derr_Vi)
    if (Q6_Guid_type == 1 or (Q6_Guid_type == 1 and Q6_flag_guid_trans == 1)):
        #.. guidance - GL -based
        Aqw_cmd     =   np.zeros(3)
        # a_x command
        err_mag_V   =   Q6_desired_speed - mag_Vqi
        dmag_Vqi    =   np.dot(Q6_Vi, Q6_Ai) / max(mag_Vqi, 0.1)
        derr_mag_V  =   0. - dmag_Vqi
        Aqw_cmd[0]  =   Q6_Kp_speed * err_mag_V + Q6_Kd_speed * derr_mag_V
        # print("err_mag_V = " + str(err_mag_V))
        # optimal pursuit guidance law
        Rqti        =   VT_Ri - Q6_Ri
        Rqtw        =   np.matmul(Q6_cI_W,Rqti)
        err_azim, err_elev    =   azim_elev_from_vec3(Rqtw)
        Aqw_cmd[1]  =   Q6_guid_eta*mag_Vqi / max(tgo, 0.001) * err_azim
        Aqw_cmd[2]  =   -Q6_guid_eta*mag_Vqi / max(tgo, 0.001) * err_elev
        # command coordinate change
        cW_I        =   np.transpose(Q6_cI_W)
        Aqi_cmd     =   Aqi_cmd + ratio_guid_type[1] * np.matmul(cW_I, Aqw_cmd)
    if Q6_Guid_type == 2 or (Q6_Guid_type == 2 and Q6_flag_guid_trans == 1):
        Aqi_cmd     =   Aqi_cmd + ratio_guid_type[2] * MPPI_ctrl_input
    if Q6_Guid_type == 3 or (Q6_Guid_type == 3 and Q6_flag_guid_trans == 1):
        #.. guidance - GL -based
        Aqw_cmd     =   np.zeros(3)
        # a_x command
        err_mag_V   =   Q6_desired_speed - mag_Vqi
        dmag_Vqi    =   np.dot(Q6_Vi, Q6_Ai) / max(mag_Vqi, 0.1)
        derr_mag_V  =   0. - dmag_Vqi
        Aqw_cmd[0]  =   Q6_Kp_speed * err_mag_V + Q6_Kd_speed * derr_mag_V
        # pursuit guidance law
        Rqti        =   VT_Ri - Q6_Ri
        Rqtw        =   np.matmul(Q6_cI_W,Rqti)
        err_azim, err_elev    =   azim_elev_from_vec3(Rqtw)
        Aqw_cmd[1]  =   Q6_guid_eta* 3. / 1.5 * err_azim
        Aqw_cmd[2]  =   -Q6_guid_eta* 3. / 1.5 * err_elev
        # command coordinate change
        cW_I        =   np.transpose(Q6_cI_W)
        Aqi_cmd     =   Aqi_cmd + ratio_guid_type[Q6_Guid_type] * np.matmul(cW_I, Aqw_cmd)
    return Aqi_cmd, Q6_flag_guid_trans

#.. compensate_Aqi_cmd
def compensate_Aqi_cmd(Aqi_cmd, Q6_Ai_est_dstb, Aqi_grav):
    # compensate disturbance
    Aqi_cmd     =   Aqi_cmd - Q6_Ai_est_dstb
    # compensate gravity
    Aqi_cmd     =   Aqi_cmd - Aqi_grav 
    return Aqi_cmd
    
#.. thrust_cmd
def thrust_cmd(Aqi_cmd, Q6_throttle_hover, MP_a_lim, Q6_mass):
    max_mag_Aqi_thru    =   9.81 / Q6_throttle_hover * MP_a_lim
    Q6_mag_Aqi_thru     =   min(norm(Aqi_cmd), max_mag_Aqi_thru)
    #.. convert_Ai_cmd_to_mag_thrust_cmd
    mag_thrust_cmd  =   Q6_mag_Aqi_thru * Q6_mass
    norm_thrust_cmd = Q6_mag_Aqi_thru / max_mag_Aqi_thru
    return Q6_mag_Aqi_thru, mag_thrust_cmd, norm_thrust_cmd

#.. att_cmd
def att_cmd(Aqi_cmd, Q6_att_ang_2, LOS_azim):
    # Psi nonzero
    euler_psi   =   np.array([0., 0., Q6_att_ang_2])
    mat_psi     =   DCM_from_euler_angle(euler_psi)
    Apsi_cmd    =   np.matmul(mat_psi , Aqi_cmd)
    Apsix_cmd   =   Apsi_cmd[0]
    Apsiy_cmd   =   Apsi_cmd[1]
    phi         =   m.asin(Apsiy_cmd/norm(Aqi_cmd))
    sintheta    =   min(max(-Apsix_cmd/m.cos(phi)/norm(Aqi_cmd), -1.), 1.)
    theta       =   m.asin(sintheta)
    # psi         =   0.
    # psi         =   FPA_azim
    psi         =   LOS_azim
    
    att_ang_cmd =   np.array([phi, theta, psi])
    return att_ang_cmd

#.. NDO_Aqi
def NDO_Aqi(Aqi_grav, Q6_mag_Aqi_thru, Q6_cI_B, Q6_thr_unitvec, Q6_gain_NDO, Q6_z_NDO, Q6_Vi, dt_NDO):
    # Aqi_thru_wo_grav for NDO
    Ab_thrust   =   np.array([0., 0., -Q6_mag_Aqi_thru])            # 현재 thrust 크기
    Ai_thrust   =   np.matmul(np.transpose(Q6_cI_B), Ab_thrust)     # 자세 반영
    Aqi_thru_wo_grav    =   Q6_mag_Aqi_thru * Q6_thr_unitvec + Aqi_grav
    Q6_thr_unitvec      =   Ai_thrust / max(norm(Ai_thrust), 0.1)
        
    # nonlinear disturbance observer
    dz_NDO      =   np.zeros(3)
    dz_NDO[0]   =   -Q6_gain_NDO[0]*Q6_z_NDO[0] - Q6_gain_NDO[0] * (Q6_gain_NDO[0]*Q6_Vi[0] + Aqi_thru_wo_grav[0])
    dz_NDO[1]   =   -Q6_gain_NDO[1]*Q6_z_NDO[1] - Q6_gain_NDO[1] * (Q6_gain_NDO[1]*Q6_Vi[1] + Aqi_thru_wo_grav[1])
    dz_NDO[2]   =   -Q6_gain_NDO[2]*Q6_z_NDO[2] - Q6_gain_NDO[2] * (Q6_gain_NDO[2]*Q6_Vi[2] + Aqi_thru_wo_grav[2])
    
    Q6_out_NDO  =   np.zeros(3)
    Q6_out_NDO[0]  =   Q6_z_NDO[0] + Q6_gain_NDO[0]*Q6_Vi[0]
    Q6_out_NDO[1]  =   Q6_z_NDO[1] + Q6_gain_NDO[1]*Q6_Vi[1]
    Q6_out_NDO[2]  =   Q6_z_NDO[2] + Q6_gain_NDO[2]*Q6_Vi[2]

    Q6_z_NDO  =   Q6_z_NDO + dz_NDO*dt_NDO
    
    return Q6_thr_unitvec, Q6_out_NDO, Q6_z_NDO
    
    
    
#.. NDO_Aqi
def NDO_Aqi_v2(Aqi_cmd, Aqi_grav, Q6_mag_Aqi_thru, Q6_cI_B, Q6_thr_unitvec, Q6_gain_NDO, Q6_z_NDO, Q6_Vi, dt_NDO):
    
    #.. 이 부분은 MPPI로 빼는게 맞는 것 같음
    # Aqi_thru_wo_grav for NDO
    Ab_thrust   =   np.array([0., 0., -Q6_mag_Aqi_thru])            # 현재 thrust 크기
    Ai_thrust   =   np.matmul(np.transpose(Q6_cI_B), Ab_thrust)     # 자세 반영
    Q6_thr_unitvec      =   Ai_thrust / max(norm(Ai_thrust), 0.1)
    
    # Aqi_thru_wo_grav    =   Q6_mag_Aqi_thru * Q6_thr_unitvec + Aqi_grav
    
    Aqi_thru_wo_grav = Aqi_cmd
        
    # nonlinear disturbance observer
    dz_NDO      =   np.zeros(3)
    dz_NDO[0]   =   -Q6_gain_NDO[0]*Q6_z_NDO[0] - Q6_gain_NDO[0] * (Q6_gain_NDO[0]*Q6_Vi[0] + Aqi_thru_wo_grav[0])
    dz_NDO[1]   =   -Q6_gain_NDO[1]*Q6_z_NDO[1] - Q6_gain_NDO[1] * (Q6_gain_NDO[1]*Q6_Vi[1] + Aqi_thru_wo_grav[1])
    dz_NDO[2]   =   -Q6_gain_NDO[2]*Q6_z_NDO[2] - Q6_gain_NDO[2] * (Q6_gain_NDO[2]*Q6_Vi[2] + Aqi_thru_wo_grav[2])
    
    Q6_out_NDO  =   np.zeros(3)
    Q6_out_NDO[0]  =   Q6_z_NDO[0] + Q6_gain_NDO[0]*Q6_Vi[0]
    Q6_out_NDO[1]  =   Q6_z_NDO[1] + Q6_gain_NDO[1]*Q6_Vi[1]
    Q6_out_NDO[2]  =   Q6_z_NDO[2] + Q6_gain_NDO[2]*Q6_Vi[2]

    Q6_z_NDO  =   Q6_z_NDO + dz_NDO*dt_NDO
    
    return Q6_thr_unitvec, Q6_out_NDO, Q6_z_NDO

#.. NDO_Aqi
def NDO_Aqi_v3(Aqi_cmd, Aqi_grav, Q6_mag_Aqi_thru, Q6_cI_B, Q6_thr_unitvec, Q6_gain_NDO, Q6_z_NDO, Q6_Vi, dt_NDO):
    
    #.. 이 부분은 MPPI로 빼는게 맞는 것 같음
    # Aqi_thru_wo_grav for NDO
    Ab_thrust   =   np.array([0., 0., -Q6_mag_Aqi_thru])            # 현재 thrust 크기
    Ai_thrust   =   np.matmul(np.transpose(Q6_cI_B), Ab_thrust)     # 자세 반영
    Q6_thr_unitvec      =   Ai_thrust / max(norm(Ai_thrust), 0.1)
    
    # Aqi_thru_wo_grav    =   Q6_mag_Aqi_thru * Q6_thr_unitvec + Aqi_grav
    
    Aqi_thru_wo_grav = Aqi_cmd + Aqi_grav
        
    # nonlinear disturbance observer
    dz_NDO      =   np.zeros(3)
    dz_NDO[0]   =   -Q6_gain_NDO[0]*Q6_z_NDO[0] - Q6_gain_NDO[0] * (Q6_gain_NDO[0]*Q6_Vi[0] + Aqi_thru_wo_grav[0])
    dz_NDO[1]   =   -Q6_gain_NDO[1]*Q6_z_NDO[1] - Q6_gain_NDO[1] * (Q6_gain_NDO[1]*Q6_Vi[1] + Aqi_thru_wo_grav[1])
    dz_NDO[2]   =   -Q6_gain_NDO[2]*Q6_z_NDO[2] - Q6_gain_NDO[2] * (Q6_gain_NDO[2]*Q6_Vi[2] + Aqi_thru_wo_grav[2])
    
    Q6_out_NDO  =   np.zeros(3)
    Q6_out_NDO[0]  =   Q6_z_NDO[0] + Q6_gain_NDO[0]*Q6_Vi[0]
    Q6_out_NDO[1]  =   Q6_z_NDO[1] + Q6_gain_NDO[1]*Q6_Vi[1]
    Q6_out_NDO[2]  =   Q6_z_NDO[2] + Q6_gain_NDO[2]*Q6_Vi[2]

    Q6_z_NDO  =   Q6_z_NDO + dz_NDO*dt_NDO
    
    return Q6_thr_unitvec, Q6_out_NDO, Q6_z_NDO
