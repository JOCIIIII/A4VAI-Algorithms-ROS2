############################################################
#
#   - Name : path_following_required_info.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m

# private libs.


#.. distance_to_path
def distance_to_path(WP_WPs, QR_WP_idx_heading, QR_Ri, QR_point_closest_on_path_i, QR_WP_idx_passed):
    dist_to_path            =   999999.
    for i_WP in range(QR_WP_idx_heading,0,-1):
        Rw1w2               =   WP_WPs[i_WP] - WP_WPs[i_WP-1]
        mag_Rw1w2           =   np.linalg.norm(Rw1w2)
        Rw1q                =   QR_Ri - WP_WPs[i_WP-1]
        mag_w1p             =   min(max(np.dot(Rw1w2, Rw1q)/max(mag_Rw1w2,0.001), 0.), mag_Rw1w2)
        p_closest_on_path   =   WP_WPs[i_WP-1] + mag_w1p * Rw1w2/max(mag_Rw1w2,0.001)
        mag_Rqp             =   np.linalg.norm(p_closest_on_path - QR_Ri)
        if dist_to_path < mag_Rqp:
            break
        else:
            dist_to_path                  =   mag_Rqp
            QR_point_closest_on_path_i    =   p_closest_on_path
            QR_WP_idx_passed              =   max(i_WP-1, 0)
            pass
        pass
    return dist_to_path, QR_point_closest_on_path_i, QR_WP_idx_passed
    
    
#.. check waypoint
def check_waypoint(WP_WPs, QR_WP_idx_heading, QR_Ri, QR_distance_change_WP):
    Rqw2i       =   WP_WPs[QR_WP_idx_heading] - QR_Ri
    mag_Rqw2i   =   np.linalg.norm(Rqw2i)
    if mag_Rqw2i < QR_distance_change_WP:
        QR_WP_idx_heading = min(QR_WP_idx_heading + 1, WP_WPs.shape[0] - 1)
    return QR_WP_idx_heading

#.. VTP_decision
def VTP_decision(dist_to_path, QR_virtual_target_distance, QR_point_closest_on_path_i, QR_WP_idx_passed, WP_WPs):
    if dist_to_path >= QR_virtual_target_distance:
        VT_Ri   =   QR_point_closest_on_path_i
    else:
        total_len   = dist_to_path
        p1  =   QR_point_closest_on_path_i
        for i_WP in range(QR_WP_idx_passed+1, WP_WPs.shape[0]):
            # check segment whether Rti exist
            p2          =   WP_WPs[i_WP]
            Rp1p2       =   p2 - p1
            mag_Rp1p2   =   np.linalg.norm(Rp1p2)
            if total_len + mag_Rp1p2 > QR_virtual_target_distance:
                mag_Rp1t    =   QR_virtual_target_distance - total_len
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

#..takeoff_to_first_WP
def takeoff_to_first_WP(WP_WPs, QR_Ri, QR_WP_idx_passed, dist_change_first_WP, VT_Ri):
    if (QR_WP_idx_passed < 1):
        dist_to_WP = np.linalg.norm(WP_WPs[1] - QR_Ri)
        if (dist_to_WP > dist_change_first_WP):
            VT_Ri = np.copy(WP_WPs[1])
        pass    
    return VT_Ri


#.. cost_function_1
def cost_function_1(Guid_type, R, u, Q0, dist_to_path, Q1, att_ang, dt):
    # uRu of LQR cost, set low value of norm(R)

    if Guid_type == 3:
        uRu = (u[1]*u[1]*R[1] + u[2]*u[2]*R[2])
    else:
        uRu = (u[0]*u[0]*R[0] + u[1]*u[1]*R[1] + u[2]*u[2]*R[2])
        pass
    
    # path following performance
    x0 = dist_to_path
    x0Q0x0 = x0 * Q0 * x0

    # attitude control stability
    x1 = np.sqrt(att_ang[0]*att_ang[0]+att_ang[1]*att_ang[1])
    x1Q1x1 = x1 * Q1 * x1
    
    # total cost
    cost_arr = np.array([uRu, x0Q0x0, x1Q1x1]) * dt
    
    return cost_arr

#.. terminal_cost_1
def terminal_cost_1(P1, WP_WPs, PF_var_init_WP_idx_passed, PF_var_final_WP_idx_passed,
                    PF_var_init_point_closest_on_path, PF_var_final_point_closest_on_path,
                    min_move_range, total_time):
    
    # calc. init remained range
    init_remianed_range = 0.
    for i_WP in range(PF_var_init_WP_idx_passed, WP_WPs.shape[0]-1):
        # check segment whether Rti exist
        rel_pos = WP_WPs[i_WP+1] - WP_WPs[i_WP]
        init_remianed_range = init_remianed_range + np.linalg.norm(rel_pos)
        pass
    rel_pos = PF_var_init_point_closest_on_path - WP_WPs[PF_var_init_WP_idx_passed]
    init_remianed_range = init_remianed_range - np.linalg.norm(rel_pos)
    
    # calc. final remained range
    final_remianed_range = 0.
    for i_WP in range(PF_var_final_WP_idx_passed, WP_WPs.shape[0]-1):
        # check segment whether Rti exist
        rel_pos = WP_WPs[i_WP+1] - WP_WPs[i_WP]
        final_remianed_range = final_remianed_range + np.linalg.norm(rel_pos)
        pass
    rel_pos = PF_var_final_point_closest_on_path - WP_WPs[PF_var_final_WP_idx_passed]
    final_remianed_range = final_remianed_range - np.linalg.norm(rel_pos)
    
    # calc. move range
    move_range = init_remianed_range - final_remianed_range
    terminal_cost = P1 * total_time / max(move_range, min_move_range)
    return terminal_cost
