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
    flag_update             =   False
    QR_WP_idx_passed_prev   =   QR_WP_idx_passed 
    for i_WP in range(QR_WP_idx_heading,max(QR_WP_idx_passed_prev-1,0),-1):
        Rw1w2               =   WP_WPs[i_WP] - WP_WPs[i_WP-1]
        mag_Rw1w2           =   np.linalg.norm(Rw1w2)
        Rw1q                =   QR_Ri - WP_WPs[i_WP-1]
        mag_w1p             =   min(max(np.dot(Rw1w2, Rw1q)/max(mag_Rw1w2,0.001), 0.), mag_Rw1w2)
        p_closest_on_path   =   WP_WPs[i_WP-1] + mag_w1p * Rw1w2/max(mag_Rw1w2,0.001)
        mag_Rqp             =   np.linalg.norm(p_closest_on_path - QR_Ri)
        if dist_to_path > mag_Rqp:
            dist_to_path                  =   mag_Rqp
            QR_point_closest_on_path_i    =   p_closest_on_path
            QR_WP_idx_passed              =   max(i_WP-1, QR_WP_idx_passed_prev)
            flag_update                   =   True 
        elif i_WP == ( QR_WP_idx_passed_prev + 1 ) and flag_update == False :
            dist_to_path                  =   mag_Rqp
            QR_point_closest_on_path_i    =   p_closest_on_path
            QR_WP_idx_passed              =   QR_WP_idx_passed_prev 
        else:
            pass
        pass
    return dist_to_path, QR_point_closest_on_path_i, QR_WP_idx_passed
    
    
#.. check waypoint
def check_waypoint(WP_WPs, QR_WP_idx_heading, QR_Ri, QR_distance_change_WP):

    Rqw2i       =   WP_WPs[QR_WP_idx_heading] - QR_Ri
    mag_Rqw2i   =   np.linalg.norm(Rqw2i)

    PF_done     =   False
    if mag_Rqw2i < QR_distance_change_WP:
        QR_WP_idx_heading = min(QR_WP_idx_heading + 1, WP_WPs.shape[0] - 1)

        # Landing
        if QR_WP_idx_heading == WP_WPs.shape[0]-1:
            if( mag_Rqw2i < 0.5 ):
                PF_done = True
            QR_WP_idx_heading = min(QR_WP_idx_heading, WP_WPs.shape[0] - 1)
            
    return QR_WP_idx_heading, PF_done

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

# #.. cost_function_1
# def cost_function(u, dist_to_path, vel_err, dt):
#     # uRu of LQR cost, set low value of norm(R)

#     uu = (u[0]*u[0] + u[1]*u[1])
    
#     # path following performance
#     x0 = dist_to_path
#     x0Q0x0 = x0 * Q0 * x0

#     # total cost
#     cost_arr = np.array([uu, x0Q0x0, x1Q1x1]) * dt
    
#     return cost_arr
