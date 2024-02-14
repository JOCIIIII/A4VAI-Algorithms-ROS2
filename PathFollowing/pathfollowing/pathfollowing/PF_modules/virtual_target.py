############################################################
#
#   - Name : model_virtual_target.py
#
#                   -   Created by E. T. Jeong, 2024.09.13
#
############################################################

#.. Library
# pulbic libs.
import numpy as np

# private libs.

# A virtual target model
class Virtual_Target():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Ri     =   np.array([0., 0., 0.])
        # self.Vi     =   np.array([0., 0., 0.])
        # self.distance_change_WP =   0.1
        # self.WP_idx_heading     =   1
        pass
    
    #.. initialize VT Position
    def init_VT_Ri(self, WP_WPs, Q6_Ri, Q6_look_ahead_distance):
        Rqwi        =   WP_WPs[0] - Q6_Ri
        mag_Rqwi    =   np.linalg.norm(Rqwi)
        if mag_Rqwi < Q6_look_ahead_distance:
            Rqwi        =   WP_WPs[1] - Q6_Ri
            mag_Rqwi    =   np.linalg.norm(Rqwi)
            self.Ri       =   Q6_Ri + Q6_look_ahead_distance*Rqwi/max(mag_Rqwi,1)
        else:
            self.Ri       =   WP_WPs[0]
        pass
    
    pass