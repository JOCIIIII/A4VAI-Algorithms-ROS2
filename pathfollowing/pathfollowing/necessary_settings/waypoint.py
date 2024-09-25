############################################################
#
#   - Name : waypoint.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m


# private libs.


#.. Waypoint
class Waypoint():
    
    #.. initialize an instance of the class
    def __init__(self, wp_type_selection=1) -> None:
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_z = []
        self.set_values(wp_type_selection, self.waypoint_x, self.waypoint_y, self.waypoint_z)
        pass
    
    def set_values(self, wp_type_selection, wpx, wpy, wpz):

        # 20240914 diy
        #.. regenerated WPs for test
        h1      =   10.   
        h2      =   10.         
        self.reWPs     =   np.array([ [55, 30, -h1], [55, 55, -h2], [25, 55, -h1], [25, 30, -h1], [10, 40, -h1]])

        #.. straight line
        if wp_type_selection == 0:
            d       =   1500.
            h1      =   10.
            wp0     =   1.
            self.WPs     =   np.array([ [0, 0, -h1], [wp0, 0., -h1], [d-wp0, 0., -h1], [d, 0., -h1] ])
        #.. rectangle
        elif wp_type_selection == 1:
            # d       =   35
            d       =   35
            wp0     =   5.
            h1      =   10.
            h2      =   10.
            self.WPs     =   np.array([ [0, 0, -h1],
                                [wp0, wp0, -h1], [wp0 + d, wp0, -h2], [wp0 + d, wp0 + d, -h1], [wp0, wp0 + d, -h2], [wp0, wp0, -h1], 
                                [0.5*wp0, 0.5*wp0, -h1], [0, 0, -h1]])

            # self.WPs     =   np.array([ [0, 0, -h1],
            #                     [wp0, wp0, -h1], [wp0 - d , wp0, -h2], [wp0 - d, wp0 + d, -h1], [wp0, wp0 + d, -h2], [wp0, wp0, -h1],
            #                     [0.5* wp0, 0.5*wp0, -h1], 
            #                     [0, 0, -h1]])
        #.. circle
        elif wp_type_selection == 2:
            # param.
            n_cycle     =   1
            R           =   20
            N           =   n_cycle * 20        # 38
            # calc.
            ang_WP              =   n_cycle * 2*m.pi*(np.arange(N) + 1)/N
            self.WPs            =   -10*np.ones((N + 1,3))
            self.WPs[0,0]       =   0.
            self.WPs[0,1]       =   0.
            self.WPs[1:N+1,0]   =   R*np.sin(ang_WP)
            self.WPs[1:N+1,1]   =   - R*np.cos(ang_WP) + R
            pass
        
        #.. designed
        elif wp_type_selection == 3:
            WPx     =   np.array([0., 7.5, 9.0,  11.9, 16.0, 42.5, 44.0, 44.6, 42.2, 21.0, \
                17.9, 15.6, 13.9, 13.5, 16.4, 21.0, 28.9, 44.4, 43.8, 40.4, 26.9, -15.0, -25.0, -20.0, -10.0
                ])
            WPy     =   np.array([0., 7.7, 44.0, 46.4, 47.0, 46.7, 43.9, 38.1, 35.2, 34.7, \
                33.4, 29.9, 23.6, 7.9,  5.0,  3.1,  4.3,  25.5, 30.8, 34.3, 38.2, 35.0,  10.0,   0.0, -5.0
                ])
            N = len(WPx)
            self.WPs        =   -10.*np.ones((N,3))
            self.WPs[:,1]   =   WPx
            self.WPs[:,0]   =   WPy
            pass

        elif wp_type_selection == 4:
            WPx     =   np.array(wpx)
            WPy     =   np.array(wpy)
            h       =   np.array(wpz)

            N = len(WPx)

            self.WPs        =   -10.*np.ones((N,3))
            self.WPs[:,0]   =   WPx
            self.WPs[:,1]   =   WPy
            self.WPs[:,2]   =   -h
        
        else:
            # straight line
            self.WPs     =   np.array([ [0, 0, -10], [30, 30, -10] ])
            pass
        pass
    
    #.. insert_WP
    def insert_WP(self, WP_index, WP_pos_i):
        self.WPs = np.insert(self.WPs, WP_index, WP_pos_i, axis=0)
        pass
    
    pass