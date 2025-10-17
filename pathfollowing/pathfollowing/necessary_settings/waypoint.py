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

        #.. Following WP
        if wp_type_selection == 0:
            WPx     =   np.array(wpx)
            WPy     =   np.array(wpy)
            h       =   np.array(wpz)

            N = len(WPx)

            self.WPs        =   -10.*np.ones((N,3))
            self.WPs[:,0]   =   WPx
            self.WPs[:,1]   =   WPy
            self.WPs[:,2]   =   -h            
        #.. straight line
        elif wp_type_selection == 1:
            d       =   50.
            h1      =   10.
            wp0     =   5.
            self.WPs     =   np.array([ [0, 0, -h1], [-wp0, 0., -h1], [-d+wp0, 0., -h1], [-d, 0., -h1] ])
            # self.WPs     =   np.array([ [0, 0, -h1], [-d, 0., -h1]])
        #.. rectangle
        elif wp_type_selection == 2:
            # d       =   35
            d       =   35
            wp0     =   5.
            h1      =   10.
            h2      =   10.
            self.WPs     =   np.array([ [0, 0, -h1],
                                [-wp0, wp0, -h1], [-wp0 - d, wp0, -h2], [-wp0 - d, wp0 + d, -h1], [-wp0, wp0 + d, -h2], [-wp0, wp0, -h1], 
                                [-0.5*wp0, 0.5*wp0, -h1], [0, 0, -h1]])            
        #.. zigzag
        elif wp_type_selection == 3:
            d1       =   15
            d2       =   5
            wp0     =   5.
            h      =   10.
            # h2      =   10.
            self.WPs = np.array([
                [0, 0, -h],
                [-wp0,  wp0, -h],
                [-wp0 - d2,  wp0 + d1, -h],
                [-wp0 - 2*d2,  wp0, -h],
                [-wp0 - 3*d2,  wp0 + d1, -h],
                [-wp0 - 4*d2,  wp0, -h],
                [-wp0 - 4*d2 - wp0,  wp0, -h],
                [-wp0 - 4*d2 - 2*wp0,  wp0, -h],
            ])
        #.. circle
        elif wp_type_selection == 4:
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
        #.. figure-8
        elif wp_type_selection == 5:
            # param
            wp0     =   5.
            n_cycle     = 1
            R           = 16.0 
            N           = 40
            n           = N // 4
            h           = 10.0 
            # angle 분배
            ang_half    = np.pi * np.linspace(0, 1, n)  # 0 to pi
            ang_full    = np.concatenate([ang_half, np.pi + ang_half])
            # 초기화
            self.WPs = -h * np.ones((N+3, 3))
            # x좌표: 왼쪽 원은 -R 중심, 오른쪽 원은 -3R 중심
            self.WPs[0, :]              = [0.0, 0.0, -h]
            self.WPs[1:n+1, 0]          = -R + R * np.cos(np.pi + ang_half[::-1]) - wp0
            self.WPs[1:n+1, 1]          =      R * np.sin(np.pi + ang_half[::-1])
            self.WPs[n+1:3*n+1, 0]      = -3*R + R * np.cos(ang_full) - wp0
            self.WPs[n+1:3*n+1, 1]      =        R * np.sin(ang_full)
            self.WPs[3*n+1:4*n+1, 0]    = -R + R * np.cos(ang_half[::-1]) - wp0
            self.WPs[3*n+1:4*n+1, 1]    =      R * np.sin(ang_half[::-1])

            self.WPs[4*n+1, :]    = [- wp0, 0.0, -h]
            self.WPs[4*n+2, :]    = [0.0, 0.0, -h]

        #.. Alt change
        elif wp_type_selection == 6:
            d       =   10.
            h1      =   10.
            h2      =   20.
            wp0     =   5.
            self.WPs     =   np.array([ [0, 0, -h1], [-d, 0., -h1], [-2*d, 0., -h2], [-3*d, 0., -h2], [-4*d, 0., -h1], [-5*d+wp0, 0., -h1], [-5*d, 0., -h1] ])
        #.. Spiral
        elif wp_type_selection == 7:
            n_turns     = 1
            points_per_turn = 20
            R           = 10.0
            h_start     = 10.0
            h_step      = 10.0
            N           = n_turns * points_per_turn
            theta       = np.linspace(0, 2 * np.pi * n_turns, N)
            # WP shape: (N, 3)
            self.WPs = np.zeros((N, 3))
            self.WPs[:, 0] = R * np.cos(theta)   # x = R cos(θ)
            self.WPs[:, 1] = R * np.sin(theta)   # y = R sin(θ)
            self.WPs[:, 2] = -(h_start + h_step * theta / (2 * np.pi))
            # 시작점 앞에 (0,0) 추가 (선택)
            self.WPs = np.vstack(([ [0., 0., -h_start] ], self.WPs))
            self.WPs = np.vstack((self.WPs, [[0., 0., -(h_start + h_step*n_turns)]]))
            self.WPs = np.vstack((self.WPs, [[0., 0., -h_start ]]))
        
        # Lissajous curve (figure 8)
        elif wp_type_selection == 8:
            # param
            wp0     =   0.
            n_cycle     = 1
            R           = 16.0 
            N           = 40
            n           = N // 4
            h           = 10.0 

            A = 30.0  # X
            B = 20.0  # Y
            N = 40    # # of waypoints
            h = 10.0  # Z
            
            # --- Waypoint generation ---
            t = np.linspace(0, 2 * np.pi, N, endpoint=False)

            x = A * np.sin(t)
            y = B * np.sin(2 * t)
            z = -h * np.ones_like(t)  

            # x, y, z -> (N, 3) 
            self.WPs = np.vstack([x, y, z]).T

            start_point = np.array([[0.0, 0.0, -h]])
            end_point   = start_point
            self.WPs = np.concatenate([start_point, self.WPs, start_point])
            pass
        
        else:
        # straight line
            self.WPs     =   np.array([ [0, 0, -10], [-10, 10, -10] ])
            pass
        pass
    
    #.. insert_WP
    def insert_WP(self, WP_index, WP_pos_i):
        self.WPs = np.insert(self.WPs, WP_index, WP_pos_i, axis=0)
        pass
    
    pass