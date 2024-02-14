############################################################
#
#   - Name : set_parameter.py
#
#                   -   Created by E. T. Jeong, 2024.09.13
#
############################################################

#.. Library
# pulbic libs.
import numpy as np
import math as m

# private libs.

# waypoint setting
class Way_Point():
    #.. initialize an instance of the class
    def __init__(self,wp_type_selection,wpx=None, wpy=None, wpz=None) -> None:
        #.. straight line
        if wp_type_selection == 0:
            d       =   150.
            h1      =   10.
            wp0     =   5.
            self.WPs     =   np.array([ [0, 0, -h1], [wp0, 0., -h1], [d-wp0, 0., -h1], [d, 0., -h1] ])
        #.. rectangle
        elif wp_type_selection == 1:
            # d       =   25.     # 25.
            d       =   35.     # 25.
            # d       =   45.     # 25.
            wp0     =   5.
            h1      =   10.
            h2      =   10.
            
            # self.WPs     =   np.array([ [0, 0, -h1],
            #                     [wp0, wp0, -h1], [wp0 + 10, wp0, -h1], [wp0 + d, wp0, -h2], [wp0 + d, wp0 + d, -h1], [wp0, wp0 + d, -h2], [wp0, wp0, -h1], 
            #                     [0, 0, -h1]])
            
            self.WPs     =   np.array([ [0, 0, -h1],
                                [wp0, wp0, -h1], [wp0 + d, wp0, -h2], [wp0 + d, wp0 + d, -h1], [wp0, wp0 + d, -h2], [wp0, wp0, -h1], 
                                [0, 0, -h1]])         
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
            WPx     =   np.array(wpx)
            WPy     =   np.array(wpy)
            h       =   np.array(wpz)

            N = len(WPx)
            # self.WPs        =   -10.*np.ones((N,3))
            # self.WPs[:,0]   =   WPx
            # self.WPs[:,1]   =   WPy
            # self.WPs[:,2]   =   -h

            self.WPs        =   -10.*np.ones((N,3))
            self.WPs[:,0]   =   WPx
            self.WPs[:,1]   =   WPy
            self.WPs[:,2]   =   -h
            pass
        
        else:
            # straight line
            self.WPs     =   np.array([ [0, 0, -10], [30, 30, -10] ])
            pass
        pass

    #.. initialize WP
    def init_WP(self, Q6_Ri):
        self.WPs[0][0]   =   Q6_Ri[0]
        self.WPs[0][1]   =   Q6_Ri[1]
        pass
    
    pass
    
# MPPI guidance parameter setting
class MPPI_Guidance_Parameter():
    #.. initialize an instance of the class
    def __init__(self,MPPI_type_selection) -> None:
        self.MPPI_type_selection = MPPI_type_selection
        #.. acceleration command MPPI (direct)
        self.Q_lim_margin = np.array([0.9])
        self.Q_lim  =   np.array([0.5])
        self.a_lim  =   1.0
        # self.Q      =   np.array([0.2, 0.02, 10.0, 0.0])
        # self.Q      =   np.array([0.05, 0.02, 0.2, 0.0])
        # self.Q      =   np.array([0.05, 0.02, 0.5, 0.0])
        self.Q      =   np.array([0.05, 0.02, 2.0, 0.0])
        self.R      =   np.array([0.001, 0.001, 0.001])
        self.K      =   2**8
        # self.K      =   2**9
        
        if MPPI_type_selection == 2:
            #.. cost
            self.flag_cost_calc     =   0
            #.. Parameters - low performance && short computation
            self.dt     =   0.05
            self.N      =   100
            self.nu     =   1000.       # 2.
            #.. u1: acmd_x_i, u2: acmd_y_i, u3: acmd_z_i
            self.var1   =   1.0 * 0.5       # 1.0
            self.var2   =   self.var1
            self.var3   =   self.var1
            self.lamb1  =   1. * 5.
            self.lamb2  =   1. * 5.
            self.lamb3  =   1. * 5.
            self.u1_init    =   0.
            self.u2_init    =   0.
            self.u3_init    =   0.
            
        #.. guidance parameter MPPI (indirect)
        elif MPPI_type_selection == 3:
            #.. cost
            self.flag_cost_calc     =   0
            
            #.. Parameters - good / stable for 6dof
            self.dt     =   0.04
            self.N      =   100
            self.nu     =   1000.
            # #.. u1: LAD, u2: desired_speed, u3: eta
            # self.var1   =   1.0 * 0.7           # 0.2
            self.var1   =   1.0 * 0.5           # 0.2
            
            self.var2   =   self.var1
            self.var3   =   self.var1
            self.lamb1  =   1.0 *1.0           # 1.0
            self.lamb2  =   self.lamb1
            self.lamb3  =   self.lamb1
            self.u1_init    =   3.0
            self.u2_init    =   2.0
            self.u3_init    =   2.
            
        
        #.. no use MPPI module
        else:
            #.. cost
            self.flag_cost_calc     =   0
            # # Q: penaty of distance[0], thrust_to_move[1]
            # self.Q      =   np.array([1.0, 0.02]) 
            # self.R      =   np.array([0.001, 0.001, 0.001])
            #.. Parameters - low performance && short computation
            self.dt     =   0.001
            self.K      =   32
            self.N      =   30000
            self.nu     =   2.
            #.. u1: acmd_x_i, u2: acmd_y_i, u3: acmd_z_i
            self.var1   =   1.0 * 2.0
            self.var2   =   1.0 * 2.0
            self.var3   =   1.0 * 1.0
            self.lamb1  =   1. * 10.
            self.lamb2  =   1. * 10.
            self.lamb3  =   1. * 10.
            self.u1_init    =   0.
            self.u2_init    =   0.
            self.u3_init    =   0.
            pass
        pass
    pass
    