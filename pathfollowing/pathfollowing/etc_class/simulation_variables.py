############################################################
#
#   - Name : simulation_variables.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np


# private libs.


#.. Simulation_Variables
class Simulation_Variables():
    #.. initialize an instance of the class
    def set_values(self):
        # simulation
        self.t_sim      =   0.
        self.flag_stop  =   False
        
        pass
    
    def __init__(self) -> None:
        self.set_values()
        pass
    pass