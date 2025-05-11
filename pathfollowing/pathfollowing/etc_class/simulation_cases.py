############################################################
#
#   - Name : simulation_cases.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np


# private libs.


#.. Simulation_Cases
class Simulation_Cases():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        #.. simulation
        self.dict_sim_cases = {}
        pass
    
    #.. add_sim_case
    def add_sim_case(self, case_name, case_value_array):
        self.dict_sim_cases[case_name] = case_value_array
        print("add the sim. case_name = '" + str(case_name) + "', value_array = " + str(case_value_array))
        pass
    
    #.. get_all_case_name
    def get_all_case_name(self, bool_print_name=False):
        keys = list(self.dict_sim_cases.keys())
        if bool_print_name == True:
            print("get_all_case_name: " + str(keys)) 
            pass
        return keys
    
    #.. get_a_case_value
    def get_a_case_value(self, case_name, bool_print_value=False):
        values = list(self.dict_sim_cases.get(case_name))
        if bool_print_value == True:
            print("get_all_case_value: " + str(case_name) + str(values))
            pass
        return values
    
    #.. get_case_combination
    def get_case_combination(self):
        list_case_name = self.get_all_case_name()
        list_value_length = []
        list_value_iter = []
        num_total_sim_case = 1
        for case_name in list_case_name:
            values = self.get_a_case_value(case_name)
            list_value_length.append(len(values))
            list_value_iter.append(0)
            num_total_sim_case = num_total_sim_case * len(values)
            pass
        arr_value_length = np.array(list_value_length)
        arr_value_iter = np.array(list_value_iter)
        return list_case_name, arr_value_length, arr_value_iter, num_total_sim_case
        
    #.. get_array_value_iter
    def get_array_value_iter(self, i_case, arr_value_iter, arr_value_length):
        num_sim_case_type = len(arr_value_iter)
        numerator = 1
        arr_numerator = np.zeros_like(arr_value_iter)
        for i_sim_case_type_rev in range(num_sim_case_type-1, -1, -1):
            numerator = numerator * arr_value_length[i_sim_case_type_rev]
            arr_numerator[i_sim_case_type_rev] = numerator
            if i_case < numerator:
                rem = i_case
                if i_sim_case_type_rev < num_sim_case_type - 1:
                    for i_sim_case_type in range(i_sim_case_type_rev + 1, num_sim_case_type):
                        quo, rem = divmod(rem, arr_numerator[i_sim_case_type])
                        arr_value_iter[i_sim_case_type-1] = quo
                        if i_sim_case_type == num_sim_case_type - 1:
                            arr_value_iter[i_sim_case_type] = rem
                else:
                    arr_value_iter[i_sim_case_type_rev] = rem
                break
            pass
        return arr_value_iter
    
    #.. check_all_sim_case_name
    def check_all_sim_case_name(self):
        all_sim_case_name = self.get_all_case_name()
        print("---- check_sim_case ----")
        if  len(all_sim_case_name) == 0:
            self.add_sim_case('defualt',[0])
        else:
            for case_name in all_sim_case_name:
                values = self.get_a_case_value(case_name)
                print("  Key:" + str(case_name) + " , Value:" + str(values) )
            pass
        pass
        print("")
        
        
    pass
