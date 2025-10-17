############################################################
#
#   - Name : GPR.py
#
#                     -   Created by D. Yoon, 2024.07.18
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m
import sys, os

# private libs.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from necessary_settings.quadrotor_iris_parameters import GPR_Parameter

#.. GPR_Modules
class GPR_Modules():
    #.. initialize an instance of the class
    def __init__(self, GPR_Param:GPR_Parameter) -> None:
        self.GP_param     =   GPR_Param
        pass
    
    #.. GPR_hyperparams_opt
    def GPR_hyperparams_opt(self):
        if (self.GP_param.count_GPR == 1):
            # x axis
            temp1 = np.zeros([2, 2])
            temp2 = np.zeros([2, 2])
            temp3 = np.zeros([2, 2])

            for j in range(len(self.GP_param.training_data_x) - 1):
                temp1 = temp1 + np.array(self.GP_param.training_data_x[j + 1]).dot(np.array(self.GP_param.training_data_x[j]).T)
                temp2 = temp2 + np.array(self.GP_param.training_data_x[j]).dot(np.array(self.GP_param.training_data_x[j]).T)

            if (np.max(temp1)<1.0 or np.max(temp2)<1.0):
                pass
            else:
                A_ml = temp1.dot(np.linalg.inv(temp2 + np.eye(2) * 1e-5))

                self.GP_param.hyp_l_GPR[0] = (1 - A_ml[1][1]) * 0.5 / self.GP_param.dt_GPR
                l_opt = np.max([np.min([self.GP_param.hyp_l_GPR[0],0.5]),1e-2])
                
                # self.GP_param.hyp_l_GPR[0] = np.sqrt(np.abs(A_ml[1][0]) / self.GP_param.dt_GPR)
                # l_opt = self.GP_param.hyp_l_GPR[0]            

                self.GP_param.F_x_GPR = np.array([[0.0, 1.0], [-pow(l_opt, 2), -2 * l_opt]])
                self.GP_param.A_x_GPR = np.array([[1.0, self.GP_param.dt_GPR], [-pow(l_opt, 2) * self.GP_param.dt_GPR, 1 - 2 * l_opt * self.GP_param.dt_GPR]])

                for j in range(len(self.GP_param.training_data_x) - 1):
                    temp = np.array(self.GP_param.training_data_x[j + 1] - self.GP_param.A_x_GPR.dot(np.array(self.GP_param.training_data_x[j])))
                    temp3 = temp3 + temp.dot(temp.T)

                S_ml = temp3 / (len(self.GP_param.training_data_x) - 1)
                self.GP_param.hyp_q_GPR[0] = S_ml[1][1] / (
                            self.GP_param.dt_GPR - 2 * l_opt * pow(self.GP_param.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.GP_param.dt_GPR, 3))

                self.GP_param.hyp_q_GPR[0] = np.max([np.min([self.GP_param.hyp_q_GPR[0],0.5]),1e-2])

                self.GP_param.Q_x_GPR = self.GP_param.hyp_q_GPR[0] * np.array(
                    [[1 / 3 * pow(self.GP_param.dt_GPR, 3), 0.5 * pow(self.GP_param.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.GP_param.dt_GPR, 3)],
                    [0.5 * pow(self.GP_param.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.GP_param.dt_GPR, 3),
                    self.GP_param.dt_GPR - 2 * l_opt * pow(self.GP_param.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.GP_param.dt_GPR, 3)]])

        if (self.GP_param.count_GPR == 2):

            # y axis
            temp1 = np.zeros([2, 2])
            temp2 = np.zeros([2, 2])
            temp3 = np.zeros([2, 2])

            for j in range(len(self.GP_param.training_data_y) - 1):
                temp1 = temp1 + np.array(self.GP_param.training_data_y[j + 1]).dot(np.array(self.GP_param.training_data_y[j]).T)
                temp2 = temp2 + np.array(self.GP_param.training_data_y[j]).dot(np.array(self.GP_param.training_data_y[j]).T)

            if (np.max(temp1)<1.0 or np.max(temp2)<1.0):
                pass
            else:
                A_ml = temp1.dot(np.linalg.inv(temp2 + np.eye(2) * 1e-5))

                self.GP_param.hyp_l_GPR[1] = (1 - A_ml[1][1]) * 0.5 / self.GP_param.dt_GPR
                l_opt = np.max([np.min([self.GP_param.hyp_l_GPR[1],0.5]),1e-2])

                # self.GP_param.hyp_l_GPR[1] = np.sqrt(np.abs(A_ml[1][0]) / self.GP_param.dt_GPR)
                # l_opt = self.GP_param.hyp_l_GPR[1]

                self.GP_param.F_y_GPR = np.array([[0.0, 1.0], [-pow(l_opt, 2), -2 * l_opt]])
                self.GP_param.A_y_GPR = np.array([[1.0, self.GP_param.dt_GPR], [-pow(l_opt, 2) * self.GP_param.dt_GPR, 1 - 2 * l_opt * self.GP_param.dt_GPR]])

                for j in range(len(self.GP_param.training_data_y) - 1):
                    temp = np.array(self.GP_param.training_data_y[j + 1] - self.GP_param.A_y_GPR.dot(np.array(self.GP_param.training_data_y[j])))
                    temp3 = temp3 + temp.dot(temp.T)

                S_ml = temp3 / (len(self.GP_param.training_data_y) - 1)
                self.GP_param.hyp_q_GPR[1] = S_ml[1][1] / (
                            self.GP_param.dt_GPR - 2 * l_opt * pow(self.GP_param.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.GP_param.dt_GPR, 3))

                self.GP_param.hyp_q_GPR[1] = np.max([np.min([self.GP_param.hyp_q_GPR[1],0.5]),1e-2])

                self.GP_param.Q_y_GPR = self.GP_param.hyp_q_GPR[1] * np.array(
                    [[1 / 3 * pow(self.GP_param.dt_GPR, 3), 0.5 * pow(self.GP_param.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.GP_param.dt_GPR, 3)],
                    [0.5 * pow(self.GP_param.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.GP_param.dt_GPR, 3),
                    self.GP_param.dt_GPR - 2 * l_opt * pow(self.GP_param.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.GP_param.dt_GPR, 3)]])

            self.GP_param.count_GPR = 0

        # if (self.GP_param.count_GPR == 3):
            # temp1 = np.zeros([2, 2])
            # temp2 = np.zeros([2, 2])
            # temp3 = np.zeros([2, 2])

            # for j in range(len(self.GP_param.training_data_z) - 1):
            #     temp1 = temp1 + np.array(self.GP_param.training_data_z[j + 1]).dot(np.array(self.GP_param.training_data_z[j]).T)
            #     temp2 = temp2 + np.array(self.GP_param.training_data_z[j]).dot(np.array(self.GP_param.training_data_z[j]).T)

            # if (np.max(temp1)<1.0 or np.max(temp2)<1.0):
            #     pass
            # else:
            #     A_ml = temp1.dot(np.linalg.inv(temp2 + np.eye(2) * 1e-5))

            #     self.GP_param.hyp_l_GPR[2] = (1 - A_ml[1][1]) * 0.5 / self.GP_param.dt_GPR
            #     l_opt = np.max([np.min([self.GP_param.hyp_l_GPR[2],5]),1e-2])

            #     # self.GP_param.hyp_l_GPR[2] = np.sqrt(np.abs(A_ml[1][0]) / self.GP_param.dt_GPR)
            #     # l_opt = self.GP_param.hyp_l_GPR[2]

            #     self.GP_param.F_z_GPR = np.array([[0.0, 1.0], [-pow(l_opt, 2), -2 * l_opt]])
            #     self.GP_param.A_z_GPR = np.array([[1.0, self.GP_param.dt_GPR], [-pow(l_opt, 2) * self.GP_param.dt_GPR, 1 - 2 * l_opt * self.GP_param.dt_GPR]])

            #     for j in range(len(self.GP_param.training_data_z) - 1):
            #         temp = np.array(self.GP_param.training_data_z[j + 1] - self.GP_param.A_z_GPR.dot(np.array(self.GP_param.training_data_z[j])))
            #         temp3 = temp3 + temp.dot(temp.T)

            #     S_ml = temp3 / (len(self.GP_param.training_data_z) - 1)
            #     self.GP_param.hyp_q_GPR[2] = S_ml[1][1] / (
            #                 self.GP_param.dt_GPR - 2 * l_opt * pow(self.GP_param.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.GP_param.dt_GPR, 3))
                
            #     self.GP_param.hyp_q_GPR[2] = np.max([np.min([self.GP_param.hyp_q_GPR[2],0.5]),1e-2])

            #     self.GP_param.Q_z_GPR = self.GP_param.hyp_q_GPR[1] * np.array(
            #         [[1 / 3 * pow(self.GP_param.dt_GPR, 3), 0.5 * pow(self.GP_param.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.GP_param.dt_GPR, 3)],
            #         [0.5 * pow(self.GP_param.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.GP_param.dt_GPR, 3),
            #         self.GP_param.dt_GPR - 2 * l_opt * pow(self.GP_param.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.GP_param.dt_GPR, 3)]])

            # self.GP_param.count_GPR = 0

        self.GP_param.count_GPR = self.GP_param.count_GPR + 1
        pass

    #.. GPR_update
    def GPR_update(self, QR_out_NDO):

        # Prediction step
        mp_x = self.GP_param.A_x_GPR.dot(self.GP_param.m_x_GPR)
        Pp_x = (self.GP_param.A_x_GPR.dot(self.GP_param.P_x_GPR)).dot(self.GP_param.A_x_GPR.T) + self.GP_param.Q_x_GPR

        mp_y = self.GP_param.A_y_GPR.dot(self.GP_param.m_y_GPR)
        Pp_y = (self.GP_param.A_y_GPR.dot(self.GP_param.P_y_GPR)).dot(self.GP_param.A_y_GPR.T) + self.GP_param.Q_y_GPR

        # mp_z = self.GP_param.A_z_GPR.dot(self.GP_param.m_z_GPR)
        # Pp_z = (self.GP_param.A_z_GPR.dot(self.GP_param.P_z_GPR)).dot(self.GP_param.A_z_GPR.T) + self.GP_param.Q_z_GPR

        # Update step
        v = QR_out_NDO[0] - self.GP_param.H_GPR.dot(mp_x)
        S = (self.GP_param.H_GPR.dot(Pp_x)).dot(self.GP_param.H_GPR.T) + self.GP_param.R_GPR_x
        K = (Pp_x.dot(self.GP_param.H_GPR.T)) / S
        self.GP_param.m_x_GPR = mp_x + K * v
        self.GP_param.P_x_GPR = Pp_x - (K.dot(K.T)) * S

        v = QR_out_NDO[1] - self.GP_param.H_GPR.dot(mp_y)
        S = (self.GP_param.H_GPR.dot(Pp_y)).dot(self.GP_param.H_GPR.T) + self.GP_param.R_GPR_y
        K = (Pp_y.dot(self.GP_param.H_GPR.T)) / S
        self.GP_param.m_y_GPR = mp_y + K * v
        self.GP_param.P_y_GPR = Pp_y - (K.dot(K.T)) * S

        # v = QR_out_NDO[2] - self.GP_param.H_GPR.dot(mp_z)
        # S = (self.GP_param.H_GPR.dot(Pp_z)).dot(self.GP_param.H_GPR.T) + self.GP_param.R_GPR_z
        # K = (Pp_z.dot(self.GP_param.H_GPR.T)) / S
        # self.GP_param.m_z_GPR = mp_z + K * v
        # self.GP_param.P_z_GPR = Pp_z - (K.dot(K.T)) * S

        # LPF data filtering
        self.GP_param.m_x_LPF      = self.GP_param.num_LPF * self.GP_param.m_x_prev - self.GP_param.den_LPF * self.GP_param.m_x_LPF_prev
        self.GP_param.m_x_prev     = mp_x
        self.GP_param.m_x_LPF_prev = self.GP_param.m_x_LPF

        self.GP_param.m_y_LPF      = self.GP_param.num_LPF * self.GP_param.m_y_prev - self.GP_param.den_LPF * self.GP_param.m_y_LPF_prev
        self.GP_param.m_y_prev     = mp_y
        self.GP_param.m_y_LPF_prev = self.GP_param.m_y_LPF

        # self.GP_param.m_z_LPF      = self.GP_param.num_LPF * self.GP_param.m_z_prev - self.GP_param.den_LPF * self.GP_param.m_z_LPF_prev
        # self.GP_param.m_z_prev     = mp_z
        # self.GP_param.m_z_LPF_prev = self.GP_param.m_z_LPF

        # 3) Training data acquisition
        self.GP_param.training_data_x.append(self.GP_param.m_x_LPF)
        self.GP_param.training_data_y.append(self.GP_param.m_y_LPF)
        # self.GP_param.training_data_z.append(self.GP_param.m_z_LPF)

        if (len(self.GP_param.training_data_x) > self.GP_param.hyp_n_GPR):
            self.GP_param.training_data_x.pop(0)
            self.GP_param.training_data_y.pop(0)
            # self.GP_param.training_data_z.pop(0)

        pass

    #.. GPR_forecasting
    def GPR_forecasting(self, curr_t):
        
        te   = curr_t
        me_x = self.GP_param.m_x_GPR
        Pe_x = self.GP_param.P_x_GPR
        me_y = self.GP_param.m_y_GPR
        Pe_y = self.GP_param.P_y_GPR
        # me_z = self.GP_param.m_z_GPR
        # Pe_z = self.GP_param.P_z_GPR 

        for i in range(self.GP_param.ne_GPR*2):
            me_x = self.GP_param.A_x_GPR.dot(me_x)
            Pe_x = (self.GP_param.A_x_GPR.dot(Pe_x)).dot(self.GP_param.A_x_GPR.T) + self.GP_param.Q_x_GPR
            me_y = self.GP_param.A_y_GPR.dot(me_y)
            Pe_y = (self.GP_param.A_y_GPR.dot(Pe_y)).dot(self.GP_param.A_y_GPR.T) + self.GP_param.Q_y_GPR
            # me_z = self.GP_param.A_y_GPR.dot(me_z)
            # Pe_z = (self.GP_param.A_y_GPR.dot(Pe_z)).dot(self.GP_param.A_z_GPR.T) + self.GP_param.Q_z_GPR

            te = te + self.GP_param.dt_GPR

            if (i % 2 == 0):
                i_save = int(i*0.5)
                self.GP_param.te_array[i_save]    = te
                self.GP_param.me_x_array[i_save]  = me_x[0]
                self.GP_param.var_x_array[i_save] = Pe_x[0][0]
                self.GP_param.me_y_array[i_save]  = me_y[0]
                self.GP_param.var_y_array[i_save] = Pe_y[0][0]
                # self.GP_param.me_z_array[i_save]  = me_z[0]
                # self.GP_param.var_z_array[i_save] = Pe_z[0][0]
        
        pass