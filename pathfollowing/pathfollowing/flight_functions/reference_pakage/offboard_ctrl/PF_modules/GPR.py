############################################################
#
#   - Name : GPR.py
#
#                      -   Created by D. Yoon, 2023.12.14
#
############################################################

#.. Library
# pulbic libs.
import numpy as np

class GPR_Modules():
    #.. initialize an instance of the class
    def __init__(self, dt_GPR) -> None:

        #.. gaussian process regression parameter
        self.dt_GPR  = dt_GPR

        self.ne_GPR  = 500  # forecasting number (ne = 2000, te = 2[sec])

        self.H_GPR   = np.array([1.0, 0.0]).reshape(1, 2)
        self.R_GPR_x = pow(0.001, 2)
        self.R_GPR_y = pow(0.001, 2)
        self.R_GPR_z = pow(0.01, 2)

        # self.R_GPR_x    =   pow(0.001, 2)
        # self.R_GPR_y    =   pow(0.005, 2)
        # self.R_GPR_z    =   pow(0.005, 2)

        self.hyp_l_GPR = 1 * np.ones(3)
        self.hyp_q_GPR = 1 * np.ones(3)
        self.hyp_n_GPR = 1000

        # self.hyp_l_GPR  =   10 * np.ones(3)
        # self.hyp_q_GPR  =   10 * np.ones(3)
        # self.hyp_n_GPR  =   1000

        hyp_l = self.hyp_l_GPR[0]

        self.F_x_GPR = np.array([[0.0, 1.0], [-pow(hyp_l, 2), -2 * hyp_l]])
        self.A_x_GPR = np.array([[1.0, self.dt_GPR], [-pow(hyp_l, 2) * self.dt_GPR, 1 - 2 * hyp_l * self.dt_GPR]])
        self.Q_x_GPR = self.hyp_q_GPR[0] * np.array(
            [[1 / 3 * pow(self.dt_GPR, 3), 0.5 * pow(self.dt_GPR, 2) - 2 * hyp_l / 3 * pow(self.dt_GPR, 3)],
             [0.5 * pow(self.dt_GPR, 2) - 2 * hyp_l / 3 * pow(self.dt_GPR, 3),
              self.dt_GPR - 2 * hyp_l * pow(self.dt_GPR, 2) + 4 / 3 * pow(hyp_l, 2) * pow(self.dt_GPR, 3)]])
        self.m_x_GPR = np.zeros([2, 1]).reshape(2, 1)
        self.P_x_GPR = np.zeros([2, 2])

        self.F_y_GPR = self.F_x_GPR[:]
        self.A_y_GPR = self.A_x_GPR[:]
        self.Q_y_GPR = self.Q_x_GPR[:]
        self.m_y_GPR = self.m_x_GPR[:]
        self.P_y_GPR = self.P_x_GPR[:]

        self.F_z_GPR = self.F_x_GPR[:]
        self.A_z_GPR = self.A_x_GPR[:]
        self.Q_z_GPR = self.Q_x_GPR[:]
        self.m_z_GPR = self.m_x_GPR[:]
        self.P_z_GPR = self.P_x_GPR[:]

        self.count_init = 0
        self.count_GPR = 1

        # Training data save
        self.training_data_x = []
        self.training_data_y = []
        self.training_data_z = []

        # Save data for plotting forecasting results
        self.te_array     =  np.zeros(self.ne_GPR)
        self.me_x_array   =  np.zeros(self.ne_GPR)
        self.var_x_array  =  np.zeros(self.ne_GPR)
        self.me_y_array   =  np.zeros(self.ne_GPR)
        self.var_y_array  =  np.zeros(self.ne_GPR)
        self.me_z_array   =  np.zeros(self.ne_GPR)
        self.var_z_array  =  np.zeros(self.ne_GPR)

    def HyperParamsOpt(self):
        if (self.count_GPR == 1):
            # x axis
            temp1 = np.zeros([2, 2])
            temp2 = np.zeros([2, 2])
            temp3 = np.zeros([2, 2])

            for j in range(len(self.training_data_x) - 1):
                temp1 = temp1 + np.array(self.training_data_x[j + 1]).dot(np.array(self.training_data_x[j]).T)
                temp2 = temp2 + np.array(self.training_data_x[j]).dot(np.array(self.training_data_x[j]).T)

            A_ml = temp1.dot(np.linalg.inv(temp2))

            self.hyp_l_GPR[0] = np.sqrt(np.abs(A_ml[1][0]) / self.dt_GPR)
            l_opt = self.hyp_l_GPR[0]

            self.F_x_GPR = np.array([[0.0, 1.0], [-pow(l_opt, 2), -2 * l_opt]])
            self.A_x_GPR = np.array([[1.0, self.dt_GPR], [-pow(l_opt, 2) * self.dt_GPR, 1 - 2 * l_opt * self.dt_GPR]])

            for j in range(len(self.training_data_x) - 1):
                temp = np.array(self.training_data_x[j + 1] - self.A_x_GPR.dot(np.array(self.training_data_x[j])))
                temp3 = temp3 + temp.dot(temp.T)

            S_ml = temp3 / (len(self.training_data_x) - 1)
            self.hyp_q_GPR[0] = S_ml[1][1] / (
                        self.dt_GPR - 2 * l_opt * pow(self.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.dt_GPR, 3))

            self.Q_x_GPR = self.hyp_q_GPR[0] * np.array(
                [[1 / 3 * pow(self.dt_GPR, 3), 0.5 * pow(self.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.dt_GPR, 3)],
                 [0.5 * pow(self.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.dt_GPR, 3),
                  self.dt_GPR - 2 * l_opt * pow(self.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.dt_GPR, 3)]])

        if (self.count_GPR == 2):
            # y axis
            temp1 = np.zeros([2, 2])
            temp2 = np.zeros([2, 2])
            temp3 = np.zeros([2, 2])

            for j in range(len(self.training_data_y) - 1):
                temp1 = temp1 + np.array(self.training_data_y[j + 1]).dot(np.array(self.training_data_y[j]).T)
                temp2 = temp2 + np.array(self.training_data_y[j]).dot(np.array(self.training_data_y[j]).T)

            A_ml = temp1.dot(np.linalg.inv(temp2))

            self.hyp_l_GPR[1] = np.sqrt(np.abs(A_ml[1][0]) / self.dt_GPR)
            l_opt = self.hyp_l_GPR[1]

            self.F_y_GPR = np.array([[0.0, 1.0], [-pow(l_opt, 2), -2 * l_opt]])
            self.A_y_GPR = np.array([[1.0, self.dt_GPR], [-pow(l_opt, 2) * self.dt_GPR, 1 - 2 * l_opt * self.dt_GPR]])

            for j in range(len(self.training_data_y) - 1):
                temp = np.array(self.training_data_y[j + 1] - self.A_y_GPR.dot(np.array(self.training_data_y[j])))
                temp3 = temp3 + temp.dot(temp.T)

            S_ml = temp3 / (len(self.training_data_y) - 1)
            self.hyp_q_GPR[1] = S_ml[1][1] / (
                        self.dt_GPR - 2 * l_opt * pow(self.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.dt_GPR, 3))

            self.Q_y_GPR = self.hyp_q_GPR[1] * np.array(
                [[1 / 3 * pow(self.dt_GPR, 3), 0.5 * pow(self.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.dt_GPR, 3)],
                 [0.5 * pow(self.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.dt_GPR, 3),
                  self.dt_GPR - 2 * l_opt * pow(self.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.dt_GPR, 3)]])

        if (self.count_GPR == 3):
            temp1 = np.zeros([2, 2])
            temp2 = np.zeros([2, 2])
            temp3 = np.zeros([2, 2])

            for j in range(len(self.training_data_z) - 1):
                temp1 = temp1 + np.array(self.training_data_z[j + 1]).dot(np.array(self.training_data_z[j]).T)
                temp2 = temp2 + np.array(self.training_data_z[j]).dot(np.array(self.training_data_z[j]).T)

            A_ml = temp1.dot(np.linalg.inv(temp2))

            self.hyp_l_GPR[2] = np.sqrt(np.abs(A_ml[1][0]) / self.dt_GPR)
            l_opt = self.hyp_l_GPR[2]

            self.F_z_GPR = np.array([[0.0, 1.0], [-pow(l_opt, 2), -2 * l_opt]])
            self.A_z_GPR = np.array([[1.0, self.dt_GPR], [-pow(l_opt, 2) * self.dt_GPR, 1 - 2 * l_opt * self.dt_GPR]])

            for j in range(len(self.training_data_z) - 1):
                temp = np.array(self.training_data_z[j + 1] - self.A_z_GPR.dot(np.array(self.training_data_z[j])))
                temp3 = temp3 + temp.dot(temp.T)

            S_ml = temp3 / (len(self.training_data_z) - 1)
            self.hyp_q_GPR[2] = S_ml[1][1] / (
                        self.dt_GPR - 2 * l_opt * pow(self.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.dt_GPR, 3))

            self.Q_z_GPR = self.hyp_q_GPR[1] * np.array(
                [[1 / 3 * pow(self.dt_GPR, 3), 0.5 * pow(self.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.dt_GPR, 3)],
                 [0.5 * pow(self.dt_GPR, 2) - 2 * l_opt / 3 * pow(self.dt_GPR, 3),
                  self.dt_GPR - 2 * l_opt * pow(self.dt_GPR, 2) + 4 / 3 * pow(l_opt, 2) * pow(self.dt_GPR, 3)]])

            self.count_GPR = 0

        self.count_GPR = self.count_GPR + 1

    def GPR_Update(self, Q6_out_NDO):
        # Prediction step
        mp_x = self.A_x_GPR.dot(self.m_x_GPR)
        Pp_x = (self.A_x_GPR.dot(self.P_x_GPR)).dot(self.A_x_GPR.T) + self.Q_x_GPR

        mp_y = self.A_y_GPR.dot(self.m_y_GPR)
        Pp_y = (self.A_y_GPR.dot(self.P_y_GPR)).dot(self.A_y_GPR.T) + self.Q_y_GPR

        mp_z = self.A_z_GPR.dot(self.m_z_GPR)
        Pp_z = (self.A_z_GPR.dot(self.P_z_GPR)).dot(self.A_z_GPR.T) + self.Q_z_GPR

        # Update step
        v = Q6_out_NDO[0] - self.H_GPR.dot(mp_x)
        S = (self.H_GPR.dot(Pp_x)).dot(self.H_GPR.T) + self.R_GPR_x
        K = (Pp_x.dot(self.H_GPR.T)) / S
        self.m_x_GPR = mp_x + K * v
        self.P_x_GPR = Pp_x - (K.dot(K.T)) * S

        v = Q6_out_NDO[1] - self.H_GPR.dot(mp_y)
        S = (self.H_GPR.dot(Pp_y)).dot(self.H_GPR.T) + self.R_GPR_y
        K = (Pp_y.dot(self.H_GPR.T)) / S
        self.m_y_GPR = mp_y + K * v
        self.P_y_GPR = Pp_y - (K.dot(K.T)) * S

        v = Q6_out_NDO[2] - self.H_GPR.dot(mp_z)
        S = (self.H_GPR.dot(Pp_z)).dot(self.H_GPR.T) + self.R_GPR_z
        K = (Pp_z.dot(self.H_GPR.T)) / S
        self.m_z_GPR = mp_z + K * v
        self.P_z_GPR = Pp_z - (K.dot(K.T)) * S

        # 3) Training data acquisition
        self.training_data_x.append(mp_x)
        self.training_data_y.append(mp_y)
        self.training_data_z.append(mp_z)

        if (len(self.training_data_x) > self.hyp_n_GPR):
            self.training_data_x.pop(0)
            self.training_data_y.pop(0)
            self.training_data_z.pop(0)

    def GPR_Forecasting(self, SP_t):

            te   = SP_t
            me_x = self.m_x_GPR
            Pe_x = self.P_x_GPR
            me_y = self.m_y_GPR
            Pe_y = self.P_y_GPR
            me_z = self.m_z_GPR
            Pe_z = self.P_z_GPR

            for i in range(self.ne_GPR):
                me_x = self.A_x_GPR.dot(me_x)
                Pe_x = (self.A_x_GPR.dot(Pe_x)).dot(self.A_x_GPR.T) + self.Q_x_GPR
                me_y = self.A_y_GPR.dot(me_y)
                Pe_y = (self.A_y_GPR.dot(Pe_y)).dot(self.A_y_GPR.T) + self.Q_y_GPR
                me_z = self.A_y_GPR.dot(me_z)
                Pe_z = (self.A_y_GPR.dot(Pe_z)).dot(self.A_z_GPR.T) + self.Q_z_GPR

                te = te + self.dt_GPR

                self.te_array[i] = te
                self.me_x_array[i] = me_x[0]
                self.var_x_array[i] = Pe_x[0][0]
                self.me_y_array[i] = me_y[0]
                self.var_y_array[i] = Pe_y[0][0]
                self.me_z_array[i] = me_z[0]
                self.var_z_array[i] = Pe_z[0][0]

    pass