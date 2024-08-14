############################################################
#
#   - Name : MPPI_guidance.py
#
#                   -   Created by E. T. Jeong, 2024.04.12
#
############################################################


#.. Library
# pulbic libs.
import numpy as np
import math as m
import time
import sys, os

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# private libs.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from necessary_settings.quadrotor_iris_parameters import MPPI_Parameter
from necessary_settings.waypoint import Waypoint
from models.quadrotor import Quadrotor_6DOF

#.. MPPI_Guidance_Modules
class MPPI_Guidance_Modules():
    #.. initialize an instance of the class
    def __init__(self, MPPI_Param:MPPI_Parameter) -> None:
        self.MP     =   MPPI_Param
        self.u1     =   self.MP.u1_init * np.ones(self.MP.N)
        self.u2     =   self.MP.u2_init * np.ones(self.MP.N)
        self.u3     =   self.MP.u3_init * np.ones(self.MP.N)
        self.Ai_est_dstb    =   np.zeros((self.MP.N,3))
        pass
    
    def run_MPPI_Guidance(self, QR:Quadrotor_6DOF, WPs:Waypoint):
        
        t0 = time.time()      
        
        #.. variable setting - MPPI Monte Carlo simulation
        # set CPU variables
        arr_u1          =   np.array(self.u1).astype(np.float64)
        arr_u2          =   np.array(self.u2).astype(np.float64)
        arr_u3          =   np.array(self.u3).astype(np.float64)
        arr_delta_u1    =   self.MP.var1*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_delta_u2    =   self.MP.var2*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_delta_u3    =   self.MP.var3*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_stk         =   np.zeros(self.MP.K).astype(np.float64)
        # res_length      =   self.MP.N
        # res_var_num     =   16
        res_length      =   1
        res_var_num     =   1
        arr_res1        =   np.zeros((res_length, res_var_num)).astype(np.float64)
        arr_int_MP      =   np.array([self.MP.K, self.MP.N, self.MP.flag_cost_calc, res_var_num]).astype(np.int32)
        arr_dbl_MP      =   np.array([self.MP.dt_MPPI, self.MP.nu, 
                                      self.MP.R[0], self.MP.Q[0], self.MP.Q[1], self.MP.P[0],
                                      self.MP.u1_init, self.MP.u2_init, self.MP.u3_init, 
                                      self.MP.cost_min_V_aligned]).astype(np.float64)
        arr_int_QR      =   np.array([QR.PF_var.WP_idx_heading, QR.PF_var.WP_idx_passed, QR.GnC_param.Guid_type]).astype(np.int32)
        arr_dbl_QR      =   np.array([QR.state_var.Ri[0], QR.state_var.Ri[1], QR.state_var.Ri[2], 
                                      QR.state_var.Vi[0], QR.state_var.Vi[1], QR.state_var.Vi[2],
                                      QR.state_var.att_ang[0], QR.state_var.att_ang[1], QR.state_var.att_ang[2], 
                                      QR.physical_param.throttle_hover, QR.physical_param.mass, 
                                      QR.GnC_param.desired_speed, QR.GnC_param.virtual_target_distance, 
                                      QR.GnC_param.distance_change_WP, QR.GnC_param.dist_change_first_WP,
                                      QR.GnC_param.Kp_vel, QR.GnC_param.Kd_vel, QR.GnC_param.Kp_speed, QR.GnC_param.Kd_speed, QR.GnC_param.guid_eta, 
                                      QR.GnC_param.tau_phi, QR.GnC_param.tau_the, QR.GnC_param.tau_psi, 
                                      QR.GnC_param.tau_p, QR.GnC_param.tau_q, QR.GnC_param.tau_r,
                                      QR.GnC_param.alpha_p, QR.GnC_param.alpha_q, QR.GnC_param.alpha_r,
                                      QR.guid_var.T_cmd, QR.physical_param.psuedo_rotor_drag_coeff, QR.GnC_param.del_psi_cmd_limit,
                                      QR.GnC_param.tau_Wb
                                      ]).astype(np.float64)
        # print(arr_dbl_QR)
        arr_dbl_WPs     =   np.ravel(WPs,order='C').astype(np.float64)
        arr_dbl_VT      =   np.array([QR.PF_var.VT_Ri[0], QR.PF_var.VT_Ri[1], QR.PF_var.VT_Ri[2]]).astype(np.float64)
        arr_Ai_est_dstb =   np.array(self.Ai_est_dstb).astype(np.float64)
        arr_ent_param_float =   np.array([self.MP.lamb1, self.MP.lamb2, self.MP.lamb3]).astype(np.float64)
        arr_numer1  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_numer2  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_numer3  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom1  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom2  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom3  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        # arr_ent_param_float =   np.array([self.MP.lamb1, self.MP.lamb2, self.MP.lamb3]).astype(np.float64)
        # occupy GPU memory space
        gpu_u1          =   cuda.mem_alloc(arr_u1.nbytes)
        gpu_u2          =   cuda.mem_alloc(arr_u2.nbytes)
        gpu_u3          =   cuda.mem_alloc(arr_u3.nbytes)
        gpu_delta_u1    =   cuda.mem_alloc(arr_delta_u1.nbytes)
        gpu_delta_u2    =   cuda.mem_alloc(arr_delta_u2.nbytes)
        gpu_delta_u3    =   cuda.mem_alloc(arr_delta_u3.nbytes)
        gpu_stk         =   cuda.mem_alloc(arr_stk.nbytes)
        gpu_res1        =   cuda.mem_alloc(arr_res1.nbytes)
        gpu_int_MP      =   cuda.mem_alloc(arr_int_MP.nbytes)
        gpu_dbl_MP      =   cuda.mem_alloc(arr_dbl_MP.nbytes)
        gpu_int_QR      =   cuda.mem_alloc(arr_int_QR.nbytes)
        gpu_dbl_QR      =   cuda.mem_alloc(arr_dbl_QR.nbytes)
        gpu_dbl_WPs     =   cuda.mem_alloc(arr_dbl_WPs.nbytes)
        gpu_dbl_VT      =   cuda.mem_alloc(arr_dbl_VT.nbytes)
        gpu_Ai_est_dstb =   cuda.mem_alloc(arr_Ai_est_dstb.nbytes)
        # gpu_ent_param_float =   cuda.mem_alloc(arr_ent_param_float.nbytes)
        gpu_numer1  =   cuda.mem_alloc(arr_numer1.nbytes)
        gpu_numer2  =   cuda.mem_alloc(arr_numer2.nbytes)
        gpu_numer3  =   cuda.mem_alloc(arr_numer3.nbytes)
        gpu_denom1  =   cuda.mem_alloc(arr_denom1.nbytes)
        gpu_denom2  =   cuda.mem_alloc(arr_denom2.nbytes)
        gpu_denom3  =   cuda.mem_alloc(arr_denom3.nbytes)
        gpu_ent_param_float =   cuda.mem_alloc(arr_ent_param_float.nbytes)
        # convert data memory from CPU to GPU
        cuda.memcpy_htod(gpu_u1,arr_u1)
        cuda.memcpy_htod(gpu_u2,arr_u2)
        cuda.memcpy_htod(gpu_u3,arr_u3)
        cuda.memcpy_htod(gpu_delta_u1,arr_delta_u1)
        cuda.memcpy_htod(gpu_delta_u2,arr_delta_u2)
        cuda.memcpy_htod(gpu_delta_u3,arr_delta_u3)
        cuda.memcpy_htod(gpu_res1,arr_res1)
        cuda.memcpy_htod(gpu_stk,arr_stk)
        cuda.memcpy_htod(gpu_int_MP,arr_int_MP)
        cuda.memcpy_htod(gpu_dbl_MP,arr_dbl_MP)
        cuda.memcpy_htod(gpu_int_QR,arr_int_QR)
        cuda.memcpy_htod(gpu_dbl_QR,arr_dbl_QR)
        cuda.memcpy_htod(gpu_dbl_WPs,arr_dbl_WPs)
        cuda.memcpy_htod(gpu_dbl_VT,arr_dbl_VT)
        cuda.memcpy_htod(gpu_Ai_est_dstb,arr_Ai_est_dstb)
        # cuda.memcpy_htod(gpu_ent_param_float, arr_ent_param_float)
        cuda.memcpy_htod(gpu_numer1, arr_numer1)
        cuda.memcpy_htod(gpu_numer2, arr_numer2)
        cuda.memcpy_htod(gpu_numer3, arr_numer3)
        cuda.memcpy_htod(gpu_denom1, arr_denom1)
        cuda.memcpy_htod(gpu_denom2, arr_denom2)
        cuda.memcpy_htod(gpu_denom3, arr_denom3)
        cuda.memcpy_htod(gpu_ent_param_float, arr_ent_param_float)
        
        #.. run MPPI Monte Carlo simulation code script
        # cuda code script function handler
        # func_MC     =   SourceModule(self.total_MPPI_code).get_function("MPPI_monte_carlo_sim")
        # run cuda script by using GPU cores
        unit_gpu_allocation = 32        # GPU SP number
        # unit_gpu_allocation = 64        # GPU SP number
        # unit_gpu_allocation = 128        # GPU SP number
        blocksz     =   (unit_gpu_allocation, 1, 1)
        gridsz      =   (round(self.MP.K/(unit_gpu_allocation)), 1)
        # gridsz      =   (round(int(np.ceil(256 / unit_gpu_allocation))), 1)
        # blocksz     =   (1, 1, 1)
        # gridsz      =   (1, 1)
        
        t1 = time.time()
        self.func_MC(gpu_u1, gpu_u2, gpu_u3, 
                gpu_delta_u1, gpu_delta_u2, gpu_delta_u3, gpu_stk, 
                gpu_res1,
                gpu_int_MP, gpu_dbl_MP, gpu_int_QR, gpu_dbl_QR, 
                gpu_dbl_WPs, gpu_dbl_VT,gpu_Ai_est_dstb, 
                gpu_numer1,gpu_numer2,gpu_numer3,gpu_denom1,gpu_denom2,gpu_denom3,
                gpu_ent_param_float,
                block=blocksz, grid=gridsz)
        t2 = time.time()

        #.. variable setting - MPPI entropy calculation
        # entropy calc. results
        res_numer1     =   np.empty_like(arr_numer1)
        res_numer2     =   np.empty_like(arr_numer2)
        res_numer3     =   np.empty_like(arr_numer3)
        res_denom1     =   np.empty_like(arr_denom1)
        res_denom2     =   np.empty_like(arr_denom2)
        res_denom3     =   np.empty_like(arr_denom3)
        cuda.memcpy_dtoh(res_numer1, gpu_numer1)
        cuda.memcpy_dtoh(res_numer2, gpu_numer2)
        cuda.memcpy_dtoh(res_numer3, gpu_numer3)
        cuda.memcpy_dtoh(res_denom1, gpu_denom1)
        cuda.memcpy_dtoh(res_denom2, gpu_denom2)
        cuda.memcpy_dtoh(res_denom3, gpu_denom3)
        
        #.. MPPI input calculation
        # entropy
        sum_numer1      =   res_numer1.sum(axis=1)
        sum_numer2      =   res_numer2.sum(axis=1)
        sum_numer3      =   res_numer3.sum(axis=1)
        sum_denom1      =   res_denom1.sum(axis=1)
        sum_denom2      =   res_denom2.sum(axis=1)
        sum_denom3      =   res_denom3.sum(axis=1)
        denom_min       =   np.zeros(np.size(sum_denom1)) + 1.0e-11
        entropy1    =   sum_numer1/np.maximum(sum_denom1, denom_min)
        entropy2    =   sum_numer2/np.maximum(sum_denom2, denom_min)
        entropy3    =   sum_numer3/np.maximum(sum_denom3, denom_min)
        # MPPI input
        self.u1     =   self.u1 + entropy1    
        self.u2     =   self.u2 + entropy2
        self.u3     =   self.u3 + entropy3
        
        # MPPI result and update
        MPPI_ctrl_input    =   np.array([self.u1[0], self.u2[0], self.u3[0]])
        
        self.u1[0:self.MP.N-1]  =   self.u1[1:self.MP.N]
        self.u2[0:self.MP.N-1]  =   self.u2[1:self.MP.N]
        self.u3[0:self.MP.N-1]  =   self.u3[1:self.MP.N]
        self.u1[self.MP.N-1]    =   self.MP.u1_init
        self.u2[self.MP.N-1]    =   self.MP.u2_init
        self.u3[self.MP.N-1]    =   self.MP.u3_init


        t3 = time.time()      
        # MPPI_calc_time = (t2 - t1) 
        MPPI_calc_time = t3 - t0

        return MPPI_ctrl_input, MPPI_calc_time
    
    def set_MPPI_entropy_calc_code(self):
        self.MPPI_entropy_calc_code = """
        __global__ void MPPI_entropy(double* arr_numer1, double* arr_numer2, double* arr_numer3, \
            double* arr_denom1, double* arr_denom2, double* arr_denom3, double *arr_ent_param_float, \
            double* arr_tmp1, double* arr_tmp2,\
            double *arr_delta_u1, double *arr_delta_u2, double *arr_delta_u3, double *arr_stk)
        {            
            // parameters
            double lamb1  =   arr_ent_param_float[0];
            double lamb2  =   arr_ent_param_float[1];
            double lamb3  =   arr_ent_param_float[2];

            // index variables  
            int k       =   threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
            int idx     =   threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;

            // calc num, den
            arr_numer1[idx]     =   exp((-1/lamb1)*arr_stk[k])*arr_delta_u1[idx];
            arr_denom1[idx]     =   exp((-1/lamb1)*arr_stk[k]);
            arr_numer2[idx]     =   exp((-1/lamb2)*arr_stk[k])*arr_delta_u2[idx];
            arr_denom2[idx]     =   exp((-1/lamb2)*arr_stk[k]);
            arr_numer3[idx]     =   exp((-1/lamb3)*arr_stk[k])*arr_delta_u3[idx];
            arr_denom3[idx]     =   exp((-1/lamb3)*arr_stk[k]);
            arr_denom3[idx]     =   exp((-1/lamb3)*arr_stk[k]);
            //arr_tmp1[idx] = arr_stk[k];
            //arr_tmp2[idx] = (-1/lamb1)*arr_stk[k];
        }
        
        """
        self.func_Ent    =   SourceModule(self.MPPI_entropy_calc_code).get_function("MPPI_entropy")
        pass
    
    def set_total_MPPI_code(self, num_WPs):
        self.total_MPPI_code = "#define nWP " + str(num_WPs) +  """
        /*.. Declaire Subfunctions ..*/
        // utility functions
        __device__ double norm_(double x[3]);
        __device__ double dot_(double x[3], double y[3]);
        __device__ void cross_(double x[3], double y[3], double res[3]);
        __device__ void azim_elev_from_vec3(double vec[3], double* azim, double* elev);
        __device__ void DCM_from_euler_angle(double ang_euler321[3], double DCM[3][3]);
        __device__ void matmul_(double mat[3][3], double vec[3], double res[3]);
        __device__ void transpose_(double mat[3][3], double res[3][3]);
        
        // simulation module functions
        __device__ void PF_required_info__distance_to_path(double WP_WPs[nWP][3], int QR_WP_idx_heading, \
            double QR_Ri[3], double QR_point_closest_on_path_i[3], int* QR_WP_idx_passed, double* dist_to_path);
        __device__ void path_following_required_info__check_waypoint(double WP_WPs[nWP][3], \
            int* QR_WP_idx_heading, double QR_Ri[3], double QR_distance_change_WP);
        __device__ void path_following_required_info__VTP_decision(double dist_to_path, \
            double QR_virtual_target_distance, double QR_point_closest_on_path_i[3], int QR_WP_idx_passed,
            double WP_WPs[nWP][3], double PF_var_VT_Ri[3]);
        __device__ void path_following_required_info__takeoff_to_first_WP(double WP_WPs[nWP][3], \
            double QR_Ri[3], int QR_WP_idx_passed, double QR_dist_change_first_WP, double PF_var_VT_Ri[3]);
        __device__ void path_following_required_info__cost_function_1(double R, double u, \
            double Q0, double dist_to_path, double Q1, double Vi[3], \
            double unit_W1W2[3], double min_V_aligned, double cost_arr[3], double dt);
        __device__ void path_following_required_info__terminal_cost_1(double P1, double WP_WPs[nWP][3], \
            int PF_var_init_WP_idx_passed, int PF_var_final_WP_idx_passed,\
            double PF_var_init_point_closest_on_path[3], double PF_var_final_point_closest_on_path[3], \
            double min_move_range, double total_time, double *terminal_cost);
        __device__ void guidance_path_following__guidance_modules(int QR_Guid_type, \
            int QR_WP_idx_passed, int QR_WP_idx_heading, int WP_WPs_shape0, double VT_Ri[3], \
            double QR_Ri[3], double QR_Vi[3], double QR_Ai[3], double QR_desired_speed, double QR_Kp_vel, double QR_Kd_vel, \
            double QR_Kp_speed, double QR_Kd_speed, double QR_guid_eta, double MPPI_ctrl_input[3], double Aqi_cmd[3]);
        __device__ void guidance_path_following__simple_rotor_drag_model(double QR_Vi[3], \
            double psuedo_rotor_drag_coeff, double cB_I[3][3], double Fi_drag[3]);
        __device__ void guidance_path_following__convert_Ai_cmd_to_thrust_and_att_ang_cmd(double Ai_cmd[3], \
            double mass, double T_max, double WP_WPs[nWP][3], int WP_idx_heading, double Ri[3], double att_ang[3], \
            double del_psi_cmd_limit, double* T_cmd, double att_ang_cmd[3]);
        __device__ void controller__attitude_controller(\
            double att_ang_cmd[3], double att_ang[3], double Wb[3], \
            double tau_phi, double tau_the, double tau_psi, double Wb_cmd[3]);
        __device__ void controller__rate_controller(double Wb_cmd[3], double Wb[3], \
            double tau_Wb, double dt_GCU, double err_Wb[3], double int_err_Wb[3]);
        __device__ void dynamics__equations_of_motions(double cI_B[3][3], double cB_I[3][3], \
            double T_cmd, double mass, double Ai_disturbance[3], double Ai_grav[3], \
            double zeta_Wb[3], double omega_Wb[3], double err_Wb[3], double int_err_Wb[3], double Wb[3], double Vb[3], double att_ang[3], \
            double dot_Ri[3], double dot_Vb[3], double dot_att_ang[3], double dot_Wb[3]);
        __device__ void update_states(double dot_Ri[3], double dot_Vb[3], double dot_att_ang[3], double dot_Wb[3], double dt, \
            double Ri[3], double Vb[3], double att_ang[3], double Wb[3], double cI_B[3][3], double cB_I[3][3], double Vi[3]);
        
        /*.. main function ..*/    
        __global__ void MPPI_monte_carlo_sim(double* arr_u1, double* arr_u2, double* arr_u3, \
            double* arr_delta_u1, double* arr_delta_u2, double* arr_delta_u3, double* arr_stk, \
            double* arr_res1, \
            int* arr_int_MP, double* arr_dbl_MP, int* arr_int_QR, double* arr_dbl_QR, \
            double* arr_dbl_WPs, double* arr_dbl_VT, double* arr_Ai_est_dstb, \
            double* arr_numer1, double* arr_numer2, double* arr_numer3, \
            double* arr_denom1, double* arr_denom2, double* arr_denom3, double *arr_ent_param_float)
        {
            //.. GPU core index for parallel computation
            int idx     =   threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;
            double m_pi =   acos(-1.);

            /*.. declare variables ..*/
            //.. set MPPI variables
            int MP_K = arr_int_MP[0];
            int MP_N = arr_int_MP[1];
            int MP_flag_cost_calc = arr_int_MP[2];
            int res_var_num = arr_int_MP[3];
            double MP_dt = arr_dbl_MP[0];
            double MP_nu = arr_dbl_MP[1]; 
            double MP_R = arr_dbl_MP[2]; 
            double MP_Q[2] = {arr_dbl_MP[3], arr_dbl_MP[4]};
            double MP_P = arr_dbl_MP[5];
            double MP_u1_init = arr_dbl_MP[6]; 
            double MP_u2_init = arr_dbl_MP[7]; 
            double MP_u3_init = arr_dbl_MP[8];
            double MP_cost_min_V_aligned = arr_dbl_MP[9];
            
            double times_N = 1.0;
            double modif_dt = MP_dt / times_N;
            
            //.. set QR variables
            int PF_var_WP_idx_heading = arr_int_QR[0];
            int PF_var_WP_idx_passed = arr_int_QR[1];
            int GnC_param_Guid_type = arr_int_QR[2];
            
            double state_var_Ri[3] = {arr_dbl_QR[0], arr_dbl_QR[1], arr_dbl_QR[2]};
            double state_var_Vi[3] = {arr_dbl_QR[3], arr_dbl_QR[4], arr_dbl_QR[5]};
            double state_var_att_ang[3] = {arr_dbl_QR[6], arr_dbl_QR[7], arr_dbl_QR[8]};
            double state_var_Ai[3] = {0.,};
            double state_var_Wb[3] = {0.,};
            
            double physical_param_throttle_hover= arr_dbl_QR[9]; 
            double physical_param_mass = arr_dbl_QR[10]; 
            double GnC_param_desired_speed= arr_dbl_QR[11]; 
            double GnC_param_virtual_target_distance= arr_dbl_QR[12]; 
            double GnC_param_distance_change_WP= arr_dbl_QR[13]; 
            double GnC_param_dist_change_first_WP= arr_dbl_QR[14]; 
            double GnC_param_Kp_vel= arr_dbl_QR[15]; 
            double GnC_param_Kd_vel= arr_dbl_QR[16]; 
            double GnC_param_Kp_speed= arr_dbl_QR[17]; 
            double GnC_param_Kd_speed= arr_dbl_QR[18]; 
            double GnC_param_guid_eta= arr_dbl_QR[19]; 
            double GnC_param_tau_phi= arr_dbl_QR[20]; 
            double GnC_param_tau_the= arr_dbl_QR[21]; 
            double GnC_param_tau_psi= arr_dbl_QR[22]; 
            double GnC_param_tau_p= arr_dbl_QR[23]; 
            double GnC_param_tau_q= arr_dbl_QR[24]; 
            double GnC_param_tau_r= arr_dbl_QR[25]; 
            double GnC_param_alpha_p= arr_dbl_QR[26]; 
            double GnC_param_alpha_q= arr_dbl_QR[27]; 
            double GnC_param_alpha_r= arr_dbl_QR[28]; 
            double guid_var_T_cmd = arr_dbl_QR[29]; 
            double physical_param_psuedo_rotor_drag_coeff = arr_dbl_QR[30]; 
            double GnC_param_del_psi_cmd_limit = arr_dbl_QR[31];
            double GnC_param_tau_Wb = arr_dbl_QR[32];
                        
            //.. set waypoints
            double WP_WPs[nWP][3]   =   {0.,};
            for(int i_WP = 0; i_WP < nWP; i_WP++){
                for(int i = 0; i < 3; i++){
                    WP_WPs[i_WP][i] = arr_dbl_WPs[i_WP*3 + i];
                }
            }
            
            //.. set others
            double env_var_grav = 9.81;
            double Ai_grav[3] = {0., 0., env_var_grav};
            double physical_param_T_max = physical_param_mass * env_var_grav / physical_param_throttle_hover; 
            double PF_var_point_closest_on_path_i[3] = {0.,};
            double PF_var_dist_to_path = 0.;
            double PF_var_VT_Ri[3] = {0.,};
            double PF_var_cost_arr[3] = {0.,};
            int PF_var_init_WP_idx_passed = 0;
            int PF_var_final_WP_idx_passed = 0;
            double PF_var_init_point_closest_on_path[3] = {0.,};
            double PF_var_final_point_closest_on_path[3] = {0.,};
            double PF_var_init_time = 0.;
            double PF_var_final_time = MP_dt * MP_N;
            double guid_var_Ai_cmd[3] = {0.,};
            double cI_B[3][3]; DCM_from_euler_angle(state_var_att_ang, cI_B);
            double cB_I[3][3]; transpose_(cI_B, cB_I);
            double Fi_drag[3] = {0.,};
            double guid_var_Ai_rotor_drag[3] = {0.,};
            double guid_var_Ai_disturbance[3] = {0.,};
            double guid_var_Ai_cmd_compensated[3] = {0.,};
            double guid_var_att_ang_cmd[3] = {0.,};
            
            double state_var_Vb[3] = {0.,}; matmul_(cI_B, state_var_Vi, state_var_Vb);
            double ctrl_var_err_Wb[3] = {0.,};
            double ctrl_var_Wb_cmd[3] = {0.,};
            double ctrl_var_int_err_Wb[3] = {0.,};
            double dyn_var_dot_Ri[3] = {0.,};
            double dyn_var_dot_Vb[3] = {0.,};
            double dyn_var_dot_att_ang[3] = {0.,};
            double dyn_var_dot_Wb[3] = {0.,};
            
            
            // set rate controller parameters
            double zeta_Wb[3] = {0.,};
            double omega_Wb[3] = {0.,};
            zeta_Wb[0] = 0.5*sqrt(GnC_param_alpha_p/GnC_param_tau_p);
            zeta_Wb[1] = 0.5*sqrt(GnC_param_alpha_q/GnC_param_tau_q);
            zeta_Wb[2] = 0.5*sqrt(GnC_param_alpha_r/GnC_param_tau_r);
            omega_Wb[0] = sqrt(1/(GnC_param_alpha_p*GnC_param_tau_p));
            omega_Wb[1] = sqrt(1/(GnC_param_alpha_q*GnC_param_tau_q));
            omega_Wb[2] = sqrt(1/(GnC_param_alpha_r*GnC_param_tau_r));
            
            
            //.. main loop
            int i_N = 0;
            for(i_N = 0; i_N < MP_N; i_N++){
                    
                //.. MPPI modules - checked
                double MPPI_ctrl_input[3] = {0.,};
                if(GnC_param_Guid_type >= 2){
                    MPPI_ctrl_input[0] = arr_u1[i_N] + arr_delta_u1[idx + MP_K*i_N];
                    MPPI_ctrl_input[1] = arr_u2[i_N] + arr_delta_u2[idx + MP_K*i_N];
                    MPPI_ctrl_input[2] = arr_u3[i_N] + arr_delta_u3[idx + MP_K*i_N];
                }
                
                if (PF_var_WP_idx_heading == nWP-1){
                    GnC_param_Guid_type = 1;
                    GnC_param_virtual_target_distance = 4;
                    GnC_param_desired_speed = 2;
                    GnC_param_guid_eta = 2;
                }
                
                //.. Environment - checked
                double Ai_disturbance[3]; for(int i=0;i<3;i++) Ai_disturbance[i] = arr_Ai_est_dstb[i + 3*i_N];
                
                //.. Path-Following-required information - checked
                PF_required_info__distance_to_path(WP_WPs, PF_var_WP_idx_heading, \
                    state_var_Ri, PF_var_point_closest_on_path_i, &PF_var_WP_idx_passed, &PF_var_dist_to_path);
                path_following_required_info__check_waypoint(WP_WPs, \
                    &PF_var_WP_idx_heading, state_var_Ri, GnC_param_distance_change_WP);
                
                //.. VT is not required to Guid_type 2
                if(GnC_param_Guid_type != 2){
                    path_following_required_info__VTP_decision(PF_var_dist_to_path, \
                        GnC_param_virtual_target_distance, PF_var_point_closest_on_path_i, PF_var_WP_idx_passed,
                        WP_WPs, PF_var_VT_Ri);
                    path_following_required_info__takeoff_to_first_WP(WP_WPs, \
                        state_var_Ri, PF_var_WP_idx_passed, GnC_param_dist_change_first_WP, PF_var_VT_Ri);
                }
                    
                double u = guid_var_T_cmd/physical_param_mass;
                double Rw1w2[3];  for(int i=0;i<3;i++) Rw1w2[i] = WP_WPs[PF_var_WP_idx_heading][i] - WP_WPs[PF_var_WP_idx_passed][i];
                double PF_var_unit_Rw1w2[3]; for(int i=0;i<3;i++) PF_var_unit_Rw1w2[i] = Rw1w2[i]/norm_(Rw1w2);
                path_following_required_info__cost_function_1(MP_R, u, MP_Q[0], PF_var_dist_to_path, \
                    MP_Q[1], state_var_Vi, PF_var_unit_Rw1w2, MP_cost_min_V_aligned, PF_var_cost_arr, MP_dt);
                    
                // for terminal cost
                if(i_N == 0){
                    for(int i=0;i<3;i++) PF_var_init_point_closest_on_path[i] = PF_var_point_closest_on_path_i[i];
                    PF_var_init_WP_idx_passed = PF_var_WP_idx_passed;
                }
                
                
                //.. skip this for terminal WP phase command
                /*
                if ( (PF_var_WP_idx_passed == 0) || (PF_var_WP_idx_heading == nWP - 1) )
                    for(int i=0;i<3;i++) PF_var_cost_arr[i] = 0.;
                */
                
                //.. Guidance - checked
                guidance_path_following__guidance_modules(GnC_param_Guid_type, \
                    PF_var_WP_idx_passed, PF_var_WP_idx_heading, (int)nWP, PF_var_VT_Ri, \
                    state_var_Ri, state_var_Vi, state_var_Ai, GnC_param_desired_speed, GnC_param_Kp_vel, GnC_param_Kd_vel, \
                    GnC_param_Kp_speed, GnC_param_Kd_speed, GnC_param_guid_eta, MPPI_ctrl_input, guid_var_Ai_cmd);
                // calc. simple rotor drag model
                guidance_path_following__simple_rotor_drag_model(state_var_Vi, physical_param_psuedo_rotor_drag_coeff, cB_I, Fi_drag);
                for(int i=0;i<3;i++) guid_var_Ai_rotor_drag[i] = Fi_drag[i]/physical_param_mass;
                // compensate disturbance
                for(int i=0;i<3;i++) guid_var_Ai_disturbance[i] = Ai_disturbance[i] + guid_var_Ai_rotor_drag[i];
                for(int i=0;i<3;i++) guid_var_Ai_cmd_compensated[i] = guid_var_Ai_cmd[i] - guid_var_Ai_disturbance[i];
                // compensate gravity
                guid_var_Ai_cmd_compensated[2] = guid_var_Ai_cmd_compensated[2] - env_var_grav;
                // convert_Ai_cmd_to_thrust_and_att_ang_cmd
                guidance_path_following__convert_Ai_cmd_to_thrust_and_att_ang_cmd(guid_var_Ai_cmd_compensated, \
                    physical_param_mass, physical_param_T_max, WP_WPs, PF_var_WP_idx_heading, state_var_Ri, state_var_att_ang, \
                    GnC_param_del_psi_cmd_limit, &guid_var_T_cmd, guid_var_att_ang_cmd);
                    
                    
                for(int i_times_N = 0; i_times_N < (int)times_N; i_times_N++){
                    //------- Start - dynamics and integration --------//
                    //.. Controller - checked
                    controller__attitude_controller(guid_var_att_ang_cmd, state_var_att_ang, state_var_Wb, \
                        GnC_param_tau_phi, GnC_param_tau_the, GnC_param_tau_psi, ctrl_var_Wb_cmd);
                    controller__rate_controller(ctrl_var_Wb_cmd, state_var_Wb, GnC_param_tau_Wb, \
                        modif_dt, ctrl_var_err_Wb, ctrl_var_int_err_Wb);
                    
                    //.. Dynamics - checked
                    dynamics__equations_of_motions(cI_B, cB_I, guid_var_T_cmd, physical_param_mass, guid_var_Ai_disturbance, Ai_grav, \
                        zeta_Wb, omega_Wb, ctrl_var_err_Wb, ctrl_var_int_err_Wb, state_var_Wb, state_var_Vb, state_var_att_ang, \
                        dyn_var_dot_Ri, dyn_var_dot_Vb, dyn_var_dot_att_ang, dyn_var_dot_Wb);
                    
                    //.. save data - skip
                    
                    //.. update_states - checked
                    update_states(dyn_var_dot_Ri, dyn_var_dot_Vb, dyn_var_dot_att_ang, dyn_var_dot_Wb, modif_dt, \
                        state_var_Ri, state_var_Vb, state_var_att_ang, state_var_Wb, cI_B, cB_I, state_var_Vi);
                    //-------  End  - dynamics and integration --------//
                }
                
                //.. stop - when arrive the terminal WP - skip
                
                //.. MPPI cost - checked
                double cost_sum = 0.;
                for(int i=0;i<3;i++) cost_sum = cost_sum + PF_var_cost_arr[i];
                arr_stk[idx]    =   arr_stk[idx] + cost_sum;
            }
            
            // terminal cost function            
            PF_var_final_time = MP_dt * i_N;
            for(int i=0;i<3;i++) PF_var_final_point_closest_on_path[i] = PF_var_point_closest_on_path_i[i];
            PF_var_final_WP_idx_passed = PF_var_WP_idx_passed;
            double total_time = PF_var_final_time - PF_var_init_time;
            double min_move_range = MP_cost_min_V_aligned*MP_N*MP_dt;
            double terminal_cost = 0.;
            path_following_required_info__terminal_cost_1(MP_P, WP_WPs, \
                PF_var_init_WP_idx_passed, PF_var_final_WP_idx_passed,\
                PF_var_init_point_closest_on_path, PF_var_final_point_closest_on_path, \
                min_move_range, total_time, &terminal_cost);
            
            arr_stk[idx] = arr_stk[idx] + terminal_cost;
            
            
            //.. MPPI_entropy
            // parameters
            double lamb1  =   arr_ent_param_float[0];
            double lamb2  =   arr_ent_param_float[1];
            double lamb3  =   arr_ent_param_float[2];
            i_N = 0;
            for(i_N = 0; i_N < MP_N; i_N++){
                arr_numer1[idx + MP_K*i_N] = exp((-1/lamb1)*arr_stk[idx])*arr_delta_u1[idx + MP_K*i_N];
                arr_denom1[idx + MP_K*i_N] = exp((-1/lamb1)*arr_stk[idx]);
                arr_numer2[idx + MP_K*i_N] = exp((-1/lamb2)*arr_stk[idx])*arr_delta_u2[idx + MP_K*i_N];
                arr_denom2[idx + MP_K*i_N] = exp((-1/lamb2)*arr_stk[idx]);
                arr_numer3[idx + MP_K*i_N] = exp((-1/lamb3)*arr_stk[idx])*arr_delta_u3[idx + MP_K*i_N];
                arr_denom3[idx + MP_K*i_N] = exp((-1/lamb3)*arr_stk[idx]);
                arr_denom3[idx + MP_K*i_N] = exp((-1/lamb3)*arr_stk[idx]);
            }
        }
        
        // utility functions
        __device__ double norm_(double x[3])
        {
            return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        }
        __device__ double dot_(double x[3], double y[3]) 
        {
            return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
        }
        __device__ void cross_(double x[3], double y[3], double res[3]) 
        {
            res[0] = x[1]*y[2] - x[2]*y[1];
            res[1] = x[2]*y[0] - x[0]*y[2];
            res[2] = x[0]*y[1] - x[1]*y[0];
        }
        __device__ void azim_elev_from_vec3(double vec[3], double* azim, double* elev)
        {
            azim[0]     =   atan2(vec[1],vec[0]);
            elev[0]     =   atan2(-vec[2], sqrt(vec[0]*vec[0]+vec[1]*vec[1]));
        }
        __device__ void DCM_from_euler_angle(double ang_euler321[3], double DCM[3][3])
        {
            double spsi     =   sin( ang_euler321[2] );
            double cpsi     =   cos( ang_euler321[2] );
            double sthe     =   sin( ang_euler321[1] );
            double cthe     =   cos( ang_euler321[1] );
            double sphi     =   sin( ang_euler321[0] );
            double cphi     =   cos( ang_euler321[0] );

            DCM[0][0]       =   cpsi * cthe ;
            DCM[1][0]       =   cpsi * sthe * sphi - spsi * cphi ;
            DCM[2][0]       =   cpsi * sthe * cphi + spsi * sphi ;
            
            DCM[0][1]       =   spsi * cthe ;
            DCM[1][1]       =   spsi * sthe * sphi + cpsi * cphi ;
            DCM[2][1]       =   spsi * sthe * cphi - cpsi * sphi ;
            
            DCM[0][2]       =   -sthe ;
            DCM[1][2]       =   cthe * sphi ;
            DCM[2][2]       =   cthe * cphi ;
        }
        __device__ void matmul_(double mat[3][3], double vec[3], double res[3])
        {
            res[0]  =   mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2];
            res[1]  =   mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2];
            res[2]  =   mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2];
        }
        __device__ void transpose_(double mat[3][3], double res[3][3])
        {
            res[0][0]   =   mat[0][0];
            res[0][1]   =   mat[1][0];
            res[0][2]   =   mat[2][0];
            res[1][0]   =   mat[0][1];
            res[1][1]   =   mat[1][1];
            res[1][2]   =   mat[2][1];
            res[2][0]   =   mat[0][2];
            res[2][1]   =   mat[1][2];
            res[2][2]   =   mat[2][2];
        }


        // simulation module functions
        __device__ void PF_required_info__distance_to_path(double WP_WPs[nWP][3], int QR_WP_idx_heading, \
            double QR_Ri[3], double QR_point_closest_on_path_i[3], int* QR_WP_idx_passed, double* dist_to_path)
        {
            // calc. variables
            dist_to_path[0] = 999999.;
            for(int i_WP = QR_WP_idx_heading; i_WP > 0; i_WP--)
            {
                double Rw1w2[3]; for(int i=0;i<3;i++) Rw1w2[i] = WP_WPs[i_WP][i] - WP_WPs[i_WP-1][i];
                double mag_Rw1w2 = norm_(Rw1w2);
                double Rw1q[3]; for(int i=0;i<3;i++) Rw1q[i] = QR_Ri[i] - WP_WPs[i_WP-1][i];
                double mag_w1p = min(max(dot_(Rw1w2, Rw1q)/max(mag_Rw1w2,0.001), 0.), mag_Rw1w2);
                double p_closest_on_path[3]; for(int i=0;i<3;i++) p_closest_on_path[i] = WP_WPs[i_WP-1][i] + mag_w1p * Rw1w2[i]/max(mag_Rw1w2,0.001);
                double tmp[3]; for(int i=0;i<3;i++) tmp[i] = p_closest_on_path[i] - QR_Ri[i];
                double mag_Rqp = norm_(tmp);
                if(dist_to_path[0] < mag_Rqp){
                    break;
                }else{
                    dist_to_path[0] = mag_Rqp;
                    for(int i=0;i<3;i++) QR_point_closest_on_path_i[i] = p_closest_on_path[i];
                    QR_WP_idx_passed[0] = max(i_WP-1, 0);
                }
            }
        }
        __device__ void path_following_required_info__check_waypoint(double WP_WPs[nWP][3], \
            int* QR_WP_idx_heading, double QR_Ri[3], double QR_distance_change_WP)
        {
            double Rqw2i[3]; for(int i=0;i<3;i++) Rqw2i[i] = WP_WPs[QR_WP_idx_heading[0]][i] - QR_Ri[i];
            double mag_Rqw2i = norm_(Rqw2i);
            if(mag_Rqw2i < QR_distance_change_WP){
                QR_WP_idx_heading[0] = min(QR_WP_idx_heading[0] + 1, nWP - 1);
            }
        }
        __device__ void path_following_required_info__VTP_decision(double dist_to_path, \
            double QR_virtual_target_distance, double QR_point_closest_on_path_i[3], int QR_WP_idx_passed,
            double WP_WPs[nWP][3], double PF_var_VT_Ri[3])
        {
            if(dist_to_path >= QR_virtual_target_distance){
                for(int i=0;i<3;i++) PF_var_VT_Ri[i] = QR_point_closest_on_path_i[i];
            }else{
                double total_len = dist_to_path;
                double p1[3]; for(int i=0;i<3;i++) p1[i] = QR_point_closest_on_path_i[i];
                for (int i_WP = QR_WP_idx_passed+1; i_WP < nWP; i_WP++){
                    // check segment whether Rti exist
                    double p2[3]; for(int i=0;i<3;i++) p2[i] = WP_WPs[i_WP][i];
                    double Rp1p2[3]; for(int i=0;i<3;i++) Rp1p2[i] = p2[i] - p1[i];
                    double mag_Rp1p2 = norm_(Rp1p2);
                    if (total_len + mag_Rp1p2 > QR_virtual_target_distance){
                        double mag_Rp1t = QR_virtual_target_distance - total_len;
                        for(int i=0;i<3;i++) PF_var_VT_Ri[i] = p1[i] + mag_Rp1t * Rp1p2[i]/max(mag_Rp1p2,0.001);
                        break;
                    }else{
                        for(int i=0;i<3;i++) p1[i] = p2[i];
                        total_len = total_len + mag_Rp1p2;
                        if (i_WP == nWP - 1)
                            for(int i=0;i<3;i++) PF_var_VT_Ri[i] = p2[i];
                    }
                }
            }
        }
        __device__ void path_following_required_info__takeoff_to_first_WP(double WP_WPs[nWP][3], \
            double QR_Ri[3], int QR_WP_idx_passed, double QR_dist_change_first_WP, double PF_var_VT_Ri[3])
        {
            if(QR_WP_idx_passed < 1){
                double dist_to_WP[3]; for(int i=0; i<3; i++) dist_to_WP[i] = WP_WPs[1][i] - QR_Ri[i];
                if(norm_(dist_to_WP) > QR_dist_change_first_WP){
                    for(int i=0; i<3; i++) PF_var_VT_Ri[i] = WP_WPs[1][i];
                }
            }
        }
        __device__ void path_following_required_info__cost_function_1(double R, double u, \
            double Q0, double dist_to_path, double Q1, double Vi[3], \
            double unit_W1W2[3], double min_V_aligned, double cost_arr[3], double dt)
        {
            // uRu of LQR cost, set low value of norm(R)
            double uRu = u * R * u ; 
            
            // path following performance
            double x0 = dist_to_path;
            //double x0Q0x0 = x0 * Q0 * x0;
            double x0Q0x0 = x0 * Q0 * x0 * x0 * x0 * x0 * x0;
            
            // energy consumption efficiency 
            //double V_aligned = max(dot_(unit_W1W2, Vi), min_V_aligned);
            //double V_aligned = max(dot_(Vi, Vi), min_V_aligned);
            double V_aligned = exp(0.5 * dot_(unit_W1W2, Vi)) + min_V_aligned;
            double x1 = 1. / V_aligned;
            double x1Q1x1 = x1 * Q1;
            //double x1Q1x1 = x1 * Q1 * x1;
            
            // total cost
            cost_arr[0] = uRu * dt;
            cost_arr[1] = x0Q0x0 * dt;
            cost_arr[2] = x1Q1x1 * dt;
        }
        __device__ void path_following_required_info__terminal_cost_1(double P1, double WP_WPs[nWP][3], \
            int PF_var_init_WP_idx_passed, int PF_var_final_WP_idx_passed,\
            double PF_var_init_point_closest_on_path[3], double PF_var_final_point_closest_on_path[3], \
            double min_move_range, double total_time, double *terminal_cost)
        {
            // calc. init remained range
            double init_remianed_range = 0.;
            double rel_pos[3] = {0.,};
            for(int i_WP = PF_var_init_WP_idx_passed; i_WP<nWP-1; i_WP++){
                for(int i=0;i<3;i++) rel_pos[i] = WP_WPs[i_WP+1][i] - WP_WPs[i_WP][i];
                init_remianed_range = init_remianed_range + norm_(rel_pos);
            }
            for(int i=0;i<3;i++) rel_pos[i] = PF_var_init_point_closest_on_path[i] - WP_WPs[PF_var_init_WP_idx_passed][i];
            init_remianed_range = init_remianed_range - norm_(rel_pos);
            
            // calc. final remained range
            double final_remianed_range = 0.;
            for(int i_WP = PF_var_final_WP_idx_passed; i_WP<nWP-1; i_WP++){
                for(int i=0;i<3;i++) rel_pos[i] = WP_WPs[i_WP+1][i] - WP_WPs[i_WP][i];
                final_remianed_range = final_remianed_range + norm_(rel_pos);
            }
            for(int i=0;i<3;i++) rel_pos[i] = PF_var_final_point_closest_on_path[i] - WP_WPs[PF_var_final_WP_idx_passed][i];
            final_remianed_range = final_remianed_range - norm_(rel_pos);
            
            // calc. move range
            double move_range = init_remianed_range - final_remianed_range;
            
            // terminal cost
            terminal_cost[0] = P1 * total_time / max(move_range, min_move_range);
        }
        __device__ void guidance_path_following__guidance_modules(int QR_Guid_type, \
            int QR_WP_idx_passed, int QR_WP_idx_heading, int WP_WPs_shape0, double VT_Ri[3], \
            double QR_Ri[3], double QR_Vi[3], double QR_Ai[3], double QR_desired_speed, double QR_Kp_vel, double QR_Kd_vel, \
            double QR_Kp_speed, double QR_Kd_speed, double QR_guid_eta, double MPPI_ctrl_input[3], double Aqi_cmd[3])
        {
            // starting phase
            if (QR_WP_idx_passed < 1){
                QR_Guid_type = 0;
                QR_desired_speed = 0.5;
            }
            // terminal phase
            if (QR_WP_idx_heading == (WP_WPs_shape0 - 1)){
                QR_Guid_type = 0;
            }
            
            // guidance command
            if ( (QR_Guid_type == 0) || (QR_Guid_type == 4) ){
                if (QR_Guid_type == 4){
                    QR_desired_speed = MPPI_ctrl_input[1];
                    QR_Kp_vel = MPPI_ctrl_input[2];
                }
                //.. guidance - position & velocity control
                // position control
                double err_Ri[3]; for(int i=0;i<3;i++) err_Ri[i] = VT_Ri[i] - QR_Ri[i];
                double Kp_pos = QR_desired_speed/max(norm_(err_Ri),QR_desired_speed); // (terminal WP, tgo < 1) --> decreasing speed
                double derr_Ri[3]; for(int i=0;i<3;i++) derr_Ri[i] = 0. - QR_Vi[i];
                double Vqi_cmd[3]; for(int i=0;i<3;i++) Vqi_cmd[i] = Kp_pos * err_Ri[i];
                double dVqi_cmd[3]; for(int i=0;i<3;i++) dVqi_cmd[i] = Kp_pos * derr_Ri[i];
                // velocity control
                double err_Vi[3]; for(int i=0;i<3;i++) err_Vi[i] = Vqi_cmd[i] - QR_Vi[i];
                double derr_Vi[3]; for(int i=0;i<3;i++) derr_Vi[i] = dVqi_cmd[i] - QR_Ai[i];
                for(int i=0;i<3;i++) Aqi_cmd[i] = QR_Kp_vel * err_Vi[i] + QR_Kd_vel * derr_Vi[i];
            }
            else if ( (QR_Guid_type == 1) || (QR_Guid_type == 3) ){
                if (QR_Guid_type == 3){
                    QR_desired_speed = MPPI_ctrl_input[1];
                    QR_guid_eta = MPPI_ctrl_input[2];
                }
                // calc. variables
                double QR_mag_Vi = norm_(QR_Vi);
                double FPA_azim, FPA_elev; azim_elev_from_vec3(QR_Vi, &FPA_azim, &FPA_elev);
                double FPA_euler[3] = {0., FPA_elev, FPA_azim};
                double QR_cI_W[3][3]; DCM_from_euler_angle(FPA_euler, QR_cI_W);
                //.. guidance - GL - parameters by MPPI
                double Aqw_cmd[3];
                // a_x command
                double err_mag_V = QR_desired_speed - QR_mag_Vi;
                double dQR_mag_Vi = dot_(QR_Vi, QR_Ai) / max(QR_mag_Vi, 0.1);
                double derr_mag_V = 0. - dQR_mag_Vi;
                Aqw_cmd[0] = QR_Kp_speed * err_mag_V + QR_Kd_speed * derr_mag_V;
                // pursuit guidance law
                double Rqti[3]; for(int i=0;i<3;i++) Rqti[i] = VT_Ri[i] - QR_Ri[i];
                double Rqtw[3]; matmul_(QR_cI_W, Rqti, Rqtw);
                double err_azim, err_elev; azim_elev_from_vec3(Rqtw, &err_azim, &err_elev);
                Aqw_cmd[1]  =   QR_guid_eta* 3. / 1.5 * err_azim;
                Aqw_cmd[2]  =   -QR_guid_eta* 3. / 1.5 * err_elev;
                // command coordinate change
                double cW_I[3][3]; transpose_(QR_cI_W, cW_I);
                matmul_(cW_I, Aqw_cmd, Aqi_cmd);
            }
            else if (QR_Guid_type == 2)
                for(int i=0;i<3;i++) Aqi_cmd[i] = MPPI_ctrl_input[i];
        }
        __device__ void guidance_path_following__simple_rotor_drag_model(double QR_Vi[3], \
            double psuedo_rotor_drag_coeff, double cB_I[3][3], double Fi_drag[3])
        {
            double joint_axis_b[3] = {0., 0., -1.};
            double joint_axis_i[3]; matmul_(cB_I, joint_axis_b, joint_axis_i);
            double tmp_value = dot_(QR_Vi, joint_axis_i);
            double velocity_parallel_to_rotor_axis[3]; 
            for(int i=0;i<3;i++) velocity_parallel_to_rotor_axis[i] = tmp_value* joint_axis_i[i];
            double velocity_perpendicular_to_rotor_axis[3]; 
            for(int i=0;i<3;i++) velocity_perpendicular_to_rotor_axis[i] = QR_Vi[i] - velocity_parallel_to_rotor_axis[i];
            for(int i=0;i<3;i++) Fi_drag[i] = - psuedo_rotor_drag_coeff * velocity_perpendicular_to_rotor_axis[i];
        }
        __device__ void guidance_path_following__convert_Ai_cmd_to_thrust_and_att_ang_cmd(double Ai_cmd[3], \
            double mass, double T_max, double WP_WPs[nWP][3], int WP_idx_heading, double Ri[3], double att_ang[3], \
            double del_psi_cmd_limit, double* T_cmd, double att_ang_cmd[3])
        {
            double pi = acos(-1.);
            
            // thrust cmd
            double norm_Ai_cmd = norm_(Ai_cmd);
            T_cmd[0] = min(norm_Ai_cmd * mass, T_max);
            double mag_Ai_cmd = T_cmd[0] / mass;
            for(int i=0;i<3;i++) Ai_cmd[i] =  Ai_cmd[i] / norm_Ai_cmd * mag_Ai_cmd;
            
            // attitude angle cmd
            double WP_heading[3]; for(int i=0;i<3;i++) WP_heading[i] = WP_WPs[WP_idx_heading][i];
            double Rqwi[3]; for(int i=0;i<3;i++) Rqwi[i] = WP_heading[i] - Ri[i];
            double psi_des, tmp; 
            if (WP_idx_heading < nWP-1){
                azim_elev_from_vec3(Rqwi, &psi_des, &tmp); // toward to the heading waypoint
            }else{
                int WP_idx_passed = max(WP_idx_heading - 1, 0);
                double WP_passed[3]; for(int i=0;i<3;i++) WP_passed[i] = WP_WPs[WP_idx_passed][i];
                double WP12[3]; for(int i=0;i<3;i++) WP12[i] = WP_heading[i] - WP_passed[i];
                azim_elev_from_vec3(WP12, &psi_des, &tmp);
            }
            
            // att_ang_cmd -  del_psi_cmd limitation
            double del_psi = psi_des - att_ang[2];
            if (abs(del_psi) > 1.0*pi){
                if (psi_des > att_ang[2])
                    psi_des = psi_des - 2.*pi;
                else
                    psi_des = psi_des + 2.*pi;
            }
            del_psi = max(min(psi_des - att_ang[2], del_psi_cmd_limit), -del_psi_cmd_limit);
            psi_des = att_ang[2] + del_psi;
            
            double euler_psi[3] = {0., 0., psi_des};
            double mat_psi[3][3]; DCM_from_euler_angle(euler_psi, mat_psi);
            double Apsi_cmd[3]; matmul_(mat_psi , Ai_cmd, Apsi_cmd);
            att_ang_cmd[0] = asin(Apsi_cmd[1]/mag_Ai_cmd);
            double sintheta = min(max(-Apsi_cmd[0]/cos(att_ang_cmd[0])/mag_Ai_cmd, -1.), 1.);
            att_ang_cmd[1] = asin(sintheta);
            att_ang_cmd[2] = psi_des;
        }
        __device__ void controller__attitude_controller(\
            double att_ang_cmd[3], double att_ang[3], double Wb[3], \
            double tau_phi, double tau_the, double tau_psi, double Wb_cmd[3])
        {
            double pi = acos(-1.);
            // yaw continuity
            if (abs(att_ang_cmd[2] - att_ang[2]) > 1.0*pi){
                if (att_ang_cmd[2] > att_ang[2]){
                    att_ang_cmd[2] = att_ang_cmd[2] - 2.*pi;
                }else{
                    att_ang_cmd[2] = att_ang_cmd[2] + 2.*pi;
                }
            }
            
            // desired error dynamics
            double desired_dot_att_angle[3] = {0.,};
            desired_dot_att_angle[0] =  1.0 / tau_phi * ( att_ang_cmd[0] - att_ang[0] );
            desired_dot_att_angle[1] =  1.0 / tau_the * ( att_ang_cmd[1] - att_ang[1] );
            desired_dot_att_angle[2] =  1.0 / tau_psi * ( att_ang_cmd[2] - att_ang[2] );
            
            // at first step, assume the initial Wb
            if ((Wb[0] == 0.) && (Wb[1] == 0.) && (Wb[2] == 0.)){
                Wb[0] = desired_dot_att_angle[0] - desired_dot_att_angle[2]*sin(att_ang[1]);
                Wb[1] = desired_dot_att_angle[1]*cos(att_ang[0]) + desired_dot_att_angle[2]*sin(att_ang[0])*cos(att_ang[1]);
                Wb[2] = -desired_dot_att_angle[1]*sin(att_ang[0]) + desired_dot_att_angle[2]*cos(att_ang[0])*cos(att_ang[1]);
            }
            
            
            // att_angle
            double cthe =   cos( att_ang[1] )   ;
            double sthe =   1.0 / cthe   ;
            double tthe =   tan( att_ang[1] )   ;
            double sphi =   sin( att_ang[0] )   ;
            double cphi =   cos( att_ang[0] )   ;
            double tphi =   tan( att_ang[0] )   ;
            
            // Outer Loop: Kinematic Relationship beween Euler Angle and Body Rate   
            double p_trim =   - Wb[1] * sphi * tthe - Wb[2] * cphi * tthe   ;
            Wb_cmd[0] =   desired_dot_att_angle[0] + p_trim;

            double q_trim =   Wb[2] * tphi   ;
            Wb_cmd[1] =   1.0 / cphi * desired_dot_att_angle[1] + q_trim;

            double r_trim =   - Wb[1] * tphi   ;
            Wb_cmd[2] =   1.0 / (  cphi * sthe ) * desired_dot_att_angle[2] + r_trim;
        }
        __device__ void controller__rate_controller(double Wb_cmd[3], double Wb[3], \
            double tau_Wb, double dt_GCU, double err_Wb[3], double int_err_Wb[3])
        {
            double pi = acos(-1.);
            //.. rate_controller
            // Wb_cmd limit in 7~8p, [https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf]
            double lim_Wb = 2./tau_Wb * pi/180.;
            double norm_Wb_cmd = norm_(Wb_cmd);
            if (norm_Wb_cmd > lim_Wb){
                for(int i=0;i<3;i++) Wb_cmd[i] = Wb_cmd[i] / norm_Wb_cmd * lim_Wb;
            }
            
            // Inner Loop: Rate Control Loop (SAS) -> PI controller (2nd order system)
            for(int i=0;i<3;i++) {
                err_Wb[i] = Wb_cmd[i] - Wb[i];
                int_err_Wb[i] = int_err_Wb[i] + err_Wb[i] * dt_GCU;
            }
        }
        __device__ void dynamics__equations_of_motions(double cI_B[3][3], double cB_I[3][3], \
            double T_cmd, double mass, double Ai_disturbance[3], double Ai_grav[3], \
            double zeta_Wb[3], double omega_Wb[3], double err_Wb[3], double int_err_Wb[3], double Wb[3], double Vb[3], double att_ang[3], \
            double dot_Ri[3], double dot_Vb[3], double dot_att_ang[3], double dot_Wb[3])
        {
            //.. dynamics
            double Ab_thrust[3] = {0.,0., -T_cmd/mass}; 
            double Ab_aero[3] = {0.,}; matmul_(cI_B, Ai_disturbance, Ab_aero);
            double Ab_grav[3] = {0.,}; matmul_(cI_B, Ai_grav, Ab_grav);
            double Ab[3] = {0.,};
            for(int i=0;i<3;i++) {
                Ab[i] = Ab_thrust[i] + Ab_aero[i] + Ab_grav[i];
            }

            
            // Computing Dynamics 
            double Wb_X_Vb[3] = {0.,};
            cross_(Wb, Vb, Wb_X_Vb);
            
            for(int i=0;i<3;i++) {
                dot_Vb[i] = - Wb_X_Vb[i] + Ab[i];
                dot_Wb[i] = 2*zeta_Wb[i]*omega_Wb[i]*err_Wb[i] + omega_Wb[i]*omega_Wb[i]*int_err_Wb[i];
            }
            matmul_(cB_I, Vb, dot_Ri);
            
            double cthe =   cos( att_ang[1] )   ;
            double sthe =   1.0 / cthe   ;
            double tthe =   tan( att_ang[1] )   ;
            double sphi =   sin( att_ang[0] )   ;
            double cphi =   cos( att_ang[0] )   ;
            
            dot_att_ang[0] = Wb[0] + Wb[1]*sphi*tthe + Wb[2]*cphi*tthe ;
            dot_att_ang[1] = Wb[1]*cphi - Wb[2]*sphi;
            dot_att_ang[2] = Wb[1]*sphi*sthe + Wb[2]*cphi*sthe;
        }
        __device__ void update_states(double dot_Ri[3], double dot_Vb[3], double dot_att_ang[3], double dot_Wb[3], double dt, \
            double Ri[3], double Vb[3], double att_ang[3], double Wb[3], double cI_B[3][3], double cB_I[3][3], double Vi[3])
        {
            for(int i=0;i<3;i++) {
                Ri[i] =   Ri[i] + dot_Ri[i] * dt;
                Vb[i] =   Vb[i] + dot_Vb[i] * dt;
                att_ang[i] =   att_ang[i] + dot_att_ang[i] * dt;
                att_ang[i] =   atan2(sin(att_ang[i]),cos(att_ang[i]));
                Wb[i] =   Wb[i] + dot_Wb[i] * dt;
            }
            DCM_from_euler_angle(att_ang, cI_B);
            transpose_(cI_B, cB_I);
            matmul_(cB_I, Vb, Vi);
        }
        
        
        """
        
        self.func_MC     =   SourceModule(self.total_MPPI_code).get_function("MPPI_monte_carlo_sim")
        pass
    
    pass

