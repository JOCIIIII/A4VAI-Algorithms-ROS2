############################################################
#
#   - Name : MPPI_guidance.py
#
#                   -   Created by E. T. Jeong, 2024.04.12
#                       Keep - E. T. ver
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
        self.MP             =   MPPI_Param
        self.u0             =   self.MP.u0_init * np.ones(self.MP.N)
        self.u1             =   self.MP.u1_init * np.ones(self.MP.N)
        self.Ai_est_dstb    =   np.zeros((self.MP.N,3))
        self.Ai_est_var     =   np.zeros((self.MP.N,1))
        pass
    
    def run_MPPI_Guidance(self, QR:Quadrotor_6DOF, WPs:Waypoint):
        
        t0 = time.time()      
        
        #.. variable setting - MPPI Monte Carlo simulation
        # set CPU variables
        arr_u0          =   np.array(self.u0).astype(np.float64)
        arr_u1          =   np.array(self.u1).astype(np.float64)
        arr_delta_u0    =   self.MP.var0*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_delta_u1    =   self.MP.var1*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_stk         =   np.zeros(self.MP.K).astype(np.float64)
        
        arr_int_MP      =   np.array([self.MP.K, self.MP.N]).astype(np.int32)
        arr_dbl_MP      =   np.array([self.MP.dt_MPPI, self.MP.R[0], self.MP.R[1], self.MP.R[2], self.MP.Q[0], self.MP.Q[1], self.MP.Q[2], self.MP.P[0], self.MP.P[1], self.MP.P[2],
                                      QR.GnC_param.virtual_target_distance, self.MP.cost_min_V_aligned, self.MP.gamma, self.MP.beta]).astype(np.float64)
        arr_int_QR      =   np.array([QR.PF_var.WP_idx_heading, QR.PF_var.WP_idx_passed, QR.GnC_param.Guid_type]).astype(np.int32)
        arr_dbl_QR      =   np.array([QR.state_var.Ri[0], QR.state_var.Ri[1], QR.state_var.Ri[2], 
                                      QR.state_var.Vi[0], QR.state_var.Vi[1], QR.state_var.Vi[2],
                                      QR.state_var.att_ang[0], QR.state_var.att_ang[1], QR.state_var.att_ang[2], 
                                      QR.guid_var.T_cmd, QR.GnC_param.desired_speed]).astype(np.float64)
        arr_dbl_WPs         =   np.ravel(WPs,order='C').astype(np.float64)
        arr_dbl_VT          =   np.array([QR.PF_var.VT_Ri[0], QR.PF_var.VT_Ri[1], QR.PF_var.VT_Ri[2]]).astype(np.float64)
        arr_Ai_est_dstb     =   np.array(self.Ai_est_dstb).astype(np.float64)
        arr_Ai_est_var      =   np.array(self.Ai_est_var).astype(np.float64)
        arr_numer0          =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_numer1          =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom0          =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom1          =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)

        # occupy GPU memory space
        gpu_u0              =   cuda.mem_alloc(arr_u0.nbytes)
        gpu_u1              =   cuda.mem_alloc(arr_u1.nbytes)
        gpu_delta_u0        =   cuda.mem_alloc(arr_delta_u0.nbytes)
        gpu_delta_u1        =   cuda.mem_alloc(arr_delta_u1.nbytes)
        gpu_stk             =   cuda.mem_alloc(arr_stk.nbytes)
        gpu_int_MP          =   cuda.mem_alloc(arr_int_MP.nbytes)
        gpu_dbl_MP          =   cuda.mem_alloc(arr_dbl_MP.nbytes)
        gpu_int_QR          =   cuda.mem_alloc(arr_int_QR.nbytes)
        gpu_dbl_QR          =   cuda.mem_alloc(arr_dbl_QR.nbytes)
        gpu_dbl_WPs         =   cuda.mem_alloc(arr_dbl_WPs.nbytes)
        gpu_dbl_VT          =   cuda.mem_alloc(arr_dbl_VT.nbytes)
        gpu_Ai_est_dstb     =   cuda.mem_alloc(arr_Ai_est_dstb.nbytes)
        gpu_Ai_est_var      =   cuda.mem_alloc(arr_Ai_est_var.nbytes)
        gpu_numer0          =   cuda.mem_alloc(arr_numer0.nbytes)
        gpu_numer1          =   cuda.mem_alloc(arr_numer1.nbytes)
        gpu_denom0          =   cuda.mem_alloc(arr_denom0.nbytes)
        gpu_denom1          =   cuda.mem_alloc(arr_denom1.nbytes)

        # convert data memory from CPU to GPU
        cuda.memcpy_htod(gpu_u0,arr_u0)
        cuda.memcpy_htod(gpu_u1,arr_u1)
        cuda.memcpy_htod(gpu_delta_u0,arr_delta_u0)
        cuda.memcpy_htod(gpu_delta_u1,arr_delta_u1)
        cuda.memcpy_htod(gpu_stk,arr_stk)
        cuda.memcpy_htod(gpu_int_MP,arr_int_MP)
        cuda.memcpy_htod(gpu_dbl_MP,arr_dbl_MP)
        cuda.memcpy_htod(gpu_int_QR,arr_int_QR)
        cuda.memcpy_htod(gpu_dbl_QR,arr_dbl_QR)
        cuda.memcpy_htod(gpu_dbl_WPs,arr_dbl_WPs)
        cuda.memcpy_htod(gpu_dbl_VT,arr_dbl_VT)
        cuda.memcpy_htod(gpu_Ai_est_dstb,arr_Ai_est_dstb)
        cuda.memcpy_htod(gpu_Ai_est_var,arr_Ai_est_var)
        cuda.memcpy_htod(gpu_numer0, arr_numer0)
        cuda.memcpy_htod(gpu_numer1, arr_numer1)
        cuda.memcpy_htod(gpu_denom0, arr_denom0)
        cuda.memcpy_htod(gpu_denom1, arr_denom1)

        # cuda.memcpy_htod(gpu_out_results, arr_out_results)
        #.. run MPPI Monte Carlo simulation code script
        # run cuda script by using GPU cores
        unit_gpu_allocation = 32        # GPU SP number
        # unit_gpu_allocation = 64        # GPU SP number
        blocksz     =   (unit_gpu_allocation, 1, 1)
        gridsz      =   (round(self.MP.K/(unit_gpu_allocation)), 1)
        # gridsz      =   (round(int(np.ceil(256 / unit_gpu_allocation))), 1)
        # blocksz     =   (1, 1, 1)
        # gridsz      =   (1, 1)
        
        t1 = time.time()
        self.func_MC(gpu_u0, gpu_u1, 
                gpu_delta_u0, gpu_delta_u1, gpu_stk,
                gpu_int_MP, gpu_dbl_MP, gpu_int_QR, gpu_dbl_QR, 
                gpu_dbl_WPs, gpu_dbl_VT,gpu_Ai_est_dstb, gpu_Ai_est_var,
                gpu_numer0,gpu_numer1,gpu_denom0,gpu_denom1,
                block=blocksz, grid=gridsz)
        t2 = time.time()

        #.. variable setting - MPPI entropy calculation
        # entropy calc. results
        res_numer0     =   np.empty_like(arr_numer0)
        res_numer1     =   np.empty_like(arr_numer1)
        res_denom0     =   np.empty_like(arr_denom0)
        res_denom1     =   np.empty_like(arr_denom1)
        cuda.memcpy_dtoh(res_numer0, gpu_numer0)
        cuda.memcpy_dtoh(res_numer1, gpu_numer1)
        cuda.memcpy_dtoh(res_denom0, gpu_denom0)
        cuda.memcpy_dtoh(res_denom1, gpu_denom1)
       
        #.. MPPI input calculation
        # entropy
        sum_numer0      =   res_numer0.sum(axis=1)
        sum_numer1      =   res_numer1.sum(axis=1)
        sum_denom0      =   res_denom0.sum(axis=1)
        sum_denom1      =   res_denom1.sum(axis=1)
        denom_min       =   np.zeros(np.size(sum_denom0)) + 1.0e-200
        entropy0    =   sum_numer0/np.maximum(sum_denom0, denom_min)
        entropy1    =   sum_numer1/np.maximum(sum_denom1, denom_min)

        print("sum_numer0 " + str(sum_numer0) +", sum_denom0 " + str(sum_denom0))

        # MPPI input
        self.u0     =   self.u0 + entropy0
        self.u1     =   self.u1 + entropy1

        # MPPI result and update
        MPPI_ctrl_input    =   np.array([self.u0[0], self.u1[0]])
        
        self.u0[0:self.MP.N-1]  =   self.u0[1:self.MP.N]
        self.u1[0:self.MP.N-1]  =   self.u1[1:self.MP.N]
        self.u0[self.MP.N-1]    =   self.MP.u0_init
        self.u1[self.MP.N-1]    =   self.MP.u1_init

        t3 = time.time()      
        # MPPI_calc_time = (t2 - t1) 
        MPPI_calc_time = t3 - t0

        return MPPI_ctrl_input, MPPI_calc_time
    
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
        __device__ void path_following_required_info__cost_function_1(double R[3], double FPA_azim_diy, double MPPI_ctrl_input[2], \
            double Q0, double dist_to_path, double Q1, double att_ang[3], double weight_by_var, \
            double unit_W1W2[3], double min_V_aligned, double cost_arr[3], double dt);
        __device__ void path_following_required_info__cost_function_diy(double R[3], double FPA_azim_diy, double MPPI_ctrl_input[2], \
            double Q[3], double dist_to_path, double cost_arr[3], double dt, double delta_u[2], double magV, \
            double refV, double MP_gamma, double att_ang[3], double weight_by_var);
        __device__ void path_following_required_info__terminal_cost_1(double P1, double WP_WPs[nWP][3], \
            int PF_var_init_WP_idx_passed, int PF_var_final_WP_idx_passed,\
            double PF_var_init_point_closest_on_path[3], double PF_var_final_point_closest_on_path[3], \
            double min_move_range, double total_time, double *terminal_cost);
        __device__ void path_following_required_info__terminal_cost_diy(double P[3], double WP_WPs[nWP][3], \
            double PF_var_final_point_closest_on_path[3], double *terminal_cost, int PF_var_WP_idx_heading, double V_tf[3], double dist_to_path, double magV, double refV); 
        __device__ void guidance_path_following__guidance_modules(int QR_Guid_type, \
            int QR_WP_idx_passed, int QR_WP_idx_heading, int WP_WPs_shape0, double VT_Ri[3], \
            double QR_Ri[3], double QR_Vi[3], double QR_Ai[3], double QR_virtual_target_distance, double QR_desired_speed, double QR_Kp_vel, double QR_Kd_vel, \
            double QR_Kp_speed, double QR_Kd_speed, double QR_guid_eta, double MPPI_ctrl_input[2], double Aqi_cmd[3]);
        __device__ void guidance_path_following__simple_rotor_drag_model(double QR_Vi[3], \
            double psuedo_rotor_drag_coeff, double cB_I[3][3], double Fi_drag[3]);
        __device__ void guidance_path_following__convert_Ai_cmd_to_thrust_and_att_ang_cmd(double cI_B[3][3], double Ai_cmd[3], \
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
        __global__ void MPPI_monte_carlo_sim(double* arr_u0, double* arr_u1, \
            double* arr_delta_u0, double* arr_delta_u1, double* arr_stk, \
            int* arr_int_MP, double* arr_dbl_MP, int* arr_int_QR, double* arr_dbl_QR, \
            double* arr_dbl_WPs, double* arr_dbl_VT, double* arr_Ai_est_dstb, double* arr_Ai_est_var, \
            double* arr_numer0, double* arr_numer1, \
            double* arr_denom0, double* arr_denom1)
        {
            //.. GPU core index for parallel computation
            int idx     =   threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;
            
            //--------------------------------------------------------------
            // QUADROTOR MODEL PARAMETERS (PLEASE UPDATE WHEN CHANGES OCCUR)
            //--------------------------------------------------------------
            double physical_param_throttle_hover          = 0.7299;
            double physical_param_mass                    = 1.972475; //241223 diy
            double GnC_param_virtual_target_distance      = 4.5; 
            double GnC_param_distance_change_WP           = GnC_param_virtual_target_distance; 
            double GnC_param_dist_change_first_WP         = 0.3; 
            double GnC_param_Kp_vel                       = 1.; 
            double GnC_param_Kd_vel                       = 0.; 
            double GnC_param_Kp_speed                     = 1.; 
            double GnC_param_Kd_speed                     = 0.; 
            double GnC_param_guid_eta                     = 3.; 
            double GnC_param_tau_phi                      = 0.6; 
            double GnC_param_tau_the                      = 0.6; 
            double GnC_param_tau_psi                      = 0.9; 
            double GnC_param_tau_p                        = 0.2; 
            double GnC_param_tau_q                        = 0.2; 
            double GnC_param_tau_r                        = 0.3; 
            double GnC_param_alpha_p                      = 0.1; 
            double GnC_param_alpha_q                      = 0.1; 
            double GnC_param_alpha_r                      = 0.1; 
            double physical_param_psuedo_rotor_drag_coeff = 0.5620; 
            double GnC_param_del_psi_cmd_limit            = 0.349065850398866;
            double GnC_param_tau_Wb                       = 0.05;
            //--------------------------------------------------------------

            /*.. declare variables ..*/
            //.. set MPPI variables
            int    MP_K                   = arr_int_MP[0];
            int    MP_N                   = arr_int_MP[1];
            double MP_dt                  = arr_dbl_MP[0];
            double MP_R[3]                = {arr_dbl_MP[1], arr_dbl_MP[2], arr_dbl_MP[3]}; 
            double MP_Q[3]                = {arr_dbl_MP[4], arr_dbl_MP[5], arr_dbl_MP[6]};
            double MP_P[3]                = {arr_dbl_MP[7], arr_dbl_MP[8], arr_dbl_MP[9]};
            double virtual_target_distance= arr_dbl_MP[10];
            double MP_cost_min_V_aligned  = arr_dbl_MP[11];
            double MP_gamma               = arr_dbl_MP[12];
            double MP_beta                = arr_dbl_MP[13];
            
            double times_N                = 1.0;
            double modif_dt               = MP_dt / times_N;
            
            //.. set PF variables
            int PF_var_WP_idx_heading    = arr_int_QR[0];
            int PF_var_WP_idx_passed     = arr_int_QR[1];
            int GnC_param_Guid_type      = arr_int_QR[2];
                        
            //.. set QR state variables
            double state_var_Ri[3]       = {arr_dbl_QR[0], arr_dbl_QR[1], arr_dbl_QR[2]};
            double state_var_Vi[3]       = {arr_dbl_QR[3], arr_dbl_QR[4], arr_dbl_QR[5]};
            double state_var_att_ang[3]  = {arr_dbl_QR[6], arr_dbl_QR[7], arr_dbl_QR[8]};
            double state_var_Ai[3]       = {0.,};
            double state_var_Wb[3]       = {0.,};
            double guid_var_T_cmd        = arr_dbl_QR[9];
            double GnC_param_desired_speed = arr_dbl_QR[10];

            //.. set waypoints
            double WP_WPs[nWP][3]   =   {0.,};
            for(int i_WP = 0; i_WP < nWP; i_WP++){
                for(int i = 0; i < 3; i++){
                    WP_WPs[i_WP][i] = arr_dbl_WPs[i_WP*3 + i];
                }
            }
            
            //.. set others
            double env_var_grav                          = 9.81;
            double Ai_grav[3]                            = {0., 0., env_var_grav};
            double physical_param_T_max                  = physical_param_mass * env_var_grav / physical_param_throttle_hover; 
            double PF_var_point_closest_on_path_i[3]     = {0.,};
            double PF_var_dist_to_path                   = 0.;
            double PF_var_VT_Ri[3]                       = {0.,};
            double PF_var_cost_arr[3]                    = {0.,};
            int PF_var_init_WP_idx_passed                = 0;
            int PF_var_final_WP_idx_passed               = 0;
            double PF_var_init_point_closest_on_path[3]  = {0.,};
            double PF_var_final_point_closest_on_path[3] = {0.,};
            double PF_var_init_time                      = 0.;
            double PF_var_final_time                     = MP_dt * MP_N;
            double guid_var_Ai_cmd[3]                    = {0.,};
            double cI_B[3][3]; DCM_from_euler_angle(state_var_att_ang, cI_B);
            double cB_I[3][3]; transpose_(cI_B, cB_I);
            double Fi_drag[3] = {0.,};
            double guid_var_Ai_rotor_drag[3]      = {0.,};
            double guid_var_Ai_disturbance[3]     = {0.,};
            double guid_var_Ai_cmd_compensated[3] = {0.,};
            double guid_var_att_ang_cmd[3]        = {0.,};
                        
            double state_var_Vb[3]        = {0.,}; matmul_(cI_B, state_var_Vi, state_var_Vb);
            double ctrl_var_err_Wb[3]     = {0.,};
            double ctrl_var_Wb_cmd[3]     = {0.,};
            double ctrl_var_int_err_Wb[3] = {0.,};
            double dyn_var_dot_Ri[3]      = {0.,};
            double dyn_var_dot_Vb[3]      = {0.,};
            double dyn_var_dot_att_ang[3] = {0.,};
            double dyn_var_dot_Wb[3]      = {0.,};
            
            
            // set rate controller parameters
            double zeta_Wb[3]  = {0.,};
            double omega_Wb[3] = {0.,};
            zeta_Wb[0]         = 0.5*sqrt(GnC_param_alpha_p/GnC_param_tau_p);
            zeta_Wb[1]         = 0.5*sqrt(GnC_param_alpha_q/GnC_param_tau_q);
            zeta_Wb[2]         = 0.5*sqrt(GnC_param_alpha_r/GnC_param_tau_r);
            omega_Wb[0]        = sqrt(1/(GnC_param_alpha_p*GnC_param_tau_p));
            omega_Wb[1]        = sqrt(1/(GnC_param_alpha_q*GnC_param_tau_q));
            omega_Wb[2]        = sqrt(1/(GnC_param_alpha_r*GnC_param_tau_r));

            // temperary settings
            double lim_var     = 0.0005;
            double magV        = 0.;
            double delta_u[2]  = {0., };
            double FPA_azim_diy= 0.; 
            double FPA_elev_diy= 0.; 

            //.. main loop
            int i_N = 0;
            for(i_N = 0; i_N < MP_N; i_N++){
                    
                //.. MPPI modules - checked
                double MPPI_ctrl_input[2] = {0.,};
                if(GnC_param_Guid_type >= 2){
                    MPPI_ctrl_input[0] = arr_u0[i_N] + arr_delta_u0[idx + MP_K*i_N];
                    MPPI_ctrl_input[1] = arr_u1[i_N] + arr_delta_u1[idx + MP_K*i_N];
                    delta_u[0]         = arr_delta_u0[idx + MP_K*i_N];
                    delta_u[1]         = arr_delta_u1[idx + MP_K*i_N];
                }
                
                if (PF_var_WP_idx_heading == nWP-1){
                    GnC_param_Guid_type               = 1;
                    GnC_param_virtual_target_distance = 4;
                    GnC_param_desired_speed           = 2;
                    GnC_param_guid_eta                = 2;
                }
                
                //.. Environment - checked
                double Ai_disturbance[3]; for(int i=0;i<3;i++) Ai_disturbance[i] = arr_Ai_est_dstb[i + 3*i_N];
                double Ai_dist_var = arr_Ai_est_var[i_N]; 

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
                //     path_following_required_info__takeoff_to_first_WP(WP_WPs, \
                //         state_var_Ri, PF_var_WP_idx_passed, GnC_param_dist_change_first_WP, PF_var_VT_Ri);
                }
                    
                // weight by variance of GPR
                double weight_by_var = 40. * min(Ai_dist_var, lim_var) ;
                double Rw1w2[3];  for(int i=0;i<3;i++) Rw1w2[i] = WP_WPs[PF_var_WP_idx_heading][i] - WP_WPs[PF_var_WP_idx_passed][i];
                double PF_var_unit_Rw1w2[3]; for(int i=0;i<3;i++) PF_var_unit_Rw1w2[i] = Rw1w2[i]/norm_(Rw1w2);
                
                magV            = norm_(state_var_Vi);
                azim_elev_from_vec3(state_var_Vi, &FPA_azim_diy, &FPA_elev_diy);

                if (GnC_param_Guid_type == 4) {
                    path_following_required_info__cost_function_diy(MP_R, FPA_azim_diy, MPPI_ctrl_input, MP_Q, PF_var_dist_to_path, PF_var_cost_arr, MP_dt,\
                        delta_u, magV, GnC_param_desired_speed, MP_gamma, state_var_att_ang, weight_by_var);
                }
                else {
                    path_following_required_info__cost_function_1(MP_R, FPA_azim_diy, MPPI_ctrl_input, MP_Q[0], PF_var_dist_to_path, \
                        MP_Q[1], state_var_att_ang, weight_by_var, PF_var_unit_Rw1w2, MP_cost_min_V_aligned, PF_var_cost_arr, MP_dt);
                }              
                    
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
                    state_var_Ri, state_var_Vi, state_var_Ai, GnC_param_virtual_target_distance, GnC_param_desired_speed, GnC_param_Kp_vel, GnC_param_Kd_vel, \
                    GnC_param_Kp_speed, GnC_param_Kd_speed, GnC_param_guid_eta, MPPI_ctrl_input, guid_var_Ai_cmd);
                // calc. simple rotor drag model
                guidance_path_following__simple_rotor_drag_model(state_var_Vi, physical_param_psuedo_rotor_drag_coeff, cB_I, Fi_drag);
                for(int i=0;i<3;i++) guid_var_Ai_rotor_drag[i] = Fi_drag[i]/physical_param_mass;
                // compensate disturbance
                for(int i=0;i<3;i++) guid_var_Ai_disturbance[i]     = Ai_disturbance[i] + guid_var_Ai_rotor_drag[i];
                for(int i=0;i<3;i++) guid_var_Ai_cmd_compensated[i] = guid_var_Ai_cmd[i] - guid_var_Ai_disturbance[i];
                // compensate gravity
                guid_var_Ai_cmd_compensated[2] = guid_var_Ai_cmd_compensated[2] - env_var_grav;
                // convert_Ai_cmd_to_thrust_and_att_ang_cmd
                guidance_path_following__convert_Ai_cmd_to_thrust_and_att_ang_cmd(cI_B, guid_var_Ai_cmd_compensated, \
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
            
            double terminal_cost = 0.;
            for(int i=0;i<3;i++) PF_var_final_point_closest_on_path[i] = PF_var_point_closest_on_path_i[i];

            // terminal cost function        
            if (GnC_param_Guid_type == 4) {
                path_following_required_info__terminal_cost_diy(MP_P, WP_WPs, PF_var_final_point_closest_on_path, &terminal_cost, PF_var_WP_idx_heading, state_var_Vi, PF_var_dist_to_path, magV, GnC_param_desired_speed);
            }
            else {
                PF_var_final_time = MP_dt * i_N;
                PF_var_final_WP_idx_passed = PF_var_WP_idx_passed;
                double total_time = PF_var_final_time - PF_var_init_time;
                double min_move_range = MP_cost_min_V_aligned*MP_N*MP_dt;
                path_following_required_info__terminal_cost_1(MP_P[0], WP_WPs, PF_var_init_WP_idx_passed, PF_var_final_WP_idx_passed,\
                    PF_var_init_point_closest_on_path, PF_var_final_point_closest_on_path, min_move_range, total_time, &terminal_cost);
            }
            
            arr_stk[idx] = arr_stk[idx] + terminal_cost;

            //.. MPPI_entropy
            i_N = 0;
            for(i_N = 0; i_N < MP_N; i_N++){
                arr_numer0[idx + MP_K*i_N] = exp((-1/MP_beta)*arr_stk[idx])*arr_delta_u0[idx + MP_K*i_N];
                arr_denom0[idx + MP_K*i_N] = exp((-1/MP_beta)*arr_stk[idx]);
                arr_numer1[idx + MP_K*i_N] = exp((-1/MP_beta)*arr_stk[idx])*arr_delta_u1[idx + MP_K*i_N];
                arr_denom1[idx + MP_K*i_N] = exp((-1/MP_beta)*arr_stk[idx]);
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
            for(int i_WP = QR_WP_idx_heading; i_WP >= QR_WP_idx_passed[0]; i_WP--)
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
        __device__ void path_following_required_info__cost_function_1(double R[3], double FPA_azim_diy, double MPPI_ctrl_input[2], \
            double Q0, double dist_to_path, double Q1, double att_ang[3], double weight_by_var, \
            double unit_W1W2[3], double min_V_aligned, double cost_arr[3], double dt)
        {
            // uRu of LQR cost, set low value of norm(R)
            double uRu = 0.;
            
            uRu = R[0] * (MPPI_ctrl_input[0]*MPPI_ctrl_input[0]) + R[1] * (MPPI_ctrl_input[1]*MPPI_ctrl_input[1]);
            
            // path following performance
            double x0     = dist_to_path;
            double x0Q0x0 = x0 * Q0 * x0;
            
            // attitude control stability
            double x1     = sqrt(att_ang[0]*att_ang[0] + att_ang[1]*att_ang[1]);
            double x1Q1x1 = x1 * Q1 * x1;
            
            // total cost
            cost_arr[0] = uRu * dt;
            cost_arr[1] = x0Q0x0 * dt;
            cost_arr[2] = weight_by_var * x1Q1x1 * dt;
        }
        __device__ void path_following_required_info__cost_function_diy(double R[3], double FPA_azim_diy, double MPPI_ctrl_input[2], \
            double Q[3], double dist_to_path, double cost_arr[3], double dt, double delta_u[2], double magV, \
            double refV, double MP_gamma, double att_ang[3], double weight_by_var)
        {
            // uRu of LQR cost, set low value of norm(R)
            double uRu  = 0.;
            double uRdu = 0.;

            double u0   = MPPI_ctrl_input[0]; 
            double u1   = MPPI_ctrl_input[1];
            double du0  = delta_u[0];
            double du1  = delta_u[1];

            uRdu        = MP_gamma * ( ( R[0] * u0 ) * du0 + ( R[1] * FPA_azim_diy * FPA_azim_diy * u1 ) * du1 ); 
            
            // path following performance
            double x0     = min( dist_to_path / 5.0, 1.0 );
            double x0Q0x0 = x0 * Q[0] * x0;

            // desired velocity
            double x1     = min( abs(magV - refV)/refV, 1.0); 
            double x1Q1x1 = x1 * Q[1] * x1;
            
            // attitude control stability
            double x2     = sqrt(att_ang[0]*att_ang[0] + att_ang[1]*att_ang[1]);
            double x2Q2x2 = x2 * Q[2] * x2;
            
            // total cost
            cost_arr[0] = uRdu * dt;
            cost_arr[1] = (x0Q0x0 + x1Q1x1) * dt ;
            cost_arr[2] = weight_by_var * x2Q2x2 * dt;
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
        __device__ void path_following_required_info__terminal_cost_diy(double P[3], double WP_WPs[nWP][3], \
            double PF_var_final_point_closest_on_path[3], double *terminal_cost, int PF_var_WP_idx_heading, double V_tf[3], double dist_to_path, double magV, double refV)
        {

            double PF_dir_tf[2]   = {0., };
            double PF_dir_tf_norm = 0.; 
            double V_tf_norm      = 0.;
            double PF_V_inner     = 0.; 
            double X_dir          = 0.;
            double x0             = 0.; 
            double x1             = 0.;

            //-------------------------------------------------------------------------------------------------------------

            PF_dir_tf[0] = WP_WPs[PF_var_WP_idx_heading][0] - PF_var_final_point_closest_on_path[0];
            PF_dir_tf[1] = WP_WPs[PF_var_WP_idx_heading][1] - PF_var_final_point_closest_on_path[1];
            
            PF_dir_tf_norm = sqrt( PF_dir_tf[0]*PF_dir_tf[0] + PF_dir_tf[1]*PF_dir_tf[1] );
            V_tf_norm      = sqrt( V_tf[0]*V_tf[0] + V_tf[1]*V_tf[1] );

            PF_V_inner     = PF_dir_tf[0] * V_tf[0] + PF_dir_tf[1] * V_tf[1];
            X_dir          = PF_V_inner / max((PF_dir_tf_norm * V_tf_norm), 1e-10); 

            x0             = min( dist_to_path / 5.0, 1.0 );
            x1             = min( abs( magV - refV )/ refV, 1.0 );
 
            //-------------------------------------------------------------------------------------------------------------
            
            // terminal cost
            terminal_cost[0] = P[0] * 0.5 * ( 1.0 - X_dir ) + P[1] * x0 + P[2] * x1 ;
        }
        __device__ void guidance_path_following__guidance_modules(int QR_Guid_type, \
            int QR_WP_idx_passed, int QR_WP_idx_heading, int WP_WPs_shape0, double VT_Ri[3], \
            double QR_Ri[3], double QR_Vi[3], double QR_Ai[3], double QR_virtual_target_distance, double QR_desired_speed, double QR_Kp_vel, double QR_Kd_vel, \
            double QR_Kp_speed, double QR_Kd_speed, double QR_guid_eta, double MPPI_ctrl_input[2], double Aqi_cmd[3])
        {
            // starting phase
            if (QR_WP_idx_passed < 1){
                QR_Guid_type = 0;
                QR_desired_speed = 3.0;
            }
            // terminal phase
            if (QR_WP_idx_heading == (WP_WPs_shape0 - 1)){
                QR_Guid_type = 0;
            }
            
            // guidance command
            if ( QR_Guid_type == 0 ){
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
            else if ( (QR_Guid_type == 3) || (QR_Guid_type == 4) ){
                QR_guid_eta = MPPI_ctrl_input[1];
                // calc. variables
                double QR_mag_Vi = norm_(QR_Vi);
                double FPA_azim, FPA_elev; azim_elev_from_vec3(QR_Vi, &FPA_azim, &FPA_elev);
                double FPA_euler[3] = {0., FPA_elev, FPA_azim};
                double QR_cI_W[3][3]; DCM_from_euler_angle(FPA_euler, QR_cI_W);
                //.. guidance - GL - parameters by MPPI
                double Aqw_cmd[3];
                
                // a_x command
                Aqw_cmd[0] = MPPI_ctrl_input[0];

                // pursuit guidance law
                double Rqti[3]; for(int i=0;i<3;i++) Rqti[i] = VT_Ri[i] - QR_Ri[i];
                double Rqtw[3]; matmul_(QR_cI_W, Rqti, Rqtw);
                double err_azim, err_elev; azim_elev_from_vec3(Rqtw, &err_azim, &err_elev);
                Aqw_cmd[1]  =   QR_guid_eta * QR_mag_Vi * sin(err_azim);
                Aqw_cmd[2]  =   -2.0 * QR_mag_Vi * sin(err_elev);
                // command coordinate change
                double cW_I[3][3]; transpose_(QR_cI_W, cW_I);
                matmul_(cW_I, Aqw_cmd, Aqi_cmd);
            }
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
        __device__ void guidance_path_following__convert_Ai_cmd_to_thrust_and_att_ang_cmd(double cI_B[3][3], double Ai_cmd[3], \
            double mass, double T_max, double WP_WPs[nWP][3], int WP_idx_heading, double Ri[3], double att_ang[3], \
            double del_psi_cmd_limit, double* T_cmd, double att_ang_cmd[3])
        {
            double pi = acos(-1.);
            
            // thrust cmd
            double mag_Ai_cmd = norm_(Ai_cmd);
            double Ab_cmd[3]; matmul_(cI_B , Ai_cmd, Ab_cmd);
            T_cmd[0] = min(abs(Ab_cmd[2]) * mass, T_max);
            
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
            att_ang_cmd[0] = min(max(asin(Apsi_cmd[1]/mag_Ai_cmd), -0.5236), 0.5236);
            double sintheta = min(max(-Apsi_cmd[0]/cos(att_ang_cmd[0])/mag_Ai_cmd, -0.5236), 0.5236);
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

