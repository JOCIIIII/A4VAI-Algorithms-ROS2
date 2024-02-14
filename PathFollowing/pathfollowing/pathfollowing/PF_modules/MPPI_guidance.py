############################################################
#
#   - Name : MPPI_guidance.py
#
#                   -   Created by E. T. Jeong, 2024.09.13
#
############################################################

#.. Library
# pulbic libs.
import numpy as np
import math as m
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# private libs.
from .quadrotor_6dof import Quadrotor_6DOF
from .virtual_target import Virtual_Target
from .set_parameter import MPPI_Guidance_Parameter

# A Q6rotor 3dof model integrated in a class which cosnsists of several modules 
class MPPI_Guidance_Modules():    
    #.. initialize an instance of the class
    def __init__(self, MPPI_parameter:MPPI_Guidance_Parameter) -> None:
        self.MP     =   MPPI_parameter
        self.u1     =   self.MP.u1_init * np.ones(self.MP.N)
        self.u2     =   self.MP.u2_init * np.ones(self.MP.N)
        self.u3     =   self.MP.u3_init * np.ones(self.MP.N)
        self.Ai_est_dstb    =   np.zeros((self.MP.N,3))
        pass
    pass

    def run_MPPI_Guidance(self, Q6:Quadrotor_6DOF, WPs:np, VT:Virtual_Target):
        #.. variable setting - MPPI Monte Carlo simulation
        tmp_float = 0.
        # set CPU variables
        arr_u1          =   np.array(self.u1).astype(np.float64)
        arr_u2          =   np.array(self.u2).astype(np.float64)
        arr_u3          =   np.array(self.u3).astype(np.float64)
        arr_delta_u1    =   self.MP.var1*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_delta_u2    =   self.MP.var2*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_delta_u3    =   self.MP.var3*np.random.randn(self.MP.N,self.MP.K).astype(np.float64)
        arr_stk         =   np.zeros(self.MP.K).astype(np.float64)
        res_length      =   1 # self.MP.N
        arr_res1        =   np.zeros(res_length).astype(np.float64)
        arr_res2        =   np.zeros(res_length).astype(np.float64)
        arr_res3        =   np.zeros(res_length).astype(np.float64)
        arr_res4        =   np.zeros(res_length).astype(np.float64)
        arr_int_MP      =   np.array([self.MP.K, self.MP.N, self.MP.flag_cost_calc]).astype(np.int32)
        arr_dbl_MP      =   np.array([self.MP.dt, self.MP.nu, 
                                      self.MP.R[0], self.MP.R[1], self.MP.R[2], 
                                      self.MP.Q[0], self.MP.Q[1], self.MP.Q[2], self.MP.Q[3], self.MP.Q_lim[0] * self.MP.Q_lim_margin[0], self.MP.a_lim,
                                      self.MP.u1_init, self.MP.u2_init, self.MP.u3_init]).astype(np.float64)
        arr_int_Q6      =   np.array([Q6.WP_idx_heading, Q6.WP_idx_passed, Q6.Guid_type, Q6.flag_guid_trans]).astype(np.int32)
        arr_dbl_Q6      =   np.array([Q6.throttle_hover, tmp_float, Q6.desired_speed, Q6.look_ahead_distance, Q6.distance_change_WP, 
                                      Q6.Kp_vel, Q6.Kd_vel, Q6.Kp_speed * 0.5, Q6.Kd_speed, Q6.guid_eta, 
                                      Q6.tau_phi, Q6.tau_the, Q6.tau_psi, 
                                      Q6.Ri[0], Q6.Ri[1], Q6.Ri[2], 
                                      Q6.Vi[0], Q6.Vi[1], Q6.Vi[2], 
                                      Q6.Ai[0], Q6.Ai[1], Q6.Ai[2], 
                                      Q6.thr_unitvec[0], Q6.thr_unitvec[1], Q6.thr_unitvec[2]
                                      ]).astype(np.float64)
        arr_int_WPs     =   np.array([WPs.shape[0]]).astype(np.int32)
        arr_dbl_WPs     =   np.ravel(WPs,order='C').astype(np.float64)
        arr_dbl_VT      =   np.array([VT.Ri[0], VT.Ri[1], VT.Ri[2],
                                      ]).astype(np.float64)
        arr_Ai_est_dstb =   np.array(self.Ai_est_dstb).astype(np.float64)
        # occupy GPU memory space
        gpu_u1          =   cuda.mem_alloc(arr_u1.nbytes)
        gpu_u2          =   cuda.mem_alloc(arr_u2.nbytes)
        gpu_u3          =   cuda.mem_alloc(arr_u3.nbytes)
        gpu_delta_u1    =   cuda.mem_alloc(arr_delta_u1.nbytes)
        gpu_delta_u2    =   cuda.mem_alloc(arr_delta_u2.nbytes)
        gpu_delta_u3    =   cuda.mem_alloc(arr_delta_u3.nbytes)
        gpu_stk         =   cuda.mem_alloc(arr_stk.nbytes)
        gpu_res1        =   cuda.mem_alloc(arr_res1.nbytes)
        gpu_res2        =   cuda.mem_alloc(arr_res2.nbytes)
        gpu_res3        =   cuda.mem_alloc(arr_res3.nbytes)
        gpu_res4        =   cuda.mem_alloc(arr_res4.nbytes)
        gpu_int_MP      =   cuda.mem_alloc(arr_int_MP.nbytes)
        gpu_dbl_MP      =   cuda.mem_alloc(arr_dbl_MP.nbytes)
        gpu_int_Q6      =   cuda.mem_alloc(arr_int_Q6.nbytes)
        gpu_dbl_Q6      =   cuda.mem_alloc(arr_dbl_Q6.nbytes)
        gpu_int_WPs     =   cuda.mem_alloc(arr_int_WPs.nbytes)
        gpu_dbl_WPs     =   cuda.mem_alloc(arr_dbl_WPs.nbytes)
        gpu_dbl_VT      =   cuda.mem_alloc(arr_dbl_VT.nbytes)
        gpu_Ai_est_dstb =   cuda.mem_alloc(arr_Ai_est_dstb.nbytes)
        # convert data memory from CPU to GPU
        cuda.memcpy_htod(gpu_u1,arr_u1)
        cuda.memcpy_htod(gpu_u2,arr_u2)
        cuda.memcpy_htod(gpu_u3,arr_u3)
        cuda.memcpy_htod(gpu_delta_u1,arr_delta_u1)
        cuda.memcpy_htod(gpu_delta_u2,arr_delta_u2)
        cuda.memcpy_htod(gpu_delta_u3,arr_delta_u3)
        cuda.memcpy_htod(gpu_res1,arr_res1)
        cuda.memcpy_htod(gpu_res2,arr_res2)
        cuda.memcpy_htod(gpu_res3,arr_res3)
        cuda.memcpy_htod(gpu_res4,arr_res4)
        cuda.memcpy_htod(gpu_stk,arr_stk)
        cuda.memcpy_htod(gpu_int_MP,arr_int_MP)
        cuda.memcpy_htod(gpu_dbl_MP,arr_dbl_MP)
        cuda.memcpy_htod(gpu_int_Q6,arr_int_Q6)
        cuda.memcpy_htod(gpu_dbl_Q6,arr_dbl_Q6)
        cuda.memcpy_htod(gpu_int_WPs,arr_int_WPs)
        cuda.memcpy_htod(gpu_dbl_WPs,arr_dbl_WPs)
        cuda.memcpy_htod(gpu_dbl_VT,arr_dbl_VT)
        cuda.memcpy_htod(gpu_Ai_est_dstb,arr_Ai_est_dstb)
        
        #.. run MPPI Monte Carlo simulation code script
        # cuda code script function handler
        func_MC     =   SourceModule(self.total_MPPI_code).get_function("MPPI_monte_carlo_sim")
        # run cuda script by using GPU cores
        unit_gpu_allocation = 32        # GPU SP number
        blocksz     =   (unit_gpu_allocation, 1, 1)
        gridsz      =   (round(self.MP.K/(unit_gpu_allocation)), 1)
        func_MC(gpu_u1, gpu_u2, gpu_u3, 
                gpu_delta_u1, gpu_delta_u2, gpu_delta_u3, gpu_stk, 
                gpu_res1, gpu_res2, gpu_res3, gpu_res4, 
                gpu_int_MP, gpu_dbl_MP, gpu_int_Q6, gpu_dbl_Q6, 
                gpu_int_WPs, gpu_dbl_WPs, gpu_dbl_VT, gpu_Ai_est_dstb,
                block=blocksz, grid=gridsz)

        #.. variable setting - MPPI Monte Carlo simulation
        # set CPU variables
        arr_numer1  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_numer2  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_numer3  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom1  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom2  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_denom3  =   np.zeros((self.MP.N, self.MP.K)).astype(np.float64)
        arr_ent_param_float =   np.array([self.MP.lamb1, self.MP.lamb2, self.MP.lamb3]).astype(np.float64)
        # occupy GPU memory space
        gpu_numer1  =   cuda.mem_alloc(arr_numer1.nbytes)
        gpu_numer2  =   cuda.mem_alloc(arr_numer2.nbytes)
        gpu_numer3  =   cuda.mem_alloc(arr_numer3.nbytes)
        gpu_denom1  =   cuda.mem_alloc(arr_denom1.nbytes)
        gpu_denom2  =   cuda.mem_alloc(arr_denom2.nbytes)
        gpu_denom3  =   cuda.mem_alloc(arr_denom3.nbytes)
        gpu_ent_param_float =   cuda.mem_alloc(arr_ent_param_float.nbytes)
        # convert data memory from CPU to GPU
        cuda.memcpy_htod(gpu_numer1, arr_numer1)
        cuda.memcpy_htod(gpu_numer2, arr_numer2)
        cuda.memcpy_htod(gpu_numer3, arr_numer3)
        cuda.memcpy_htod(gpu_denom1, arr_denom1)
        cuda.memcpy_htod(gpu_denom2, arr_denom2)
        cuda.memcpy_htod(gpu_denom3, arr_denom3)
        cuda.memcpy_htod(gpu_ent_param_float, arr_ent_param_float)
        
        #.. run entropy calculation code script
        func_Ent    =   SourceModule(self.MPPI_entropy_calc_code).get_function("MPPI_entropy")
        # run cuda script by using GPU cores
        unit_gpu_allocation = 32        # GPU SP number
        blocksz     =   (unit_gpu_allocation, 1, 1)
        gridsz      =   (round(self.MP.K/(unit_gpu_allocation)), self.MP.N)
        func_Ent(gpu_numer1,gpu_numer2,gpu_numer3,gpu_denom1,gpu_denom2,gpu_denom3,
                 gpu_ent_param_float,gpu_delta_u1,gpu_delta_u2,gpu_delta_u3,gpu_stk, 
                 block=blocksz, grid=gridsz)
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
        
        if Q6.Guid_type == 3:
            # low pass fileter & lead phase
            u1 = self.u1.copy()
            u2 = self.u2.copy()
            u3 = self.u3.copy()
            tau_LPF     =   self.MP.dt * 2.
            # tau_LPF     =   0.1
            for i_u in range(self.MP.N - 1):
                du1     =   1/tau_LPF * (u1[i_u + 1] - u1[i_u])
                u1[i_u + 1] = u1[i_u] + du1 * self.MP.dt
                du2     =   1/tau_LPF * (u2[i_u + 1] - u2[i_u])
                u2[i_u + 1] = u2[i_u] + du2 * self.MP.dt
                du3     =   1/tau_LPF * (u1[i_u + 1] - u3[i_u])
                u3[i_u + 1] = u3[i_u] + du3 * self.MP.dt
                
            # N_tau_u     =   m.floor(0.7 * tau_LPF / self.MP.dt)
            # for i_N in range(N_tau_u):
            #     u1[0:self.MP.N - 1] = u1[1:self.MP.N]
            #     u2[0:self.MP.N - 1] = u2[1:self.MP.N]
            #     u3[0:self.MP.N - 1] = u3[1:self.MP.N]
            #     u1[self.MP.N-1]   =   self.MP.u1_init
            #     u2[self.MP.N-1]   =   self.MP.u2_init
            #     u3[self.MP.N-1]   =   self.MP.u3_init
                
            self.u1 = u1.copy()
            self.u2 = u2.copy()
            self.u3 = u3.copy()
            
        
        # MPPI result and update
        Aqi_cmd1    =   np.array([self.u1[0], self.u2[0], self.u3[0]])
        Aqi_cmd2    =   np.array([self.u1[1], self.u2[1], self.u3[1]])
        
        self.u1[0:self.MP.N-1]  =   self.u1[1:self.MP.N]
        self.u2[0:self.MP.N-1]  =   self.u2[1:self.MP.N]
        self.u3[0:self.MP.N-1]  =   self.u3[1:self.MP.N]
        self.u1[self.MP.N-1]    =   self.MP.u1_init
        self.u2[self.MP.N-1]    =   self.MP.u2_init
        self.u3[self.MP.N-1]    =   self.MP.u3_init
        
        return Aqi_cmd1, Aqi_cmd2
    
    def set_MPPI_entropy_calc_code(self):
        self.MPPI_entropy_calc_code = """
        __global__ void MPPI_entropy(double* arr_numer1, double* arr_numer2, double* arr_numer3, \
            double* arr_denom1, double* arr_denom2, double* arr_denom3, double *arr_ent_param_float, \
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
        }
        
        """
        pass

    # def set_total_MPPI_code(self, num_WPs):
        # self.total_MPPI_code = "#define nWP " + str(num_WPs) +  """
    def set_total_MPPI_code(self):
        self.total_MPPI_code = """
        /*.. Declaire Subfunctions ..*/
        // utility functions    
        __device__ double norm_(double x[3]);
        __device__ double dot(double x[3], double y[3]);
        __device__ void matmul_vec(double mat[3][3], double vec[3], double res[3]);
        __device__ void transpose(double mat[3][3], double res[3][3]);
        __device__ void azim_elev_from_vec3(double vec[3], double* azim, double* elev);
        __device__ void DCM_from_euler_angle(double ang_euler321[3], double DCM[3][3]);
        
        // simulation module functions
        
        /*.. main function ..*/    
        __global__ void MPPI_monte_carlo_sim(double* arr_u1, double* arr_u2, double* arr_u3, \
            double* arr_delta_u1, double* arr_delta_u2, double* arr_delta_u3, double* arr_stk, \
            double* arr_res1, double* arr_res2, double* arr_res3, double* arr_res4, \
            int* arr_int_MP, double* arr_dbl_MP, int* arr_int_Q6, double* arr_dbl_Q6, \
            int* arr_int_WPs, double* arr_dbl_WPs, double* arr_dbl_VT, double* arr_Ai_est_dstb)
        {
            //.. GPU core index for parallel computation
            int idx     =   threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;

            /*.. declare variables ..*/
            //.. MPPI variables
            int K       =   arr_int_MP[0];
            int N       =   arr_int_MP[1];
            int MP_flag_cost_calc   =   arr_int_MP[2];
            double SP_dt    =   arr_dbl_MP[0];
            double nu       =   arr_dbl_MP[1];
            double MP_R[3]  =   {arr_dbl_MP[2], arr_dbl_MP[3], arr_dbl_MP[4]};
            double MP_Q[4]  =   {arr_dbl_MP[5], arr_dbl_MP[6], arr_dbl_MP[7], arr_dbl_MP[8]};
            double MP_Q_lim     =   arr_dbl_MP[9];
            double MP_a_lim     =   arr_dbl_MP[10];
            double MP_u1_init   =   arr_dbl_MP[11];
            double MP_u2_init   =   arr_dbl_MP[12];
            double MP_u3_init   =   arr_dbl_MP[13];
            double R_mat[3][3] = {0.,};
            for(int i_3 = 0; i_3 < 3; i_3++)
                R_mat[i_3][i_3] = MP_R[i_3];
            
            //.. global constant
            double pi   =   acos(-1.);
            double grav_i[3]    =   {0., 0., 9.81};
            
            //.. quadrotor variables
            // parameters
            double Q6_throttle_hover    =    arr_dbl_Q6[0];
            double Q6_desired_speed     =   arr_dbl_Q6[2];
            double Q6_look_ahead_distance   =   arr_dbl_Q6[3];
            double Q6_distance_change_WP    =   arr_dbl_Q6[4];
            double Q6_Kp_vel    =   arr_dbl_Q6[5];
            double Q6_Kd_vel    =   arr_dbl_Q6[6];
            double Q6_Kp_speed  =   arr_dbl_Q6[7];
            double Q6_Kd_speed  =   arr_dbl_Q6[8];
            double Q6_guid_eta  =   arr_dbl_Q6[9];
            double Q6_tau_phi   =   arr_dbl_Q6[10];
            double Q6_tau_the   =   arr_dbl_Q6[11];
            double Q6_tau_psi   =   arr_dbl_Q6[12];
            // variables
            double Q6_Ri[3]     =   {arr_dbl_Q6[13], arr_dbl_Q6[14], arr_dbl_Q6[15]};
            double Q6_Vi[3]     =   {arr_dbl_Q6[16], arr_dbl_Q6[17], arr_dbl_Q6[18]};
            double Q6_Ai[3]     =   {arr_dbl_Q6[19], arr_dbl_Q6[20], arr_dbl_Q6[21]};
            double Q6_thr_unitvec[3]    =   {arr_dbl_Q6[22], arr_dbl_Q6[23], arr_dbl_Q6[24]};
            double Q6_cI_W[3][3]    =   {0.,};
            int Q6_WP_idx_heading   =   arr_int_Q6[0];
            int Q6_WP_idx_passed    =   arr_int_Q6[1];
            int Q6_Guid_type        =   arr_int_Q6[2];
            int Q6_flag_guid_trans  =   arr_int_Q6[3]; 
            double Q6_p_closest_on_path[3]  =   {0.,};
            double Q6_cost_total    =   0.;
            int flag_stop       =   0;
            double Q6_Ai_est_dstb[3]    =   {0.,};
            
            //.. virtual target variables
            // parameters
            // variables
            double VT_Ri[3]     =   {arr_dbl_VT[0], arr_dbl_VT[1], arr_dbl_VT[2]};
            
            // set waypoint
            int nWP = arr_int_WPs[0];
            double WP_WPs[500][3]   =   {0.,};
            for(int i_WP = 0; i_WP < nWP; i_WP++){
                for(int i_3 = 0; i_3 < 3; i_3++){
                    WP_WPs[i_WP][i_3] = arr_dbl_WPs[i_WP*3 + i_3];
                }
            }
                
            
            /*.. main loop ..*/
            //.. declare variables
            int tmp_WP_idx_passed = 0;
            
            //.. loop start
            for(int i_N = 0; i_N < N; i_N++){
                
                //.. MPPI module
                double MPPI_ctrl_input[3] = {0.,};
                if(Q6_Guid_type >= 2){
                    MPPI_ctrl_input[0] = arr_u1[i_N] + arr_delta_u1[idx + K*i_N];
                    MPPI_ctrl_input[1] = arr_u2[i_N] + arr_delta_u2[idx + K*i_N];
                    MPPI_ctrl_input[2] = arr_u3[i_N] + arr_delta_u3[idx + K*i_N];
                }
                
                
                //.. Environment
                double Aqi_aero[3] = {0.,};
                for (int i_3 = 0; i_3 < 3; i_3++){
                    Q6_Ai_est_dstb[i_3] = arr_Ai_est_dstb[i_3 + 3*i_N];
                    Aqi_aero[i_3] = Q6_Ai_est_dstb[i_3];
                }
                double Aqi_grav[3]  =   {grav_i[0], grav_i[1], grav_i[2]};
                
                //.. kinematics
                double Rqti[3] = {0.,};
                double mag_Rqti = 0.;
                double mag_Vqi = 0.;
                double LOS_azim = 0., LOS_elev = 0.;
                double tgo = 0.;
                double FPA_azim = 0., FPA_elev = 0.;
                double LA_azim = 0., LA_elev = 0.;
                for (int i_3 = 0; i_3 < 3; i_3++)
                    Rqti[i_3] = VT_Ri[i_3] - Q6_Ri[i_3];
                mag_Rqti    =   norm_(Rqti);
                mag_Vqi     =   norm_(Q6_Vi);
                azim_elev_from_vec3(Rqti, &LOS_azim, &LOS_elev);
                tgo         =   mag_Rqti / max(mag_Vqi, 0.001);
                azim_elev_from_vec3(Q6_Vi, &FPA_azim, &FPA_elev);
                double FPA_euler321[3] =   {0., FPA_elev, FPA_azim};
                DCM_from_euler_angle(FPA_euler321, Q6_cI_W);
                LA_azim     =   FPA_azim - LOS_azim;
                LA_elev     =   FPA_elev - LOS_elev;
                
                
                //.. distance from quadrotor to ref. path
                double Rw1w2[3]     =   {0.,};
                double mag_Rw1w2    =   0.;
                double Rw1q[3]      =   {0.,};
                double mag_w1p      =   0.;
                double p_closest_on_path[3] =   {0.,};
                double Rqp[3]       =   {0.,};
                double mag_Rqp      =   0.;
                double unit_Rw1w2[3]    =   {0.,};
                
                double dist_to_path =   999999.;
                for (int i_WP = Q6_WP_idx_heading; i_WP>0; i_WP--){
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        Rw1w2[i_3] = WP_WPs[i_WP][i_3] - WP_WPs[i_WP-1][i_3];
                    mag_Rw1w2   =   norm_(Rw1w2);
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        Rw1q[i_3] = Q6_Ri[i_3] - WP_WPs[i_WP-1][i_3];
                    mag_w1p     =   min(max(dot(Rw1w2, Rw1q)/max(mag_Rw1w2,0.001), 0.), mag_Rw1w2);
                    for (int i_3 = 0; i_3 < 3; i_3++){
                        p_closest_on_path[i_3]  =   WP_WPs[i_WP-1][i_3] + mag_w1p * Rw1w2[i_3]/max(mag_Rw1w2,0.001);
                        Rqp[i_3] = p_closest_on_path[i_3] - Q6_Ri[i_3];
                    }
                    mag_Rqp     =   norm_(Rqp);
                    if (dist_to_path < mag_Rqp)
                        break;
                    else{
                        for (int i_3 = 0; i_3 < 3; i_3++)
                            unit_Rw1w2[i_3] = Rw1w2[i_3]/max(mag_Rw1w2,0.001);
                        dist_to_path    =   mag_Rqp;
                        for (int i_3 = 0; i_3 < 3; i_3++)
                            Q6_p_closest_on_path[i_3] = p_closest_on_path[i_3];
                        Q6_WP_idx_passed    =   max(i_WP-1, tmp_WP_idx_passed);
                    }
                }
                
                
                //.. setting guidance module
                if (Q6_Guid_type == 3){
                    if(Q6_flag_guid_trans == 0){
                        Q6_look_ahead_distance  =   MPPI_ctrl_input[0];
                        Q6_desired_speed        =   MPPI_ctrl_input[1];
                        Q6_guid_eta             =   MPPI_ctrl_input[2];
                    }
                    else{
                        Q6_look_ahead_distance  =   MP_u1_init;
                        Q6_desired_speed        =   MP_u2_init;
                        Q6_guid_eta             =   MP_u3_init;
                    }
                }
                        
                //.. virtual target modules
                //.. directly decide a position of the virtual target 
                double Rqw2i[3] = {0.,};
                double mag_Rqw2i = 0.;
                double total_len = 0.;
                double p1[3] = {0.,};
                double p2[3] = {0.,};
                double Rp1p2[3] = {0.,};
                double mag_Rp1p2 = 0.;
                double mag_Rp1t = 0.;
                
                // check waypoint - quadrotor
                for (int i_3 = 0; i_3 < 3; i_3++)
                    Rqw2i[i_3] = WP_WPs[Q6_WP_idx_heading][i_3] - Q6_Ri[i_3];
                mag_Rqw2i   =   norm_(Rqw2i);
                if (mag_Rqw2i < Q6_distance_change_WP){
                    Q6_WP_idx_heading = min(Q6_WP_idx_heading + 1, nWP - 1);
                }
                // virtual target position       
                if (dist_to_path >= Q6_look_ahead_distance){
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        VT_Ri[i_3]   =   Q6_p_closest_on_path[i_3];
                }
                else{
                    total_len   = dist_to_path;
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        p1[i_3]  =   Q6_p_closest_on_path[i_3];
                    for (int i_WP = Q6_WP_idx_passed+1; i_WP < nWP; i_WP++)
                    {
                        // check segment whether Rti exist
                        for (int i_3 = 0; i_3 < 3; i_3++){
                            p2[i_3]     =   WP_WPs[i_WP][i_3];
                            Rp1p2[i_3]  =   p2[i_3] - p1[i_3];
                        }
                        mag_Rp1p2   =   norm_(Rp1p2);
                        if (total_len + mag_Rp1p2 > Q6_look_ahead_distance){
                            mag_Rp1t    =   Q6_look_ahead_distance - total_len;
                            for (int i_3 = 0; i_3 < 3; i_3++)
                                VT_Ri[i_3] = p1[i_3] + mag_Rp1t * Rp1p2[i_3]/max(mag_Rp1p2,0.001);
                            break;
                        }
                        else{
                            for (int i_3 = 0; i_3 < 3; i_3++)
                                p1[i_3] = p2[i_3];
                            total_len   =   total_len + mag_Rp1p2;
                            if (i_WP == nWP - 1)
                                for (int i_3 = 0; i_3 < 3; i_3++)
                                    VT_Ri[i_3]   =   p2[i_3];
                        }
                    }
                }
                
                //.. guidance modules
                // guidance transition
                //Q6_flag_guid_trans = 0;
                double ratio_guid_type[6] = {0.,};
                ratio_guid_type[Q6_Guid_type] = 1.;
                if (Q6_Guid_type != 0){
                    if (mag_Vqi < 0.2 || Q6_WP_idx_passed < 1){
                        // low speed --> generate FPA firstly for efficiently using guidance law
                        Q6_flag_guid_trans = 1;
                    }
                    else if (mag_Vqi > 0.5){
                        // high speed --> enough speed to avoid flag chattering
                        Q6_flag_guid_trans = 0;
                    }
                    else{
                        // gap between mag_Vqi (0.2, 0.5)
                    }
                    if (Q6_WP_idx_heading == (nWP - 1)){
                        // low speed approach to terminal WP
                        Q6_flag_guid_trans = 1;
                        ratio_guid_type[0] = 1.;
                        ratio_guid_type[Q6_Guid_type]   =   1. - ratio_guid_type[0];
                    }
                    if (Q6_flag_guid_trans == 1){
                        ratio_guid_type[0] = 1.;
                        ratio_guid_type[Q6_Guid_type]   =   1. - ratio_guid_type[0];
                    }
                }
                
                // guidance command
                double Aqi_cmd[3]   =   {0.,};
                if (Q6_Guid_type == 0 || Q6_flag_guid_trans == 1){
                    //.. guidance - position & velocity control
                    double err_Ri[3] = {0.,};
                    double Kp_pos = 0.;
                    double derr_Ri[3] = {0.,};
                    double Vqi_cmd[3] = {0.,};
                    double dVqi_cmd[3] = {0.,};
                    double err_Vi[3] = {0.,};
                    double derr_Vi[3] = {0.,};
                    // position control
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        err_Ri[i_3] = VT_Ri[i_3] - Q6_Ri[i_3];
                    Kp_pos = Q6_desired_speed/max(norm_(err_Ri),Q6_desired_speed);   // (terminal WP, tgo < 1) --> decreasing speed
                    for (int i_3 = 0; i_3 < 3; i_3++){
                        derr_Ri[i_3] = 0. - Q6_Vi[i_3];
                        Vqi_cmd[i_3] = Kp_pos * err_Ri[i_3];        // set (Vqi_cmd = Q6.desired_speed)
                        dVqi_cmd[i_3] = Kp_pos * derr_Ri[i_3];
                    }
                    // velocity control
                    for (int i_3 = 0; i_3 < 3; i_3++){
                        err_Vi[i_3] = Vqi_cmd[i_3] - Q6_Vi[i_3];
                        derr_Vi[i_3] = dVqi_cmd[i_3] - Q6_Ai[i_3];
                        Aqi_cmd[i_3] = Aqi_cmd[i_3] + ratio_guid_type[0] * (Q6_Kp_vel * err_Vi[i_3] + Q6_Kd_vel * derr_Vi[i_3]);
                    }
                }
                if (Q6_Guid_type == 1 || (Q6_Guid_type == 1 && Q6_flag_guid_trans == 1)){
                    //.. guidance - GL -based
                    double err_mag_V = 0.;
                    double dmag_Vqi = 0.;
                    double derr_mag_V = 0.;
                    double Rqtw[3] = {0.,};
                    double err_azim = 0., err_elev = 0.;
                    double Aqw_cmd[3] = {0.,};
                    double cW_I[3][3] = {0.,};
                    double Aqi_cmd_tmp[3] = {0.,};
                    // a_x command
                    err_mag_V   =   Q6_desired_speed - mag_Vqi;
                    dmag_Vqi    =   dot(Q6_Vi, Q6_Ai) / max(mag_Vqi, 0.1);
                    derr_mag_V  =   0. - dmag_Vqi;
                    Aqw_cmd[0]  =   Q6_Kp_speed * err_mag_V + Q6_Kd_speed * derr_mag_V;
                    // optimal pursuit guidance law
                    matmul_vec(Q6_cI_W, Rqti, Rqtw);
                    azim_elev_from_vec3(Rqtw, &err_azim, &err_elev);
                    Aqw_cmd[1]  =   Q6_guid_eta*mag_Vqi / max(tgo, 0.001) * err_azim;
                    Aqw_cmd[2]  =   -Q6_guid_eta*mag_Vqi / max(tgo, 0.001) * err_elev;
                    // command coordinate change
                    transpose(Q6_cI_W, cW_I);
                    matmul_vec(cW_I, Aqw_cmd, Aqi_cmd_tmp);
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        Aqi_cmd[i_3] =   Aqi_cmd[i_3] + ratio_guid_type[1] * Aqi_cmd_tmp[i_3];
                }
                if (Q6_Guid_type == 2 || (Q6_Guid_type == 2 && Q6_flag_guid_trans == 1)){
                    //.. guidance - MPPI direct method
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        Aqi_cmd[i_3]  =   Aqi_cmd[i_3] + ratio_guid_type[2] * MPPI_ctrl_input[i_3];
                }
                if (Q6_Guid_type == 3 || (Q6_Guid_type == 3 && Q6_flag_guid_trans == 1)){
                    //.. guidance - GL -based
                    double err_mag_V = 0.;
                    double dmag_Vqi = 0.;
                    double derr_mag_V = 0.;
                    double Rqtw[3] = {0.,};
                    double err_azim = 0., err_elev = 0.;
                    double Aqw_cmd[3] = {0.,};
                    double cW_I[3][3] = {0.,};
                    double Aqi_cmd_tmp[3] = {0.,};
                    // a_x command
                    err_mag_V   =   Q6_desired_speed - mag_Vqi;
                    dmag_Vqi    =   dot(Q6_Vi, Q6_Ai) / max(mag_Vqi, 0.1);
                    derr_mag_V  =   0. - dmag_Vqi;
                    Aqw_cmd[0]  =   Q6_Kp_speed * err_mag_V + Q6_Kd_speed * derr_mag_V;
                    // optimal pursuit guidance law
                    matmul_vec(Q6_cI_W, Rqti, Rqtw);
                    azim_elev_from_vec3(Rqtw, &err_azim, &err_elev);
                    Aqw_cmd[1]  =   Q6_guid_eta* 3. / 1.5 * err_azim;
                    Aqw_cmd[2]  =   -Q6_guid_eta* 3. / 1.5 * err_elev;
                    // command coordinate change
                    transpose(Q6_cI_W, cW_I);
                    matmul_vec(cW_I, Aqw_cmd, Aqi_cmd_tmp);
                    for (int i_3 = 0; i_3 < 3; i_3++)
                        Aqi_cmd[i_3] =   Aqi_cmd[i_3] + ratio_guid_type[Q6_Guid_type] * Aqi_cmd_tmp[i_3];
                }
                
                // calc. cost_uRu
                double cost_uRu     =    0.;
                double tmp_Ru[3] = {0.,};
                matmul_vec(R_mat, Aqi_cmd, tmp_Ru);
                cost_uRu    =   dot(Aqi_cmd, tmp_Ru);
                
                // compensate disturbance
                for (int i_3 = 0; i_3 < 3; i_3++){
                    Aqi_cmd[i_3] = Aqi_cmd[i_3] - Q6_Ai_est_dstb[i_3];
                }
            
                // compensate gravity
                for (int i_3 = 0; i_3 < 3; i_3++)
                    Aqi_cmd[i_3] = Aqi_cmd[i_3] - Aqi_grav[i_3];

                //.. thrust & attitude control - no delay
                double Aqi_thru[3] = {0.,};

                //.. pseudo thrust & attitude control
                double max_mag_Aqi_thru =   9.81 / Q6_throttle_hover * MP_a_lim;
                double mag_Aqi_thru     =   min(norm_(Aqi_cmd), max_mag_Aqi_thru);
                double thr_unitvec_cmd[3] = {0.,};
                double dot_thr_unitvec[3] = {0.,};
                
                for (int i_3 = 0; i_3 < 3; i_3++){
                    thr_unitvec_cmd[i_3] = Aqi_cmd[i_3] / norm_(Aqi_cmd);
                    dot_thr_unitvec[i_3] = 1./Q6_tau_phi*(thr_unitvec_cmd[i_3] - Q6_thr_unitvec[i_3]);
                    Aqi_thru[i_3] = mag_Aqi_thru * Q6_thr_unitvec[i_3];
                }
                
                //.. dynamics
                double dot_Rqi[3] = {0.,};
                double dot_Vqi[3] = {0.,};
                for (int i_3 = 0; i_3 < 3; i_3++){
                    Q6_Ai[i_3] = Aqi_thru[i_3] + Aqi_aero[i_3] + Aqi_grav[i_3];
                    dot_Rqi[i_3] = Q6_Vi[i_3];
                    dot_Vqi[i_3] = Q6_Ai[i_3];
                }
                
                
                //.. cost function
                double mag_Vqi_alinged_path = max(dot(unit_Rw1w2, Q6_Vi), 0.);
                double cost_distance    =   MP_Q[0] * (dist_to_path * dist_to_path);
                if(dist_to_path > MP_Q_lim)
                    cost_distance = cost_distance + MP_Q[2] * (dist_to_path * dist_to_path);
                
                double energy_cost1    =   norm_(Aqi_thru)/max(mag_Vqi_alinged_path, 0.1);
                
                
                double cost_a   =   MP_Q[1] * energy_cost1;
                double cost[3]  =   {cost_distance, cost_a, cost_uRu};
                
                double cost_sum =   cost[0] + cost[1] + cost[2];
                
                Q6_cost_total   =   Q6_cost_total + cost_sum*SP_dt;
                
                //.. MPPI cost
                arr_stk[idx]    =   arr_stk[idx] + cost_sum;
                
                /*
                //.. save data
                arr_res1[i_N]   =   cost[0];
                arr_res2[i_N]   =   cost[1];
                arr_res3[i_N]   =   cost[2];
                arr_res4[i_N]   =   cost_sum;
                */

                //.. update variables
                for (int i_3 = 0; i_3 < 3; i_3++){
                    Q6_Ri[i_3] = Q6_Ri[i_3] + dot_Rqi[i_3] * SP_dt;
                    Q6_Vi[i_3] = Q6_Vi[i_3] + dot_Vqi[i_3] * SP_dt;
                    Q6_thr_unitvec[i_3] = Q6_thr_unitvec[i_3] + dot_thr_unitvec[i_3] * SP_dt;
                }
                
                
                /*
                //.. check stop
                // stop around terminal WP, (due to using transition mode, high cost)
                if (Q6_WP_idx_heading == (nWP - 1)) {
                    flag_stop       =   1;
                }
                
                if (flag_stop == 1){
                    double stk_weight = (double)N /(double)(i_N+1);
                    arr_stk[idx]    =   stk_weight * arr_stk[idx];
                    break;
                }
                */
                    
                
            }   // main loop end
        }
        
        __device__ double norm_(double x[3])
        {
            return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        }
        __device__ double dot(double x[3], double y[3]) 
        {
            return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
        }
        __device__ void matmul_vec(double mat[3][3], double vec[3], double res[3])
        {
            res[0]  =   mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2];
            res[1]  =   mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2];
            res[2]  =   mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2];
        }
        __device__ void transpose(double mat[3][3], double res[3][3])
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
            
        """
        pass

    def set_commom_code__initialze_variables(self):
        self.commom_code__initialze_variables = 1
        pass