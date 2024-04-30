import cv2
import numpy as np
import math

import onnx
import onnxruntime as rt
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from .utils_gray import preprocess
from rclpy.qos import ReliabilityPolicy, QoSProfile, LivelinessPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import VehicleCommand, OffboardControlMode , TrajectorySetpoint, VehicleAttitude, VehicleLocalPosition
#############################################################################################################
# added by controller
from custom_msgs.msg import Heartbeat
#############################################################################################################
class JBNU_Collision(Node):
    def __init__(self):
        super().__init__('jbnu_collision')
        
#############################################################################################################
# added by controller
        self.qosProfileGen()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        ## publiser and subscriber
        # declare publisher from ROS2 to PX4
        self.vehicle_command_publisher              =   self.create_publisher(VehicleCommand,               '/fmu/in/vehicle_command',           qos_profile)
        self.offboard_control_mode_publisher        =   self.create_publisher(OffboardControlMode,          '/fmu/in/offboard_control_mode',     qos_profile)
        self.trajectory_setpoint_publisher          =   self.create_publisher(TrajectorySetpoint,           '/fmu/in/trajectory_setpoint',       qos_profile)

        self.vehicle_attitude_subscriber            =   self.create_subscription(VehicleAttitude,           '/fmu/out/vehicle_attitude',          self.vehicle_attitude_callback,         qos_profile)
        self.vehicle_local_position_subscriber      =   self.create_subscription(VehicleLocalPosition,      '/fmu/out/vehicle_local_position',    self.vehicle_local_position_callback,   qos_profile)
        period_offboard_control_mode =   0.2         # required about 5Hz for attitude control (proof that the external controller is healthy
        self.offboard_main_timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_main)

        period_offboard_vel_ctrl    =   0.02         # required 50Hz at least for velocity control
        self.velocity_control_call_timer =  self.create_timer(period_offboard_vel_ctrl, self.publish_vehicle_velocity_setpoint)

        ###.. - Start - set variable of publisher msg for PX4 - ROS2  ..###
        #
        #.. parameter - vehicle command 
        class prm_msg_veh_com:
            def __init__(self):
                self.CMD_mode   =   np.NaN
                self.params     =   np.NaN * np.ones(2)

        # offboard mode command in ref. [3]
        self.prm_offboard_mode            =   prm_msg_veh_com()
        self.prm_offboard_mode.CMD_mode   =   VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        self.prm_offboard_mode.params[0]  =   1
        self.prm_offboard_mode.params[1]  =   6

        #.. parameter - offboard control mode
        class prm_msg_off_con_mod:
            def __init__(self):        
                self.position        =   False
                self.velocity        =   False
                self.acceleration    =   False
                self.attitude        =   False
                self.body_rate       =   False

        self.prm_off_con_mod            =   prm_msg_off_con_mod()
        self.prm_off_con_mod.velocity   =   True

        class msg_veh_trj_set:
            def __init__(self):
                self.pos_NED        =   np.NaN * np.ones(3)   # meters
                self.vel_NED        =   np.zeros(3)     # meters/second
                self.yaw_rad        =   np.NaN
                self.yaw_vel_rad    =   0.                    # radians/second
        
        self.veh_trj_set    =   msg_veh_trj_set()

        self.vel_cmd_body_x = 0
        self.vel_cmd_body_y = 0
        self.vel_cmd_body_z = 0
        self.psi    =   0
        self.theta  =   0
        self.phi    =   0
        self.x      =   0
        self.y      =   0
        self.z      =   0
        self.v_x    =   0
        self.v_y    =   0
        self.v_z    =   0
        self.u   =   0
        self.v   =   0
        self.w   =   0        
        #.. callback state_logger
        self.period_state_logger = 0.1
        self.timer  =   self.create_timer(self.period_state_logger, self.state_logger)
        self.flightlogFile = open("/home/user/ros_ws/log/flight_log.txt",'w')


        self.current_frame = []
#############################################################################################################
        model_pretrained = onnx.load("/home/user/ros_ws/src/collision_avoidance/model/Inha.onnx")
        self.sess = rt.InferenceSession(model_pretrained.SerializeToString(), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.subscription = self.create_subscription(Image, '/depth/raw', self.depth_sub, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))


        # self.CameraSubscriber_ = self.create_subscription(Image, '/airsim_node/Typhoon_1/DptCamera/DepthPerspective', self.depth_sub, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.bridge = CvBridge()
        self.get_logger().info("Learning_based_feedforward node initialized")

        self.collision_avoidance_period = 0.1
        self.collision_avoidance_timer =  self.create_timer(self.collision_avoidance_period, self.collision_avoidance)


    def collision_avoidance(self):
        if len(self.current_frame) > 0:

            image = preprocess(self.current_frame)

            image = np.array([image])  # The model expects a 4D array
            self.image = image.astype(np.float32)

            infer = self.sess.run([self.output_name], {self.input_name: self.image})
            infer = infer[0][0]
            # self.get_logger().info("infer =" + str(infer) )
            vx = infer[0]
            vy = infer[1]
            vz = infer[2]
            vyaw = infer[3] * 1.0

            cmd = Twist()
            self.vel_cmd_body_x = float(vx) # [meters/second]
            self.vel_cmd_body_y = float(vy) # [meters/second]
            self.vel_cmd_body_z = float(vz) # [meters/second]
            self.veh_trj_set.yaw_vel_rad = vyaw
            self.BodytoNED()
        else :
            pass


    def depth_sub(self, msg):
        # check another module nodes alive
        try:
            # Convert the ROS Image message to OpenCV format
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting Image message: {e}")
            return

        self.current_frame = np.interp(image, (0, 10.0), (0, 255))

        # valid_mask = image < 100

        # valid_depths = image[valid_mask]

        # scaled_depths = np.interp(valid_depths, (valid_depths.min(), 20), (0, 255))

        # self.current_frame = np.full(image.shape, 255, dtype=np.uint8)
            
        # self.current_frame[valid_mask] = scaled_depths.astype(np.uint8)
        
        # cv2.imshow('walid', image.astype(np.uint8))
        # cv2.waitKey(0)

    def qosProfileGen(self):
    #   Reliability : 데이터 전송에 있어 속도를 우선시 하는지 신뢰성을 우선시 하는지를 결정하는 QoS 옵션
    #   History : 데이터를 몇 개나 보관할지를 결정하는 QoS 옵션
    #   Durability : 데이터를 수신하는 서브스크라이버가 생성되기 전의 데이터를 사용할지 폐기할지에 대한 QoS 옵션
        self.QOS_Sub_Sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        
        self.QOS_Service = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE)
#############################################################################################################

    def offboard_control_main(self):
        # offboard mode cmd to px4
        self.publish_vehicle_command(self.prm_offboard_mode)

        # send offboard heartbeat signal to px4 
        self.publish_offboard_control_mode(self.prm_off_con_mod)

    ## publish to px4
    # publish_vehicle_command to px4
    def publish_vehicle_command(self, prm_veh_com):
        msg                 =   VehicleCommand()
        msg.param1          =   prm_veh_com.params[0]
        msg.param2          =   prm_veh_com.params[1]
        msg.command         =   prm_veh_com.CMD_mode
        # values below are in [3]
        msg.target_system   =   1
        msg.target_component=   1
        msg.source_system   =   1
        msg.source_component=   1
        msg.from_external   =   True
        self.vehicle_command_publisher.publish(msg)

    # publish offboard control mode to px4
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg                 =   OffboardControlMode()
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher.publish(msg)

    # publish velocity offboard command to px4
    def publish_vehicle_velocity_setpoint(self):
        msg                 =   TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   np.float32(self.veh_trj_set.pos_NED)
        msg.velocity        =   np.float32(self.veh_trj_set.vel_NED)  # [meters/second]
        msg.yaw             =   self.veh_trj_set.yaw_rad
        msg.yawspeed        =   self.veh_trj_set.yaw_vel_rad
        self.trajectory_setpoint_publisher.publish(msg)

    def DCM(self, _phi, _theta, _psi):
        PHI = math.radians(_phi)  
        THETA = math.radians(_theta)
        PSI = math.radians(_psi)
        # print(PHI, THETA, PSI)

        mtx_DCM = np.array([[math.cos(PSI)*math.cos(THETA), math.sin(PSI)*math.cos(THETA), -math.sin(THETA)], 
                            [(-math.sin(PSI)*math.cos(PHI))+(math.cos(PSI)*math.sin(THETA)*math.sin(PHI)), (math.cos(PSI)*math.cos(PHI))+(math.sin(PSI)*math.sin(THETA)*math.sin(PHI)), math.cos(THETA)*math.sin(PHI)], 
                            [(math.sin(PSI)*math.sin(PHI))+(math.cos(PSI)*math.sin(THETA)*math.cos(PHI)), (-math.cos(PSI)*math.sin(PHI))+(math.sin(PSI)*math.sin(THETA)*math.cos(PHI)), math.cos(THETA)*math.cos(PHI)]])
       
        return mtx_DCM

    # update attitude from px4
    def vehicle_attitude_callback(self, msg):
        self.psi , self.theta, self.phi     =   self.Quaternion2Euler(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
        self.DCM_nb = self.DCM(self.phi, self.theta, self.psi)
        self.DCM_bn = np.transpose(self.DCM_nb)

    ## subscribe from px4
    # update position and velocity from px4
    def vehicle_local_position_callback(self, msg):
        # update NED position 
        self.x      =   msg.x
        self.y      =   msg.y
        self.z      =   msg.z
        # update NED velocity
        self.v_x    =   msg.vx
        self.v_y    =   msg.vy
        self.v_z    =   msg.vz

        vel_NED = np.array([self.v_x,self.v_y,self.v_z])
        self.u,self.v,self.w = np.array((self.DCM_nb @ vel_NED).tolist())

        
    # quaternion to euler
    def Quaternion2Euler(self, w, x, y, z):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        Roll = math.atan2(t0, t1) * 57.2958

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Pitch = math.asin(t2) * 57.2958

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Yaw = math.atan2(t3, t4) * 57.2958

        return Roll, Pitch, Yaw
    
    def BodytoNED(self):
        vel_cmd_body = np.array([self.vel_cmd_body_x,self.vel_cmd_body_y,self.vel_cmd_body_z])
        
        self.veh_trj_set.vel_NED = np.array((self.DCM_bn @ vel_cmd_body).tolist())


    def state_logger(self):
        self.get_logger().info("-----------------")
        self.get_logger().info("NED Velocity CMD:   [cmd_x]=" + str(self.veh_trj_set.vel_NED[0]) +", [cmd_y]=" + str(self.veh_trj_set.vel_NED[1]) +", [cmd_z]=" + str(self.veh_trj_set.vel_NED[2]))
        self.get_logger().info("Body Velocity CMD:  [cmd_x]=" + str(self.vel_cmd_body_x) +",    [cmd_y]=" + str(self.vel_cmd_body_y) +",    [cmd_z]=" + str(self.vel_cmd_body_z))
        self.get_logger().info("NED Velocity:       [x]=" + str(self.v_x) +", [y]=" + str(self.v_y) +", [z]=" + str(self.v_z))
        self.get_logger().info("Body Velocity:      [x]=" + str(self.u) +", [y]=" + str(self.v) +", [z]=" + str(self.w))
        flightlog = "%f %f %f %f %f %f %f %f %f %f %f %f\n" %(
            self.veh_trj_set.vel_NED[0], self.veh_trj_set.vel_NED[1], self.veh_trj_set.vel_NED[2],
            self.vel_cmd_body_x,  self.vel_cmd_body_y,  self.vel_cmd_body_z,
            self.v_x, self.v_y, self.v_z, self.u, self.v, self.w)
        self.flightlogFile.write(flightlog)


def main(args=None):
    rclpy.init(args=args)
    tensor = JBNU_Collision()
    rclpy.spin(tensor)
    tensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
