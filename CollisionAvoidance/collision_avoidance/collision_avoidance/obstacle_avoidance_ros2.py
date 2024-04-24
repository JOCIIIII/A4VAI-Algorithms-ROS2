import cv2
import numpy as np
import onnx
import onnxruntime as rt
import rclpy
from rclpy.node import Node
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
#############################################################################################################
# added by controller
from custom_msgs.msg import Heartbeat
#############################################################################################################
class JBNU_Collision(Node):
    def __init__(self):
        super().__init__('jbnu_collision')
        
#############################################################################################################
# added by controller
        self.path_planning_heartbeat            =   False
        self.path_following_heartbeat           =   False
        self.controller_heartbeat               =   False

        # declare heartbeat_publisher 
        self.heartbeat_publisher                        =   self.create_publisher(Heartbeat,    '/collision_avoidance_heartbeat', 10)
        # declare heartbeat_subscriber 
        self.controller_heartbeat_subscriber            =   self.create_subscription(Heartbeat, '/controller_heartbeat',            self.controller_heartbeat_call_back,            10)
        self.path_following_heartbeat_subscriber        =   self.create_subscription(Heartbeat, '/path_following_heartbeat',        self.path_following_heartbeat_call_back,        10)
        self.path_planning_heartbeat_subscriber         =   self.create_subscription(Heartbeat, '/path_planning_heartbeat',         self.path_planning_heartbeat_call_back,         10)

        # declare heartbeat_timer
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)


        self.image = []
#############################################################################################################
        model_pretrained = onnx.load("/home/user/ros_ws/src/collision_avoidance/model/Inha.onnx")
        self.sess = rt.InferenceSession(model_pretrained.SerializeToString(), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.subscription = self.create_subscription(Image, '/depth/raw', self.depth_sub, 1)


        # self.CameraSubscriber_ = self.create_subscription(Image, '/airsim_node/Typhoon_1/DptCamera/DepthPerspective', self.depth_sub, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.publisher_cmd = self.create_publisher(Twist, '/ca_vel_2_control', 1)
        self.bridge = CvBridge()
        self.get_logger().info("Learning_based_feedforward node initialized")

        self.collision_avoidance_period = 0.02
        self.collision_avoidance_timer =  self.create_timer(self.collision_avoidance_period, self.collision_avoidance)


    def collision_avoidance(self):
        if self.path_following_heartbeat == True and self.path_planning_heartbeat == True and self.controller_heartbeat == True:
            if len(self.image) > 0:
                infer = self.sess.run([self.output_name], {self.input_name: self.image})
                infer = infer[0][0]
                # self.get_logger().info("infer =" + str(infer) )
                vx = infer[0]
                vy = infer[1]
                vz = infer[2]
                vyaw = infer[3] * 1.0

                cmd = Twist()
                cmd.linear.x = float(vx)
                cmd.linear.y = float(vy)
                cmd.linear.z = float(vz)
                cmd.angular.z = vyaw * 2.0
                self.publisher_cmd.publish(cmd)
            else :
                pass
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

        # Your preprocessing steps here
        # image = np.interp(image, (0, 10.0), (0, 255))
        
        valid_mask = image < 100

        valid_depths = image[valid_mask]

        scaled_depths = np.interp(valid_depths, (valid_depths.min(), valid_depths.max()), (0, 255))

        output_image = np.full(image.shape, 255, dtype=np.uint8)
            
        output_image[valid_mask] = scaled_depths.astype(np.uint8)
        image = preprocess(output_image)

        # cv2.imshow('walid', image.astype(np.uint8))
        # cv2.waitKey(0)

        image = np.array([image])  # The model expects a 4D array
        self.image = image.astype(np.float32)

# heartbeat check function
    # heartbeat publish
    def publish_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.heartbeat_publisher.publish(msg)

    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self,msg):
        self.controller_heartbeat = msg.heartbeat

    # heartbeat subscribe from path following
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.heartbeat

    # heartbeat subscribe from collision avoidance
    def path_following_heartbeat_call_back(self,msg):
        self.path_following_heartbeat = msg.heartbeat
#############################################################################################################

def main(args=None):
    rclpy.init(args=args)
    tensor = JBNU_Collision()
    rclpy.spin(tensor)
    tensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
