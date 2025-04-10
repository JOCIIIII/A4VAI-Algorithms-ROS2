import cv2
import numpy as np
import onnx
import onnxruntime as rt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
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
        self.heartbeat_publisher                        =   self.create_publisher(Bool,    '/collision_avoidance_heartbeat', 1)
        # declare heartbeat_subscriber 
        self.controller_heartbeat_subscriber            =   self.create_subscription(Bool, '/controller_heartbeat',            self.controller_heartbeat_call_back,            10)
        self.path_following_heartbeat_subscriber        =   self.create_subscription(Bool, '/path_following_heartbeat',        self.path_following_heartbeat_call_back,        10)
        self.path_planning_heartbeat_subscriber         =   self.create_subscription(Bool, '/path_planning_heartbeat',         self.path_planning_heartbeat_call_back,         10)
        self.pub = self.create_publisher(Image, '/image4', 1)

        self.image = []
#############################################################################################################
        model_pretrained = onnx.load("/home/user/workspace/ros2/ros2_ws/src/collision_avoidance/model/Inha.onnx")
        self.sess = rt.InferenceSession(model_pretrained.SerializeToString(), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.subscription = self.create_subscription(Image, '/airsim_node/SimpleFlight/Depth_Camera_DepthPerspective/image', self.depth_sub, 1)


        # self.CameraSubscriber_ = self.create_subscription(Image, '/airsim_node/Typhoon_1/DptCamera/DepthPerspective', self.depth_sub, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.publisher_cmd = self.create_publisher(Twist, '/ca_vel_2_control', 1)
        self.bridge = CvBridge()
        self.get_logger().info("Learning_based_feedforward node initialized")

        self.collision_avoidance_period = 0.02
        self.collision_avoidance_timer =  self.create_timer(self.collision_avoidance_period, self.collision_avoidance)
        
        
        # declare heartbeat_timer
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)


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
                # print('cmd2cnt:', cmd.angular.z)
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
        
        valid_image = np.ones(image.shape)*12.0
        
        valid_mask = (image <= 12) & (image > 0.3)

        valid_image[valid_mask] = image[valid_mask]

        image = np.interp(image, (0, 12.0), (0, 255))

        image = preprocess(image)

        image = np.array([image])  # The model expects a 4D array
        self.image = image.astype(np.float32)


    def publish_image4(self, image):
        msg = Image()

        msg.header.frame_id = 'depth_image'
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = image.shape[1] * 4

        # Convert and publish the message
        msg.data = image.tobytes()
        self.pub.publish(msg)

# heartbeat check function
    # heartbeat publish
    def publish_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.heartbeat_publisher.publish(msg)

    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self,msg):
        self.controller_heartbeat = msg.data

    # heartbeat subscribe from path following
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.data

    # heartbeat subscribe from collision avoidance
    def path_following_heartbeat_call_back(self,msg):
        self.path_following_heartbeat = msg.data
#############################################################################################################

def main(args=None):
    rclpy.init(args=args)
    tensor = JBNU_Collision()
    rclpy.spin(tensor)
    tensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
