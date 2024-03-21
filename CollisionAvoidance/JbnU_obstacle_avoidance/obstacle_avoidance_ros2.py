import cv2
import numpy as np
import onnx
import onnxruntime as rt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import utils_gray

class JBNU_Collision(Node):
    def __init__(self):
        super().__init__('jbnu_collision')

        model_pretrained = onnx.load("/home/walid/Documents/INha_nov/Inha.onnx")
        self.sess = rt.InferenceSession(model_pretrained.SerializeToString(), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

        self.subscription = self.create_subscription(Image, '/depth/raw', self.depth_sub, 1)
        self.publisher_cmd = self.create_publisher(Twist, '/cmd_jbnu', 1)
        self.bridge = CvBridge()
        self.get_logger().info("Learning_based_feedforward node initialized")

    def depth_sub(self, msg):
        try:
            # Convert the ROS Image message to OpenCV format
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting Image message: {e}")
            return

        # Your preprocessing steps here
        image = np.interp(image, (0, 6.0), (0, 255))
        image = utils_gray.preprocess(image)

        cv2.imshow('walid', image.astype(np.uint8))
        cv2.waitKey(1)

        image = np.array([image])  # The model expects a 4D array
        image = image.astype(np.float32)

        infer = self.sess.run([self.output_name], {self.input_name: image})
        infer = infer[0][0]

        vx = 0.8
        vy = 0.0
        vz = 0.0
        vyaw = infer[3] * 1.0

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.linear.z = vz
        cmd.angular.z = vyaw * 2.0
        self.publisher_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    tensor = JBNU_Collision()
    rclpy.spin(tensor)
    tensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
