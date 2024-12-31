import rclpy
from rclpy.node import Node
from .data_class import ModeFlag

import numpy as np
from std_msgs.msg import Float32MultiArray

from custom_msgs.msg import StateFlag

class CollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('collision_avoidance_node')

        self.mode_flag = ModeFlag()

        # obstacle info subscriber
        self.obstacle_subscription = self.create_subscription(
            Float32MultiArray,
            '/obstacle_info',
            self.obstacle_callback,
            1
        )

        self.flag_subscription = self.create_subscription(
            StateFlag,
            '/mode_flag2collision',
            self.flag_callback,
            1
        )
        # flag publisher
        self.flag_publisher = self.create_publisher(StateFlag, '/mode_flag2control', 1)
        self.obstacle_publisher_ = self.create_publisher(Float32MultiArray, '/obstacle', 1)

    def obstacle_callback(self, msg):
        # self.get_logger().info(f"is_pf: {self.mode_flag.is_pf}, is_ca: {self.mode_flag.is_ca}")
        obstacle_info = np.array(msg.data).reshape(-1, 6)  # [distance, azimuth, elevation]
        if self.mode_flag.is_offboard:
            if self.mode_flag.is_pf:
                for obstacle in obstacle_info:
                    azimuth, elevation, distance, x,y,z = obstacle
                    if distance < 8.0 and np.deg2rad(-18) <= azimuth <= np.deg2rad(18):
                        
                        self.mode_flag.is_ca = True
                        self.mode_flag.is_pf = False
                        self.publish_flags()
                        # self.publish_obstacle_info(x,y,z)
                        # self.get_logger().info(f"is_ca: distance={distance}, azimuth={np.degrees(azimuth)}")
                        break

            if self.mode_flag.is_ca:
                obstacle_detected = False
                for obstacle in obstacle_info:
                    azimuth, elevation, distance, x,y,z = obstacle
                    if distance < 4.:
                        obstacle_detected = True
                        break
                    elif np.deg2rad(-30) <= azimuth <= np.deg2rad(30):  
                        if distance < 10:  
                            obstacle_detected = True
                            break
                if not obstacle_detected:
                    self.mode_flag.is_ca = False
                    self.mode_flag.is_pf = True
                    self.publish_flags()
                    # self.publish_obstacle_info(x,y,z)
                    # self.get_logger().info(f"is_pf: distance={distance}, azimuth={np.degrees(azimuth)}")
 
    def publish_obstacle_info(self, obstacle_x, obstacle_y, obstacle_z):
        msg = Float32MultiArray()
        msg.data = [
                float(obstacle_x),
                float(obstacle_y),
                float(obstacle_z)
            ]
        self.obstacle_publisher_.publish(msg)
        
    def flag_callback(self, msg):
        self.mode_flag.is_offboard = msg.is_offboard
        self.mode_flag.is_ca = msg.is_ca
        self.mode_flag.is_pf = msg.is_pf
    
    def publish_flags(self):
        msg = StateFlag()
        msg.is_pf = self.mode_flag.is_pf
        msg.is_ca = self.mode_flag.is_ca
        print(f"CA: is_pf: {msg.is_pf}, is_ca: {msg.is_ca}")
        self.flag_publisher.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    collision_avoidance_node = CollisionAvoidanceNode()
    rclpy.spin(collision_avoidance_node)
    collision_avoidance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
