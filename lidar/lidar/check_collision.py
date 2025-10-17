import rclpy
from rclpy.node import Node
from .data_class import ModeStatus

import numpy as np
from std_msgs.msg import Float32MultiArray

from custom_msgs.msg import StateFlag

class COLLISION_AVOIDANCENode(Node):
    def __init__(self):
        super().__init__('collision_avoidance_node')

        self.mode_status = ModeStatus()

        # obstacle info subscriber
        self.obstacle_subscription = self.create_subscription(
            Float32MultiArray,
            '/obstacle_info',
            self.obstacle_callback,
            1
        )

        self.flag_subscription = self.create_subscription(
            StateFlag,
            '/mode_flag_to_CC',
            self.flag_callback,
            1
        )
        # flag publisher
        self.flag_publisher = self.create_publisher(StateFlag, '/mode_flag2control', 1)
        self.obstacle_publisher_ = self.create_publisher(Float32MultiArray, '/obstacle', 1)

    def obstacle_callback(self, msg):
        # self.get_logger().info(f"PATH_FOLLOWING: {self.mode_status.PATH_FOLLOWING}, COLLISION_AVOIDANCE: {self.mode_status.COLLISION_AVOIDANCE}")
        obstacle_info = np.array(msg.data).reshape(-1, 6)  # [distance, azimuth, elevation]
        if self.mode_status.OFFBOARD:
            if self.mode_status.PATH_FOLLOWING:
                for obstacle in obstacle_info:
                    azimuth, elevation, distance, x,y,z = obstacle
                    if distance < 8.0 and np.deg2rad(-18) <= azimuth <= np.deg2rad(18):
                        
                        self.mode_status.COLLISION_AVOIDANCE = True
                        self.mode_status.PATH_FOLLOWING = False
                        self.publish_flags()
                        # self.publish_obstacle_info(x,y,z)
                        # self.get_logger().info(f"COLLISION_AVOIDANCE: distance={distance}, azimuth={np.degrees(azimuth)}")
                        break

            if self.mode_status.COLLISION_AVOIDANCE:
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
                    self.mode_status.COLLISION_AVOIDANCE = False
                    self.mode_status.PATH_FOLLOWING = True
                    self.publish_flags()
                    # self.publish_obstacle_info(x,y,z)
                    # self.get_logger().info(f"PATH_FOLLOWING: distance={distance}, azimuth={np.degrees(azimuth)}")
 
    def publish_obstacle_info(self, obstacle_x, obstacle_y, obstacle_z):
        msg = Float32MultiArray()
        msg.data = [
                float(obstacle_x),
                float(obstacle_y),
                float(obstacle_z)
            ]
        self.obstacle_publisher_.publish(msg)
        
    def flag_callback(self, msg):
        self.mode_status.OFFBOARD = msg.OFFBOARD
        self.mode_status.COLLISION_AVOIDANCE = msg.COLLISION_AVOIDANCE
        self.mode_status.PATH_FOLLOWING = msg.PATH_FOLLOWING
    
    def publish_flags(self):
        msg = StateFlag()
        msg.PATH_FOLLOWING = self.mode_status.PATH_FOLLOWING
        msg.COLLISION_AVOIDANCE = self.mode_status.COLLISION_AVOIDANCE
        print(f"CA: PATH_FOLLOWING: {msg.PATH_FOLLOWING}, COLLISION_AVOIDANCE: {msg.COLLISION_AVOIDANCE}")
        self.flag_publisher.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    collision_avoidance_node = COLLISION_AVOIDANCENode()
    rclpy.spin(collision_avoidance_node)
    collision_avoidance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
