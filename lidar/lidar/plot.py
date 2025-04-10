import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Python 3.9에서도 이 방식이 호환
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleLocalPosition

class ObstacleVisualizer(Node):
    def __init__(self):
        super().__init__('obstacle_visualizer')




        self.qos_profile_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS2 Subscriber for obstacle info
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/obstacle',
            self.obstacle_callback,
            1
        )
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            self.qos_profile_px4,
        )

        # Initialize obstacle data
        self.obstacle_info = []
        self.vehicle_x = np.array([])  # [m]
        self.vehicle_y = np.array([])  # [m]
        self.vehicle_z = np.array([])  # [m]
        self.vehicle_heading = 0       # [rad]

        self.obstacle_x = np.array([])  # [m]
        self.obstacle_y = np.array([])  # [m]
        self.obstacle_z = np.array([])  # [m]


        # Create Matplotlib figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.timer = self.create_timer(0.1, self.update_plot)

    def obstacle_callback(self, msg):
        obstacle = np.array(msg.data).reshape(-1, 3)  # [x, y, z, distance]
        self.get_logger().info(f"Obstacle: {obstacle}")
        self.get_logger().info(f"Obstacle x: {obstacle[:, 0]}")
        # Extract obstacle data
        point_x = obstacle[:, 0]
        point_y = obstacle[:, 1]
        point_z = obstacle[:, 2]

        self.obstacle_x = np.append(self.obstacle_x, point_x).flatten()
        self.obstacle_y = np.append(self.obstacle_y, point_y).flatten()
        self.obstacle_z = np.append(self.obstacle_z, -point_z).flatten()

    def update_plot(self):
        """
        Update the 3D visualization of obstacles in the NED frame.
        """
        # Clear the existing plot
        self.ax.clear()

        # Plot the minimum distance point
        # if len(self.obstacle_x) > 0:
        #     self.ax.scatter(
        #         self.obstacle_x,
        #         self.obstacle_y,
        #         self.obstacle_z,
        #         color="red",
        #         s=4,
        #     )

        # if len(self.vehicle_x) > 0:
        #     self.ax.scatter(
        #         self.vehicle_x,
        #         self.vehicle_y,
        #         self.vehicle_z,
        #         color="green",
        #         s=4,
        #         )
            
        #     x_center = self.vehicle_x[-1]
        #     y_center = self.vehicle_y[-1]
        #     z_center = self.vehicle_z[-1]

        #     margin = 15  # 드론의 주변을 보기 위한 마진

        #     self.ax.set_xlim([x_center - margin, x_center + margin])
        #     self.ax.set_ylim([y_center - margin, y_center + margin])
        #     self.ax.set_zlim([z_center - margin, z_center + margin])

        # # Set plot properties
        # self.ax.set_xlabel("X")
        # self.ax.set_ylabel("Y")
        # self.ax.set_zlabel("Z")

        # Plot the minimum distance point
        if len(self.obstacle_x) > 0:
            self.ax.scatter(
                self.obstacle_x,
                self.obstacle_y,
                color="red",
                s=4,
            )

        if len(self.vehicle_x) > 0:
            self.ax.scatter(
                self.vehicle_x,
                self.vehicle_y,
                color="green",
                s=4,
                )
            
            x_center = self.vehicle_x[-1]
            y_center = self.vehicle_y[-1]

            # margin = 15  # 드론의 주변을 보기 위한 마진

            # self.ax.set_xlim([x_center - margin, x_center + margin])
            # self.ax.set_ylim([y_center - margin, y_center + margin])

        # Set plot properties
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        # Draw and pause to update the plot
        plt.draw()
        plt.pause(0.001)

    def vehicle_local_position_callback(self, msg):
        # convert data to list
        vehicle_x = np.array(msg.x)
        vehicle_y = np.array(msg.y)
        vehicle_z = np.array(-msg.z)
        self.vehicle_heading = msg.heading

        # append to list
        self.vehicle_x = np.append(self.vehicle_x, vehicle_x).flatten()
        self.vehicle_y = np.append(self.vehicle_y, vehicle_y).flatten()
        self.vehicle_z = np.append(self.vehicle_z, vehicle_z).flatten()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
