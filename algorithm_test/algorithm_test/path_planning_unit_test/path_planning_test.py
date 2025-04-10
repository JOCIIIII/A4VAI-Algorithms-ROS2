# Librarys

# Library for common
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# ROS libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

# custom msgs libararies
from custom_msgs.msg import GlobalWaypointSetpoint
from custom_msgs.msg import LocalWaypointSetpoint


class PathPlanningTest(Node):
    def __init__(self):
        super().__init__("give_global_waypoint")

        ############################################################################################################
        # input: start_point, goal_point
        # set start and goal point
        self.start_point = [200.0, 400.0, 11.0]
        self.goal_point = [600.0, 200.0, 14.0]
        ############################################################################################################

        # Publisher for global waypoint setpoint
        self.global_waypoint_publisher = self.create_publisher(
            GlobalWaypointSetpoint, "/global_waypoint_setpoint", 10
        )

        # Subscriber for local waypoint setpoint from path planning
        self.local_waypoint_subscriber = self.create_subscription(
            LocalWaypointSetpoint,
            "/local_waypoint_setpoint_from_PP",
            self.local_waypoint_callback,
            10,
        )

        # set waypoint
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_z = []

        # set flag
        self.path_planning_complete = False
        self.local_waypoint_set = False

        # initialize figure
        self.fig = plt.figure(figsize=(10, 10))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122, projection="3d")

        self.controller_heartbeat_publisher = self.create_publisher(
            Bool, "/controller_heartbeat", 10
        )

        self.path_following_heartbeat_publisher = self.create_publisher(
            Bool, "/path_following_heartbeat", 10
        )

        self.collision_avoidance_heartbeat_publisher = self.create_publisher(
            Bool, "/collision_avoidance_heartbeat", 10
        )

        # create timer for global waypoint publish
        self.timer = self.create_timer(1, self.global_waypoint_publish)

        period_heartbeat_mode = 1
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_collision_avoidance_heartbeat
        )
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_path_following_heartbeat
        )
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_controller_heartbeat
        )

    # publish global waypoint
    def global_waypoint_publish(self):

        if self.path_planning_complete == False:

            # input : global start waypoint [x, y, z]
            #         global goal waypoint  [x, y, z]

            msg = GlobalWaypointSetpoint()
            msg.start_point = self.start_point
            msg.goal_point = self.goal_point

            self.global_waypoint_publisher.publish(msg)

            self.get_logger().info(
                "======================================================"
            )
            self.get_logger().info("Global Waypoint Publish")
            self.get_logger().info("start_point :" + str(self.start_point))
            self.get_logger().info("goal_point  :" + str(self.goal_point))
            self.get_logger().info(
                "======================================================"
            )

            #####
            # QOS setting is needed
            # It will be added after further study
            #####

    # local waypoint callback function
    def local_waypoint_callback(self, msg):
        self.waypoint_x = msg.waypoint_x
        self.waypoint_y = msg.waypoint_y
        self.waypoint_z = msg.waypoint_z
        # self.waypoint_z = [(x+10) * 0.1 for x in self.waypoint_z]

        self.path_planning_complete = msg.path_planning_complete

        with open(
            "/home/user/workspace/ros2/logs/pathplanning.log", "a"
        ) as log_file:  # 'a' 모드는 append 모드로, 기존 내용에 추가합니다.
            log_file.write("\n")
            log_file.write(f"Waypoint X: {self.waypoint_x}\n")
            log_file.write(f"Waypoint Y: {self.waypoint_y}\n")
            log_file.write(f"Waypoint Z: {self.waypoint_z}\n")
            log_file.write("\n")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region Plot 1 2D full trajectory

        # Plot global waypoints with blue color
        self.ax1.scatter(
            [self.start_point[0], self.goal_point[0]],
            [self.start_point[1], self.goal_point[1]],
            color="blue",
            label="Global Waypoint",
            s=70,
        )

        # Plot local waypoints with red color
        self.ax1.scatter(
            self.waypoint_x,
            self.waypoint_y,
            color="red",
            label="Local Waypoints",
            s=6,
        )
        self.ax1.plot(
            self.waypoint_x,
            self.waypoint_y,
            color="red",
            linewidth=1,
        )

        # set the title, x and y labels
        self.ax1.set_title("full trajectory")
        self.ax1.set_xlabel("X Coordinate")
        self.ax1.set_ylabel("Y Coordinate")
        self.ax1.legend()

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Plot height map on 3D plot
        image_path = "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/map/512-001.png"
        img = Image.open(image_path).convert("L")
        height_map = np.array(img)

        # X, Y 축 생성
        x = np.linspace(0, height_map.shape[1] - 1, height_map.shape[1])
        y = np.linspace(0, height_map.shape[0] - 1, height_map.shape[0])
        x, y = np.meshgrid(x, y)

        # Z 축은 높이 맵의 픽셀 값으로 설정
        z = height_map * 0.1

        # 3D 플롯 생성
        self.ax2.plot_surface(x, y, z, cmap="viridis", alpha=1)

        # region Plot 2 altutude
        # Plot global waypoints with blue color
        self.ax2.scatter(
            [self.start_point[0], self.goal_point[0]],
            [self.start_point[1], self.goal_point[1]],
            [self.start_point[2] * 0.1, self.goal_point[2]],
            color="blue",
            label="Local Waypoints",
            s=70,
        )

        # Plot local waypoints with red color
        self.ax2.plot(
            self.waypoint_x,
            self.waypoint_y,
            self.waypoint_z,
            color="red",
            linewidth=4,
        )

        # set the title, x and y labels

        self.ax2.set_title("3D Height Map")
        self.ax2.set_xlabel("X axis")
        self.ax2.set_ylabel("Y axis")
        self.ax2.set_zlabel("Height (Z)")
        self.ax2.legend()

        self.get_logger().info("======================================================")
        self.get_logger().info("plot")
        self.get_logger().info("======================================================")

        self.ax2.set_box_aspect([1, 1, 1])
        self.ax2.set_zlim(0, 100)

        plt.tight_layout()
        plt.draw()
        plt.pause(10000)

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # heartbeat publish
    def publish_collision_avoidance_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.collision_avoidance_heartbeat_publisher.publish(msg)

    def publish_path_following_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.path_following_heartbeat_publisher.publish(msg)

    def publish_controller_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.controller_heartbeat_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    path_planning_test = PathPlanningTest()
    rclpy.spin(path_planning_test)
    path_planning_test.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
