import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from lidar.lib.data_class import StateVariable
from lidar.lib.subscriber import PX4Subscriber

import numpy as np
from sklearn.cluster import DBSCAN

from visualization_msgs.msg import MarkerArray

class LidarProcessor(Node):
    def __init__(self):
        super().__init__("lidar_processor")

        self.state_var = StateVariable()

        self.qos_profile_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_px4 = PX4Subscriber(self)
        self.sub_px4.declareVehicleLocalPositionSubscriber()
        self.sub_px4.declareVehicleAttitudeSubscriber()

        # lidar qos profile
        self.qos_profile_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10
        )

        # lidar subscriber
        self.LidarSubscriber_ = self.create_subscription(
            PointCloud2,
            "/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3",
            self.lidar_callback,
            self.qos_profile_lidar,
        )
        # filter lidar points publisher
        self.filtered_lidar_publisher_ = self.create_publisher(
            PointCloud2, "/airsim_node/SimpleFlight/lidar/points/velo", 10)

        # obstacle info publishers
        self.obstacle_publisher_ = self.create_publisher(
            Float32MultiArray, "/obstacle_info", 1
        )
    # obstacle marker publisher
        self.obstacle_marker_publisher_ = self.create_publisher(
        MarkerArray, "/obstacle_markers", 10)


    def lidar_callback(self, pc_msg):
        if pc_msg.is_dense:
            input_points = point_cloud2.read_points(
                    pc_msg, field_names=("x", "y", "z"), skip_nans=True)
            # preprocess points
            x, y, z = self.preprocess_points(input_points)

            # publish filtered lidar points
            self.publish_filtered_lidar(x, y, z, pc_msg)

            # obstacle clustering
            obstacle_info = self.cluster_obstacles(x, y, z)

            # publish obstacle info
            self.publish_obstacle_info(obstacle_info)

    def preprocess_points(self, input_points):
        # convert generator to list
        input_points = list(input_points)
        # convert list to numpy array
        points = np.array(input_points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        # sort by x, y, z
        x = points["x"]
        y = points["y"]
        z = points["z"]

        # filter out points inside the vehicle and ground points
        vehicle_radius = 0.01  
        # mask = (np.sqrt((x)**2 + (y)**2 + (z)**2) > vehicle_radius) & (z > 0.5)
        mask = (np.sqrt((x)**2 + (y)**2 + (z)**2) > vehicle_radius)
        x = x[mask]
        y = y[mask]
        z = z[mask]

        return x, y, z

    # publish filtered lidar points
    def publish_filtered_lidar(self, x, y, z, point_cloud_msg):

        filtered_lidar_msg = PointCloud2()
        filtered_lidar_msg.header = point_cloud_msg.header
        points = np.column_stack((x, y, z)).astype(np.float32)
        filtered_lidar_msg = point_cloud2.create_cloud_xyz32(point_cloud_msg.header, points)
        self.filtered_lidar_publisher_.publish(filtered_lidar_msg)


        
    

    def cluster_obstacles(self, x, y, z):
        # initialize variables
        obstacle_info = []

        ned_position = np.array([self.state_var.x, self.state_var.y, -self.state_var.z]) # [1 x 3]

        # arrange points
        points = np.column_stack((x, y, z))  # [n x 3]

        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
        labels = clustering.labels_
        
        # extract unique labels
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # exclude noise points
                continue
            rel_body_cluster_points = points[labels == label] # [n x 3]



            # distances = np.linalg.norm(cluster_points, axis=1)
            # min_distance = np.min(distances)
            # min_point = cluster_points[np.argmin(distances)]

            # Transform cluster points to NED frame
            rel_ned_cluster_points = self.state_var.dcm_b2n @ (rel_body_cluster_points.T) # [3 x n]

            cluster_points_ned = ned_position + rel_ned_cluster_points.T # [n x 3]

            ground_mask = -cluster_points_ned[:, 2] > 0
            cluster_points_ned = cluster_points_ned[ground_mask]    # [n x 3]
            height_mask = abs((cluster_points_ned - ned_position)[:, 2]) < 0.5
            cluster_points_ned = cluster_points_ned[height_mask]    # [n x 3]

            rel_pos = cluster_points_ned - ned_position # [n x 3]

            rel_distances = np.linalg.norm(rel_pos, axis=1) # [1 X n]
            
            if len(rel_distances) > 0:
                min_rel_distance = np.min(rel_distances)

                if min_rel_distance < 15:
                    min_ned_point = cluster_points_ned[np.argmin(min_rel_distance)] # [1 x 3]

                    min_rel_pose = min_ned_point - ned_position   # [1 x 3]

                    azimuth = np.arctan2(min_rel_pose[1], min_rel_pose[0])
                    elevation = np.arctan2(min_rel_pose[2], np.linalg.norm(min_rel_pose[:2]))
                    rel_heading = azimuth - self.state_var.heading
                    rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
                    obstacle_info.append({"azimuth": rel_heading,
                                  "elevation": elevation,
                                  "distance": min_rel_distance,
                                  "position": min_ned_point
                                  })
                                
                else:
                    continue
            else:
                continue

        return obstacle_info

    def publish_obstacle_info(self, obstacle_info):
        msg = Float32MultiArray()
        for obstacle in obstacle_info:
            msg.data.extend(
                [
                    obstacle["azimuth"],
                    obstacle["elevation"],
                    obstacle["distance"],
                    obstacle["position"][0],
                    obstacle["position"][1],
                    obstacle["position"][2]
                ]
            )
        self.obstacle_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()
    rclpy.spin(lidar_processor)
    lidar_processor.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()