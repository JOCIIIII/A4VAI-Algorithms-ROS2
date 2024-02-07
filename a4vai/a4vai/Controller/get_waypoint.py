import numpy as np

import rclpy
from rclpy.node import Node

from .give_global_waypoint import GiveGlobalWaypoint

from custom_msgs.msg import LocalWaypointSetpoint


class Controller(Node):

    def __init__(self):
        super().__init__("controller")
        
        # initialize waypoint parameter
        self.waypoint_index     =   0
        self.waypoint_x         =   []
        self.waypoint_y         =   []
        self.waypoint_z         =   []
        self.path_planning_complete = False
        # path planning position
        # [x, z, y]
        self.start_point        =   [1.0, 5.0, 1.0]
        self.goal_point         =   [950.0, 10.0, 950.0]

        # NED position status
        self.x                  =   0
        self.y                  =   0
        self.z                  =   0

        # path planning start
        give_global_waypoint = GiveGlobalWaypoint()
        give_global_waypoint.global_waypoint_publish(self.start_point, self.goal_point)
        give_global_waypoint.destroy_node()


        self.local_waypoint_subscriber = self.create_subscription(LocalWaypointSetpoint, '/local_waypoint_setpoint_from_PP',self.path_planning_call_back, 10)
       

    def path_planning_call_back(self, msg):
        self.path_planning_complete = msg.path_planning_complete
        self.waypoint_x             = msg.waypoint_x 
        self.waypoint_y             = msg.waypoint_y
        self.waypoint_z             = msg.waypoint_z
        print(self.waypoint_x)
        print(self.waypoint_y)
        print(self.waypoint_z)
        print("                                          ")
        print("=====   Path Planning Complete!!     =====")
        print("                                          ")

 
            

    #     # period of publishing waypoint from Controller to PathPlanning
    #     ##### can change period 
    #     self.waypoint_index_publish_period = 0.1

    #     self.waypoint_index_publisher = self.create_publisher(WaypointIndex,'/waypoint_index',10)

    #     self.waypoint_index_publish_timer = self.create_timer(self.waypoint_publish_period, self.publish_waypoint_index)


    # def publish_waypoint_index(self):
    #     msg = WaypointIndex()
    #     msg.waypoint_index = self.waypoint_index
    #     self.waypoint_index_publisher.publish(msg)

    # def check_waypoint_index(self):
    #     remaining_distance = np.sqrt((self.x - self.waypoint_x[self.waypoint_index])**2 - (self.y - self.waypoint_y[self.waypoint_index])**2)
        
    #     if remaining_distance <= 3.0 :
    #         self.waypoint_index += 1
    #     else:
    #         pass


def main(args=None):
    print("======================================================")
    print("------------- main() in get_waypoint.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()