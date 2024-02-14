#.. ROS libraries
import rclpy
from rclpy.node import Node

#.. custom msgs libararies - srv.
from custom_msgs.msg  import GlobalWaypointSetpoint

class GiveGlobalWaypoint(Node):
    def __init__(self):
        super().__init__('give_global_waypoint')

        # create global waypoint publisher
        self.global_waypoint_publisher = self.create_publisher(GlobalWaypointSetpoint, '/global_waypoint_setpoint', 10)
 
        # publish global waypoint
    def global_waypoint_publish(self, start_point, goal_point):

        # input : global start waypoint [x, z, y]
        #         global goal waypoint  [x, z, y]

        msg                            =  GlobalWaypointSetpoint()
        msg.start_point                =  start_point
        msg.goal_point                 =  goal_point
        
        self.global_waypoint_publisher.publish(msg)

        #####
        # QOS setting is needed
        # It will be added after further study
        #####