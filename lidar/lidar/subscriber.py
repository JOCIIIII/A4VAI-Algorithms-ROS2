from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude

# custom msgs libararies
from custom_msgs.msg import ConveyLocalWaypointComplete

from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

from .callback_functions import *

class PX4Subscriber(object):
    def __init__(self, node):
        self.node = node

    def declareVehicleLocalPositionSubscriber(self, state_var):
        self.node.vehicle_local_position_subscriber = self.node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            lambda msg: vehicle_local_position_callback(state_var, msg),
            self.node.qos_profile_px4,
        )

    def declareVehicleAttitudeSubscriber(self, state_var):
        self.node.vehicle_attitude_subscriber = self.node.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            lambda msg: vehicle_attitude_callback(state_var, msg),
            self.node.qos_profile_px4,
        )

class FlagSubscriber(object):
    def __init__(self, node):
        self.node = node
    
    def declareConveyLocalWaypointCompleteSubscriber(self, mode_flag):
        self.node.convey_local_waypoint_complete_subscriber = self.node.create_subscription(
            ConveyLocalWaypointComplete,
            "/convey_local_waypoint_complete",
            lambda msg: convey_local_waypoint_complete_call_back(mode_flag, msg),
            1,
        )
    def declarePFCompleteSubscriber(self, mode_flag):
        self.node.pf_complete_subscriber = self.node.create_subscription(
            Bool,
            "/path_following_complete",
            lambda msg: pf_complete_callback(mode_flag, msg),
            1,
        )

class CmdSubscriber(object):
    def __init__(self, node):
        self.node = node
            
    def declarePFAttitudeSetpointSubscriber(self, veh_att_set):
        self.node.PF_attitude_setpoint_subscriber = self.node.create_subscription(
            VehicleAttitudeSetpoint,
            "/pf_att_2_control",
            lambda msg: PF_Att2Control_callback(veh_att_set, msg),
            1,
        )
    
    def declareCAVelocitySetpointSubscriber(self, veh_vel_set, stateVar, ca_var):
        self.node.CA_velocity_setpoint_subscriber = self.node.create_subscription(
            Twist,
            "/ca_vel_2_control",
            lambda msg: CA2Control_callback(veh_vel_set, stateVar, ca_var, msg),
            1
        )

class MissionSubscriber(object):
    def __init__(self, node):
        self.node = node

    def declareLidarSubscriber(self, state_var, guid_var, mode_flag, ca_var, pub_func):
        self.node.LidarSubscriber_ = self.node.create_subscription(
            PointCloud2,
            '/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3',
            lambda msg: lidar_callback(state_var, guid_var, mode_flag, ca_var, pub_func, msg),
            self.node.qos_profile_lidar
        )
    
    def declareDepthSubscriber(self, mode_flag, ca_var):
        self.node.DepthSubscriber_ = self.node.create_subscription(
            Image,
            "/airsim_node/SimpleFlight/Depth_Camera_DepthPerspective/image",
            lambda msg: depth_callback(mode_flag, ca_var, msg),
            1,
        )

class EtcSubscriber(object):
    def __init__(self, node):
        self.node = node

    def declareHeadingWPIdxSubscriber(self, guid_var):
        self.node.heading_wp_idx_subscriber = self.node.create_subscription(
            Int32,
            "/heading_waypoint_index",
            lambda msg: heading_wp_idx_callback(guid_var, msg),
            1,
        )

class HeartbeatSubscriber(object):
    def __init__(self, node):
        self.node = node

    def declareControllerHeartbeatSubscriber(self, offboard_var):
        self.node.controller_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/controller_heartbeat",
            lambda msg: controller_heartbeat_callback(offboard_var, msg),
            1,
        )

    def declarePathPlanningHeartbeatSubscriber(self, offboard_var):
        self.node.path_planning_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/path_planning_heartbeat",
            lambda msg: path_planning_heartbeat_callback(offboard_var, msg),
            1,
        )

    def declareCollisionAvoidanceHeartbeatSubscriber(self, offboard_var):
        self.node.collision_avoidance_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/collision_avoidance_heartbeat",
            lambda msg: collision_avoidance_heartbeat_callback(offboard_var, msg),
            1,
        )
    
    def declarePathFollowingHeartbeatSubscriber(self, offboard_var):
        self.node.path_following_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/path_following_heartbeat",
            lambda msg: path_following_heartbeat_callback(offboard_var, msg),
            1,
        )