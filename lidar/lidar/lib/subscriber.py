# common libraries
import numpy as np
import math
import rclpy
from rclpy.clock import Clock
# custom libraries
from .common_fuctions import convert_quaternion2euler, BodytoNED, DCM_from_euler_angle, NEDtoBody

# PX4 message libraries
from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude

# custom message libraries
from custom_msgs.msg import ConveyLocalWaypointComplete

# ROS2 message libraries
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs_py import point_cloud2
#-------------------------------------------------------------------------------------------#
# region: SUBSCRIBER CLASSES
class PX4Subscriber(object):
    def __init__(self, node):
        self.node = node

    # declare vehicle local position subscriber
    def declareVehicleLocalPositionSubscriber(self):
        self.node.vehicle_local_position_subscriber = self.node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            lambda msg: vehicle_local_position_callback(self.node, msg),
            self.node.qos_profile_px4,
        )

    # declare vehicle attitude subscriber
    def declareVehicleAttitudeSubscriber(self):
        self.node.vehicle_attitude_subscriber = self.node.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            lambda msg: vehicle_attitude_callback(self.node, msg),
            self.node.qos_profile_px4,
        )

# flag subscriber
class FlagSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare convey local waypoint complete subscriber
    def declareConveyLocalWaypointCompleteSubscriber(self):
        self.node.convey_local_waypoint_complete_subscriber = self.node.create_subscription(
            ConveyLocalWaypointComplete,
            "/convey_local_waypoint_complete",
            lambda msg: convey_local_waypoint_complete_call_back(self.node, msg),
            1,
        )

    # declare path following complete subscriber
    def declarePFCompleteSubscriber(self):
        self.node.pf_complete_subscriber = self.node.create_subscription(
            Bool,
            "/path_following_complete",
            lambda msg: pf_complete_callback(self.node, msg),
            1,
        )

# command subscriber
class CmdSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare path following attitude setpoint subscriber
    def declarePFAttitudeSetpointSubscriber(self):
        self.node.PF_attitude_setpoint_subscriber = self.node.create_subscription(
            VehicleAttitudeSetpoint,
            "/pf_att_2_control",
            lambda msg: PF_Att2Control_callback(self.node, msg),
            1,
        )
    
    # declare collision avoidance velocity setpoint subscriber
    def declareCAVelocitySetpointSubscriber(self):
        self.node.CA_velocity_setpoint_subscriber = self.node.create_subscription(
            Twist,
            "/ca_vel_2_control",
            lambda msg: CA2Control_callback(self.node, msg),
            1
        )

# etc subscriber
class EtcSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare heading waypoint index subscriber
    def declareHeadingWPIdxSubscriber(self):
        self.node.heading_wp_idx_subscriber = self.node.create_subscription(
            Int32,
            "/heading_waypoint_index",
            lambda msg: heading_wp_idx_callback(self.node, msg),
            1,
        )

# heartbeat subscriber
class HeartbeatSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare controller heartbeat subscriber
    def declareControllerHeartbeatSubscriber(self):
        self.node.controller_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/controller_heartbeat",
            lambda msg: controller_heartbeat_callback(self.node, msg),
            1,
        )

    # declare path planning heartbeat subscriber
    def declarePathPlanningHeartbeatSubscriber(self):
        self.node.path_planning_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/path_planning_heartbeat",
            lambda msg: path_planning_heartbeat_callback(self.node, msg),
            1,
        )

    # declare collision avoidance heartbeat subscriber
    def declareCollisionAvoidanceHeartbeatSubscriber(self):
        self.node.collision_avoidance_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/collision_avoidance_heartbeat",
            lambda msg: collision_avoidance_heartbeat_callback(self.node, msg),
            1,
        )

    # declare path following heartbeat subscriber
    def declarePathFollowingHeartbeatSubscriber(self):
        self.node.path_following_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/path_following_heartbeat",
            lambda msg: path_following_heartbeat_callback(self.node, msg),
            1,
        )
# endregion
#-------------------------------------------------------------------------------------------#
# region: CALLBACK FUNCTIONS
# update attitude offboard command from path following
def PF_Att2Control_callback(node, msg):
    node.veh_att_set = node.veh_att_set
    node.veh_att_set.roll_body = msg.roll_body
    node.veh_att_set.pitch_body = msg.pitch_body
    node.veh_att_set.yaw_body = msg.yaw_body
    node.veh_att_set.yaw_sp_move_rate = msg.yaw_sp_move_rate
    node.veh_att_set.q_d[0] = msg.q_d[0]
    node.veh_att_set.q_d[1] = msg.q_d[1]
    node.veh_att_set.q_d[2] = msg.q_d[2]
    node.veh_att_set.q_d[3] = msg.q_d[3]
    node.veh_att_set.thrust_body[0] = msg.thrust_body[0]
    node.veh_att_set.thrust_body[1] = msg.thrust_body[1]
    node.veh_att_set.thrust_body[2] = msg.thrust_body[2]
    
# update velocity offboard command from collision avoidance
def CA2Control_callback(node, msg):

    # K_p = 1.1
    # if stateVar.vx_b > 1.0:
    #     veh_vel_set.body_velocity = np.array([stateVar.vx_b+3, msg.linear.y + ca_var.vy_offset, msg.linear.z + ca_var.vz_offset])
    # else:
    #     veh_vel_set.body_velocity = np.array([msg.linear.x, msg.linear.y + ca_var.vy_offset, msg.linear.z + ca_var.vz_offset])


    
    

    # veh_vel_set.ned_velocity = BodytoNED(veh_vel_set.body_velocity, stateVar.dcm_b2n)

    # # if stateVar.vx_b < 2.0:
    #     gain_yawrate = 1.0
    #     gain_vy = 1.0
    # elif stateVar.vx_b > 2.0 and stateVar.vx_b < 4.0:
    #     gain_yawrate = 0.7  
    #     gain_vy = 1.1
    # elif stateVar.vx_b > 4.0 and stateVar.vx_b < 6.0:
    #     gain_yawrate = 0.8
    #     gain_vy = 1.0
    # elif stateVar.vx_b > 6.0 and stateVar.vx_b < 7.0:
    #     gain_yawrate = 0.65
    #     gain_vy = 1.0
    # elif stateVar.vx_b > 7.0 and stateVar.vx_b < 8.0:
    #     gain_yawrate = 0.5
    #     gain_vy = 1.0
    # elif stateVar.vx_b > 8.0 and stateVar.vx_b < 9.0:
    #     gain_yawrate = 0.65
    #     gain_vy = 1.5
    # elif stateVar.vx_b > 9.0 and stateVar.vx_b < 10.0:
    #     gain_yawrate = 0.9






    # 0820
    node.veh_vel_set.body_velocity = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
    node.veh_vel_set.ned_velocity = BodytoNED(node.veh_vel_set.body_velocity, node.state_var.dcm_b2n)
    node.veh_vel_set.yawspeed = -msg.angular.z

    # node.get_logger().info('vx: ' + str(msg.linear.x))
    # node.get_logger().info('vy: ' + str(msg.linear.y))
    # node.get_logger().info('vz: ' + str(msg.linear.z))
    # node.get_logger().info('veh_vel_set.yawspeed: ' + str(msg.angular.z))






# subscribe convey local waypoint complete flag from path following
def vehicle_local_position_callback(node, msg):
    # update NED position
    node.state_var.x = msg.x
    node.state_var.y = msg.y
    node.state_var.z = -msg.z
    
    # update NED velocity
    node.state_var.vx_n = msg.vx
    node.state_var.vy_n = msg.vy
    node.state_var.vz_n = -msg.vz

    vel_n = np.array([msg.vx, msg.vy, msg.vz])
    vel_b = NEDtoBody(vel_n, node.state_var.dcm_n2b)

    node.state_var.vx_b = vel_b[0]
    node.state_var.vy_b = vel_b[1]
    node.state_var.vz_b = vel_b[2]

# update attitude from vehicle attitude
def vehicle_attitude_callback(node, msg):
    # node.state_var.q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
    node.state_var.roll, node.state_var.pitch, node.state_var.yaw = convert_quaternion2euler(
        msg.q[0], msg.q[1], msg.q[2], msg.q[3]
    )
    node.state_var.dcm_n2b = DCM_from_euler_angle(np.array([node.state_var.roll, node.state_var.pitch, node.state_var.yaw]))
    node.state_var.dcm_b2n = node.state_var.dcm_n2b.T

# update heading waypoint index
def heading_wp_idx_callback(node, msg):
    node.guid_var.cur_wp = msg.data

# update path following complete flag
def pf_complete_callback(node, msg):
    node.flags.pf_done = msg.data

# update convey local waypoint complete flag
def convey_local_waypoint_complete_call_back(node, msg):
    node.flags.pf_get_local_waypoint = msg.convey_local_waypoint_is_complete

# update controller heartbeat
def controller_heartbeat_callback(node, msg):
    offboard_var.ct_heartbeat = msg.data

# update path planning heartbeat
def path_planning_heartbeat_callback(node, msg):
    offboard_var.pp_heartbeat = msg.data

# update collision avoidance heartbeat
def collision_avoidance_heartbeat_callback(node, msg):
    node.offboard_var.ca_heartbeat = msg.data

# update path following heartbeat
def path_following_heartbeat_callback(node, msg):
    node.offboard_var.pf_heartbeat = msg.data
# endregion
#-------------------------------------------------------------------------------------------#