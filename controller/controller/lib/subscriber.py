# common libraries
import numpy as np

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
    def declareVehicleLocalPositionSubscriber(self, state_var):
        self.node.vehicle_local_position_subscriber = self.node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            lambda msg: vehicle_local_position_callback(state_var, msg),
            self.node.qos_profile_px4,
        )

    # declare vehicle attitude subscriber
    def declareVehicleAttitudeSubscriber(self, state_var):
        self.node.vehicle_attitude_subscriber = self.node.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            lambda msg: vehicle_attitude_callback(state_var, msg),
            self.node.qos_profile_px4,
        )

# flag subscriber
class FlagSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare convey local waypoint complete subscriber
    def declareConveyLocalWaypointCompleteSubscriber(self, mode_flag):
        self.node.convey_local_waypoint_complete_subscriber = self.node.create_subscription(
            ConveyLocalWaypointComplete,
            "/convey_local_waypoint_complete",
            lambda msg: convey_local_waypoint_complete_call_back(mode_flag, msg),
            1,
        )

    # declare path following complete subscriber
    def declarePFCompleteSubscriber(self, mode_flag):
        self.node.pf_complete_subscriber = self.node.create_subscription(
            Bool,
            "/path_following_complete",
            lambda msg: pf_complete_callback(mode_flag, msg),
            1,
        )

# command subscriber
class CmdSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare path following attitude setpoint subscriber
    def declarePFAttitudeSetpointSubscriber(self, veh_att_set):
        self.node.PF_attitude_setpoint_subscriber = self.node.create_subscription(
            VehicleAttitudeSetpoint,
            "/pf_att_2_control",
            lambda msg: PF_Att2Control_callback(veh_att_set, msg),
            1,
        )
    
    # declare collision avoidance velocity setpoint subscriber
    def declareCAVelocitySetpointSubscriber(self, veh_vel_set, stateVar, ca_var):
        self.node.CA_velocity_setpoint_subscriber = self.node.create_subscription(
            Twist,
            "/ca_vel_2_control",
            lambda msg: CA2Control_callback(veh_vel_set, stateVar, ca_var, msg),
            1
        )

# etc subscriber
class EtcSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare heading waypoint index subscriber
    def declareHeadingWPIdxSubscriber(self, guid_var):
        self.node.heading_wp_idx_subscriber = self.node.create_subscription(
            Int32,
            "/heading_waypoint_index",
            lambda msg: heading_wp_idx_callback(guid_var, msg),
            1,
        )

# heartbeat subscriber
class HeartbeatSubscriber(object):
    def __init__(self, node):
        self.node = node

    # declare controller heartbeat subscriber
    def declareControllerHeartbeatSubscriber(self, offboard_var):
        self.node.controller_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/controller_heartbeat",
            lambda msg: controller_heartbeat_callback(offboard_var, msg),
            1,
        )

    # declare path planning heartbeat subscriber
    def declarePathPlanningHeartbeatSubscriber(self, offboard_var):
        self.node.path_planning_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/path_planning_heartbeat",
            lambda msg: path_planning_heartbeat_callback(offboard_var, msg),
            1,
        )

    # declare collision avoidance heartbeat subscriber
    def declareCollisionAvoidanceHeartbeatSubscriber(self, offboard_var):
        self.node.collision_avoidance_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/collision_avoidance_heartbeat",
            lambda msg: collision_avoidance_heartbeat_callback(offboard_var, msg),
            1,
        )

    # declare path following heartbeat subscriber
    def declarePathFollowingHeartbeatSubscriber(self, offboard_var):
        self.node.path_following_heartbeat_subscriber = self.node.create_subscription(
            Bool,
            "/path_following_heartbeat",
            lambda msg: path_following_heartbeat_callback(offboard_var, msg),
            1,
        )
# endregion
#-------------------------------------------------------------------------------------------#
# region: CALLBACK FUNCTIONS
# update attitude offboard command from path following
def PF_Att2Control_callback(veh_att_set, msg):
    veh_att_set.roll_body = msg.roll_body
    veh_att_set.pitch_body = msg.pitch_body
    veh_att_set.yaw_body = msg.yaw_body
    veh_att_set.yaw_sp_move_rate = msg.yaw_sp_move_rate
    veh_att_set.q_d[0] = msg.q_d[0]
    veh_att_set.q_d[1] = msg.q_d[1]
    veh_att_set.q_d[2] = msg.q_d[2]
    veh_att_set.q_d[3] = msg.q_d[3]
    veh_att_set.thrust_body[0] = msg.thrust_body[0]
    veh_att_set.thrust_body[1] = msg.thrust_body[1]
    veh_att_set.thrust_body[2] = msg.thrust_body[2]
    
# update velocity offboard command from collision avoidance
def CA2Control_callback(veh_vel_set, stateVar, ca_var, msg):

    total_body_cmd = np.array([msg.linear.x + 0.8, msg.linear.y, msg.linear.z])

    veh_vel_set.body_velocity = total_body_cmd
    veh_vel_set.ned_velocity = BodytoNED(veh_vel_set.body_velocity, stateVar.dcm_b2n)
    if abs(msg.angular.z) > np.deg2rad(45):
        ca_var.sign = np.sign(-msg.angular.z)
    veh_vel_set.yawspeed = -msg.angular.z + np.deg2rad(7) + ca_var.sign*np.deg2rad(5)

# subscribe convey local waypoint complete flag from path following
def vehicle_local_position_callback(state_var, msg):
    # update NED position
    state_var.x = msg.x
    state_var.y = msg.y
    state_var.z = -msg.z
    
    # update NED velocity
    state_var.vx_n = msg.vx
    state_var.vy_n = msg.vy
    state_var.vz_n = -msg.vz

    vel_n = np.array([state_var.vx_n, state_var.vy_n, state_var.vz_n])
    vel_b = NEDtoBody(vel_n, state_var.dcm_b2n)

    state_var.vx_b = vel_b[0]
    state_var.vy_b = vel_b[1]
    state_var.vz_b = vel_b[2]

# update attitude from vehicle attitude
def vehicle_attitude_callback(state_var, msg):
    state_var.phi, state_var.theta, state_var.psi = convert_quaternion2euler(
        msg.q[0], msg.q[1], msg.q[2], msg.q[3]
    )
    state_var.dcm_n2b = DCM_from_euler_angle([state_var.phi, state_var.theta, state_var.psi])
    state_var.dcm_b2n = state_var.dcm_n2b.T

# update heading waypoint index
def heading_wp_idx_callback(guid_var, msg):
    guid_var.cur_wp = msg.data

# update path following complete flag
def pf_complete_callback(mode_flag, msg):
    mode_flag.pf_done = msg.data

# update convey local waypoint complete flag
def convey_local_waypoint_complete_call_back(mode_flag, msg):
    mode_flag.pf_recieved_lw = msg.convey_local_waypoint_is_complete

# update controller heartbeat
def controller_heartbeat_callback(offboard_var, msg):
    offboard_var.ct_heartbeat = msg.data

# update path planning heartbeat
def path_planning_heartbeat_callback(offboard_var, msg):
    offboard_var.pp_heartbeat = msg.data

# update collision avoidance heartbeat
def collision_avoidance_heartbeat_callback(offboard_var, msg):
    offboard_var.ca_heartbeat = msg.data

# update path following heartbeat
def path_following_heartbeat_callback(offboard_var, msg):
    offboard_var.pf_heartbeat = msg.data
# endregion
#-------------------------------------------------------------------------------------------#