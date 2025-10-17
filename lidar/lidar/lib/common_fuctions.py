# This file is used to set the waypoint
# ----------------------------------------------------------------------------------------#
# region LIBRARIES
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime, timedelta

from .data_class import *

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.clock import Clock
from px4_msgs.msg import FusionWeight
# endregion
# ----------------------------------------------------------------------------------------#



def state_logger(self):
    flightlog = "%f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (
        int(Clock().now().nanoseconds / 1000),
        self.state_var.x,
        self.state_var.y,
        self.state_var.z,
        # ned velocity [m/s]
        self.state_var.vx_b,
        self.state_var.vy_b,
        self.state_var.vz_b,
        # attitude [rad]
        self.state_var.roll,
        self.state_var.pitch,
        self.state_var.yaw,

        self.guid_var.cur_wp,
        # self.guid_var.waypoint_y,
        # self.guid_var.waypoint_z,
        self.mode_status.PATH_FOLLOWING,
        self.mode_status.COLLISION_AVOIDANCE,
    
    )
    # self.sim_var.flight_log.write(flightlog)

def set_initial_variables(classIn, dir, sim_name):
    
    classIn.state_var      = StateVariable()
    classIn.offboard_var   = OffboardVariable()
    classIn.guid_var       = GuidVariable()
    classIn.mode_status      = ModeStatus()
    classIn.flags          = Flags()
    classIn.offboard_mode  = OffboardControlModeState()
    classIn.modes          = VehicleModes()
    classIn.veh_att_set    = VehicleAttitudeSetpointState()
    classIn.veh_vel_set    = VehicleVelocitySetpointState()
    classIn.ca_var         = CollisionAvoidanceVariable()
    classIn.sim_var        = SimulationVariable(sim_name, dir)

    classIn.qos_profile_px4 = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )

    classIn.qos_profile_lidar = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        depth=10
    )

    classIn.weight = FusionWeight()
    set_logging_file(classIn)

def set_wp(self):

    wp_path = os.path.join(self.sim_var.dir, "wp.csv")
    data = pd.read_csv(wp_path)
    self.state_var.x = data["x"][0]
    self.state_var.y = data["y"][0]
    self.state_var.z = data["z"][0]

    self.guid_var.waypoint_x = list(data["x"])
    self.guid_var.waypoint_y = list(data["y"])
    self.guid_var.waypoint_z = list(data["z"])
    
    self.guid_var.real_wp_x = list(data["x"])
    self.guid_var.real_wp_y = list(data["y"])
    self.guid_var.real_wp_z = list(data["z"])

def set_logging_file(self):
    # file name like 20241216_225040_path_following.csv
    current_time = datetime.now() + timedelta(hours=9)
    log_file_name = current_time.strftime("%Y%m%d_%H%M%S_") + self.sim_var.sim_name + ".csv"
    log_path = os.path.join((self.sim_var.dir + '/log'), log_file_name)
    # self.sim_var.flight_log = open(log_path, "w")

def publish_to_plotter(self):
    self.pub_func_plotter.publish_global_waypoint_to_plotter(self.guid_var)
    self.pub_func_plotter.publish_local_waypoint_to_plotter(self.guid_var)
    self.pub_func_plotter.publish_obstacle_min_distance(self.ca_var)
    self.pub_func_plotter.publish_heading(self.state_var)
    self.pub_func_plotter.publish_control_mode(self.mode_status)

def set_waypoint():
    waypoint = pd.read_csv("waypoint.csv")
    return waypoint


def convert_quaternion2euler(w, x, y, z):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    Roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Yaw = math.atan2(t3, t4)

    return Roll, Pitch, Yaw

# convert Body frame to NED frame
def BodytoNED(vel_body_cmd, dcm):
    vel_ned_cmd = (dcm @ vel_body_cmd).tolist()
    return vel_ned_cmd

# convert NED frame to Body frame
def NEDtoBody(vel_ned, dcm):
    vel_body = (dcm @ vel_ned).tolist()
    return vel_body


# calculate DCM matrix
#.. get DCM from Euler angle
def DCM_from_euler_angle( EulerAng ):
    # Local Variable 
    spsi            =   math.sin( EulerAng[2] )
    cpsi            =   math.cos( EulerAng[2] )
    sthe            =   math.sin( EulerAng[1] )
    cthe            =   math.cos( EulerAng[1] )
    sphi            =   math.sin( EulerAng[0] )
    cphi            =   math.cos( EulerAng[0] )
    
    # DCM from 1-frame to 2-frame
    c1_2            =   np.zeros((3,3))
    
    c1_2[0,0]       =   cpsi * cthe 
    c1_2[1,0]       =   cpsi * sthe * sphi - spsi * cphi 
    c1_2[2,0]       =   cpsi * sthe * cphi + spsi * sphi 
    
    c1_2[0,1]       =   spsi * cthe 
    c1_2[1,1]       =   spsi * sthe * sphi + cpsi * cphi 
    c1_2[2,1]       =   spsi * sthe * cphi - cpsi * sphi 
    
    c1_2[0,2]       =   -sthe 
    c1_2[1,2]       =   cthe * sphi 
    c1_2[2,2]       =   cthe * cphi 

    return c1_2
