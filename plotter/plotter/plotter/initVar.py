# Library
# Library for common
import numpy as np
import matplotlib.pyplot as plt

# Library for ros2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def setInitialVariables(classIn):

    # initialize figure
    classIn.fig, ((classIn.ax1, classIn.ax2), (classIn.ax3, classIn.ax4)) = (
        plt.subplots(2, 2, figsize=(10, 10))
    )

    # initialize flag
    classIn.global_waypoint_set = False  # flag whether global waypoint is subscribed
    classIn.local_waypoint_set = False  # flag whether local waypoint is subscribed
    classIn.is_ca  = False # flag whether state is subscribed

    # initialize global waypoint
    classIn.start_global_waypoint = []  # start global waypoint
    classIn.goal_global_waypoint = []  # goal global waypoint

    # initialize local waypoint
    classIn.waypoint_x = []
    classIn.waypoint_y = []
    classIn.waypoint_z = []

    # initialize vehicle position
    classIn.vehicle_x = np.array([])  # [m]
    classIn.vehicle_y = np.array([])  # [m]
    classIn.vehicle_z = np.array([])  # [m]
    classIn.vehicle_heading = 0       # [rad]
    classIn.min_distance    = 0       # [m]


    classIn.current_heading_waypoint_callback_counter = 1
    # set qos profile
    classIn.qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
