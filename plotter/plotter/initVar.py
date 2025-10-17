# Library
# Library for common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Library for ros2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def setInitialVariables(classIn):

    classIn.fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(6, 4)

    classIn.ax1 = classIn.fig.add_subplot(gs[0:3, 0])
    classIn.ax2 = classIn.fig.add_subplot(gs[0:3, 1])
    classIn.ax3 = classIn.fig.add_subplot(gs[3:6, 0])
    classIn.ax4 = classIn.fig.add_subplot(gs[3:6, 1])

    classIn.ax5 = classIn.fig.add_subplot(gs[0:2, 2])
    classIn.ax6 = classIn.fig.add_subplot(gs[2:4, 2])
    classIn.ax7 = classIn.fig.add_subplot(gs[4:6, 2])

    classIn.ax8 = classIn.fig.add_subplot(gs[0:2, 3])
    classIn.ax9 = classIn.fig.add_subplot(gs[2:4, 3])
    classIn.ax10 = classIn.fig.add_subplot(gs[4:6, 3])

    # initialize flag
    classIn.global_waypoint_set = False  # flag whether global waypoint is subscribed
    classIn.local_waypoint_set = False  # flag whether local waypoint is subscribed
    classIn.is_ca  = False # flag whether state is subscribed
    classIn.is_complete = False
    
    classIn.pf_waypoint_set = False
    classIn.pf_attitude_cmd_set = False

    classIn.was_armed = False   # 이전 arming 상태
    classIn.stop_plotting = False  # 플롯 중단 조건
    classIn.plot_saved = False

    # initialize global waypoint
    classIn.start_global_waypoint = []  # start global waypoint
    classIn.goal_global_waypoint = []  # goal global waypoint

    # initialize local waypoint
    classIn.waypoint_x = []
    classIn.waypoint_y = []
    classIn.waypoint_z = []

    # initialize attitude
    classIn.roll_body   = np.array([])
    classIn.pitch_body  = np.array([])
    classIn.yaw_body    = np.array([])
    classIn.time_att = np.array([])

    classIn.roll_body_cmd   = np.array([])
    classIn.pitch_body_cmd  = np.array([])
    classIn.yaw_body_cmd    = np.array([])
    classIn.time_sim    = np.array([])
    classIn.MPPI_ax     = np.array([])
    classIn.MPPI_eta    = np.array([])
    classIn.MPPI_cal_time    = np.array([])
    classIn.VT_x       = np.array([])
    classIn.VT_y       = np.array([])
    classIn.VT_z       = np.array([])
    classIn.p_x       = np.array([])
    classIn.p_y       = np.array([])
    classIn.p_z       = np.array([])
    classIn.vel_tot    = np.array([])
    classIn.time_pf = np.array([])

    # initialize vehicle position
    classIn.vehicle_x = np.array([])  # [m]
    classIn.vehicle_y = np.array([])  # [m]
    classIn.vehicle_z = np.array([])  # [m]
    classIn.vehicle_heading = 0       # [rad]
    classIn.min_distance    = 0       # [m]

    classIn.vehicle_vx = np.array([])  # [m]
    classIn.vehicle_vy = np.array([])  # [m]
    classIn.vehicle_vz = np.array([])  # [m]
    classIn.time_vel = np.array([])

    classIn.current_heading_waypoint_callback_counter = 1
    # set qos profile
    classIn.qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
