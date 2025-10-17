# Library

import matplotlib.pyplot as plt
import numpy as np
import math
from math import degrees

# Library for ros2
import rclpy
from rclpy.node import Node

# Library for custom message
from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint

# Library for px4 message
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus, VehicleAttitude

# Library for std_msgs
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray

# submodule for initial variables
from .initVar import *

#.. private libs.
from .pf_funcs import Quaternion2Euler

class Plotter(Node):
    def __init__(self):
        super().__init__("plotter")

        setInitialVariables(self)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region Subscriber

        # # Subscriber for global waypoint setpoint from controller
        # self.global_waypoint_subscriber = self.create_subscription(
        #     GlobalWaypointSetpoint,
        #     "/global_waypoint_setpoint_to_plotter",
        #     self.global_waypoint_callback,
        #     1,
        # )

        # # Subscriber for local waypoint setpoint from path planning
        # self.local_waypoint_subscriber = self.create_subscription(
        #     LocalWaypointSetpoint,
        #     "/local_waypoint_setpoint_to_plotter",
        #     self.local_waypoint_callback,
        #     1,
        # )


        # Subscriber 
        self.pf_waypoint_subscriber = self.create_subscription(
            Float64MultiArray,
            "/path_following_waypoint_to_plotter",
            self.path_following_waypoint_callback,
            10,
        )

        self.pf_subscriber = self.create_subscription(
            Float64MultiArray,
            "/path_following_to_plotter",
            self.path_following_callback,
            1,
        )


        # Subscriber for vehicle local position from px4
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            self.qos_profile,
        )
        # Subscriber for vehicle attitude from px4
        self.vehicle_attitude_subscriber        =   self.create_subscription(
            VehicleAttitude,         
            '/fmu/out/vehicle_attitude',         
            self.vehicle_attitude_callback,         
            self.qos_profile)
        
        # Subscriber for vehicle local position from px4
        self.vehicle_state_subscriber = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            self.qos_profile,
        )

        self.state_subscriber = self.create_subscription(
            Bool, "/controller_state", self.state_callback, 1
        )

        self.pf_state_subscriber = self.create_subscription(
            Bool, "/path_following_complete", self.pf_state_callback, 1
        )
        # self.current_heading_waypoint_subscriber =   self.create_subscription(
        #     Float32, '/heading', self.current_heading_waypoint_callback, 1)


        self.min_distance_subscriber = self.create_subscription(
            Float64MultiArray,
            '/min_distance',
            self.min_distance_callback,
            1,
        )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region timer

        # update plot timer
        period_update_plot = 1
        self.timer = self.create_timer(period_update_plot, self.update_plot)

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Callback functions

    # global waypoint callback function
    # def global_waypoint_callback(self, msg):
    #     self.start_global_waypoint = msg.start_point
    #     self.goal_global_waypoint = msg.goal_point
    #     self.global_waypoint_set = True

    # # local waypoint callback function
    # def local_waypoint_callback(self, msg):
    #     self.waypoint_x = msg.waypoint_x
    #     self.waypoint_y = msg.waypoint_y
    #     self.waypoint_z = msg.waypoint_z
    #     self.local_waypoint_set = msg.path_planning_complete

    # Path Following waypoint callback function
    def path_following_waypoint_callback(self, msg):

        data = msg.data        
        waypoints = np.array(data).reshape(-1, 3)

        self.waypoint_x = waypoints[:, 0]
        self.waypoint_y = waypoints[:, 1]
        self.waypoint_z = waypoints[:, 2]

        self.pf_waypoint_set = True

    def path_following_callback(self, msg):

        if not math.isnan(msg.data[1]) and not self.is_complete:

            roll_body_cmd       = np.degrees(np.array(msg.data[0]))
            pitch_body_cmd      = np.degrees(np.array(msg.data[1]))
            yaw_body_cmd        = np.degrees(np.array(msg.data[2]))    # deg

            MPPI_eta            = np.array(msg.data[3])
            MPPI_cal_time       = np.array(msg.data[4])
            
            self.VT_x 		= np.array(msg.data[5])
            self.VT_y 		= np.array(msg.data[6])
            self.VT_z 		= np.array(msg.data[7])
            
            self.p_x 		= np.array(msg.data[8])
            self.p_y 		= np.array(msg.data[9])
            self.p_z 		= np.array(msg.data[10])

            self.pf_attitude_cmd_set = True

            # append to list
            self.roll_body_cmd  = np.append(self.roll_body_cmd, roll_body_cmd)
            self.pitch_body_cmd = np.append(self.pitch_body_cmd, pitch_body_cmd)
            self.yaw_body_cmd   = np.append(self.yaw_body_cmd, yaw_body_cmd)
            
            self.MPPI_eta       = np.append(self.MPPI_eta, MPPI_eta)
            self.MPPI_cal_time  = np.append(self.MPPI_cal_time, MPPI_cal_time)
            
            time_pf = self.get_clock().now().nanoseconds / 1e9
            self.time_pf = np.append(self.time_pf, time_pf)

        else:
            pass


    # vehicle local position callback function
    def vehicle_local_position_callback(self, msg):
        # convert data to list
        vehicle_x = np.array(msg.x)
        vehicle_y = np.array(msg.y)
        vehicle_z = np.array(-msg.z)
        self.vehicle_heading = msg.heading

        vehicle_vx    =   np.array(msg.vx)
        vehicle_vy    =   np.array(msg.vy)
        vehicle_vz    =   np.array(msg.vz)
        vehicle_vt    =   np.linalg.norm([vehicle_vx, vehicle_vy, vehicle_vz])


        # append to list
        self.vehicle_x = np.append(self.vehicle_x, vehicle_x).flatten()
        self.vehicle_y = np.append(self.vehicle_y, vehicle_y).flatten()
        self.vehicle_z = np.append(self.vehicle_z, vehicle_z).flatten()

        self.vehicle_vx = np.append(self.vehicle_vx, vehicle_vx).flatten()
        self.vehicle_vy = np.append(self.vehicle_vy, vehicle_vy).flatten()
        self.vehicle_vz = np.append(self.vehicle_vz, vehicle_vz).flatten()
        self.vel_tot    = np.append(self.vel_tot, vehicle_vt).flatten()

        time_vel = self.get_clock().now().nanoseconds / 1e9
        self.time_vel = np.append(self.time_vel, time_vel)
        
    # vehicle local attitude callback function
    def vehicle_attitude_callback(self, msg):
        
        roll_body, pitch_body, yaw_body = Quaternion2Euler(msg.q[0], msg.q[1], msg.q[2], msg.q[3])

        roll_body       = np.degrees(np.array(roll_body))
        pitch_body       = np.degrees(np.array(pitch_body))
        yaw_body       = np.degrees(np.array(yaw_body))

        self.roll_body  = np.append(self.roll_body, roll_body)
        self.pitch_body = np.append(self.pitch_body, pitch_body)
        self.yaw_body   = np.append(self.yaw_body, yaw_body)

        time_att = self.get_clock().now().nanoseconds / 1e9
        self.time_att = np.append(self.time_att, time_att)

    def vehicle_status_callback(self, msg):
        if msg.arming_state == 2:
            # Armed 상태
            self.was_armed = True
        elif msg.arming_state == 1 and self.was_armed:
            # Disarmed 되었고, 이전에는 armed였음 → 전환 감지
            self.get_logger().info("Detected disarm event. Stopping plot update.")
            self.stop_plotting = True
            self.was_armed = False

    def state_callback(self, msg):
        self.is_ca = msg.data

    def pf_state_callback(self, msg):
        self.is_complete = msg.data
        

    def min_distance_callback(self, msg):
        if msg.data[0] > 7.0:
            self.min_distance = msg.data[1]
        else:
            self.min_distance = msg.data[0]

    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region MAIN CODE
    def update_plot(self):
        # check if global and local waypoints are set
        # if self.global_waypoint_set == True and self.local_waypoint_set == True:
        if not self.stop_plotting:
    
            if self.pf_waypoint_set == True:
                
                # self.get_logger().info("Plot Update")
                # ----------------------------------------------------------------------------------------#
                # region Plot 1 full trajectory

                # Clear the previous plot
                self.ax1.clear()

                # Plot local waypoints with red color
                self.ax1.scatter(
                    self.VT_x,
                    self.VT_y,
                    color="blue",
                    label="VT",
                    s=30,
                )
                
                self.ax1.scatter(
                    self.p_x,
                    self.p_y,
                    color="red",
                    label="p",
                    s=30,
                )
                
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
                
                # Plot vehicle positions
                if len(self.vehicle_x) > 0:
                    self.ax1.plot(
                        self.vehicle_x,
                        self.vehicle_y,
                        color="green",
                        label="Vehicle Position",
                        linewidth=4,
                    )
                # set the title, x and y labels
                self.ax1.set_title("Full trajectory")
                self.ax1.set_xlabel("X Coordinate")
                self.ax1.set_ylabel("Y Coordinate")
                self.ax1.legend()
                self.ax1.xlim = [0, 1000]
                self.ax1.ylim = [0, 1000]
                self.ax1.grid()
                self.ax1.axis('equal')

                # endregion
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # region Plot 2 vihicle fixed trajectory

                # Clear the previous plot
                self.ax2.clear()

                # Plot local waypoints with red color
                self.ax2.scatter(
                    self.waypoint_x,
                    self.waypoint_y,
                    color="red",
                    s=6,
                )

                self.ax2.plot(
                    self.waypoint_x,
                    self.waypoint_y,
                    color="red",
                    label="Local Waypoints",
                    linewidth=4,
                    alpha=0.5,
                )
                
                # Plot local waypoints with red color
                self.ax2.scatter(
                    self.VT_x,
                    self.VT_y,
                    color="blue",
                    label="VT",
                    s=30,
                )
                
                self.ax2.scatter(
                    self.p_x,
                    self.p_y,
                    color="red",
                    label="p",
                    s=30,
                )
                

                if len(self.vehicle_x) > 0:
                    self.ax2.plot(
                        self.vehicle_x,
                        self.vehicle_y,
                        color="green",
                        label="Vehicle Position",
                        linewidth=4,
                    )
                    # 드론의 위치에 따라 시점 고정
                    x_center = self.vehicle_x[-1]
                    y_center = self.vehicle_y[-1]

                    margin = 10  # 드론의 주변을 보기 위한 마진

                    self.ax2.set_xlim([x_center - margin, x_center + margin])
                    self.ax2.set_ylim([y_center - margin, y_center + margin])

                    # Calculate the components of the direction vectors
                    u = np.cos(self.vehicle_heading)
                    v = np.sin(self.vehicle_heading)

                    # Plot arrows representing vehicle heading and save the quiver object
                    self.quiver_obj = self.ax2.quiver(
                        self.vehicle_x[-1],
                        self.vehicle_y[-1],
                        u,
                        v,
                        angles="xy",
                        scale_units="xy",
                        scale=0.3,
                        label="heading",
                        color="blue",
                    )

                # set the title, x and y labels
                self.ax2.set_title("Vehicle Position")
                self.ax2.set_xlabel("X Coordinate")
                self.ax2.set_ylabel("Y Coordinate")
                self.ax2.legend(loc="upper left")
                self.ax2.grid()

                # endregion
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # region Plot 3 altitude

                # Clear the previous plot
                self.ax3.clear()

                # Plot local waypoints with red color
                self.ax3.scatter(
                    self.waypoint_x,
                    -self.waypoint_z,
                    color="red",
                    label="Local Waypoints",
                    s=6,
                )
                               
                self.ax3.plot(
                    self.waypoint_x,
                    -self.waypoint_z,
                    color="red",
                    linewidth=4,
                )
                
		# Plot local waypoints with red color
                #self.ax3.scatter(
                #    self.VT_x,
                #    self.VT_z,
                #    color="blue",
                #    label="VT",
                #    s=30,
                #    )

                if len(self.vehicle_x) > 0:
                    self.ax3.plot(
                        self.vehicle_x,
                        self.vehicle_z,
                        color="green",
                        label="Vehicle Position",
                        linewidth=4,
                    )
                    
                    # 드론의 위치에 따라 시점 고정
                    x_center = self.vehicle_x[-1]
                    z_center = self.vehicle_z[-1]

                    margin = 10  # 드론의 주변을 보기 위한 마진

                    self.ax3.set_xlim([x_center - margin, x_center + margin])
                    self.ax3.set_ylim([z_center - margin, z_center + margin])
                    
                

                # set the title, x and y labels
                self.ax3.set_title("Altitude X-Z plane")
                self.ax3.set_xlabel("X Coordinate")
                self.ax3.set_ylabel("Z Coordinate")
                self.ax3.legend()
                self.ax3.grid()

                # endregion
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # region Plot 4 altitude

                # Clear the previous plot
                self.ax4.clear()

                # Plot local waypoints with red color
                self.ax4.scatter(
                    self.waypoint_y,
                    -self.waypoint_z,
                    color="red",
                    label="Local Waypoints",
                    s=6,
                )

                self.ax4.plot(
                    self.waypoint_y,
                    -self.waypoint_z,
                    color="red",
                    linewidth=4,
                )
                
                # Plot local waypoints with red color
                #self.ax4.scatter(
                #    self.VT_y,
                #    self.VT_z,
                #    color="blue",
                #    label="VT",
                #    s=30,
                #)

                if len(self.vehicle_x) > 0:
                    self.ax4.plot(
                        self.vehicle_y,
                        self.vehicle_z,
                        color="green",
                        label="Vehicle Position",
                        linewidth=4,
                    )
                    # 드론의 위치에 따라 시점 고정
                    y_center = self.vehicle_y[-1]
                    z_center = self.vehicle_z[-1]

                    margin = 10  # 드론의 주변을 보기 위한 마진

                    self.ax4.set_xlim([y_center - margin, y_center + margin])
                    self.ax4.set_ylim([z_center - margin, z_center + margin])

                    
                # set the title, x and y labels
                self.ax4.set_title("Altitude Y-Z plane")
                self.ax4.set_xlabel("Y Coordinate")
                self.ax4.set_ylabel("Z Coordinate")
                self.ax4.legend()
                self.ax4.grid()
                
                # endregion
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 5 att-Roll

            if len(self.time_pf) > 0:

                self.ax5.clear()
                
                self.ax5.plot(
                    self.time_pf-self.time_pf[0],
                    self.roll_body_cmd,
                    color="red",
                    label="Cmd",
                    linewidth=2,
                )
                
                self.ax5.plot(
                    self.time_att-self.time_pf[0],
                    self.roll_body,
                    color="green",
                    label="Response",
                    linewidth=1,
                )

                # set the title, x and y labels
                self.ax5.set_title("Vehicle Attitude - Roll")
                self.ax5.set_xlabel("Time [s]")
                self.ax5.set_ylabel("Roll attitude [deg]")
                self.ax5.legend()
                self.ax5.grid()

                self.ax5.autoscale_view()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 6 att-Pitch

                self.ax6.clear()

                self.ax6.plot(
                    self.time_pf-self.time_pf[0],
                    self.pitch_body_cmd,
                    color="red",
                    label="Cmd",
                    linewidth=2,
                )
                self.ax6.plot(
                    self.time_att-self.time_pf[0],
                    self.pitch_body,
                    color="green",
                    label="Response",
                    linewidth=1,
                )

                # set the title, x and y labels
                self.ax6.set_title("Vehicle Attitude - Pitch")
                self.ax6.set_xlabel("Time [s]")
                self.ax6.set_ylabel("Pitch attitude [deg]")
                self.ax6.legend()
                self.ax6.grid()

                self.ax6.autoscale_view()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 7 att-Yaw

                self.ax7.clear()

                self.ax7.plot(
                    self.time_pf-self.time_pf[0],
                    self.yaw_body_cmd,
                    color="red",
                    label="Cmd",
                    linewidth=2,
                )

                self.ax7.plot(
                    self.time_att-self.time_pf[0],
                    self.yaw_body,
                    color="green",
                    label="Response",
                    linewidth=1,
                )
                # set the title, x and y labels
                self.ax7.set_title("Vehicle Attitude - Yaw")
                self.ax7.set_xlabel("Time [s]")
                self.ax7.set_ylabel("Yaw attitude [deg]")
                self.ax7.legend()
                self.ax7.grid()

                self.ax7.autoscale_view()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 8 MPPI Gain

                self.ax8.clear()

                self.ax8.plot(
                    self.time_pf-self.time_pf[0],
                    self.MPPI_eta,
                    color="blue",
                    linewidth=2,
                )

                # set the title, x and y labels
                self.ax8.set_title("MPPI Gain")
                self.ax8.set_xlabel("Time [s]")
                self.ax8.set_ylabel("MPPI Gain, eta")
                self.ax8.grid()

                self.ax8.autoscale_view()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 9 Total Velocity

                self.ax9.clear()

                self.ax9.plot(
                    self.time_vel-self.time_pf[0],
                    self.vel_tot,
                    color="blue",
                    linewidth=2,
                )

                # set the title, x and y labels
                self.ax9.set_title("Vehicle Tolal Velocity")
                self.ax9.set_xlabel("Time [s]")
                self.ax9.set_ylabel("Tolal Velocity [m/s]")
                self.ax9.grid()

                self.ax9.autoscale_view()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 10 MPPI Cal time

                self.ax10.clear()

                self.ax10.plot(
                    self.time_pf-self.time_pf[0],
                    self.MPPI_cal_time,
                    color="blue",
                    linewidth=2,
                )

                # set the title, x and y labels
                self.ax10.set_title("MPPI Calculation time")
                self.ax10.set_xlabel("Time [s]")
                self.ax10.set_ylabel("MPPI Cal time [s]")
                self.ax10.grid()

                self.ax10.autoscale_view()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
        elif not self.plot_saved:

            self.ax1.axis('equal')
            self.ax2.axis('equal')
            self.ax3.axis('equal')
            self.ax4.axis('equal')

            limmargin = 2
            limmax = 20
            self.ax5.set_ylim(
                min(np.min(self.roll_body_cmd) - limmargin, -limmax),
                max(np.max(self.roll_body_cmd) + limmargin, limmax),
            )
            self.ax6.set_ylim(
                min(np.min(self.pitch_body_cmd) - limmargin, -limmax),
                max(np.max(self.pitch_body_cmd) + limmargin, limmax),
            )

            self.ax5.xlim = ([0,self.time_pf[-1]])
            self.ax6.xlim = ([0,self.time_pf[-1]])
            self.ax7.xlim = ([0,self.time_pf[-1]])
            self.ax8.xlim = ([0,self.time_pf[-1]])
            self.ax9.xlim = ([0,self.time_pf[-1]])
            self.ax10.xlim = ([0,self.time_pf[-1]])

            # 그래프 저장
            save_path = "/home/user/workspace/ros2/logs/plot_result_sim.png"
            self.fig.savefig(save_path)
            self.get_logger().info(f"Plot saved to {save_path}")

            self.plot_saved = True

        else:
            return


        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
            

def main(args=None):
    rclpy.init(args=args)
    node = Plotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
