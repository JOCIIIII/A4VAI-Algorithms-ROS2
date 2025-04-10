import numpy as np

from .common_fuctions import convert_quaternion2euler, BodytoNED, DCM_from_euler_angle
from sensor_msgs_py import point_cloud2

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

def CA2Control_callback(veh_vel_set, stateVar, ca_var, msg):

    total_body_cmd = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    # if total_body_cmd[0] > 3.0:
    #     total_body_cmd[0] = 3.0
    # if total_body_cmd[1] > 3.0:
    #     total_body_cmd[1] = 3.0
    # if total_body_cmd[2] > 3.0:
    #     total_body_cmd[2] = 3.0

    veh_vel_set.body_velocity = total_body_cmd
    veh_vel_set.ned_velocity = BodytoNED(veh_vel_set.body_velocity, stateVar.dcm_b2n)
    veh_vel_set.yawspeed = msg.angular.z

# subscribe convey local waypoint complete flag from path following
def vehicle_local_position_callback(state_var, msg):
    # update NED position
    state_var.x = msg.x
    state_var.y = msg.y
    state_var.z = -msg.z
    
    # update NED velocity
    state_var.vx = msg.vx
    state_var.vy = msg.vy
    state_var.vz = -msg.vz

    state_var.heading = msg.heading
    
def vehicle_attitude_callback(state_var, msg):
    state_var.phi, state_var.theta, state_var.psi = convert_quaternion2euler(
        msg.q[0], msg.q[1], msg.q[2], msg.q[3]
    )
    state_var.dcm_n2b = DCM_from_euler_angle([state_var.phi, state_var.theta, state_var.psi])
    state_var.dcm_b2n = state_var.dcm_n2b.T

def heading_wp_idx_callback(guid_var, msg):
    guid_var.cur_wp = msg.data

def depth_callback(mode_flag, ca_var, msg):
    if mode_flag.is_offboard:
        try:
            # Convert the ROS Image message to OpenCV format
            image = ca_var.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            ca_var.depth_min_distance = 50
            return

        valid_image = np.ones(image.shape)*12.0
        # np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample1.csv',valid_image,delimiter=",")
        valid_mask = (image <= 12) & (image > 0.3)
        # np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample2.csv',valid_mask,delimiter=",")
        valid_image[valid_mask] = image[valid_mask]
        # np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample3.csv',valid_image,delimiter=",")

        ca_var.depth_min_distance = valid_image.min()
        
        if ca_var.depth_min_distance < 10.0 and mode_flag.non_ca == False:
            mode_flag.is_pf  = False
            mode_flag.is_ca  = True

def lidar_callback(state_var, guid_var, mode_flag, ca_var, pub_func, pc_msg):
    if mode_flag.is_offboard:
        if pc_msg.is_dense:

            input_points = list(point_cloud2.read_points(
                pc_msg, field_names=("x", "y", "z"), skip_nans=True
            ))

            point = np.array(input_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

            x = point['x']
            y = point['y']
            z = point['z']

            radius = 0.6  # 동체 반경

            # 동체 영역을 제외한 포인트 필터링
            mask = np.sqrt((x)**2 + (y)**2 + (z)**2) > radius
            x = x[mask]
            y = y[mask]
            z = z[mask]

            # 거리 계산
            dist = np.sqrt(x**2 + y**2 + z**2)

            # Lidar 최소 거리 계산
            if len(dist) > 0:  # 필터링 후 데이터가 있을 경우에만 최소 거리 계산
                ca_var.lidar_min_distance = np.min(dist)
            else:
                ca_var.lidar_counter += 1

            if ca_var.lidar_counter == 3:
                ca_var.lidar_min_distance = 50
                ca_var.lidar_counter = 0

            if mode_flag.is_ca == True:
                if ca_var.lidar_min_distance > 7:

                    guid_var.waypoint_x = guid_var.waypoint_x[guid_var.cur_wp:]
                    guid_var.waypoint_y = guid_var.waypoint_y[guid_var.cur_wp:]
                    guid_var.waypoint_z = guid_var.waypoint_z[guid_var.cur_wp:]

                    guid_var.waypoint_x = list(np.insert(guid_var.waypoint_x, 0, state_var.x))
                    guid_var.waypoint_x = list(np.insert(guid_var.waypoint_x, 0, state_var.x))
                    guid_var.waypoint_y = list(np.insert(guid_var.waypoint_y, 0, state_var.y))
                    guid_var.waypoint_y = list(np.insert(guid_var.waypoint_y, 0, state_var.y))
                    guid_var.waypoint_z = list(np.insert(guid_var.waypoint_z, 0, state_var.z))
                    guid_var.waypoint_z = list(np.insert(guid_var.waypoint_z, 0, state_var.z))

                    # print(guid_var.waypoint_x[guid_var.cur_wp:])
                    # print(guid_var.cur_wp)

                    guid_var.real_wp_x = guid_var.waypoint_x
                    guid_var.real_wp_y = guid_var.waypoint_y
                    guid_var.real_wp_z = guid_var.waypoint_z

                    pub_func.local_waypoint_publish(False)

                    mode_flag.is_ca = False
                    mode_flag.is_pf = True

def pf_complete_callback(mode_flag, msg):
    mode_flag.pf_done = msg.data

def convey_local_waypoint_complete_call_back(mode_flag, msg):
    mode_flag.pf_recieved_lw = msg.convey_local_waypoint_is_complete

def controller_heartbeat_callback(offboard_var, msg):
    offboard_var.ct_heartbeat = msg.data

def path_planning_heartbeat_callback(offboard_var, msg):
    offboard_var.pp_heartbeat = msg.data

def collision_avoidance_heartbeat_callback(offboard_var, msg):
    offboard_var.ca_heartbeat = msg.data

def path_following_heartbeat_callback(offboard_var, msg):
    offboard_var.pf_heartbeat = msg.data

