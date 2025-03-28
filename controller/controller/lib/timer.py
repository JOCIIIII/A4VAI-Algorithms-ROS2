class MainTimer:
    def __init__(self, node, OffboardVariable):
        self.node = node
        self.OffboardVariable = OffboardVariable

    def declareOffboardControlTimer(self, offboard_control_main):
        self.node.offboard_main_timer = self.node.create_timer(
            self.OffboardVariable.period_offboard_control,
            offboard_control_main
        )

class CommandPubTimer:
    def __init__(self, node, OffboardVariable):
        self.node = node
        self.OffboardVariable = OffboardVariable

    def declareOffboardAttitudeControlTimer(self, mode_flag, veh_att_set, pub_func_px4):
        self.node.attitude_control_call_timer = self.node.create_timer(
            self.OffboardVariable.period_offboard_att_ctrl,
            lambda: pub_func_px4.publish_vehicle_attitude_setpoint(mode_flag, veh_att_set)
        )
    
    def declareOffboardVelocityControlTimer(self, mode_flag, veh_att_set, pub_func_px4):
        self.node.velocity_control_call_timer = self.node.create_timer(
            self.OffboardVariable.period_offboard_vel_ctrl,
            lambda: pub_func_px4.publish_vehicle_velocity_setpoint(mode_flag, veh_att_set)
        )

class HeartbeatTimer:
    def __init__(self, node, OffboardVariable, pub_func_heartbeat):
        self.node = node
        self.OffboardVariable = OffboardVariable
        self.pub_func_heartbeat = pub_func_heartbeat
    
    def declareControllerHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.OffboardVariable.period_heartbeat,
            self.pub_func_heartbeat.publish_controller_heartbeat
        )
    
    def declarePathPlanningHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.OffboardVariable.period_heartbeat,
            self.pub_func_heartbeat.publish_path_planning_heartbeat
        )
    
    def declareCollisionAvoidanceHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.OffboardVariable.period_heartbeat,
            self.pub_func_heartbeat.publish_collision_avoidance_heartbeat
        )
    
    def declarePathFollowingHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.OffboardVariable.period_heartbeat,
            self.pub_func_heartbeat.publish_path_following_heartbeat
        )