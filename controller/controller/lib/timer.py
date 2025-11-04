class MainTimer:
    def __init__(self, node):
        self.node = node

    def declareOffboardControlTimer(self, offboard_control_main):
        self.node.offboard_main_timer = self.node.create_timer(
            self.node.offboard_var.period_offboard_control,
            offboard_control_main
        )

class CommandPubTimer:
    def __init__(self, node):
        self.node = node
    
    def declareOffboardVelocityControlTimer(self):
        self.node.velocity_control_call_timer = self.node.create_timer(
            self.node.offboard_var.period_offboard_vel_ctrl,
            lambda: self.node.pub_func_px4.publish_vehicle_velocity_setpoint(self.node.mode_status, self.node.veh_vel_set)
        )
    def declareAttitudeCommandTimer(self):
        self.node.attitude_command_call_timer = self.node.create_timer(
            self.node.offboard_var.period_offboard_att_ctrl,
            lambda: self.node.pub_func_px4.publish_att_command(self.node.veh_att_set)
        )
    def declareFusionWeightTimer(self):
        self.node.fusion_weight_call_timer = self.node.create_timer(
            self.node.offboard_var.period_fusion_weight,
            lambda: self.node.pub_func_px4.publish_fusion_weight(self.node.weight)
        )

class HeartbeatTimer:
    def __init__(self, node):
        self.node = node
    
    def declareControllerHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.node.offboard_var.period_heartbeat,
            self.node.pub_func_heartbeat.publish_controller_heartbeat
        )
    
    def declarePathPlanningHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.node.offboard_var.period_heartbeat,
            self.node.pub_func_heartbeat.publish_path_planning_heartbeat
        )
    
    def declareCollisionAvoidanceHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.node.offboard_var.period_heartbeat,
            self.node.pub_func_heartbeat.publish_collision_avoidance_heartbeat
        )
    
    def declarePathFollowingHeartbeatTimer(self):
        self.node.heartbeat_timer = self.node.create_timer(
            self.node.offboard_var.period_heartbeat,
            self.node.pub_func_heartbeat.publish_path_following_heartbeat
        )