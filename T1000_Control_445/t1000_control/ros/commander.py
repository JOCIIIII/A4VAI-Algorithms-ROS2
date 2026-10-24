from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleCommand, VehicleStatus

from .. import settings as CFG

SYSTEM_ID    = CFG.PX4_TARGET_SYSTEM_ID
COMPONENT_ID = CFG.PX4_TARGET_COMPONENT_ID

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class PX4Commander:

    def __init__(self, node: Node):
        self._node = node

        self._cmd_pub = node.create_publisher(
            VehicleCommand,
            CFG.TOPIC_PX4_COMMAND,
            PX4_QOS,
        )

        self._status_sub = node.create_subscription(
            VehicleStatus,
            CFG.TOPIC_PX4_STATUS,
            self._status_cb,
            PX4_QOS,
        )

        self.armed = False
        self.nav_state = 0
        self.status_seen = False
        self._t_status = 0.0


    def _status_cb(self, msg: VehicleStatus):
        self.armed     = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state
        self.status_seen = True
        self._t_status = self._node.get_clock().now().nanoseconds * 1e-9

    @property
    def is_offboard(self) -> bool:
        return self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD


    def arm(self):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self._node.get_logger().info("ARM command sent")

    def disarm(self):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
        )
        self._node.get_logger().info("DISARM command sent")

    def set_offboard(self):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self._node.get_logger().info("OFFBOARD mode command sent")

    def land(self):
        self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self._node.get_logger().info("LAND command sent")

    def takeoff(self, altitude_m: float = 2.0):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            param7=altitude_m,
        )
        self._node.get_logger().info(f"TAKEOFF to {altitude_m}m command sent")


    def _send_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ):
        msg = VehicleCommand()
        msg.timestamp       = int(self._node.get_clock().now().nanoseconds / 1000)
        msg.command         = command
        msg.param1          = param1
        msg.param2          = param2
        msg.param3          = param3
        msg.param4          = param4
        msg.param5          = param5
        msg.param6          = param6
        msg.param7          = param7
        msg.target_system   = CFG.PX4_TARGET_SYSTEM_ID
        msg.target_component = CFG.PX4_TARGET_COMPONENT_ID
        msg.source_system   = CFG.PX4_SOURCE_SYSTEM_ID
        msg.source_component = CFG.PX4_SOURCE_COMPONENT_ID
        msg.from_external   = True
        self._cmd_pub.publish(msg)
