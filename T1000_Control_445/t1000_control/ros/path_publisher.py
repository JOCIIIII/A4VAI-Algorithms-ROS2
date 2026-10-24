import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from .. import settings as CFG

ROUTES = {
    "straight": [(0.0, 0.0), (90.0, 0.0)],
    "bend": [(0.0, 0.0), (45.0, 0.0), (90.0, 30.0)],
    "zigzag": [(0.0, 0.0), (25.0, 10.0), (50.0, -10.0), (75.0, 10.0), (90.0, 0.0)],
    "right_angle": [(0.0, 0.0), (50.0, 0.0), (50.0, 40.0)],
}


class PathPublisherStub(Node):

    def __init__(self, pts, rate_hz: float, frame: str, rate_source='cli'):
        rate_hz, _ = resolve_publish_rate(rate_hz)
        super().__init__("path_publisher_stub")
        self.pts = pts
        self.frame = frame
        self.flight_altitude_m = CFG.FLIGHT_ALTITUDE_M
        self.pub = self.create_publisher(Path, CFG.TOPIC_PATH, 10)
        self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info(
            f"Path source started: {len(pts)} points at {rate_hz}Hz ({rate_source}), "
            f"minimum={CFG.PATH_PUBLISH_MIN_HZ}Hz (settings), frame='{frame}'")
        self.get_logger().info(f"  {' -> '.join(f'({x:.0f},{y:.0f})' for x, y in pts)}")
        self.get_logger().warn(
            "This node does not plan. It keeps sending the same path.")

    def _publish(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        for x, y in self.pts:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = float(self.flight_altitude_m)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.pub.publish(msg)


def parse_points(s: str):
    out = []
    for tok in s.split():
        try:
            x, y = tok.split(",")
            point = (float(x), float(y))
            if not all(map(math.isfinite, point)):
                raise ValueError('coordinates must be finite')
            out.append(point)
        except ValueError:
            raise SystemExit(f"bad point: '{tok}'; use finite 'x,y'")
    if len(out) < 2:
        raise SystemExit(f"path has {len(out)} points; need at least 2")
    return out


def resolve_publish_rate(rate):
    source = 'settings' if rate is None else 'cli'
    rate = CFG.PATH_PUBLISH_HZ if rate is None else rate
    minimum = CFG.PATH_PUBLISH_MIN_HZ
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError('PATH_PUBLISH_MIN_HZ must be finite and positive')
    if not math.isfinite(rate) or rate < minimum:
        raise ValueError(f'--rate must be finite and at least {minimum}Hz')
    return rate, source


def main():
    ap = argparse.ArgumentParser(description="T1000 path source")
    ap.add_argument("--route", choices=sorted(ROUTES), default="straight")
    ap.add_argument("--points", help='set points: "x,y x,y ..."; overrides --route')
    ap.add_argument("--rate", type=float, default=None,
                    help="send rate in Hz; see settings.py")
    ap.add_argument("--frame", default="map", help="header.frame_id")
    argv = sys.argv[1:]
    a = ap.parse_args(args=remove_ros_args(args=argv))

    try:
        rate, source = resolve_publish_rate(a.rate)
    except ValueError as exc:
        ap.error(str(exc))

    pts = parse_points(a.points) if a.points is not None else ROUTES[a.route]

    if '--ros-args' in argv:
        rclpy.init(args=argv)
    else:
        rclpy.init()
    node = PathPublisherStub(pts, rate, a.frame, rate_source=source)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
