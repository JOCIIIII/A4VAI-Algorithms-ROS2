import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import sys
import time

from .. import settings as CFG
from ..lio.odometry import OdometryProcessor, summary_passed


def build_parser():
    parser = argparse.ArgumentParser(description="Check or publish LIO odometry for PX4.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--frd-mount-confirmed", action="store_true")
    return parser


def _validate_args(parser, args):
    if not 5.0 <= args.duration <= 600.0:
        parser.error("--duration must be between 5 and 600 seconds")
    if args.publish and not args.frd_mount_confirmed:
        parser.error("--publish requires --frd-mount-confirmed")
    if args.output.exists():
        parser.error("--output must not already exist")


def _run(args):
    settings_errors = CFG.validate_settings()
    if settings_errors:
        raise ValueError("Invalid control settings:\n- " + "\n- ".join(settings_errors))
    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from px4_msgs.msg import VehicleOdometry
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
    except ImportError as error:
        raise SystemExit(
            "ROS 2 Humble, nav_msgs and matching PX4 px4_msgs are required"
        ) from error

    class FastlioPx4Bridge(Node):
        def __init__(self):
            super().__init__("t1000_fastlio_px4_bridge")
            self.processor = OdometryProcessor(
                CFG.LIO_VELOCITY_WINDOW_S,
                CFG.VIO_MAX_JUMP_M,
                CFG.VIO_MAX_SPEED_MPS,
            )
            self.rows = []
            self.received_times = []
            self.sample_ages_s = []
            self.rejected = {}
            input_qos = QoSProfile(depth=20)
            self.create_subscription(Odometry, CFG.TOPIC_LIO_ODOM, self.receive, input_qos)
            self.publisher = None
            if args.publish:
                output_qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self.publisher = self.create_publisher(
                    VehicleOdometry, CFG.TOPIC_PX4_VISUAL_ODOM, output_qos
                )

        def reject(self, reason):
            self.rejected[reason] = self.rejected.get(reason, 0) + 1

        def receive(self, message):
            if (
                message.header.frame_id != CFG.LIO_FRAME_ID
                or message.child_frame_id != CFG.LIO_CHILD_FRAME_ID
            ):
                self.reject("frame")
                return

            stamp = float(message.header.stamp.sec) + float(
                message.header.stamp.nanosec
            ) * 1e-9
            now_s = self.get_clock().now().nanoseconds * 1e-9
            age_s = now_s - stamp
            self.received_times.append(time.monotonic())
            self.sample_ages_s.append(age_s)
            position_msg = message.pose.pose.position
            position = (position_msg.x, position_msg.y, position_msg.z)
            orientation = message.pose.pose.orientation
            converted, reason = self.processor.process(
                now_s,
                stamp,
                position,
                (orientation.x, orientation.y, orientation.z, orientation.w),
                message.pose.covariance,
            )
            if reason:
                self.reject(reason)
                return
            if converted is None:
                return

            row = {
                "received_monotonic": self.received_times[-1],
                "sample_timestamp_us": int(stamp * 1e6),
                "sample_age_s": age_s,
                **converted,
            }
            self.rows.append(row)
            if self.publisher is None:
                return

            output = VehicleOdometry()
            output.timestamp = int(now_s * 1e6)
            output.timestamp_sample = row["sample_timestamp_us"]
            output.pose_frame = VehicleOdometry.POSE_FRAME_FRD
            output.position = converted["position"]
            output.q = converted["q_wxyz"]
            output.velocity_frame = VehicleOdometry.VELOCITY_FRAME_FRD
            output.velocity = converted["velocity"]
            output.angular_velocity = [math.nan] * 3
            output.position_variance = converted["position_variance"]
            output.orientation_variance = converted["orientation_variance"]
            output.velocity_variance = converted["velocity_variance"]
            output.reset_counter = 0
            output.quality = 50
            self.publisher.publish(output)

        def summary(self):
            times = list(self.received_times)
            intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
            rate_hz = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0
            converted_times = [row["received_monotonic"] for row in self.rows]
            converted_intervals = [
                b - a for a, b in zip(converted_times, converted_times[1:]) if b > a
            ]
            converted_rate_hz = (
                1.0 / (sum(converted_intervals) / len(converted_intervals))
                if converted_intervals
                else 0.0
            )
            return {
                "passed": summary_passed(
                    len(self.rows), converted_rate_hz, self.rejected
                ),
                "publish_enabled": args.publish,
                "received_count": len(times),
                "converted_count": len(self.rows),
                "rate_hz": rate_hz,
                "converted_rate_hz": converted_rate_hz,
                "px4_documented_rate_warning": rate_hz < CFG.LIO_PX4_RECOMMENDED_RATE_HZ,
                "mean_sample_age_s": (
                    sum(self.sample_ages_s) / len(self.sample_ages_s)
                    if self.sample_ages_s
                    else math.nan
                ),
                "min_sample_age_s": min(self.sample_ages_s, default=math.nan),
                "max_sample_age_s": max(self.sample_ages_s, default=math.nan),
                "rejected": self.rejected,
                "reset_locked": self.processor.reset_locked,
                "frames": [CFG.LIO_FRAME_ID, CFG.LIO_CHILD_FRAME_ID],
            }

    args.output.mkdir(parents=True)
    ros_args = getattr(args, "ros_args", [])
    if "--ros-args" in ros_args:
        rclpy.init(args=ros_args)
    else:
        rclpy.init()
    node = FastlioPx4Bridge()
    started = time.monotonic()
    deadline = math.inf if args.continuous else started + args.duration
    if args.continuous:
        node.rows = deque(maxlen=6000)
        node.received_times = deque(maxlen=6000)
        node.sample_ages_s = deque(maxlen=6000)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        summary = node.summary()
        fields = [
            "received_monotonic",
            "sample_timestamp_us",
            "sample_age_s",
            "position",
            "q_wxyz",
            "velocity",
            "position_variance",
            "orientation_variance",
            "velocity_variance",
        ]
        with (args.output / "converted.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(node.rows)
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if summary["passed"] else 1


def main(argv=None):
    from rclpy.utilities import remove_ros_args

    parser = build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(remove_ros_args(args=raw_args))
    args.ros_args = raw_args
    _validate_args(parser, args)
    try:
        return _run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
