from collections import deque
import math

from .. import settings as CFG

POSITION_VARIANCE_FLOOR = 0.01**2
VELOCITY_VARIANCE_FLOOR = 0.05**2
ORIENTATION_VARIANCE_FLOOR = math.radians(5.0) ** 2
MAX_SAMPLE_AGE_S = CFG.LIO_MAX_SAMPLE_AGE_S
MAX_FUTURE_SAMPLE_S = CFG.LIO_MAX_FUTURE_SAMPLE_S


def sample_age_valid(now_s, sample_s):
    age_s = float(now_s) - float(sample_s)
    return -CFG.LIO_MAX_FUTURE_SAMPLE_S <= age_s <= CFG.LIO_MAX_SAMPLE_AGE_S


class VelocityWindow:

    def __init__(self, window_s=0.5):
        self.window_s = float(window_s)
        self.samples = deque()
        self.high_watermark = None

    def add(self, timestamp, position):
        timestamp = float(timestamp)
        position = tuple(float(value) for value in position)
        if self.high_watermark is not None and timestamp <= self.high_watermark:
            self.samples.clear()
            return None
        if self.high_watermark is not None and timestamp - self.high_watermark > self.window_s:
            self.samples.clear()
        self.high_watermark = timestamp
        self.samples.append((timestamp, position))
        cutoff = timestamp - self.window_s
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if (
            len(self.samples) < 3
            or self.samples[-1][0] - self.samples[0][0] < self.window_s * 0.8
        ):
            return None

        mean_t = sum(row[0] for row in self.samples) / len(self.samples)
        denominator = sum((row[0] - mean_t) ** 2 for row in self.samples)
        if denominator <= 1e-12:
            return None

        velocity = []
        for axis in range(3):
            mean_position = sum(row[1][axis] for row in self.samples) / len(
                self.samples
            )
            numerator = sum(
                (row[0] - mean_t) * (row[1][axis] - mean_position)
                for row in self.samples
            )
            velocity.append(numerator / denominator)
        return tuple(velocity)


def convert_odometry_sample(
    position_xyz, quaternion_xyzw, velocity_xyz, pose_covariance
):
    position = [float(value) for value in position_xyz]
    velocity = [float(value) for value in velocity_xyz]
    x, y, z, w = [float(value) for value in quaternion_xyzw]
    values = position + velocity + [w, x, y, z]
    norm = math.hypot(w, x, y, z)
    if not all(math.isfinite(value) for value in values) or norm <= 1e-9:
        raise ValueError("FAST-LIO sample is not finite")

    covariance = [float(value) for value in pose_covariance]
    if len(covariance) != 36 or not all(math.isfinite(value) for value in covariance):
        raise ValueError("pose covariance must contain 36 finite values")

    return {
        "position": position,
        "q_wxyz": [w / norm, x / norm, y / norm, z / norm],
        "velocity": velocity,
        "position_variance": [
            max(POSITION_VARIANCE_FLOOR, covariance[index])
            for index in (0, 7, 14)
        ],
        "orientation_variance": [
            max(ORIENTATION_VARIANCE_FLOOR, covariance[index])
            for index in (21, 28, 35)
        ],
        "velocity_variance": [VELOCITY_VARIANCE_FLOOR] * 3,
    }


class OdometryProcessor:

    def __init__(self, window_s, max_jump_m, max_speed_mps):
        self.velocity_window = VelocityWindow(window_s)
        self.max_jump_m = float(max_jump_m)
        self.max_speed_mps = float(max_speed_mps)
        self.last_timestamp = None
        self.last_position = None
        self.reset_locked = False

    def process(
        self,
        now_s,
        stamp_s,
        position,
        quaternion_xyzw,
        pose_covariance,
    ):
        stamp_s = float(stamp_s)
        position = tuple(float(value) for value in position)
        if not math.isfinite(stamp_s) or not all(map(math.isfinite, position)):
            return None, "invalid_sample"
        if not sample_age_valid(now_s, stamp_s):
            return None, "sample_age"
        if self.last_timestamp is not None and stamp_s <= self.last_timestamp:
            return None, "timestamp"
        if self.reset_locked:
            return None, "reset_required"

        self.last_timestamp = stamp_s
        if (
            self.last_position is not None
            and math.dist(position, self.last_position) > self.max_jump_m
        ):
            self.reset_locked = True
            return None, "position_jump"
        self.last_position = position

        velocity = self.velocity_window.add(stamp_s, position)
        if velocity is None:
            return None, None
        if math.hypot(*velocity) > self.max_speed_mps:
            return None, "speed"
        try:
            return (
                convert_odometry_sample(
                    position, quaternion_xyzw, velocity, pose_covariance
                ),
                None,
            )
        except ValueError:
            return None, "invalid_sample"


def summary_passed(converted_count, converted_rate_hz, rejected):
    return (
        int(converted_count) >= 10
        and float(converted_rate_hz) >= CFG.LIO_MIN_RATE_HZ
        and not any(int(count) > 0 for count in rejected.values())
    )
