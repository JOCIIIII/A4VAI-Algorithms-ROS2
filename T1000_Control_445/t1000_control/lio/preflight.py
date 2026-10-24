import argparse
import json
import math
from pathlib import Path

from .. import settings as CFG

FLOAT32 = 7
FLOAT64 = 8


def evaluate_probe(probe):
    settings_errors = CFG.validate_settings()
    if settings_errors:
        return {"passed": False, "checks": {"settings": False},
                "warnings": [], "errors": settings_errors}
    cloud = probe.get("pointcloud", {})
    odom = probe.get("odometry", {})
    fields = {field.get("name"): field for field in cloud.get("fields", [])}
    header_ns = float(cloud.get("header_stamp_s", math.nan)) * 1e9
    point_min_ns = float(cloud.get("point_timestamp_min_ns", math.nan))
    point_max_ns = float(cloud.get("point_timestamp_max_ns", math.nan))
    duration_ns = point_max_ns - point_min_ns
    min_age = float(odom.get("min_sample_age_s", math.nan))
    max_age = float(odom.get("max_sample_age_s", math.nan))
    checks = {
        "xyz_layout": all(
            fields.get(name, {}).get("offset") == offset
            and fields.get(name, {}).get("datatype") == FLOAT32
            for name, offset in (("x", 0), ("y", 4), ("z", 8))
        ),
        "timestamp_field": fields.get("timestamp", {}).get("datatype") == FLOAT64,
        "point_timestamp_semantics": (
            cloud.get("point_timestamp_semantics") == "absolute_ns"
        ),
        "point_timestamps_vary": (
            math.isfinite(point_min_ns)
            and math.isfinite(point_max_ns)
            and 0.0 <= point_min_ns <= point_max_ns
            and 0.0 < duration_ns <= 200_000_000.0
        ),
        "point_time_matches_header": (
            math.isfinite(header_ns)
            and math.isfinite(point_min_ns)
            and abs(point_min_ns - header_ns) <= 20_000_000.0
        ),
        "odom_frames": (
            odom.get("frame_id") == CFG.LIO_FRAME_ID
            and odom.get("child_frame_id") == CFG.LIO_CHILD_FRAME_ID
        ),
        "odom_rate": float(odom.get("rate_hz", 0.0)) >= CFG.LIO_MIN_RATE_HZ,
        "odom_age": (
            math.isfinite(min_age)
            and math.isfinite(max_age)
            and min_age >= -CFG.LIO_MAX_FUTURE_SAMPLE_S
            and max_age <= CFG.LIO_MAX_SAMPLE_AGE_S
            and min_age <= max_age
        ),
    }
    warnings = []
    if float(odom.get("rate_hz", 0.0)) < CFG.LIO_PX4_RECOMMENDED_RATE_HZ:
        warnings.append(
            "FAST-LIO odometry is below the PX4 documented "
            f"{CFG.LIO_PX4_RECOMMENDED_RATE_HZ:g} Hz lower recommendation; "
            "verify EKF fusion flags and innovations instead of repeating samples."
        )
    return {"passed": all(checks.values()), "checks": checks, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check captured LIO data.")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_probe(json.loads(args.probe.read_text(encoding="utf-8")))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
