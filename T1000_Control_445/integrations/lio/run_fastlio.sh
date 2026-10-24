#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${LIVOX_WS:?Set LIVOX_WS to the livox_ros_driver2 workspace}"
: "${FASTLIO_WS:?Set FASTLIO_WS to the FAST-LIO ROS 2 workspace}"
ros_distro=${ROS_DISTRO:-humble}

source "/opt/ros/${ros_distro}/setup.bash"
source "${LIVOX_WS}/install/local_setup.bash"
source "${FASTLIO_WS}/install/local_setup.bash"

exec ros2 run fast_lio fastlio_mapping --ros-args \
  --params-file "${bundle_dir}/config/mid360_t1000_realtime.yaml"

