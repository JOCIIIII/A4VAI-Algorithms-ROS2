# T1000 Control 445

This package reads a path and LiDAR data, then sends PX4 velocity commands.
The policy is `models/policy_445.onnx`.
`models/checkpoint_445.pt` is the saved training checkpoint; it is not used at runtime.

Tested setup: Ubuntu 22.04, ROS 2 Humble, PX4 1.16, Python 3.10.
PX4, `px4_msgs`, Livox ROS driver, and FAST-LIO are not included.

Run these commands from the extracted folder.

```bash
source /opt/ros/humble/setup.bash
source <px4_msgs_workspace>/install/setup.bash
python3 -m pip install -e '.[inference]'
```

Keep `models/` beside `t1000_control/`. Use the editable install; a wheel does not
include the models.

## Check first

```bash
python3 -m t1000_control.preflight --root .
```

The report must show `445` inputs, `2` outputs, and no errors.

## ROS topics

| Data | Default topic |
|---|---|
| Path in world ENU | `/t1000/path` |
| Livox point cloud | `/livox/lidar` |
| PX4 position and velocity | `/fmu/out/vehicle_local_position` |
| PX4 attitude | `/fmu/out/vehicle_attitude` |
| PX4 state | `/fmu/out/vehicle_status_v1` |
| PX4 velocity command | `/fmu/in/trajectory_setpoint` |
| PX4 OFFBOARD mode | `/fmu/in/offboard_control_mode` |
| Dry-run command | `/t1000/cmd_preview` |

Change topic names with ROS remapping:

```bash
python3 -m t1000_control --dry-run --allow-no-rc --ros-args \
  -r /t1000/path:=/planner/path \
  -r /livox/lidar:=/sensor/lidar
```

## Run in SITL

Start PX4, the simulator, and its bridge. Keep publishing these inputs and the path:

| Simulator input | Message | Contract |
|---|---|---|
| `/t1000/pose` | `geometry_msgs/PoseStamped` | world ENU position and quaternion `x,y,z,w` |
| `/t1000/velocity` | `geometry_msgs/TwistStamped` | world ENU linear velocity in m/s |
| `/t1000/lidar_bins` | `std_msgs/Float32MultiArray` | 432 normalized bins from `t1000_control.sensors.binning.xyz_to_432bins()`; `1` means clear |

Path X/Y uses the same world-ENU frame. Path Z is ignored; height comes from
`FLIGHT_ALTITUDE_M`.

```bash
python3 -m t1000_control.ros.path_publisher \
  --points "0,0 10,0 10,10" --frame map --rate 2

python3 -m t1000_control --isaac-sim --sim-auto-start \
  --allow-no-rc --confirm-sitl --duration 120 \
  --vx-cap 3 --vy-cap 2 \
  --corner-mode rounded --policy-authority obstacle \
  --ordered-waypoints --goal-stop --until-goal --land-on-goal
```

Use `--confirm-sitl` only with local simulation. Only one node may publish the
PX4 trajectory setpoint. Use a mux or mode switch when another planner is active.

Use the same world-ENU `map` frame for the path and vehicle position. Put the
first point within `PATH_START_TOLERANCE_M` of the current position, or set
`PATH_START_MODE = "prepend_current"`. Repeated copies refresh the path without
resetting progress.

## Change values

Edit only `t1000_control/settings.py` for normal tuning.

| Change | Setting |
|---|---|
| Height | `FLIGHT_ALTITUDE_M` |
| Forward and side speed | `PATH_FORWARD_COMMAND_CAP_MPS`, `PATH_LATERAL_COMMAND_CAP_MPS` |
| Simulation speed limits | `SITL_ENU_X_CAP_MPS`, `SITL_ENU_Y_CAP_MPS` |
| Fast path return | `RECOVERY_LATERAL_CAP_MPS`, `RECOVERY_FORWARD_CAP_MPS` |
| Corner speed | `CORNER_SPEED_MPS`, `CORNER_DECEL_MPS2` |
| Obstacle response | `AVOID_ACTIVATE_DISTANCE_M`, `AVOID_FULL_DISTANCE_M` |
| Goal stop | `GOAL_STOP_RADIUS_M`, `GOAL_DECEL_MPS2`, `GOAL_DWELL_MAX_SPEED_MPS`, `GOAL_DWELL_S` |
| Path handoff | `PATH_START_MODE`, `PATH_START_TOLERANCE_M` |
| Height control | `KP_ALT_VEL`, `VZ_CMD_MAX` |
| Yaw control | `AVOIDANCE_YAW_MODE`, `KP_YAW_RATE`, `YAW_RATE_MAX`, `YAW_COMMAND_MIN_SPEED_MPS` |
| Loop rates | `CONTROL_HZ`, `PATH_PUBLISH_HZ` |
| Topics | `TOPIC_*` |

`AVOIDANCE_YAW_MODE = "path"` keeps the nose on the path. Use
`"command_velocity"` to turn the nose toward the avoidance command.

Do not change the model contract values: `OBS_DIM`, `ACT_DIM`, `VX_TARGET`,
`MODEL_FORWARD_ACTION_SCALE_MPS`, or `MODEL_LATERAL_ACTION_SCALE_MPS`.

## Livox IMU and FAST-LIO

Use the point cloud for the 432 LiDAR bins. FAST-LIO uses the cloud and Livox IMU;
the policy does not read the IMU directly. Check LIO odometry in shadow mode:

```bash
python3 -m t1000_control.ros.lio_px4_bridge \
  --duration 60 --output lio_shadow_check
```

Shadow mode does not send odometry to PX4. LIO settings are in
`integrations/lio/config/mid360_t1000_realtime.yaml` and
`t1000_control/settings.py`.

Keep the existing `(x, -y, -z)` mapping in
`t1000_control/core/coordinates.py`; the simulator output mapping is separate.

The policy was trained in Isaac Lab and checked in Isaac Sim with PX4 SITL;
these checks do not establish physical-flight validation.
