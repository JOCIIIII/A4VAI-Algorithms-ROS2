import argparse
import math
import os

from . import settings as CFG


def resolve_sitl_settings(args=None, *, environ=None):
    environ = os.environ if environ is None else environ
    resolved = {}
    for name, env, default in (
        ('duration', 'T1000_SIM_DURATION', CFG.SITL_DURATION_S),
        ('vx_cap', 'T1000_SIM_VX_CAP', CFG.SITL_ENU_X_CAP_MPS),
        ('vy_cap', 'T1000_SIM_VY_CAP', CFG.SITL_ENU_Y_CAP_MPS),
    ):
        value = getattr(args, name, None)
        source = 'cli'
        if value is None:
            value, source = (environ[env], 'environment') if env in environ else (default, 'settings')
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name} ({source}) must be a finite number') from None
        resolved[name] = (value, source)
    validate_sitl_settings(resolved)
    return resolved


def validate_sitl_settings(resolved):
    for name, lower in (('duration', 2.0), ('vx_cap', 0.0), ('vy_cap', 0.0)):
        value, source = resolved[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= lower):
            raise ValueError(f'{name} ({source}) must be finite and greater than {lower}')


def build_parser():
    parser = argparse.ArgumentParser(description='T1000 ROS 2 control node')
    parser.add_argument('--dry-run', action='store_true', help='Use the original command preview mode')
    parser.add_argument('--allow-no-rc', action='store_true', help='Use the original bench-only RC override')
    parser.add_argument('--isaac-sim', action='store_true',
                        help='Use /t1000 pose, velocity and pre-binned LiDAR inputs')
    parser.add_argument('--sim-auto-start', action='store_true',
                        help='SIMULATION ONLY: request PX4 offboard, arm, then land')
    parser.add_argument('--confirm-sitl', action='store_true',
                        help='Explicitly confirm no physical flight controller is connected')
    parser.add_argument('--duration', type=float, default=None,
                        help='Simulation auto-start duration before LAND [s]')
    parser.add_argument('--vx-cap', type=float, default=None,
                        help='SITL ENU x command cap after low-pass [m/s]')
    parser.add_argument('--vy-cap', type=float, default=None,
                        help='SITL ENU y command cap after low-pass [m/s]')
    parser.add_argument('--corner-mode', choices=('sharp', 'rounded'), default=None,
                        help='Override CORNER_MODE from settings.py for this run')
    parser.add_argument('--policy-authority', choices=('always', 'obstacle'), default=None,
                        help='Override POLICY_AUTHORITY_MODE from settings.py for this run')
    parser.add_argument('--goal-stop', action=argparse.BooleanOptionalAction, default=None,
                        help='Override GOAL_STOP_ENABLED from settings.py for this run')
    parser.add_argument('--ordered-waypoints', action=argparse.BooleanOptionalAction, default=None,
                        help='Require original waypoints to be reached in order')
    parser.add_argument('--until-goal', action='store_true',
                        help='Confirmed Isaac SITL only: disable the mission deadline')
    parser.add_argument('--land-on-goal', action='store_true',
                        help='Confirmed Isaac SITL only: LAND after ordered goal stop dwell')
    parser.add_argument('--settings-snapshot', default=None, metavar='PATH',
                        help='Save startup effective settings as a new JSON file')
    return parser


def validate_goal_lifecycle(*, until_goal, land_on_goal, isaac_sim, sim_auto_start,
                            allow_no_rc, confirm_sitl, dry_run,
                            ordered_waypoints=None, goal_stop=None):
    if not (until_goal or land_on_goal):
        return
    if not land_on_goal:
        raise ValueError('--until-goal requires --land-on-goal')
    if not (isaac_sim and sim_auto_start and allow_no_rc and confirm_sitl and not dry_run):
        raise ValueError('--land-on-goal requires --isaac-sim --sim-auto-start '
                         '--allow-no-rc --confirm-sitl and no --dry-run')
    ordered = CFG.ORDERED_WAYPOINTS_ENABLED if ordered_waypoints is None else ordered_waypoints
    stopped = CFG.GOAL_STOP_ENABLED if goal_stop is None else goal_stop
    if ordered is not True or stopped is not True:
        raise ValueError('--land-on-goal requires effective --ordered-waypoints and --goal-stop')


def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.sim_auto_start and not args.isaac_sim:
        parser.error('--sim-auto-start requires --isaac-sim')
    if args.sim_auto_start and args.dry_run:
        parser.error('--sim-auto-start cannot be combined with --dry-run')
    if args.sim_auto_start and not args.allow_no_rc:
        parser.error('--sim-auto-start requires --allow-no-rc (SITL only)')
    if args.sim_auto_start and not args.confirm_sitl:
        parser.error('--sim-auto-start requires --confirm-sitl')
    try:
        validate_goal_lifecycle(until_goal=args.until_goal, land_on_goal=args.land_on_goal,
            isaac_sim=args.isaac_sim, sim_auto_start=args.sim_auto_start,
            allow_no_rc=args.allow_no_rc, confirm_sitl=args.confirm_sitl,
            dry_run=args.dry_run, ordered_waypoints=args.ordered_waypoints, goal_stop=args.goal_stop)
        sitl_settings = resolve_sitl_settings(args)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ['T1000_ISAAC_SIM'] = '1' if args.isaac_sim else '0'
    os.environ['T1000_SIM_AUTO_START'] = '1' if args.sim_auto_start else '0'
    os.environ['T1000_SITL_CONFIRMED'] = '1' if args.confirm_sitl else '0'

    from .ros.px4_control_node import main as run_node
    run_node(sitl_settings=sitl_settings, corner_mode=args.corner_mode,
             settings_snapshot=args.settings_snapshot, policy_authority_mode=args.policy_authority,
             goal_stop=args.goal_stop, ordered_waypoints=args.ordered_waypoints,
             until_goal=args.until_goal, land_on_goal=args.land_on_goal)


if __name__ == '__main__':
    main()
