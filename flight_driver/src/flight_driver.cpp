// flight_driver: flight FSM for the A4VAI apps on the real vehicle (and the same in SILS / PILS).
//
// From A4VAI-Flight-Test's path_following_test (flown on the X500): PX4 AUTO_TAKEOFF to a height above the
// ground the vehicle rests on, OFFBOARD entered in the air, PathFollowing's attitude forwarded to PX4,
// landing. Every height and position is relative to where the vehicle rests at take-off - PX4's local
// origin is not there on the vehicle (13-45 m off in z in the X500's logs).
//
// mission:
//   wp_local  wp_csv_path rows "x,y,z" = north, east, height above the take-off point [m] (path_following_test)
//   wp_gps    wp_csv_path rows "lat,lon,alt" = WGS84 degrees, height above the take-off point [m]; projected
//             with PX4's own reference (vehicle_local_position ref_lat/ref_lon, PX4's MapProjection)
//   ca        learned collision avoidance (scripts/ros2/apps/noclimb_ca.sh): once the vehicle hovers in
//             OFFBOARD the goal (goal_north_m / goal_east_m from the take-off point, or goal_lat / goal_lon)
//             goes to the policy node on goal_topic (PointStamped, PX4 local ENU, z = path height); the node
//             makes the path; within goal_handover_radius_m PX4 position control stops at the goal, then lands.
//
// Safety, beyond path_following_test:
//   - arms once: a disarm after that (kill switch, PX4) ends the run; it never re-arms
//   - the pilot first: a mode other than the driver's (AUTO_TAKEOFF / AUTO_LOITER during the take-off,
//     OFFBOARD after it, AUTO_LAND once landing) = the pilot (COM_RC_OVERRIDE) or a PX4 failsafe -> every
//     output stops for good: no setpoint, no OFFBOARD heartbeat (OFFBOARD cannot be re-entered on stale
//     setpoints), no command
//   - PathFollowing's attitude is forwarded only as it arrives and only while PX4 is in OFFBOARD
//     (path_following_test re-sent the last one every 100 ms, also after the pilot had taken over)
//   - lands in place on: PathFollowing silent (pf_timeout_s), the CA path silent (wp_timeout_s), the fence
//     (distance from the take-off point, height), speed over speed_limit_mps, a PX4 position reset (xy over
//     abort_xy_reset_m, z over abort_z_reset_m, a new reference), an impact (IMU), dead reckoning, the
//     operator (Bool on /flight_driver/land)
//   - arms only when: PX4's pre-flight checks pass, the estimate has not reset for ekf_settle_s, PathFollowing
//     is up, the CA node is ready (ca), and - require_go - the operator said go (Bool on /flight_driver/go)
//   - writes no PX4 parameter
// State at 1 Hz on /flight_driver/state; per-tick log <log_dir>/flight_driver_<time>.csv.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/sensor_combined.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_global_position.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

#include <custom_msgs/msg/convey_local_waypoint_complete.hpp>
#include <custom_msgs/msg/local_waypoint_setpoint.hpp>

using namespace std::chrono_literals;
using px4_msgs::msg::VehicleCommand;
using px4_msgs::msg::VehicleStatus;

namespace
{
constexpr double kEarthRadius = 6371000.0;  // PX4 CONSTANTS_RADIUS_OF_EARTH

// PX4 MapProjection::project (src/lib/geo/geo.cpp): azimuthal equidistant about the reference point
void project(double ref_lat, double ref_lon, double lat, double lon, double & north, double & east)
{
    const double ref_lat_rad = ref_lat * M_PI / 180.0;
    const double ref_lon_rad = ref_lon * M_PI / 180.0;
    const double lat_rad = lat * M_PI / 180.0;
    const double lon_rad = lon * M_PI / 180.0;
    const double sin_lat = std::sin(lat_rad);
    const double cos_lat = std::cos(lat_rad);
    const double cos_d_lon = std::cos(lon_rad - ref_lon_rad);
    const double arg = std::clamp(
        std::sin(ref_lat_rad) * sin_lat + std::cos(ref_lat_rad) * cos_lat * cos_d_lon, -1.0, 1.0);
    const double c = std::acos(arg);
    const double k = (std::fabs(c) > 0.0) ? c / std::sin(c) : 1.0;
    north = k * (std::cos(ref_lat_rad) * sin_lat - std::sin(ref_lat_rad) * cos_lat * cos_d_lon) * kEarthRadius;
    east = k * cos_lat * std::sin(lon_rad - ref_lon_rad) * kEarthRadius;
}

enum class Phase { INIT, PREFLIGHT, WAIT_GO, ARMING, TAKEOFF, OFFBOARD_ENTRY, STABILIZE, FOLLOW, GOAL,
                   LANDING, YIELDED, DONE };

const char * phase_name(Phase p)
{
    switch (p) {
        case Phase::INIT: return "INIT";
        case Phase::PREFLIGHT: return "PREFLIGHT";
        case Phase::WAIT_GO: return "WAIT_GO";
        case Phase::ARMING: return "ARMING";
        case Phase::TAKEOFF: return "TAKEOFF";
        case Phase::OFFBOARD_ENTRY: return "OFFBOARD_ENTRY";
        case Phase::STABILIZE: return "STABILIZE";
        case Phase::FOLLOW: return "FOLLOW";
        case Phase::GOAL: return "GOAL";
        case Phase::LANDING: return "LANDING";
        case Phase::YIELDED: return "YIELDED";
        case Phase::DONE: return "DONE";
    }
    return "?";
}

// the pilot's own modes: a landing that turns into one of these was taken over
bool pilot_mode(uint8_t nav)
{
    return nav == VehicleStatus::NAVIGATION_STATE_MANUAL || nav == VehicleStatus::NAVIGATION_STATE_ALTCTL ||
           nav == VehicleStatus::NAVIGATION_STATE_POSCTL || nav == VehicleStatus::NAVIGATION_STATE_ACRO ||
           nav == VehicleStatus::NAVIGATION_STATE_STAB;
}
}  // namespace

class FlightDriver : public rclcpp::Node
{
public:
    FlightDriver() : Node("flight_driver")
    {
        system_id_ = declare_parameter<int>("system_id", 1);
        const std::string ns = declare_parameter<std::string>("vehicle_namespace", "");
        mission_ = declare_parameter<std::string>("mission", "wp_local");
        wp_csv_path_ = declare_parameter<std::string>("wp_csv_path", "");
        takeoff_alt_ = declare_parameter<double>("takeoff_altitude", 3.0);
        takeoff_alt_tol_ = declare_parameter<double>("takeoff_alt_tol", 0.5);
        stabilize_s_ = declare_parameter<double>("stabilize_s", 2.0);
        // wp modes: PathFollowing's cruise speed, on /desired_speed at 2 Hz (0 = PathFollowing's yaml value)
        desired_speed_ = declare_parameter<double>("desired_speed", 0.0);
        // ca
        goal_north_ = declare_parameter<double>("goal_north_m", 0.0);
        goal_east_ = declare_parameter<double>("goal_east_m", 0.0);
        goal_lat_ = declare_parameter<double>("goal_lat", 0.0);
        goal_lon_ = declare_parameter<double>("goal_lon", 0.0);
        goal_topic_ = declare_parameter<std::string>("goal_topic", "/flight_driver/goal");
        ready_topic_ = declare_parameter<std::string>("ready_topic", mission_ == "ca" ? "/noclimb/ready" : "");
        goal_handover_ = declare_parameter<double>("goal_handover_radius_m", 3.0);
        goal_settle_radius_ = declare_parameter<double>("goal_settle_radius_m", 0.3);
        goal_settle_speed_ = declare_parameter<double>("goal_settle_speed", 0.3);
        goal_settle_s_ = declare_parameter<double>("goal_settle_time_s", 1.0);
        goal_timeout_s_ = declare_parameter<double>("goal_timeout_s", 20.0);
        // ca: /path_following_complete within this distance of the goal = arrived (fly to it and land there);
        // further away it is the node's landing request (contact, lidar lost) -> land in place. 0 = always in
        // place (the handover radius covers arrival)
        goal_pf_done_max_ = declare_parameter<double>("goal_pf_done_max_m", 0.0);
        // arming gate
        require_go_ = declare_parameter<bool>("require_go", true);
        ekf_settle_s_ = declare_parameter<double>("ekf_settle_s", 30.0);
        max_mission_radius_ = declare_parameter<double>("max_mission_radius_m", 300.0);
        min_wp_alt_ = declare_parameter<double>("min_wp_alt_m", 1.0);
        // in flight
        pf_timeout_s_ = declare_parameter<double>("pf_timeout_s", 0.5);
        pf_start_timeout_s_ = declare_parameter<double>("pf_start_timeout_s", 20.0);
        // ca: the node re-sends its path every 50 ms, 1 Hz in its final approach (3 m) -> 2.5 s
        wp_timeout_s_ = declare_parameter<double>("wp_timeout_s", mission_ == "ca" ? 2.5 : 0.0);
        fence_margin_ = declare_parameter<double>("fence_margin_m", 20.0);
        fence_alt_margin_ = declare_parameter<double>("fence_alt_margin_m", 10.0);
        speed_limit_ = declare_parameter<double>("speed_limit_mps", 0.0);
        abort_xy_reset_ = declare_parameter<double>("abort_xy_reset_m", 3.0);
        abort_z_reset_ = declare_parameter<double>("abort_z_reset_m", 1.0);
        // real X500 logs: |a| <= 19.4 m/s^2 in 20 normal flights, 150-250 in the two crashes; SITL the same
        impact_accel_ = declare_parameter<double>("impact_accel_m_s2", 30.0);
        impact_samples_ = static_cast<int>(declare_parameter<int>("impact_samples", 2));
        impact_window_ = declare_parameter<double>("impact_window_s", 0.2);
        abort_on_dead_reckoning_ = declare_parameter<bool>("abort_on_dead_reckoning", true);
        log_dir_ = declare_parameter<std::string>("log_dir", "");

        if (mission_ != "wp_local" && mission_ != "wp_gps" && mission_ != "ca") {
            throw std::runtime_error("mission must be wp_local, wp_gps or ca, not '" + mission_ + "'");
        }
        if (mission_ != "ca" && !load_csv()) {
            throw std::runtime_error("no usable waypoints in '" + wp_csv_path_ + "'");
        }
        needs_gps_ = mission_ == "wp_gps" || (mission_ == "ca" && (goal_lat_ != 0.0 || goal_lon_ != 0.0));

        const std::string p = ns.empty() ? "/fmu/" : "/" + ns + "/fmu/";
        auto qos_sensor = rclcpp::SensorDataQoS();
        rclcpp::QoS qos(10);
        ocm_pub_ = create_publisher<px4_msgs::msg::OffboardControlMode>(p + "in/offboard_control_mode", qos);
        traj_pub_ = create_publisher<px4_msgs::msg::TrajectorySetpoint>(p + "in/trajectory_setpoint", qos);
        att_pub_ = create_publisher<px4_msgs::msg::VehicleAttitudeSetpoint>(p + "in/vehicle_attitude_setpoint", qos);
        cmd_pub_ = create_publisher<VehicleCommand>(p + "in/vehicle_command", qos);
        wp_pub_ = create_publisher<custom_msgs::msg::LocalWaypointSetpoint>("/local_waypoint_setpoint_to_PF", qos);
        speed_pub_ = create_publisher<std_msgs::msg::Float32>("/desired_speed", qos);
        goal_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(goal_topic_, qos);
        state_pub_ = create_publisher<std_msgs::msg::String>("/flight_driver/state", qos);
        ctrl_hb_pub_ = create_publisher<std_msgs::msg::Bool>("/controller_heartbeat", qos);
        pp_hb_pub_ = create_publisher<std_msgs::msg::Bool>("/path_planning_heartbeat", qos);
        ca_hb_pub_ = create_publisher<std_msgs::msg::Bool>("/collision_avoidance_heartbeat", qos);

        status_sub_ = create_subscription<VehicleStatus>(p + "out/vehicle_status_v1", qos_sensor,
            [this](const VehicleStatus::SharedPtr m) { status_ = *m; t_status_ = now_s(); });
        lpos_sub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(p + "out/vehicle_local_position",
            qos_sensor, [this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr m) { lpos_cb(*m); });
        gpos_sub_ = create_subscription<px4_msgs::msg::VehicleGlobalPosition>(p + "out/vehicle_global_position",
            qos_sensor, [this](const px4_msgs::msg::VehicleGlobalPosition::SharedPtr m) { gpos_ = *m; t_gpos_ = now_s(); });
        att_sub_ = create_subscription<px4_msgs::msg::VehicleAttitude>(p + "out/vehicle_attitude", qos_sensor,
            [this](const px4_msgs::msg::VehicleAttitude::SharedPtr m) { att_ = *m; t_att_ = now_s(); });
        imu_sub_ = create_subscription<px4_msgs::msg::SensorCombined>(p + "out/sensor_combined", qos_sensor,
            [this](const px4_msgs::msg::SensorCombined::SharedPtr m) { imu_cb(*m); });
        pf_att_sub_ = create_subscription<px4_msgs::msg::VehicleAttitudeSetpoint>("/pf_att_2_control", qos,
            [this](const px4_msgs::msg::VehicleAttitudeSetpoint::SharedPtr m) { pf_att_cb(*m); });
        pf_done_sub_ = create_subscription<std_msgs::msg::Bool>("/path_following_complete", qos,
            [this](const std_msgs::msg::Bool::SharedPtr m) {
                if (m->data && !pf_done_ && (phase_ == Phase::FOLLOW || phase_ == Phase::GOAL)) {
                    pf_done_ = true;
                    RCLCPP_INFO(get_logger(), "/path_following_complete received");
                }
            });
        pf_hb_sub_ = create_subscription<std_msgs::msg::Bool>("/path_following_heartbeat", qos,
            [this](const std_msgs::msg::Bool::SharedPtr) { t_pf_hb_ = now_s(); });
        wp_ack_sub_ = create_subscription<custom_msgs::msg::ConveyLocalWaypointComplete>(
            "/convey_local_waypoint_complete", qos,
            [this](const custom_msgs::msg::ConveyLocalWaypointComplete::SharedPtr m) {
                if (m->convey_local_waypoint_is_complete) wp_acked_ = true;
            });
        // the CA node's path to PathFollowing (ca: its freshness is watched)
        wp_watch_sub_ = create_subscription<custom_msgs::msg::LocalWaypointSetpoint>(
            "/local_waypoint_setpoint_to_PF", qos,
            [this](const custom_msgs::msg::LocalWaypointSetpoint::SharedPtr) { t_wp_ = now_s(); });
        if (!ready_topic_.empty()) {
            ready_sub_ = create_subscription<std_msgs::msg::Bool>(ready_topic_, qos,
                [this](const std_msgs::msg::Bool::SharedPtr m) { if (m->data) t_ready_ = now_s(); });
        }
        go_sub_ = create_subscription<std_msgs::msg::Bool>("/flight_driver/go", qos,
            [this](const std_msgs::msg::Bool::SharedPtr m) {
                if (m->data && phase_ == Phase::WAIT_GO) {
                    go_ = true;
                }
            });
        land_sub_ = create_subscription<std_msgs::msg::Bool>("/flight_driver/land", qos,
            [this](const std_msgs::msg::Bool::SharedPtr m) {
                if (m->data) {
                    abort_flight("operator: /flight_driver/land");
                }
            });

        timer_ = create_wall_timer(100ms, std::bind(&FlightDriver::tick, this));
        hb_timer_ = create_wall_timer(1000ms, std::bind(&FlightDriver::heartbeats, this));

        std::ostringstream m;
        m << "mission " << mission_ << ", take-off " << takeoff_alt_ << " m above the take-off point";
        if (mission_ == "ca") {
            if (needs_gps_) {
                m << ", goal lat " << goal_lat_ << " lon " << goal_lon_;
            } else {
                m << ", goal " << goal_north_ << " m north " << goal_east_ << " m east of the take-off point";
            }
            m << " (to the policy on " << goal_topic_ << ")";
        } else {
            m << ", " << wp_.size() << " waypoints from " << wp_csv_path_;
            if (desired_speed_ > 0.0) m << " at " << desired_speed_ << " m/s";
        }
        m << (require_go_ ? ", arms on the operator's go" : ", arms when ready");
        RCLCPP_INFO(get_logger(), "%s", m.str().c_str());
    }

    ~FlightDriver() override
    {
        if (csv_.is_open()) csv_.close();
    }

private:
    struct Wp { double a, b, h; };   // wp_local: north, east, height; wp_gps: lat, lon, height

    double now_s() { return get_clock()->now().nanoseconds() * 1e-9; }
    uint64_t now_us() { return get_clock()->now().nanoseconds() / 1000; }
    uint8_t nav() const { return status_.nav_state; }
    bool armed() const { return status_.arming_state == VehicleStatus::ARMING_STATE_ARMED; }
    bool in_offboard() const { return nav() == VehicleStatus::NAVIGATION_STATE_OFFBOARD; }
    double agl() const { return -(static_cast<double>(lpos_.z) - ground_z_); }
    bool flying() const
    {
        return phase_ == Phase::TAKEOFF || phase_ == Phase::OFFBOARD_ENTRY || phase_ == Phase::STABILIZE ||
               phase_ == Phase::FOLLOW || phase_ == Phase::GOAL;
    }
    // the phases that command PX4 in OFFBOARD (and keep its OFFBOARD heartbeat up)
    bool commanding() const
    {
        return phase_ == Phase::OFFBOARD_ENTRY || phase_ == Phase::STABILIZE || phase_ == Phase::FOLLOW ||
               phase_ == Phase::GOAL;
    }

    bool load_csv()
    {
        std::ifstream f(wp_csv_path_);
        if (!f.is_open()) {
            RCLCPP_ERROR(get_logger(), "cannot open %s", wp_csv_path_.c_str());
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            const auto c = line.find('#');
            if (c != std::string::npos) line = line.substr(0, c);
            std::stringstream ss(line);
            std::string tok;
            std::vector<double> v;
            bool numeric = true;
            while (std::getline(ss, tok, ',')) {
                try {
                    size_t used = 0;
                    v.push_back(std::stod(tok, &used));
                } catch (...) {
                    numeric = false;
                }
            }
            if (!numeric || v.size() < 3) continue;   // header, comments, blank lines
            wp_.push_back({v[0], v[1], v[2]});
        }
        if (wp_.empty()) return false;
        for (const auto & w : wp_) {
            if (mission_ == "wp_gps" && (std::fabs(w.a) > 90.0 || std::fabs(w.b) > 180.0)) {
                RCLCPP_ERROR(get_logger(), "not a lat,lon,alt row: %.7f, %.7f, %.2f", w.a, w.b, w.h);
                return false;
            }
        }
        return true;
    }

    // ---- inputs ----
    void lpos_cb(const px4_msgs::msg::VehicleLocalPosition & m)
    {
        const double t = now_s();
        if (t_lpos_ > 0.0) {
            if (m.xy_reset_counter != lpos_.xy_reset_counter || m.z_reset_counter != lpos_.z_reset_counter ||
                m.ref_timestamp != lpos_.ref_timestamp) {
                t_ekf_change_ = t;
                const double dxy = std::hypot(m.delta_xy[0], m.delta_xy[1]);
                if (flying()) {
                    if (m.ref_timestamp != lpos_.ref_timestamp) {
                        abort_flight("PX4 changed its position reference in flight");
                    } else if (m.xy_reset_counter != lpos_.xy_reset_counter && dxy > abort_xy_reset_) {
                        abort_flight("PX4 position reset " + fmt(dxy) + " m horizontally");
                    } else if (m.z_reset_counter != lpos_.z_reset_counter && std::fabs(m.delta_z) > abort_z_reset_) {
                        abort_flight("PX4 position reset " + fmt(m.delta_z) + " m vertically");
                    } else {
                        RCLCPP_WARN(get_logger(), "PX4 position reset in flight: xy %.2f m, z %.2f m (kept flying)",
                                    dxy, m.delta_z);
                    }
                }
            }
        } else {
            t_ekf_change_ = t;
        }
        lpos_ = m;
        t_lpos_ = t;
        if (flying() && abort_on_dead_reckoning_ && m.dead_reckoning) {
            abort_flight("PX4 dead reckoning (no GNSS)");
        }
    }

    void imu_cb(const px4_msgs::msg::SensorCombined & m)
    {
        const double a = std::sqrt(m.accelerometer_m_s2[0] * m.accelerometer_m_s2[0] +
                                   m.accelerometer_m_s2[1] * m.accelerometer_m_s2[1] +
                                   m.accelerometer_m_s2[2] * m.accelerometer_m_s2[2]);
        imu_seen_ = true;
        if (impact_accel_ <= 0.0 || !commanding()) return;
        const double t = now_s();
        if (a > impact_accel_) {
            impacts_.push_back(t);
            max_impact_ = std::max(max_impact_, a);
        }
        while (!impacts_.empty() && t - impacts_.front() > impact_window_) impacts_.pop_front();
        if (static_cast<int>(impacts_.size()) >= impact_samples_) {
            abort_flight("impact: IMU " + fmt(max_impact_) + " m/s^2 (" + std::to_string(impacts_.size()) +
                         " samples in " + fmt(impact_window_) + " s)");
        }
    }

    void pf_att_cb(const px4_msgs::msg::VehicleAttitudeSetpoint & m)
    {
        pf_att_ = m;
        t_pf_att_ = now_s();
        // forward as it arrives (~250 Hz), only while the driver flies the vehicle in OFFBOARD
        if (use_attitude_ && phase_ == Phase::FOLLOW && in_offboard()) {
            px4_msgs::msg::VehicleAttitudeSetpoint out = m;
            out.timestamp = now_us();
            att_pub_->publish(out);
            ++forwarded_;
        }
    }

    // ---- outputs ----
    void command(uint16_t cmd, float p1 = 0.f, float p2 = 0.f, float p4 = NAN, double p5 = NAN, double p6 = NAN,
                 float p7 = NAN)
    {
        VehicleCommand c{};
        c.timestamp = now_us();
        c.command = cmd;
        c.param1 = p1;
        c.param2 = p2;
        c.param4 = p4;
        c.param5 = p5;
        c.param6 = p6;
        c.param7 = p7;
        c.target_system = static_cast<uint8_t>(system_id_);
        c.target_component = 1;
        c.source_system = 1;
        c.source_component = 1;
        c.confirmation = 0;
        c.from_external = true;
        cmd_pub_->publish(c);
    }
    void send_arm(bool arm) { command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, arm ? 1.f : 0.f); }
    void send_offboard() { command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.f, 6.f); }
    void send_land() { command(VehicleCommand::VEHICLE_CMD_NAV_LAND); }
    void send_takeoff()
    {
        // PX4 AUTO_TAKEOFF to an AMSL altitude: here + takeoff_altitude (path_following_test)
        command(VehicleCommand::VEHICLE_CMD_NAV_TAKEOFF, -1.f, 0.f, lpos_.heading, gpos_.lat, gpos_.lon,
                gpos_.alt + static_cast<float>(takeoff_alt_));
    }
    void send_setpoint(double n, double e, double d, double yaw)
    {
        px4_msgs::msg::TrajectorySetpoint s{};
        s.timestamp = now_us();
        s.position = {static_cast<float>(n), static_cast<float>(e), static_cast<float>(d)};
        s.velocity = {NAN, NAN, NAN};
        s.acceleration = {NAN, NAN, NAN};
        s.jerk = {NAN, NAN, NAN};
        s.yaw = static_cast<float>(yaw);
        s.yawspeed = NAN;
        traj_pub_->publish(s);
    }

    void heartbeats()
    {
        std_msgs::msg::Bool hb;
        hb.data = true;
        ctrl_hb_pub_->publish(hb);
        if (mission_ != "ca") {          // ca: the policy node sends these once it flies
            pp_hb_pub_->publish(hb);
            ca_hb_pub_->publish(hb);
        }
    }

    // ---- flight ----
    void set_phase(Phase p, const std::string & why = "")
    {
        if (p == phase_) return;
        RCLCPP_INFO(get_logger(), "%s -> %s%s%s", phase_name(phase_), phase_name(p), why.empty() ? "" : ": ",
                    why.c_str());
        phase_ = p;
        t_phase_ = now_s();
        if (p != Phase::FOLLOW) use_attitude_ = false;
    }

    // lands in place (in flight) or gives up before take-off
    void abort_flight(const std::string & why)
    {
        if (phase_ == Phase::LANDING || phase_ == Phase::YIELDED || phase_ == Phase::DONE) return;
        if (!flying()) {
            if (phase_ == Phase::ARMING && armed()) send_arm(false);
            RCLCPP_ERROR(get_logger(), "not flying: %s", why.c_str());
            reason_ = why;
            finish("aborted before take-off: " + why);
            return;
        }
        RCLCPP_ERROR(get_logger(), "ABORT - landing in place: %s", why.c_str());
        reason_ = why;
        start_landing(why);
    }

    void start_landing(const std::string & why)
    {
        set_phase(Phase::LANDING, why);
        send_land();
        land_sends_ = 1;
        t_land_sent_ = now_s();
    }

    void yield(const std::string & why)
    {
        RCLCPP_ERROR(get_logger(), "YIELD (no more commands): %s", why.c_str());
        reason_ = why;
        set_phase(Phase::YIELDED, why);
    }

    void finish(const std::string & why)
    {
        RCLCPP_INFO(get_logger(), "done: %s (%ld attitude setpoints forwarded)", why.c_str(), forwarded_);
        set_phase(Phase::DONE, why);
        publish_state(true);
        if (csv_.is_open()) csv_.flush();
        rclcpp::shutdown();
    }

    // what still keeps it from arming ("" = ready)
    std::string readiness(double t)
    {
        std::vector<std::string> miss;
        if (t - t_status_ > 1.0) miss.push_back("PX4 status");
        if (t - t_lpos_ > 1.0) miss.push_back("PX4 local position");
        else if (!lpos_.xy_valid || !lpos_.z_valid || !lpos_.v_xy_valid) miss.push_back("PX4 position not valid");
        if (t - t_gpos_ > 2.0 || !gpos_.lat_lon_valid || !gpos_.alt_valid) miss.push_back("PX4 global position");
        if (needs_gps_ && !lpos_.xy_global) miss.push_back("PX4 GPS reference");
        if (t - t_status_ <= 1.0 && !status_.pre_flight_checks_pass) miss.push_back("PX4 pre-flight checks");
        if (t - t_status_ <= 1.0 && armed()) miss.push_back("disarmed vehicle (it is armed)");
        if (t - t_ekf_change_ < ekf_settle_s_) {
            miss.push_back("estimate steady " + fmt(t - t_ekf_change_) + "/" + fmt(ekf_settle_s_) + " s");
        }
        if (t - t_pf_hb_ > 2.5) miss.push_back("PathFollowing heartbeat");
        if (!ready_topic_.empty() && t - t_ready_ > 2.5) miss.push_back(ready_topic_);
        std::string s;
        for (size_t i = 0; i < miss.size(); ++i) s += (i ? ", " : "") + miss[i];
        return s;
    }

    // waypoints / goal in PX4's local NED frame, relative to the take-off point (home) when flying,
    // to the current position before
    bool build_mission(double home_n, double home_e, double ground_z, std::string & err)
    {
        mission_ned_.clear();
        if (mission_ == "ca") {
            double n = home_n + goal_north_, e = home_e + goal_east_;
            if (needs_gps_) project(lpos_.ref_lat, lpos_.ref_lon, goal_lat_, goal_lon_, n, e);
            goal_ned_ = {n, e, ground_z - takeoff_alt_};
            mission_ned_.push_back(goal_ned_);
            if (std::hypot(n - home_n, e - home_e) < 5.0) {
                err = "the goal is " + fmt(std::hypot(n - home_n, e - home_e)) + " m from the take-off point (< 5 m)";
                return false;
            }
        } else {
            for (const auto & w : wp_) {
                double n = home_n + w.a, e = home_e + w.b;
                if (mission_ == "wp_gps") project(lpos_.ref_lat, lpos_.ref_lon, w.a, w.b, n, e);
                if (w.h < min_wp_alt_) {
                    err = "waypoint height " + fmt(w.h) + " m below min_wp_alt_m " + fmt(min_wp_alt_);
                    return false;
                }
                mission_ned_.push_back({n, e, ground_z - w.h});
            }
        }
        double r = 0.0, h = takeoff_alt_;
        for (const auto & p : mission_ned_) {
            r = std::max(r, std::hypot(p[0] - home_n, p[1] - home_e));
            h = std::max(h, ground_z - p[2]);
        }
        if (r > max_mission_radius_) {
            err = "mission reaches " + fmt(r) + " m from the take-off point (> max_mission_radius_m " +
                  fmt(max_mission_radius_) + ")";
            return false;
        }
        fence_r_ = r + fence_margin_;
        fence_h_ = h + fence_alt_margin_;
        return true;
    }

    void print_mission(double home_n, double home_e, double ground_z)
    {
        RCLCPP_INFO(get_logger(), "mission from here (N, E, height above the take-off point), fence %.0f m / %.0f m up:",
                    fence_r_, fence_h_);
        for (size_t i = 0; i < mission_ned_.size(); ++i) {
            const auto & p = mission_ned_[i];
            RCLCPP_INFO(get_logger(), "  %s %zu: %+8.1f %+8.1f  %5.1f m  (%.1f m away)", mission_ == "ca" ? "goal" : "wp",
                        i, p[0] - home_n, p[1] - home_e, ground_z - p[2], std::hypot(p[0] - home_n, p[1] - home_e));
        }
    }

    void send_waypoints()
    {
        custom_msgs::msg::LocalWaypointSetpoint m;
        m.path_planning_complete = true;
        std::vector<std::array<double, 3>> pts;
        // start the path where the vehicle is when the first waypoint is not here
        const auto & f = mission_ned_.front();
        if (std::sqrt(std::pow(f[0] - lpos_.x, 2) + std::pow(f[1] - lpos_.y, 2) + std::pow(f[2] - lpos_.z, 2)) > 1.0) {
            pts.push_back({lpos_.x, lpos_.y, lpos_.z});
        }
        pts.insert(pts.end(), mission_ned_.begin(), mission_ned_.end());
        for (const auto & p : pts) {
            m.waypoint_x.push_back(p[0]);
            m.waypoint_y.push_back(p[1]);
            m.waypoint_z.push_back(-p[2]);   // PathFollowing takes z up
        }
        wp_pub_->publish(m);
        t_wp_sent_ = now_s();
        ++wp_sends_;
        RCLCPP_INFO(get_logger(), "sent %zu waypoints to PathFollowing (try %d)", pts.size(), wp_sends_);
    }

    void publish_goal()
    {
        geometry_msgs::msg::PointStamped g;
        g.header.stamp = get_clock()->now();
        g.header.frame_id = "px4_local_enu";
        g.point.x = goal_ned_[1];
        g.point.y = goal_ned_[0];
        g.point.z = -goal_ned_[2];
        goal_pub_->publish(g);
    }

    void tick()
    {
        const double t = now_s();
        ++ticks_;
        step(t);
        if (commanding()) {
            px4_msgs::msg::OffboardControlMode o{};
            o.timestamp = now_us();
            o.position = !use_attitude_;
            o.attitude = use_attitude_;
            ocm_pub_->publish(o);
        }
        if (mission_ != "ca" && desired_speed_ > 0.0 && ticks_ % 5 == 0 && phase_ != Phase::DONE) {
            std_msgs::msg::Float32 v;
            v.data = static_cast<float>(desired_speed_);
            speed_pub_->publish(v);
        }
        if (mission_ == "ca" && goal_sent_ && ticks_ % 5 == 0 && (phase_ == Phase::FOLLOW || phase_ == Phase::GOAL)) {
            publish_goal();
        }
        if (ticks_ % 10 == 0) publish_state(false);
        log_row(t);
    }

    void step(double t)
    {
        // a disarm after arming ends the run, whatever the phase (kill switch, PX4 failsafe, landed)
        if (armed_once_ && t - t_status_ < 1.0 && !armed() && phase_ != Phase::DONE) {
            finish(phase_ == Phase::LANDING ? "landed and disarmed" : std::string("disarmed in ") + phase_name(phase_));
            return;
        }
        if (flying() && t - t_lpos_ > 1.0) {
            // no PX4 data: nothing can be commanded safely; PX4's own OFFBOARD loss failsafe acts
            if (!px4_silent_warned_) RCLCPP_ERROR(get_logger(), "no PX4 local position for %.1f s", t - t_lpos_);
            px4_silent_warned_ = true;
        } else {
            px4_silent_warned_ = false;
        }

        switch (phase_) {
        case Phase::INIT:
            if (ticks_ >= 10) set_phase(Phase::PREFLIGHT);
            break;

        case Phase::PREFLIGHT:
        case Phase::WAIT_GO: {
            const std::string miss = readiness(t);
            if (!miss.empty()) {
                if (phase_ == Phase::WAIT_GO) {
                    set_phase(Phase::PREFLIGHT, "not ready any more: " + miss);
                    go_ = false;
                }
                if (ticks_ % 20 == 0) RCLCPP_INFO(get_logger(), "waiting for: %s", miss.c_str());
                break;
            }
            if (phase_ == Phase::PREFLIGHT) {
                std::string err;
                if (!build_mission(lpos_.x, lpos_.y, lpos_.z, err)) {
                    abort_flight("mission rejected: " + err);
                    break;
                }
                print_mission(lpos_.x, lpos_.y, lpos_.z);
                go_ = false;
                if (require_go_) {
                    set_phase(Phase::WAIT_GO, "READY - waiting for the operator's go (Bool on /flight_driver/go)");
                    break;
                }
                go_ = true;
            }
            if (go_) {
                set_phase(Phase::ARMING);
                arm_sends_ = 0;
            }
            break;
        }

        case Phase::ARMING:
            if (armed()) {
                armed_once_ = true;
                // the take-off point: home of the waypoints, the ground of every height
                ground_z_ = lpos_.z;
                home_n_ = lpos_.x;
                home_e_ = lpos_.y;
                yaw0_ = lpos_.heading;
                std::string err;
                if (!build_mission(home_n_, home_e_, ground_z_, err)) {
                    abort_flight("mission rejected: " + err);
                    break;
                }
                set_phase(Phase::TAKEOFF, "armed at N " + fmt(home_n_) + " E " + fmt(home_e_) + ", ground z " + fmt(ground_z_));
                send_takeoff();
                takeoff_sends_ = 1;
                t_takeoff_sent_ = t;
                break;
            }
            if (t - t_phase_ > 10.0) {
                abort_flight("did not arm within 10 s");
                break;
            }
            if (ticks_ % 10 == 0 && arm_sends_ < 5) {
                send_arm(true);
                ++arm_sends_;
            }
            break;

        case Phase::TAKEOFF: {
            if (nav() == VehicleStatus::NAVIGATION_STATE_AUTO_TAKEOFF) seen_takeoff_ = true;
            if (seen_takeoff_ && nav() != VehicleStatus::NAVIGATION_STATE_AUTO_TAKEOFF &&
                nav() != VehicleStatus::NAVIGATION_STATE_AUTO_LOITER) {
                yield("mode " + std::to_string(nav()) + " during the take-off");
                break;
            }
            const double h = agl();
            // PX4 ends AUTO_TAKEOFF in AUTO_LOITER within its acceptance (~0.8 m) of the height; LOITER lower
            // down after the take-off began = a hold from the pilot / ground station
            const bool loiter = seen_takeoff_ && nav() == VehicleStatus::NAVIGATION_STATE_AUTO_LOITER;
            if (loiter && h < takeoff_alt_ - std::max(1.0, 0.3 * takeoff_alt_)) {
                yield("hold at " + fmt(h) + " m during the take-off to " + fmt(takeoff_alt_) + " m");
                break;
            }
            if (std::fabs(h - takeoff_alt_) < takeoff_alt_tol_ || loiter) {
                // hold here, at the take-off height above the ground it rested on (PX4 ends AUTO_TAKEOFF up to
                // ~0.8 m short of it; never an absolute z - PX4's origin is not on the ground)
                hover_ = {lpos_.x, lpos_.y, ground_z_ - takeoff_alt_};
                set_phase(Phase::OFFBOARD_ENTRY, "at " + fmt(h) + " m");
                break;
            }
            if (!seen_takeoff_ && t - t_takeoff_sent_ > 2.0) {
                if (takeoff_sends_ >= 3) {
                    abort_flight("PX4 did not take the take-off command");
                    break;
                }
                send_takeoff();
                ++takeoff_sends_;
                t_takeoff_sent_ = t;
            }
            if (t - t_phase_ > 40.0) abort_flight("take-off not done within 40 s (" + fmt(h) + " m)");
            break;
        }

        case Phase::OFFBOARD_ENTRY:
            send_setpoint(hover_[0], hover_[1], hover_[2], yaw0_);
            if (in_offboard()) {
                set_phase(Phase::STABILIZE);
                break;
            }
            if (nav() != VehicleStatus::NAVIGATION_STATE_AUTO_LOITER &&
                nav() != VehicleStatus::NAVIGATION_STATE_AUTO_TAKEOFF) {
                yield("mode " + std::to_string(nav()) + " before OFFBOARD");
                break;
            }
            if (ticks_ % 10 == 0) send_offboard();
            if (t - t_phase_ > 10.0) abort_flight("PX4 did not enter OFFBOARD within 10 s");
            break;

        case Phase::STABILIZE:
        case Phase::FOLLOW:
        case Phase::GOAL:
            if (!in_offboard()) {
                yield("PX4 left OFFBOARD (mode " + std::to_string(nav()) + ") - pilot or PX4 failsafe");
                break;
            }
            if (fence_r_ > 0.0 && std::hypot(lpos_.x - home_n_, lpos_.y - home_e_) > fence_r_) {
                abort_flight("fence: " + fmt(std::hypot(lpos_.x - home_n_, lpos_.y - home_e_)) + " m from the take-off point");
                break;
            }
            if (agl() > fence_h_) {
                abort_flight("fence: " + fmt(agl()) + " m up");
                break;
            }
            if (phase_ == Phase::STABILIZE) {
                send_setpoint(hover_[0], hover_[1], hover_[2], yaw0_);
                // at the take-off height for stabilize_s (at most 10 s more)
                at_height_ticks_ = std::fabs(agl() - takeoff_alt_) < 0.3 ? at_height_ticks_ + 1 : 0;
                if (at_height_ticks_ >= static_cast<int>(std::lround(stabilize_s_ * 10.0))) {
                    start_mission(t);
                } else if (t - t_phase_ > stabilize_s_ + 10.0) {
                    RCLCPP_WARN(get_logger(), "not within 0.3 m of %.1f m after %.0f s (%.2f m) - starting anyway",
                                takeoff_alt_, t - t_phase_, agl());
                    start_mission(t);
                }
            } else if (phase_ == Phase::FOLLOW) {
                follow(t);
            } else {
                goal_phase(t);
            }
            break;

        case Phase::LANDING:
            if (nav() == VehicleStatus::NAVIGATION_STATE_AUTO_LAND) seen_land_ = true;
            if (seen_land_ && nav() != VehicleStatus::NAVIGATION_STATE_AUTO_LAND && armed()) {
                yield("mode " + std::to_string(nav()) + " during the landing");
                break;
            }
            if (!seen_land_ && pilot_mode(nav())) {
                yield("pilot mode " + std::to_string(nav()) + " instead of the landing");
                break;
            }
            // NAV_LAND can be lost (one message): re-send until PX4 is landing, at most 5 times
            if (!seen_land_ && t - t_land_sent_ > 1.0 && land_sends_ < 5) {
                RCLCPP_WARN(get_logger(), "LAND not taken (mode %u) - re-sending", nav());
                send_land();
                ++land_sends_;
                t_land_sent_ = t;
            }
            // PX4 disarms by itself on the ground (COM_DISARM_LAND); after 60 s ask once (refused in the air)
            if (t - t_phase_ > 60.0 && armed() && !disarm_sent_) {
                RCLCPP_WARN(get_logger(), "still armed 60 s after LAND - sending disarm (PX4 refuses it in the air)");
                send_arm(false);
                disarm_sent_ = true;
            }
            break;

        case Phase::YIELDED:
            // the pilot / PX4 flies it; the run ends at the disarm (above)
            break;

        case Phase::DONE:
            break;
        }

        // speed limit (in the phases the driver flies)
        if ((phase_ == Phase::FOLLOW || phase_ == Phase::GOAL) && speed_limit_ > 0.0) {
            const double v = std::hypot(lpos_.vx, lpos_.vy);
            over_speed_ticks_ = v > speed_limit_ ? over_speed_ticks_ + 1 : 0;
            if (over_speed_ticks_ >= 5) abort_flight("speed " + fmt(v) + " m/s over " + fmt(speed_limit_) + " for 0.5 s");
        }
    }

    void start_mission(double t)
    {
        if (mission_ == "ca") {
            goal_sent_ = true;
            publish_goal();
            RCLCPP_INFO(get_logger(), "goal to the policy: N %.1f E %.1f (%.1f m away), path %.1f m up",
                        goal_ned_[0] - home_n_, goal_ned_[1] - home_e_,
                        std::hypot(goal_ned_[0] - lpos_.x, goal_ned_[1] - lpos_.y), ground_z_ - goal_ned_[2]);
        } else {
            wp_acked_ = false;
            wp_sends_ = 0;
            send_waypoints();
        }
        t_follow_ = t;
        set_phase(Phase::FOLLOW);
    }

    void follow(double t)
    {
        const bool pf_fresh = t_pf_att_ > t_follow_ && t - t_pf_att_ < pf_timeout_s_;
        if (use_attitude_ && !pf_fresh) {
            abort_flight("PathFollowing silent for " + fmt(t - t_pf_att_) + " s");
            return;
        }
        if (!use_attitude_) {
            if (pf_fresh) {
                use_attitude_ = true;
                RCLCPP_INFO(get_logger(), "PathFollowing flies it (attitude forwarded)");
            } else {
                send_setpoint(hover_[0], hover_[1], hover_[2], yaw0_);
                if (mission_ != "ca" && !wp_acked_ && t - t_wp_sent_ > 2.0 && wp_sends_ < 3) send_waypoints();
                if (t - t_follow_ > pf_start_timeout_s_) abort_flight("PathFollowing did not start");
                return;
            }
        }
        if (mission_ == "ca") {
            if (wp_timeout_s_ > 0.0 && t - t_wp_ > wp_timeout_s_) {
                abort_flight("no path from the policy node for " + fmt(t - t_wp_) + " s");
                return;
            }
            const double d = std::hypot(goal_ned_[0] - lpos_.x, goal_ned_[1] - lpos_.y);
            if (goal_handover_ > 0.0 && d < goal_handover_) {
                goal_yaw_ = lpos_.heading;
                set_phase(Phase::GOAL, fmt(d) + " m from the goal: PX4 position control to it");
                return;
            }
            if (pf_done_) {
                if (d <= goal_pf_done_max_) {
                    goal_yaw_ = lpos_.heading;
                    set_phase(Phase::GOAL, "path done " + fmt(d) + " m from the goal");
                } else {
                    abort_flight("landing requested (/path_following_complete) " + fmt(d) + " m from the goal");
                }
            }
        } else if (pf_done_) {
            start_landing("path complete");
        }
    }

    void goal_phase(double t)
    {
        send_setpoint(goal_ned_[0], goal_ned_[1], goal_ned_[2], goal_yaw_);
        const double d = std::hypot(goal_ned_[0] - lpos_.x, goal_ned_[1] - lpos_.y);
        const double v = std::sqrt(lpos_.vx * lpos_.vx + lpos_.vy * lpos_.vy + lpos_.vz * lpos_.vz);
        settle_ticks_ = (d < goal_settle_radius_ && v < goal_settle_speed_) ? settle_ticks_ + 1 : 0;
        const bool timeout = t - t_phase_ > goal_timeout_s_;
        if (settle_ticks_ >= static_cast<int>(std::lround(goal_settle_s_ * 10.0)) || timeout) {
            start_landing(std::string(timeout ? "goal approach timed out" : "at the goal") + " (" + fmt(d) + " m, " +
                          fmt(v) + " m/s)");
        }
    }

    // ---- reporting ----
    static std::string fmt(double v)
    {
        char b[32];
        std::snprintf(b, sizeof(b), "%.2f", v);
        return b;
    }

    void publish_state(bool force)
    {
        (void)force;
        std::ostringstream s;
        s << "phase=" << phase_name(phase_) << " nav=" << static_cast<int>(nav()) << " armed=" << armed();
        if (armed_once_) {
            s << " height=" << fmt(agl()) << " from_home=" << fmt(std::hypot(lpos_.x - home_n_, lpos_.y - home_e_))
              << " speed=" << fmt(std::hypot(lpos_.vx, lpos_.vy));
        }
        if (mission_ == "ca" && goal_sent_) s << " to_goal=" << fmt(std::hypot(goal_ned_[0] - lpos_.x, goal_ned_[1] - lpos_.y));
        s << " pf_age=" << fmt(t_pf_att_ > 0 ? now_s() - t_pf_att_ : -1.0) << " forwarded=" << forwarded_;
        if (!reason_.empty()) s << " reason=\"" << reason_ << "\"";
        if (phase_ == Phase::PREFLIGHT) s << " waiting=\"" << readiness(now_s()) << "\"";
        std_msgs::msg::String m;
        m.data = s.str();
        state_pub_->publish(m);
    }

    void log_row(double t)
    {
        if (!armed_once_ || csv_failed_) return;
        if (!csv_.is_open()) {
            std::string dir = log_dir_;
            if (dir.empty()) {
                const char * ws = std::getenv("WORKSPACE_DIR");
                dir = ws ? std::string(ws) + "/logs" : ".";
            }
            std::time_t now = std::time(nullptr);
            char ts[32];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
            const std::string path = dir + "/flight_driver_" + ts + ".csv";
            csv_.open(path);
            if (!csv_.is_open()) {
                RCLCPP_ERROR(get_logger(), "cannot write %s - no log", path.c_str());
                csv_failed_ = true;
                return;
            }
            RCLCPP_INFO(get_logger(), "log: %s", path.c_str());
            csv_ << "t,phase,nav,armed,x,y,z,vx,vy,vz,height,from_home,heading,q0,q1,q2,q3,"
                    "cmd_q0,cmd_q1,cmd_q2,cmd_q3,cmd_thrust,pf_age,wp_age,use_attitude,forwarded,"
                    "xy_resets,z_resets,dead_reckoning,to_goal\n";
        }
        csv_ << std::fixed;
        csv_.precision(3);
        csv_ << t << "," << phase_name(phase_) << "," << static_cast<int>(nav()) << "," << armed() << ","
             << lpos_.x << "," << lpos_.y << "," << lpos_.z << "," << lpos_.vx << "," << lpos_.vy << "," << lpos_.vz
             << "," << agl() << "," << std::hypot(lpos_.x - home_n_, lpos_.y - home_e_) << "," << lpos_.heading
             << "," << att_.q[0] << "," << att_.q[1] << "," << att_.q[2] << "," << att_.q[3] << "," << pf_att_.q_d[0]
             << "," << pf_att_.q_d[1] << "," << pf_att_.q_d[2] << "," << pf_att_.q_d[3] << "," << pf_att_.thrust_body[2]
             << "," << (t_pf_att_ > 0 ? t - t_pf_att_ : -1.0) << "," << (t_wp_ > 0 ? t - t_wp_ : -1.0) << ","
             << use_attitude_ << "," << forwarded_ << "," << static_cast<int>(lpos_.xy_reset_counter) << ","
             << static_cast<int>(lpos_.z_reset_counter) << "," << lpos_.dead_reckoning << ","
             << (goal_sent_ ? std::hypot(goal_ned_[0] - lpos_.x, goal_ned_[1] - lpos_.y) : -1.0) << "\n";
        if (ticks_ % 10 == 0) csv_.flush();
    }

    // parameters
    int system_id_{1};
    std::string mission_, wp_csv_path_, goal_topic_, ready_topic_, log_dir_;
    double takeoff_alt_{3.0}, takeoff_alt_tol_{0.5}, stabilize_s_{2.0}, desired_speed_{0.0};
    double goal_north_{0.0}, goal_east_{0.0}, goal_lat_{0.0}, goal_lon_{0.0};
    double goal_handover_{3.0}, goal_settle_radius_{0.3}, goal_settle_speed_{0.3}, goal_settle_s_{1.0};
    double goal_timeout_s_{20.0}, goal_pf_done_max_{0.0};
    bool require_go_{true}, needs_gps_{false}, abort_on_dead_reckoning_{true};
    double ekf_settle_s_{30.0}, max_mission_radius_{300.0}, min_wp_alt_{1.0};
    double pf_timeout_s_{0.5}, pf_start_timeout_s_{20.0}, wp_timeout_s_{0.0};
    double fence_margin_{20.0}, fence_alt_margin_{10.0}, speed_limit_{0.0};
    double abort_xy_reset_{3.0}, abort_z_reset_{1.0}, impact_accel_{30.0}, impact_window_{0.2};
    int impact_samples_{2};

    // mission
    std::vector<Wp> wp_;
    std::vector<std::array<double, 3>> mission_ned_;
    std::array<double, 3> goal_ned_{0.0, 0.0, 0.0};
    std::array<double, 3> hover_{0.0, 0.0, 0.0};
    double fence_r_{0.0}, fence_h_{1e9};

    // state
    Phase phase_{Phase::INIT};
    long ticks_{0}, forwarded_{0};
    double t_phase_{0.0};
    bool armed_once_{false}, go_{false}, seen_takeoff_{false}, seen_land_{false}, use_attitude_{false};
    bool pf_done_{false}, wp_acked_{false}, goal_sent_{false}, disarm_sent_{false}, imu_seen_{false};
    bool px4_silent_warned_{false}, csv_failed_{false};
    int arm_sends_{0}, takeoff_sends_{0}, land_sends_{0}, wp_sends_{0}, settle_ticks_{0}, over_speed_ticks_{0};
    int at_height_ticks_{0};
    double ground_z_{0.0}, home_n_{0.0}, home_e_{0.0}, yaw0_{0.0}, goal_yaw_{0.0};
    double t_status_{-1e9}, t_lpos_{-1e9}, t_gpos_{-1e9}, t_att_{-1e9}, t_pf_att_{-1e9}, t_pf_hb_{-1e9};
    double t_ready_{-1e9}, t_wp_{-1e9}, t_wp_sent_{-1e9}, t_ekf_change_{0.0}, t_takeoff_sent_{0.0};
    double t_land_sent_{0.0}, t_follow_{0.0}, max_impact_{0.0};
    std::deque<double> impacts_;
    std::string reason_;
    std::ofstream csv_;

    VehicleStatus status_{};
    px4_msgs::msg::VehicleLocalPosition lpos_{};
    px4_msgs::msg::VehicleGlobalPosition gpos_{};
    px4_msgs::msg::VehicleAttitude att_{};
    px4_msgs::msg::VehicleAttitudeSetpoint pf_att_{};

    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr ocm_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr att_pub_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr cmd_pub_;
    rclcpp::Publisher<custom_msgs::msg::LocalWaypointSetpoint>::SharedPtr wp_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ctrl_hb_pub_, pp_hb_pub_, ca_hb_pub_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr lpos_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleGlobalPosition>::SharedPtr gpos_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr att_sub_;
    rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr imu_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr pf_att_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pf_done_sub_, pf_hb_sub_, ready_sub_, go_sub_, land_sub_;
    rclcpp::Subscription<custom_msgs::msg::ConveyLocalWaypointComplete>::SharedPtr wp_ack_sub_;
    rclcpp::Subscription<custom_msgs::msg::LocalWaypointSetpoint>::SharedPtr wp_watch_sub_;
    rclcpp::TimerBase::SharedPtr timer_, hb_timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<FlightDriver>());
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("flight_driver"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
