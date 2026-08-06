#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

#include <custom_msgs/msg/local_waypoint_setpoint.hpp>
#include <custom_msgs/msg/convey_local_waypoint_complete.hpp>

using namespace std::chrono_literals;

class PathFollowingTest : public rclcpp::Node
{
public:
    PathFollowingTest()
    : Node("path_following_test"),
      tick_count_(0), last_command_("idle"),
      test_phase_("WAIT"), ctrl_mode_str_("POSITION"),
      armed_and_offboard_(false), test_tick_(-1),
      land_sent_(false), land_tick_(-1), disarmed_(false),
      initial_yaw_(0.0f), home_x_(0.0f), home_y_(0.0f),
      target_z_ned_(-5.0f),
      use_attitude_(false),
      waypoints_sent_(false),
      pf_att_received_(false), pf_done_(false), wp_ack_received_(false),
      status_received_(false), local_pos_received_(false), attitude_received_(false),
      desired_speed_(3.0f)
    {
        this->declare_parameter<int>("system_id", 1);
        system_id_ = this->get_parameter("system_id").as_int();

        // 기본 wp.csv 는 패키지 share 디렉토리에 설치됨. wp_csv_path 파라미터로 덮어쓰기 가능.
        std::string default_wp_csv;
        try {
            default_wp_csv =
                ament_index_cpp::get_package_share_directory("path_following_test") + "/wp.csv";
        } catch (...) {
            default_wp_csv = "wp.csv";
        }
        this->declare_parameter<std::string>("wp_csv_path", default_wp_csv);
        wp_csv_path_ = this->get_parameter("wp_csv_path").as_string();
        load_waypoints_from_csv(wp_csv_path_);

        // 웨이포인트 주기적 재전송 (in-flight replan 스트레스 테스트).
        //   wp_republish_hz = 0  → 기존 동작 (PATH FOLLOWING 진입 시 1회만 전송)
        //   wp_republish_hz > 0  → 해당 주파수로 /local_waypoint_setpoint_to_PF 재전송
        //   wp_reload_csv = true → 재전송할 때마다 wp.csv 를 다시 읽음 (파일을 바꾸면 반영)
        this->declare_parameter<double>("wp_republish_hz", 0.0);
        wp_republish_hz_ = this->get_parameter("wp_republish_hz").as_double();
        this->declare_parameter<bool>("wp_reload_csv", false);
        wp_reload_csv_ = this->get_parameter("wp_reload_csv").as_bool();

        this->declare_parameter<double>("desired_speed", 3.0);
        desired_speed_ = static_cast<float>(this->get_parameter("desired_speed").as_double());

        // CSV 로그 — 볼륨 마운트된 경로 (호스트에서도 접근 가능)
        // log_dir 파라미터로 덮어쓸 수 있음 (기본: 워크스페이스 내 path_following_test/logs)
        this->declare_parameter<std::string>(
            "log_dir",
            "/home/user/realgazebo/RealGazebo-ROS2/src/path_following_test/logs");
        std::string log_dir = this->get_parameter("log_dir").as_string();
        (void)std::system(("mkdir -p " + log_dir).c_str());
        std::string log_path = log_dir + "/path_following_log_v" +
                               std::to_string(static_cast<int>(desired_speed_ * 10)) + ".csv";
        csv_file_.open(log_path);
        if (csv_file_.is_open()) {
            csv_file_ << "time_s,"
                      << "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,speed,"
                      << "act_roll_deg,act_pitch_deg,act_yaw_deg,"
                      << "cmd_qw,cmd_qx,cmd_qy,cmd_qz,cmd_thrust,"
                      << "act_qw,act_qx,act_qy,act_qz,"
                      << "desired_speed,phase" << std::endl;
            RCLCPP_INFO(this->get_logger(), "Logging to: %s", log_path.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "FAILED to open: %s", log_path.c_str());
        }

        std::string prefix = "vehicle" + std::to_string(system_id_) + "/fmu/";
        RCLCPP_INFO(this->get_logger(),
            "Configure path_following_test (system_id: %d, desired_speed: %.1f)",
            system_id_, desired_speed_);

        auto qos_sensor = rclcpp::SensorDataQoS();
        rclcpp::QoS qos_pub(10);

        // ===== PX4 Publishers =====
        ocm_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
            prefix + "in/offboard_control_mode", qos_pub);
        traj_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
            prefix + "in/trajectory_setpoint", qos_pub);
        att_pub_ = this->create_publisher<px4_msgs::msg::VehicleAttitudeSetpoint>(
            prefix + "in/vehicle_attitude_setpoint", qos_pub);
        cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
            prefix + "in/vehicle_command", qos_pub);

        // ===== Module Publishers =====
        wp_pub_ = this->create_publisher<custom_msgs::msg::LocalWaypointSetpoint>(
            "/local_waypoint_setpoint_to_PF", qos_pub);
        ctrl_hb_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/controller_heartbeat", qos_pub);
        pp_hb_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/path_planning_heartbeat", qos_pub);
        ca_hb_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/collision_avoidance_heartbeat", qos_pub);

        // ===== PX4 Subscribers =====
        status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
            prefix + "out/vehicle_status_v1", qos_sensor,
            [this](const px4_msgs::msg::VehicleStatus::SharedPtr msg) {
                if (!status_received_) {
                    RCLCPP_INFO(this->get_logger(), "First VehicleStatus received!");
                    status_received_ = true;
                }
                vehicle_status_ = *msg;
            });

        local_pos_sub_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            prefix + "out/vehicle_local_position", qos_sensor,
            [this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) {
                if (!local_pos_received_) {
                    RCLCPP_INFO(this->get_logger(), "First VehicleLocalPosition received!");
                    local_pos_received_ = true;
                }
                local_pos_ = *msg;
            });

        attitude_sub_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
            prefix + "out/vehicle_attitude", qos_sensor,
            [this](const px4_msgs::msg::VehicleAttitude::SharedPtr msg) {
                if (!attitude_received_) {
                    RCLCPP_INFO(this->get_logger(), "First VehicleAttitude received!");
                    attitude_received_ = true;
                }
                vehicle_att_ = *msg;
            });

        // ===== Path Following Subscribers =====
        pf_att_sub_ = this->create_subscription<px4_msgs::msg::VehicleAttitudeSetpoint>(
            "/pf_att_2_control", qos_pub,
            [this](const px4_msgs::msg::VehicleAttitudeSetpoint::SharedPtr msg) {
                pf_att_cmd_ = *msg;
                pf_att_received_ = true;
                // PF 명령이 오는 즉시 PX4로 포워딩 (250Hz) — 10Hz 타이머 기다리지 않음
                if (use_attitude_) {
                    forward_pf_attitude();
                }
            });

        pf_complete_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/path_following_complete", qos_pub,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                if (msg->data && !pf_done_) {
                    RCLCPP_INFO(this->get_logger(), "Path following COMPLETE! Starting landing.");
                    pf_done_ = true;
                }
            });

        wp_ack_sub_ = this->create_subscription<custom_msgs::msg::ConveyLocalWaypointComplete>(
            "/convey_local_waypoint_complete", qos_pub,
            [this](const custom_msgs::msg::ConveyLocalWaypointComplete::SharedPtr msg) {
                if (msg->convey_local_waypoint_is_complete && !wp_ack_received_) {
                    RCLCPP_INFO(this->get_logger(), "Waypoint ack received from PF node.");
                    wp_ack_received_ = true;
                }
            });

        // ===== Timers =====
        timer_ = this->create_wall_timer(
            100ms, std::bind(&PathFollowingTest::timer_callback, this));
        heartbeat_timer_ = this->create_wall_timer(
            1000ms, std::bind(&PathFollowingTest::heartbeat_callback, this));

        // 웨이포인트 재전송 타이머 (wp_republish_hz > 0 일 때만).
        if (wp_republish_hz_ > 0.0) {
            auto period = std::chrono::duration<double>(1.0 / wp_republish_hz_);
            wp_timer_ = this->create_wall_timer(
                std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                std::bind(&PathFollowingTest::wp_republish_callback, this));
            RCLCPP_INFO(this->get_logger(),
                "Waypoint republish ENABLED at %.1f Hz (reload_csv=%s)",
                wp_republish_hz_, wp_reload_csv_ ? "true" : "false");
        }
    }

    ~PathFollowingTest()
    {
        if (csv_file_.is_open()) {
            csv_file_.close();
            RCLCPP_INFO(this->get_logger(), "CSV log closed.");
        }
    }

private:
    // 재호출(in-flight reload) 안전: 임시 벡터에 파싱한 뒤 유효할 때만 교체.
    // 파싱 실패/파일 없음 시 기존 wp 를 그대로 유지한다.
    void load_waypoints_from_csv(const std::string & path)
    {
        std::ifstream file(path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open wp.csv: %s", path.c_str());
            return;
        }

        std::vector<double> nx, ny, nz;
        std::string line;
        std::getline(file, line);  // 헤더 스킵

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string token;
            std::vector<double> vals;
            while (std::getline(ss, token, ',')) {
                try { vals.push_back(std::stod(token)); } catch (...) {}
            }
            if (vals.size() < 3) continue;
            nx.push_back(vals[0]);
            ny.push_back(vals[1]);
            nz.push_back(vals[2]);
        }

        if (nx.size() < 2) {
            RCLCPP_ERROR(this->get_logger(),
                "wp.csv must have >= 2 waypoints. Got %zu. Keeping previous wp.", nx.size());
            return;
        }

        const bool first_load = wp_x_.empty();
        wp_x_ = std::move(nx);
        wp_y_ = std::move(ny);
        wp_z_ = std::move(nz);
        target_z_ned_ = -static_cast<float>(wp_z_[0]);

        if (first_load) {
            RCLCPP_INFO(this->get_logger(),
                "Loaded %zu waypoints from %s. Target altitude: %.1f m (NED z: %.1f)",
                wp_x_.size(), path.c_str(), wp_z_[0], target_z_ned_);
        }
    }

    void timer_callback()
    {
        run_flight_sequence();
        publish_offboard_heartbeat();
        print_status();
        ++tick_count_;
    }

    void heartbeat_callback()
    {
        std_msgs::msg::Bool hb;
        hb.data = true;
        ctrl_hb_pub_->publish(hb);
        pp_hb_pub_->publish(hb);
        ca_hb_pub_->publish(hb);
    }

    void publish_offboard_heartbeat()
    {
        px4_msgs::msg::OffboardControlMode msg{};
        msg.timestamp       = timestamp_us();
        msg.position        = !use_attitude_;
        msg.velocity        = false;
        msg.acceleration    = false;
        msg.attitude        = use_attitude_;
        msg.body_rate       = false;
        msg.direct_actuator = false;
        ocm_pub_->publish(msg);
    }

    void run_flight_sequence()
    {
        bool is_armed = (vehicle_status_.arming_state ==
                         px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED);

        // Phase 0: PX4 파라미터 설정 (tick 0~9)
        if (tick_count_ < 10) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            initial_yaw_ = local_pos_.heading;
            test_phase_ = "SET PX4 PARAMS";
            send_trajectory_setpoint(local_pos_.x, local_pos_.y, local_pos_.z, initial_yaw_);
            if (tick_count_ == 1 || tick_count_ == 5) {
                set_px4_param("NAV_DLL_ACT",    0.0f);
                set_px4_param("COM_RCL_EXCEPT", 31.0f);
                set_px4_param("COM_RC_IN_MODE", 4.0f);
                RCLCPP_INFO(this->get_logger(), "Setting PX4 params");
            }
            return;
        }

        // Phase 1: Setpoint 스트리밍 (tick 10~19)
        if (tick_count_ < 20) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            initial_yaw_ = local_pos_.heading;
            test_phase_ = "STREAMING setpoints";
            send_trajectory_setpoint(local_pos_.x, local_pos_.y, local_pos_.z, initial_yaw_);
            return;
        }

        // Phase 2: OFFBOARD + ARM
        if (!armed_and_offboard_) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            send_trajectory_setpoint(local_pos_.x, local_pos_.y, local_pos_.z, initial_yaw_);
            if (tick_count_ % 10 == 0) {
                send_offboard();
                send_arm();
            }
            test_phase_ = "OFFBOARD+ARM...";
            if (is_armed) {
                armed_and_offboard_ = true;
                test_tick_ = tick_count_;
                home_x_ = local_pos_.x;
                home_y_ = local_pos_.y;
                RCLCPP_INFO(this->get_logger(),
                    "Armed! Home=(%.2f, %.2f), Target NED Z=%.1f, Yaw=%.1f deg",
                    home_x_, home_y_, target_z_ned_, initial_yaw_ * 180.0f / M_PI);
            }
            return;
        }

        int t = tick_count_ - test_tick_;

        // Phase 3: 이륙 — Position 제어 (7초)
        if (t < 70) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            test_phase_ = "CLIMB";
            send_trajectory_setpoint(home_x_, home_y_, target_z_ned_, initial_yaw_);
            return;
        }

        // Phase 4: 안정화 — Position 제어 (3초)
        if (t < 100) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            test_phase_ = "HOVER (stabilize)";
            send_trajectory_setpoint(home_x_, home_y_, target_z_ned_, initial_yaw_);
            return;
        }

        // Phase 5: waypoint 최초 전송. (재전송은 wp_republish_hz>0 시 wp_timer_ 가 담당)
        if (!waypoints_sent_) {
            send_waypoints();
            waypoints_sent_ = true;
            following_active_ = true;   // 재전송 타이머가 전송을 시작해도 되는 시점
            RCLCPP_INFO(this->get_logger(),
                "Waypoints sent. %zu points. Home=(%.2f, %.2f) NED Z=%.1f",
                wp_x_.size(), home_x_, home_y_, target_z_ned_);
        }

        // Phase 6: Path Following — PF cmd 포워딩
        if (!pf_done_) {
            test_phase_ = "PATH FOLLOWING";
            if (pf_att_received_) {
                use_attitude_ = true;
                ctrl_mode_str_ = "ATTITUDE (PF)";
                forward_pf_attitude();
            } else {
                use_attitude_ = false;
                ctrl_mode_str_ = "POSITION (wait PF)";
                test_phase_ = "WAIT PF CMD";
                send_trajectory_setpoint(home_x_, home_y_, target_z_ned_, initial_yaw_);
            }
            return;
        }

        // Phase 7: 착륙
        if (!land_sent_) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            send_land();
            land_sent_ = true;
            land_tick_ = tick_count_;
            test_phase_ = "LANDING";
            return;
        }

        // Phase 8: DISARM (착륙 명령 후 10초)
        if (!disarmed_ && land_tick_ >= 0 && (tick_count_ - land_tick_) >= 100) {
            send_disarm();
            disarmed_ = true;
            test_phase_ = "DISARMED";
            RCLCPP_INFO(this->get_logger(), "Disarmed. Shutting down.");
            rclcpp::shutdown();
        }
    }

    // wp_republish_hz 주기로 호출. 경로추종이 시작된 뒤(following_active_)부터,
    // 착륙(pf_done_) 전까지 웨이포인트를 계속 재전송한다. PathFollowing core 는
    // 비행 중 재수신 시 relocalize(부드러운 replan)로 처리한다.
    void wp_republish_callback()
    {
        if (!following_active_ || pf_done_ || land_sent_) {
            return;
        }
        if (wp_reload_csv_) {
            // 파일을 바꿔도 반영되도록 재읽기 (파싱 실패 시 기존 wp 유지).
            load_waypoints_from_csv(wp_csv_path_);
        }
        send_waypoints();
        ++wp_republish_count_;
    }

    void send_waypoints()
    {
        if (wp_x_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No waypoints loaded. Cannot send.");
            return;
        }

        custom_msgs::msg::LocalWaypointSetpoint msg;
        msg.path_planning_complete = true;
        msg.waypoint_x = wp_x_;
        msg.waypoint_y = wp_y_;
        msg.waypoint_z = wp_z_;
        wp_pub_->publish(msg);
    }

    void forward_pf_attitude()
    {
        px4_msgs::msg::VehicleAttitudeSetpoint msg = pf_att_cmd_;
        msg.timestamp = timestamp_us();
        att_pub_->publish(msg);
    }

    void send_trajectory_setpoint(float x, float y, float z, float yaw)
    {
        px4_msgs::msg::TrajectorySetpoint msg{};
        msg.timestamp    = timestamp_us();
        msg.position     = {x, y, z};
        msg.velocity     = {NAN, NAN, NAN};
        msg.acceleration = {NAN, NAN, NAN};
        msg.jerk         = {NAN, NAN, NAN};
        msg.yaw          = yaw;
        msg.yawspeed     = NAN;
        traj_pub_->publish(msg);
    }

    void set_px4_param(const char * param_name, float value)
    {
        char cmd_str[512];
        const char * pfx = (getuid() == 0) ? "sudo -u user " : "";
        std::snprintf(cmd_str, sizeof(cmd_str),
            "%s/home/user/realgazebo/RealGazebo-PX4/build/px4_sitl_default/bin/px4-param "
            "--instance 0 set %s %d 2>/dev/null &",
            pfx, param_name, static_cast<int>(value));
        (void)std::system(cmd_str);
    }

    void send_arm()
    {
        last_command_ = "ARM";
        px4_msgs::msg::VehicleCommand cmd{};
        cmd.timestamp        = timestamp_us();
        cmd.target_system    = static_cast<uint8_t>(system_id_);
        cmd.target_component = 1;
        cmd.source_system    = 1;
        cmd.source_component = 1;
        cmd.command          = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM;
        cmd.param1           = 1.0f;
        cmd.confirmation     = 1;
        cmd.from_external    = true;
        cmd_pub_->publish(cmd);
    }

    void send_offboard()
    {
        last_command_ = "OFFBOARD";
        px4_msgs::msg::VehicleCommand cmd{};
        cmd.timestamp        = timestamp_us();
        cmd.target_system    = static_cast<uint8_t>(system_id_);
        cmd.target_component = 1;
        cmd.source_system    = 1;
        cmd.source_component = 1;
        cmd.command          = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE;
        cmd.param1           = 1.0f;
        cmd.param2           = 6.0f;
        cmd.from_external    = true;
        cmd_pub_->publish(cmd);
    }

    void send_land()
    {
        last_command_ = "LAND";
        px4_msgs::msg::VehicleCommand cmd{};
        cmd.timestamp        = timestamp_us();
        cmd.target_system    = static_cast<uint8_t>(system_id_);
        cmd.target_component = 1;
        cmd.source_system    = 1;
        cmd.source_component = 1;
        cmd.command          = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND;
        cmd.from_external    = true;
        cmd_pub_->publish(cmd);
    }

    void send_disarm()
    {
        last_command_ = "DISARM";
        px4_msgs::msg::VehicleCommand cmd{};
        cmd.timestamp        = timestamp_us();
        cmd.target_system    = static_cast<uint8_t>(system_id_);
        cmd.target_component = 1;
        cmd.source_system    = 1;
        cmd.source_component = 1;
        cmd.command          = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM;
        cmd.param1           = 0.0f;
        cmd.confirmation     = 1;
        cmd.from_external    = true;
        cmd_pub_->publish(cmd);
    }

    void print_status()
    {
        std::printf("\033[2J\033[H");

        const char * arm_str =
            (vehicle_status_.arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED)
            ? "ARM" : "DISARM";

        float roll_deg = std::atan2(
            2.0f * (vehicle_att_.q[0] * vehicle_att_.q[1] +
                    vehicle_att_.q[2] * vehicle_att_.q[3]),
            1.0f - 2.0f * (vehicle_att_.q[1] * vehicle_att_.q[1] +
                           vehicle_att_.q[2] * vehicle_att_.q[2])
        ) * 180.0f / M_PI;

        float pitch_deg = std::asin(
            std::clamp(
                2.0f * (vehicle_att_.q[0] * vehicle_att_.q[2] -
                        vehicle_att_.q[3] * vehicle_att_.q[1]),
                -1.0f, 1.0f)
        ) * 180.0f / M_PI;

        float yaw_deg = local_pos_.heading * 180.0f / M_PI;

        RCLCPP_INFO(this->get_logger(),
            "PX4 link: status=%s  local_pos=%s  att=%s",
            status_received_ ? "OK" : "NO DATA",
            local_pos_received_ ? "OK" : "NO DATA",
            attitude_received_ ? "OK" : "NO DATA");

        RCLCPP_INFO(this->get_logger(),
            "PF link:  att_cmd=%s  done=%s  wp_ack=%s  waypoints_sent=%s  wp_republish=%ld@%.0fHz",
            pf_att_received_ ? "OK" : "WAIT",
            pf_done_         ? "YES" : "NO",
            wp_ack_received_ ? "YES" : "NO",
            waypoints_sent_  ? "YES" : "NO",
            wp_republish_count_, wp_republish_hz_);

        RCLCPP_INFO(this->get_logger(),
            "nav_state: %u (%s) | tick: %d | mode: %s",
            vehicle_status_.nav_state, arm_str, tick_count_, ctrl_mode_str_.c_str());

        RCLCPP_INFO(this->get_logger(),
            "NED pos: (%.2f, %.2f, %.2f)  vel: (%.2f, %.2f, %.2f)",
            local_pos_.x, local_pos_.y, local_pos_.z,
            local_pos_.vx, local_pos_.vy, local_pos_.vz);

        RCLCPP_INFO(this->get_logger(),
            "altitude: %.2f m (target %.1f m) | RPY: %.1f  %.1f  %.1f deg | wp_count: %zu",
            -local_pos_.z, -target_z_ned_, roll_deg, pitch_deg, yaw_deg, wp_x_.size());

        if (pf_att_received_) {
            RCLCPP_INFO(this->get_logger(),
                "PF cmd: q=[%.3f %.3f %.3f %.3f]  thrust=%.3f",
                pf_att_cmd_.q_d[0], pf_att_cmd_.q_d[1],
                pf_att_cmd_.q_d[2], pf_att_cmd_.q_d[3],
                pf_att_cmd_.thrust_body[2]);
        }

        RCLCPP_INFO(this->get_logger(),
            "phase: %s | last_cmd: %s", test_phase_.c_str(), last_command_.c_str());

        // CSV 로그 (ARM 이후)
        if (armed_and_offboard_ && csv_file_.is_open()) {
            float time_s = (tick_count_ - test_tick_) * 0.1f;
            float speed = std::sqrt(local_pos_.vx * local_pos_.vx +
                                    local_pos_.vy * local_pos_.vy +
                                    local_pos_.vz * local_pos_.vz);
            csv_file_
                << time_s << ","
                << local_pos_.x << "," << local_pos_.y << "," << local_pos_.z << ","
                << local_pos_.vx << "," << local_pos_.vy << "," << local_pos_.vz << ","
                << speed << ","
                << roll_deg << "," << pitch_deg << "," << yaw_deg << ","
                << pf_att_cmd_.q_d[0] << "," << pf_att_cmd_.q_d[1] << ","
                << pf_att_cmd_.q_d[2] << "," << pf_att_cmd_.q_d[3] << ","
                << pf_att_cmd_.thrust_body[2] << ","
                << vehicle_att_.q[0] << "," << vehicle_att_.q[1] << ","
                << vehicle_att_.q[2] << "," << vehicle_att_.q[3] << ","
                << desired_speed_ << "," << test_phase_ << std::endl;
        }
    }

    uint64_t timestamp_us()
    {
        return this->get_clock()->now().nanoseconds() / 1000;
    }

    // ===== 멤버 변수 =====

    int system_id_;
    int tick_count_;
    std::string last_command_;
    std::string test_phase_;
    std::string ctrl_mode_str_;

    bool armed_and_offboard_;
    int  test_tick_;
    bool land_sent_;
    int  land_tick_;
    bool disarmed_;

    float initial_yaw_;
    float home_x_, home_y_;
    float target_z_ned_;
    float desired_speed_;
    bool  use_attitude_;

    std::vector<double> wp_x_, wp_y_, wp_z_;
    std::string wp_csv_path_;

    // 웨이포인트 재전송 (in-flight replan)
    double wp_republish_hz_{0.0};
    bool   wp_reload_csv_{false};
    bool   following_active_{false};
    long   wp_republish_count_{0};

    bool waypoints_sent_;
    bool pf_att_received_;
    bool pf_done_;
    bool wp_ack_received_;

    bool status_received_;
    bool local_pos_received_;
    bool attitude_received_;

    std::ofstream csv_file_;

    px4_msgs::msg::VehicleStatus        vehicle_status_{};
    px4_msgs::msg::VehicleLocalPosition local_pos_{};
    px4_msgs::msg::VehicleAttitude      vehicle_att_{};
    px4_msgs::msg::VehicleAttitudeSetpoint pf_att_cmd_{};

    // ===== Publishers =====
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr     ocm_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr      traj_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr att_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr          cmd_pub_;
    rclcpp::Publisher<custom_msgs::msg::LocalWaypointSetpoint>::SharedPtr wp_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ctrl_hb_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pp_hb_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ca_hb_pub_;

    // ===== Subscribers =====
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr        status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr      attitude_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr        pf_att_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr                           pf_complete_sub_;
    rclcpp::Subscription<custom_msgs::msg::ConveyLocalWaypointComplete>::SharedPtr wp_ack_sub_;

    // ===== Timers =====
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;
    rclcpp::TimerBase::SharedPtr wp_timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathFollowingTest>());
    rclcpp::shutdown();
    return 0;
}
