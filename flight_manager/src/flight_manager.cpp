// RealGazebo 비행 매니저 (fast_planner 통합용).
//
// path_following_test 에서 "웨이포인트 발행" 책임만 제거한 버전이다.
// arm/offboard/이륙/호버/PF attitude 포워딩/착륙/disarm 의 비행 FSM 은 동일하게 유지한다.
//
// 책임 분리:
//   - 웨이포인트(/local_waypoint_setpoint_to_PF)  : fast_planner + fastplanner_pf_bridge 가 공급
//   - path_planning / collision_avoidance heartbeat : bridge 가 10Hz 로 발행
//   - controller_heartbeat                          : 본 노드가 발행 (PF 가 게이트하므로 필수)
//   - arm/이륙/호버/PF→PX4 포워딩/착륙               : 본 노드 (검증된 path_following_test 로직 재사용)
//
// 흐름: PARAMS → STREAM → OFFBOARD+ARM → CLIMB → HOVER → (PF cmd 대기) →
//       PATH FOLLOWING(PF attitude 포워딩) → LAND → DISARM.
// 기존과의 차이: Phase 5 가 "wp 1회 전송" 이 아니라 "PF 가 attitude 를 내기 시작할 때까지
// 호버하며 대기" 다. wp 는 bridge 가 외부에서 비동기로 넣어준다.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

using namespace std::chrono_literals;

class FlightManager : public rclcpp::Node
{
public:
    FlightManager()
    : Node("flight_manager"),
      tick_count_(0), last_command_("idle"),
      test_phase_("WAIT"), ctrl_mode_str_("POSITION"),
      armed_and_offboard_(false), test_tick_(-1),
      land_sent_(false), land_tick_(-1), disarmed_(false),
      initial_yaw_(0.0f), home_x_(0.0f), home_y_(0.0f),
      target_z_ned_(-5.0f),
      use_attitude_(false),
      pf_att_received_(false), pf_done_(false),
      status_received_(false), local_pos_received_(false), attitude_received_(false),
      desired_speed_(3.0f)
    {
        this->declare_parameter<int>("system_id", 1);
        system_id_ = this->get_parameter("system_id").as_int();

        // 이륙 목표 고도 (양수 m, 내부에서 NED-down 으로 부호 반전).
        this->declare_parameter<double>("takeoff_altitude", 5.0);
        target_z_ned_ = -static_cast<float>(this->get_parameter("takeoff_altitude").as_double());

        this->declare_parameter<double>("desired_speed", 3.0);
        desired_speed_ = static_cast<float>(this->get_parameter("desired_speed").as_double());

        std::string prefix = "vehicle" + std::to_string(system_id_) + "/fmu/";
        RCLCPP_INFO(this->get_logger(),
            "Configure flight_manager (system_id: %d, takeoff_alt: %.1f m)",
            system_id_, -target_z_ned_);

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
        // controller_heartbeat 만 발행한다. path_planning / collision_avoidance heartbeat 는
        // fastplanner_pf_bridge 가 담당하므로 여기서 발행하면 충돌(중복 게시자)이 된다.
        ctrl_hb_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/controller_heartbeat", qos_pub);

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

        // ===== Timers =====
        timer_ = this->create_wall_timer(
            100ms, std::bind(&FlightManager::timer_callback, this));
        heartbeat_timer_ = this->create_wall_timer(
            1000ms, std::bind(&FlightManager::heartbeat_callback, this));
    }

private:
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

        // Phase 5/6: Path Following.
        // wp 는 fast_planner+bridge 가 /local_waypoint_setpoint_to_PF 로 비동기 공급한다.
        // PF 가 경로를 받아 attitude 명령(/pf_att_2_control)을 내기 시작하면 포워딩으로
        // 전환하고, 그 전까지는 호버하며 PF cmd 를 대기한다.
        if (!pf_done_) {
            if (pf_att_received_) {
                use_attitude_ = true;
                ctrl_mode_str_ = "ATTITUDE (PF)";
                test_phase_ = "PATH FOLLOWING";
                forward_pf_attitude();
            } else {
                use_attitude_ = false;
                ctrl_mode_str_ = "POSITION (wait PF)";
                test_phase_ = "WAIT PF CMD (wp from bridge)";
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

        float yaw_deg = local_pos_.heading * 180.0f / M_PI;

        RCLCPP_INFO(this->get_logger(),
            "PX4 link: status=%s  local_pos=%s  att=%s",
            status_received_ ? "OK" : "NO DATA",
            local_pos_received_ ? "OK" : "NO DATA",
            attitude_received_ ? "OK" : "NO DATA");

        RCLCPP_INFO(this->get_logger(),
            "PF link:  att_cmd=%s  done=%s",
            pf_att_received_ ? "OK" : "WAIT",
            pf_done_ ? "YES" : "no");

        RCLCPP_INFO(this->get_logger(),
            "PHASE: %-28s  MODE: %-18s  ARM: %s",
            test_phase_.c_str(), ctrl_mode_str_.c_str(), arm_str);

        RCLCPP_INFO(this->get_logger(),
            "POS(NED): x=%.2f y=%.2f z=%.2f  yaw=%.1f deg  last_cmd=%s",
            local_pos_.x, local_pos_.y, local_pos_.z, yaw_deg, last_command_.c_str());
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

    bool pf_att_received_;
    bool pf_done_;

    bool status_received_;
    bool local_pos_received_;
    bool attitude_received_;

    px4_msgs::msg::VehicleStatus        vehicle_status_{};
    px4_msgs::msg::VehicleLocalPosition local_pos_{};
    px4_msgs::msg::VehicleAttitude      vehicle_att_{};
    px4_msgs::msg::VehicleAttitudeSetpoint pf_att_cmd_{};

    // ===== Publishers =====
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr     ocm_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr      traj_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr att_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr          cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ctrl_hb_pub_;

    // ===== Subscribers =====
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr        status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr      attitude_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr pf_att_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr                   pf_complete_sub_;

    // ===== Timers =====
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FlightManager>());
    rclcpp::shutdown();
    return 0;
}
