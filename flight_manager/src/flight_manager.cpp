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
//       PATH FOLLOWING(PF attitude 포워딩) → [GOAL APPROACH(PX4 위치제어, goal_enu 가 주어질 때)]
//       → LAND → DISARM(PX4 착지 후 자동 disarm 대기).
// 기존과의 차이: Phase 5 가 "wp 1회 전송" 이 아니라 "PF 가 attitude 를 내기 시작할 때까지
// 호버하며 대기" 다. wp 는 bridge 가 외부에서 비동기로 넣어준다.

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <string>
#include <unistd.h>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/sensor_combined.hpp>
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

        // 최종 목표 근처는 PF 대신 PX4 위치제어(offboard position)로 목표점에 정지시킨 뒤 착륙.
        // goal_enu = [x, y, z] (PX4 원점 기준 ENU, 앱의 goal CSV 마지막 줄). 비어 있으면 기존 동작
        // (PF done → 그 자리 착륙). goal_handover_radius_m > 0 이면 수평거리가 그 안에 들면 넘기고,
        // 0 이면 PF done 때 넘긴다. 목표 settle 반경·속도 안에 settle 시간 동안 있으면 LAND.
        this->declare_parameter<std::vector<double>>("goal_enu", std::vector<double>{});
        this->declare_parameter<double>("goal_handover_radius_m", 0.0);
        this->declare_parameter<double>("goal_settle_radius_m", 0.3);
        this->declare_parameter<double>("goal_settle_speed", 0.3);
        this->declare_parameter<double>("goal_settle_time_s", 1.0);
        this->declare_parameter<double>("goal_timeout_s", 20.0);
        // PF done 은 목표에서 이 수평거리 안일 때만 도착으로 본다. 더 멀면 그 자리 착륙.
        // 플래너가 막혀 PF 가 경로 끝에서 멈춘 것이라 PX4 위치제어로 목표까지 직진하면 장애물을
        // 무시하고 최대속도로 날아간다(ctrack_123_d: 42 m 앞에서 PF done → 11.5 m/s 로 풍선 충돌·전복).
        this->declare_parameter<double>("goal_pf_done_max_m", 6.0);
        // 비행 중 PX4 위치 추정이 이 거리 이상 리셋되면(EKF 가 GPS 로 재정렬) 그 자리 착륙. 리셋 전
        // 추정으로 쌓은 플래너 맵·경로는 쓸 수 없다(ctrack_123_d: 충돌 스핀 뒤 65 m 리셋). 0 = 끔.
        this->declare_parameter<double>("abort_xy_reset_m", 3.0);
        // 충돌 감지: 착륙 명령 전 비행 중 IMU 가속도 크기가 impact_accel_m_s2 를 impact_window_s 안에
        // impact_samples 번 넘으면 그 자리 착륙. 충돌 후 PX4 추정이 몇 초 뒤부터 어긋나는데
        // dead_reckoning 표시는 6 s 뒤에야 떠서 그사이 80 m 를 날아갔다(ctrack_123_h). 정상 비행
        // 최대 11-13 m/s^2(ctrack_123_e/f/g, sp_c/d 착륙 명령 전), 충돌 130-260 m/s^2 가 160-220 ms.
        // 수신 주기와 무관하게 창 안 표본 수로 센다. 0 = 끔.
        this->declare_parameter<double>("impact_accel_m_s2", 30.0);
        this->declare_parameter<int>("impact_samples", 2);
        this->declare_parameter<double>("impact_window_s", 0.2);
        // PX4 가 GNSS 없이 추정(dead reckoning)하기 시작하면 그 자리 착륙 (충돌 감지가 놓쳤을 때의 백업)
        this->declare_parameter<bool>("abort_on_dead_reckoning", true);
        goal_pf_done_max_ = this->get_parameter("goal_pf_done_max_m").as_double();
        abort_xy_reset_ = this->get_parameter("abort_xy_reset_m").as_double();
        impact_accel_ = this->get_parameter("impact_accel_m_s2").as_double();
        impact_samples_ = static_cast<int>(this->get_parameter("impact_samples").as_int());
        impact_window_ = this->get_parameter("impact_window_s").as_double();
        abort_on_dead_reckoning_ = this->get_parameter("abort_on_dead_reckoning").as_bool();
        const auto goal = this->get_parameter("goal_enu").as_double_array();
        goal_known_ = goal.size() == 3;
        if (goal_known_) {
            goal_n_ = static_cast<float>(goal[1]);    // ENU -> NED
            goal_e_ = static_cast<float>(goal[0]);
            goal_d_ = -static_cast<float>(goal[2]);
        }
        goal_handover_radius_ = this->get_parameter("goal_handover_radius_m").as_double();
        goal_settle_radius_ = this->get_parameter("goal_settle_radius_m").as_double();
        goal_settle_speed_ = this->get_parameter("goal_settle_speed").as_double();
        goal_settle_ticks_ = static_cast<int>(std::lround(this->get_parameter("goal_settle_time_s").as_double() * 10.0));
        goal_timeout_ticks_ = static_cast<int>(std::lround(this->get_parameter("goal_timeout_s").as_double() * 10.0));
        if (goal_known_) {
            RCLCPP_INFO(this->get_logger(),
                "Goal NED (%.2f, %.2f, %.2f): PX4 position control within %.1f m (0 = on PF done), "
                "land when within %.2f m below %.2f m/s", goal_n_, goal_e_, goal_d_,
                goal_handover_radius_, goal_settle_radius_, goal_settle_speed_);
        }

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
                if (armed_and_offboard_ && !land_sent_ && !abort_ && abort_xy_reset_ > 0.0 &&
                    msg->xy_reset_counter != local_pos_.xy_reset_counter) {
                    const double jump = std::hypot(msg->delta_xy[0], msg->delta_xy[1]);
                    if (jump > abort_xy_reset_) {
                        RCLCPP_WARN(this->get_logger(),
                            "Position estimate reset by %.1f m in flight - landing in place.", jump);
                        abort_ = true;
                    }
                }
                if (armed_and_offboard_ && !land_sent_ && !abort_ && abort_on_dead_reckoning_ &&
                    msg->dead_reckoning) {
                    RCLCPP_WARN(this->get_logger(),
                        "Position estimate dead-reckoning (GNSS no longer fused) - landing in place.");
                    abort_ = true;
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

        imu_sub_ = this->create_subscription<px4_msgs::msg::SensorCombined>(
            prefix + "out/sensor_combined", qos_sensor,
            [this](const px4_msgs::msg::SensorCombined::SharedPtr msg) {
                const double t = this->now().seconds();
                if (imu_first_t_ < 0.0) {
                    imu_first_t_ = t;
                    RCLCPP_INFO(this->get_logger(), "First SensorCombined received!");
                }
                if (++imu_count_ == 200) {
                    RCLCPP_INFO(this->get_logger(), "SensorCombined at %.0f Hz (impact check %s)",
                        199.0 / std::max(t - imu_first_t_, 1e-3), impact_accel_ > 0.0 ? "on" : "off");
                }
                if (impact_accel_ <= 0.0 || !armed_and_offboard_ || land_sent_ || abort_) {
                    return;
                }
                const double a = std::sqrt(msg->accelerometer_m_s2[0] * msg->accelerometer_m_s2[0] +
                                           msg->accelerometer_m_s2[1] * msg->accelerometer_m_s2[1] +
                                           msg->accelerometer_m_s2[2] * msg->accelerometer_m_s2[2]);
                if (a <= impact_accel_) {
                    return;
                }
                impact_times_.push_back(t);
                while (!impact_times_.empty() && t - impact_times_.front() > impact_window_) {
                    impact_times_.pop_front();
                }
                if (static_cast<int>(impact_times_.size()) >= impact_samples_) {
                    RCLCPP_WARN(this->get_logger(),
                        "Impact detected (|a| %.0f m/s^2, %zu samples > %.0f in %.2f s) - landing in place.",
                        a, impact_times_.size(), impact_accel_, impact_window_);
                    abort_ = true;
                }
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

        // 충돌·위치 추정 이상(abort_) → 어느 비행 단계든 그 자리 착륙 (Phase 8 이 disarm 까지 처리)
        if (abort_ && !land_sent_) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION";
            send_land();
            land_sent_ = true;
            land_tick_ = tick_count_;
            test_phase_ = "LANDING (abort)";
            return;
        }

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

        // Phase 6b: 최종 목표 접근 — PX4 위치제어로 목표점까지 가서 멈춘 뒤 착륙.
        // PF 가 목표 근처에서 멈추고 done 을 내는 데 의존하면, 거기서 착륙이 엇나갔다.
        if (goal_known_ && !goal_phase_ && !land_sent_ && pf_att_received_) {
            const float dist = std::hypot(goal_n_ - local_pos_.x, goal_e_ - local_pos_.y);
            if (goal_handover_radius_ > 0.0 && dist < goal_handover_radius_) {
                start_goal_phase(dist, "within the handover radius");
            } else if (pf_done_ && dist <= goal_pf_done_max_) {
                start_goal_phase(dist, "PF done");
            } else if (pf_done_) {
                // 아래 Phase 7 이 그 자리에 착륙시킨다
                RCLCPP_WARN(this->get_logger(),
                    "PF done %.1f m from the goal (> %.1f m) - landing in place, not flying to the goal.",
                    dist, goal_pf_done_max_);
                goal_known_ = false;
            }
        }
        if (goal_phase_ && !land_sent_) {
            use_attitude_ = false;
            ctrl_mode_str_ = "POSITION (goal, PX4)";
            test_phase_ = "GOAL APPROACH (PX4)";
            send_trajectory_setpoint(goal_n_, goal_e_, goal_d_, goal_yaw_);
            const float dist = std::hypot(goal_n_ - local_pos_.x, goal_e_ - local_pos_.y);
            const float speed = std::sqrt(local_pos_.vx * local_pos_.vx + local_pos_.vy * local_pos_.vy +
                                          local_pos_.vz * local_pos_.vz);
            settle_count_ = (dist < goal_settle_radius_ && speed < goal_settle_speed_) ? settle_count_ + 1 : 0;
            const bool timeout = (tick_count_ - goal_tick_) >= goal_timeout_ticks_;
            if (settle_count_ >= goal_settle_ticks_ || timeout) {
                RCLCPP_INFO(this->get_logger(), "%s: %.2f m from the goal at %.2f m/s - landing.",
                    timeout ? "Goal approach timed out" : "At the goal", dist, speed);
                send_land();
                land_sent_ = true;
                land_tick_ = tick_count_;
                test_phase_ = "LANDING";
            }
            return;
        }

        // Phase 5/6: Path Following.
        // wp 는 fast_planner+bridge 가 /local_waypoint_setpoint_to_PF 로 비동기 공급한다.
        // PF 가 경로를 받아 attitude 명령(/pf_att_2_control)을 내기 시작하면 포워딩으로
        // 전환하고, 그 전까지는 호버하며 PF cmd 를 대기한다.
        if (!pf_done_ && !land_sent_) {
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

        // Phase 8: DISARM. PX4 가 착지 후 스스로 disarm 한다(COM_DISARM_LAND). 고정 10초 뒤
        // disarm 은 X8 처럼 내려오는 데 더 걸리면 공중에서 거부됐다 - 그때까지 기다리고, 60초가
        // 지나도 무장 상태면 disarm 을 한 번 보낸다(공중이면 PX4 가 거부).
        // NAV_LAND goes out once over uXRCE-DDS (UDP): when that one message was lost PX4 kept holding
        // the goal, the vehicle hovered 60 s and the disarm below was refused in the air (ctrack_tau7_1).
        // Resend it every second until PX4 is in AUTO_LAND.
        if (!disarmed_ && land_tick_ >= 0 && is_armed && tick_count_ > land_tick_ &&
            (tick_count_ - land_tick_) % 10 == 0 &&
            vehicle_status_.nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_LAND) {
            RCLCPP_WARN(this->get_logger(), "LAND not taken (nav_state %u) - resending.",
                        static_cast<unsigned>(vehicle_status_.nav_state));
            send_land();
        }
        if (!disarmed_ && land_tick_ >= 0) {
            const bool timeout = (tick_count_ - land_tick_) >= 600;
            if (!is_armed || timeout) {
                if (is_armed) {
                    send_disarm();
                }
                disarmed_ = true;
                test_phase_ = "DISARMED";
                RCLCPP_INFO(this->get_logger(), "%s Shutting down.",
                    is_armed ? "Landing not confirmed after 60 s - sent disarm." : "Landed and disarmed.");
                rclcpp::shutdown();
            }
        }
    }

    void start_goal_phase(float dist, const char * reason)
    {
        goal_phase_ = true;
        goal_tick_ = tick_count_;
        settle_count_ = 0;
        goal_yaw_ = local_pos_.heading;
        RCLCPP_INFO(this->get_logger(),
            "Goal approach (%s): %.2f m from the goal - PX4 position control to NED (%.2f, %.2f, %.2f).",
            reason, dist, goal_n_, goal_e_, goal_d_);
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

    // 최종 목표 접근 (PX4 위치제어)
    bool  goal_known_{false};
    bool  goal_phase_{false};
    float goal_n_{0.0f}, goal_e_{0.0f}, goal_d_{0.0f}, goal_yaw_{0.0f};
    double goal_handover_radius_{0.0}, goal_settle_radius_{0.3}, goal_settle_speed_{0.3};
    int   goal_settle_ticks_{10}, goal_timeout_ticks_{200};
    int   goal_tick_{-1}, settle_count_{0};
    double goal_pf_done_max_{6.0};

    // 충돌(IMU 충격)·위치 추정 리셋·dead reckoning → 그 자리 착륙
    double abort_xy_reset_{3.0};
    bool   abort_{false};
    double impact_accel_{30.0}, impact_window_{0.2};
    int    impact_samples_{2};
    bool   abort_on_dead_reckoning_{true};
    std::deque<double> impact_times_;
    double imu_first_t_{-1.0};
    long   imu_count_{0};

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
    rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr       imu_sub_;
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
