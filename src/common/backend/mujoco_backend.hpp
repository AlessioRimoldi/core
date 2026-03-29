#pragma once

#include "common/backend/backend.hpp"

#include <set>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

#include "absl/synchronization/mutex.h"
#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"
#include "hardware_interface/system_interface.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace common {

class MujocoBackend : public Backend {
public:
    MujocoBackend() : logger_(rclcpp::get_logger("MujocoBackend")) {}
    ~MujocoBackend() override;

    hardware_interface::CallbackReturn init(
        const hardware_interface::HardwareInfo& info,
        rclcpp::Node::SharedPtr node) override;
    hardware_interface::CallbackReturn activate() override;
    hardware_interface::CallbackReturn deactivate() override;

    void step(double control_period) override;
    void set_controller_active(bool active) override;

    bool read(std::vector<MotorState>& states) override;
    void write(const std::vector<MotorCommand>& commands) override;

    // MuJoCo control callback -- compute PD torques and write to ctrl
    void control(const mjModel* model, mjData* data);

    // Accessors
    mjModel* model() const { return mj_model_; }
    mjData* data() const { return mj_data_; }

private:
    rclcpp::Logger logger_;

    mjModel* mj_model_{nullptr};
    mjData* mj_data_{nullptr};

    // Joint names from hardware info (defines joint ordering)
    std::vector<std::string> joint_names_;

    std::unordered_map<std::string, int> name_to_mj_q_index_;
    std::unordered_map<std::string, int> name_to_mj_dof_index_;

    absl::Mutex control_mu_;
    std::unordered_map<std::string, double> pos_setpoint_;
    std::unordered_map<std::string, double> vel_setpoint_;
    std::unordered_map<std::string, double> tau_feedforward_;
    std::unordered_map<std::string, double> kp_;
    std::unordered_map<std::string, double> kd_;

    // Instability detection
    int control_tick_{0};
    bool unstable_{false};

    // When false, control() holds pose via qfrc_bias instead of PD
    bool controller_active_{false};

    // ── Scene body TF publishing (background thread, off the control loop) ──
    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr scene_timer_;
    rclcpp::CallbackGroup::SharedPtr scene_cb_group_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr scene_executor_;
    std::thread scene_thread_;

    // Scene body name → MuJoCo body ID
    std::vector<std::string> scene_body_names_;
    std::vector<int> scene_body_ids_;

    void setup_scene_tf_publisher();
    void publish_scene_tf();
};

}  // namespace common