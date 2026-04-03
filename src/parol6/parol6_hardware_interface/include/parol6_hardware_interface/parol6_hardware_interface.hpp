#pragma once

#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/macros.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/backend/backend.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace parol6_hardware_interface {

class Parol6HardwareInterface : public hardware_interface::SystemInterface {
 public:
    RCLCPP_SHARED_PTR_DEFINITIONS(Parol6HardwareInterface)

    Parol6HardwareInterface() : logger_(rclcpp::get_logger("parol6_hardware_interface")) {}

    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo& system_info) override;
    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_error(const rclcpp_lifecycle::State& previous_state) override;

    hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

    hardware_interface::return_type perform_command_mode_switch(
        const std::vector<std::string>& start_interfaces, const std::vector<std::string>& stop_interfaces) override;

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

    ~Parol6HardwareInterface() {}

 private:
    rclcpp::Logger logger_;
    rclcpp::Node::SharedPtr node_;

    std::unique_ptr<common::Backend> backend_;

    std::vector<std::string> joint_names_;

    // Hardware state (joint space)
    std::vector<double> hw_positions_;
    std::vector<double> hw_velocities_;
    std::vector<double> hw_efforts_;

    // Hardware commands (joint space)
    std::vector<double> hw_commands_positions_;
    std::vector<double> hw_commands_velocities_;
    std::vector<double> hw_commands_efforts_;
    std::vector<double> hw_commands_p_gains_;
    std::vector<double> hw_commands_d_gains_;

    bool is_sim_{false};
    bool state_received_{false};

    // Per-joint PD gains loaded from gains YAML
    std::unordered_map<std::string, float> gains_kp_;
    std::unordered_map<std::string, float> gains_kd_;

    // Per-joint gain clamping ranges
    std::unordered_map<std::string, float> gains_kp_min_;
    std::unordered_map<std::string, float> gains_kp_max_;
    std::unordered_map<std::string, float> gains_kd_min_;
    std::unordered_map<std::string, float> gains_kd_max_;

    // Per-joint command limits from URDF
    std::unordered_map<std::string, double> joint_pos_min_;
    std::unordered_map<std::string, double> joint_pos_max_;
    std::unordered_map<std::string, double> joint_vel_min_;
    std::unordered_map<std::string, double> joint_vel_max_;
    std::unordered_map<std::string, double> joint_eff_min_;
    std::unordered_map<std::string, double> joint_eff_max_;

    // Debug publisher
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr debug_publisher_;

    // Populate a MotorCommand with clamped pos/vel/eff and resolved gains
    void populate_motor_command(common::MotorCommand& cmd, const std::string& joint_name, size_t i, float default_kp,
                                float default_kd);

    // Initial positions captured at activation
    std::vector<double> initial_positions_;

    // Defers backend activation by one cycle so the controller runs first
    bool controller_pending_{false};
};

}  // namespace parol6_hardware_interface
