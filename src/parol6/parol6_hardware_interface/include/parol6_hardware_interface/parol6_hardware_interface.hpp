#ifndef PAROL6_HARDWARE_INTERFACE__HPP_
#define PAROL6_HARDWARE_INTERFACE__HPP_
#pragma once

#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "common/backend/backend.hpp"

#include <rclcpp/macros.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msg/msg/joint_state.hpp>

namespace parol6_hardware_interface {

class Parol6HardwareInterface : public hardware_interface::SystemInterface {
    public:

        Parol6HardwareInterface(): logger_(rclcpp::get_logger("parol6_hardware_interface")) {}

        hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo &info) override;
        hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
        hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
        hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State $ previous_state) override;

        std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
        std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

        hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
        hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

        ~Parol6HardwareInterface() {}
    private:    
        rclcpp::Logger logger_;
        rclcpp::Node::SharedPtr node_;

        const char backend_type_{};
        const common::Backend backend_;

        std::vector<std::string> joint_names_{};

        std::vector<hardware_interface::StateInterface> state_interfaces_{};
        std::vector<hardware_interface::CommandInterface> command_interfaces_{};
        
        std::vector<double> hw_positions_{};
        std::vector<double> hw_velocities_{};
        std::vector<double> hw_efforts_{};

        std::vector<double> hw_positions_commands_{};
        std::vector<double> hw_velocities_commands_{};
        std::vector<double> hw_efforts_commands_{};
        std::vector<double> hw_p_gains_commands_{};
        std::vector<double> hw_d_gains_commands_{};

        // Gains and Limits
        std::unordered_map<std::string, float> gains_kp_{};
        std::unordered_map<std::string, float> gains_kd_{};
        std::unordered_map<std::string, float> gains_kp_min_{};
        std::unordered_map<std::string, float> gains_kd_min_{};
        std::unordered_map<std::string, float> gains_kp_max_{};
        std::unordered_map<std::string, float> gains_kd_max_{};
};
} // parol6_hardware_interface

#endif // PAROL6_HARDWARE_INTERFACE__HPP