#ifndef PAROL6_HARDWARE_INTERFACE__HPP_
#define PAROL6_HARDWARE_INTERFACE__HPP_
#pragma once

#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace parol6_hardware_interface {

class Parol6HardwareInterface : public hardware_interface::SystemInterface {

    public:
    
        hardware_interface::CallbackReturn on_init(
            const hardware_interface::HardwareInfo &info
        ) override;

        hardware_interface::CallbackReturn on_configure(
            const rclcpp_lifecycle::State & previous_state
        ) override;

        hardware_interface::CallbackReturn on_activate(
            const rclcpp_lifecycle::State & previous_state
        ) override;

        hardware_interface::CallbackReturn on_deactivate(
            const rclcpp_lifecycle::State $ previous_state
        ) override;

        std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

        std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

        hardware_interface::return_type read(
            const rclcpp::Time & time, const rclcpp::Duration & period
        ) override;

        hardware_interface::return_type write(
            const rclcpp::Time & time, const rclcpp::Duration & period
        ) override;

    private:

        const char backend_type_{};

        std::vector<hardware_interface::StateInterface> state_interfaces_{};
        std::vector<hardware_interface::CommandInterface> command_interfaces_{};
        
        std::vector<double> hw_positions_{};
        std::vector<double> hw_velocities_{};
        std::vector<double> hw_efforts_{};

        std::vector<double> hw_positions_commanded_{};
        std::vector<double> hw_velocities_commanded_{};
        std::vector<double> hw_efforts_commanded_{};
};
} // parol6_hardware_interface

#endif // PAROL6_HARDWARE_INTERFACE__HPP