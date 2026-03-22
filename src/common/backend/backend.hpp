#pragma once

#include <array>

#include "hardware_interface/hardware_info.hpp"
#include "rclcpp/node.hpp"

namespace common {

    static constexpr size_t knumberOfMotors;

    struct MotorCommand {
        double q = 0.0;
        double dq = 0.0;
        double tau = 0.0;
        double kp = 0.0;
        double kd = 0.0;
        bool enabled = false;
    }

    struct MotorState {
        double q = 0.0;
        double dq = 0.0;
        double tau = 0.0;
        uint8_t = 0;
    }

    class Backend {
        public:
        virtual ~Backend() = default;

        virtual hardware_interface::CallbackReturn init(const hardware_interface::HardwareInfo& info, rclcpp::Node::SharePtr node) = 0;
        virtual hardware_interface::CallbackReturn activate() = 0;
        virtual hardware_interface::CallbackReturn deactivate() = 0;

        virtual bool read(std::array<MotorState, knumberOfMotors>);
        virtual void write(std::array<MotorState, knumberOfMotors>);

        // Advance physics by one control period (no-op for real hardware)
        virtual void step(double) {};

        // Notify backend when a controller is active 
        virtual void set_controller_active(bool /*active*/) {};
    };
}
