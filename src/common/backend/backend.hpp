#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "hardware_interface/hardware_component_interface.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "rclcpp/node.hpp"

namespace common {

struct MotorCommand {
    double q = 0.0;
    double dq = 0.0;
    double tau = 0.0;
    double kp = 0.0;
    double kd = 0.0;
    bool enabled = false;
};

struct MotorState {
    double q = 0.0;
    double dq = 0.0;
    double tau = 0.0;
    uint8_t status = 0;
};

struct BodyPose {
    std::string name;
    std::array<double, 3> pos{0, 0, 0};      // x, y, z
    std::array<double, 4> quat{1, 0, 0, 0};  // w, x, y, z (MuJoCo convention)
};

class Backend {
 public:
    virtual ~Backend() = default;

    virtual hardware_interface::CallbackReturn init(const hardware_interface::HardwareInfo& info,
                                                    rclcpp::Node::SharedPtr node) = 0;
    virtual hardware_interface::CallbackReturn activate() = 0;
    virtual hardware_interface::CallbackReturn deactivate() = 0;

    virtual bool read(std::vector<MotorState>& states) = 0;
    virtual void write(const std::vector<MotorCommand>& commands) = 0;

    // Advance physics by one control period (no-op for real hardware)
    virtual void step(double /*dt*/) {}

    // Notify backend when a controller is active
    virtual void set_controller_active(bool /*active*/) {}
};

}  // namespace common
