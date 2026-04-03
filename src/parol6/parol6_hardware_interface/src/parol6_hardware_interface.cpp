#include "parol6_hardware_interface/parol6_hardware_interface.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "common/backend/mujoco_backend.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "parol6_hardware_interface/parol6_backend.hpp"
#include "rclcpp/rclcpp.hpp"

namespace parol6_hardware_interface {

// ---------------------------------------------------------------------------
// ros2_control interface exports
// ---------------------------------------------------------------------------

std::vector<hardware_interface::StateInterface> Parol6HardwareInterface::export_state_interfaces() {
    std::vector<hardware_interface::StateInterface> state_interfaces;

    for (size_t i = 0; i < joint_names_.size(); i++) {
        state_interfaces.emplace_back(
            hardware_interface::StateInterface(joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
        state_interfaces.emplace_back(
            hardware_interface::StateInterface(joint_names_[i], hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]));
    }
    return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> Parol6HardwareInterface::export_command_interfaces() {
    std::vector<hardware_interface::CommandInterface> command_interfaces;

    for (size_t i = 0; i < joint_names_.size(); i++) {
        command_interfaces.emplace_back(hardware_interface::CommandInterface(
            joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_commands_positions_[i]));
        command_interfaces.emplace_back(hardware_interface::CommandInterface(
            joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_commands_velocities_[i]));
        command_interfaces.emplace_back(hardware_interface::CommandInterface(
            joint_names_[i], hardware_interface::HW_IF_EFFORT, &hw_commands_efforts_[i]));
        command_interfaces.emplace_back(
            hardware_interface::CommandInterface(joint_names_[i], "p_gain", &hw_commands_p_gains_[i]));
        command_interfaces.emplace_back(
            hardware_interface::CommandInterface(joint_names_[i], "d_gain", &hw_commands_d_gains_[i]));
    }

    return command_interfaces;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

hardware_interface::CallbackReturn Parol6HardwareInterface::on_init(
    const hardware_interface::HardwareInfo& system_info) {
    // Init base interface
    if (auto result = hardware_interface::SystemInterface::on_init(system_info);
        result != hardware_interface::CallbackReturn::SUCCESS) {
        RCLCPP_FATAL(logger_, "Error initialising base interface");
        return result;
    }

    // Create backend based on hardware_interface_type parameter
    if (info_.hardware_parameters.count("hardware_interface_type") == 0) {
        RCLCPP_ERROR(logger_, "Missing 'hardware_interface_type' hardware parameter");
        return hardware_interface::CallbackReturn::ERROR;
    }

    auto hardware_interface_type_str = info_.hardware_parameters["hardware_interface_type"];
    if (hardware_interface_type_str == "real") {
        RCLCPP_INFO(logger_, "Operating in Real mode — PAROL6 serial backend");
        backend_ = std::make_unique<Parol6Backend>();
    } else if (hardware_interface_type_str == "sim") {
        RCLCPP_INFO(logger_, "Operating in Sim mode — MuJoCo backend");
        backend_ = std::make_unique<common::MujocoBackend>();
        is_sim_ = true;
    } else {
        RCLCPP_ERROR(logger_, "Unsupported hardware_interface_type: '%s'", hardware_interface_type_str.c_str());
        return hardware_interface::CallbackReturn::ERROR;
    }

    // Populate joint names from the hardware info
    for (const auto& joint : info_.joints) {
        joint_names_.push_back(joint.name);
    }

    // Resize state and command vectors
    size_t num_joints = joint_names_.size();
    hw_positions_.resize(num_joints, 0.0);
    hw_velocities_.resize(num_joints, 0.0);
    hw_efforts_.resize(num_joints, 0.0);
    hw_commands_positions_.resize(num_joints, 0.0);
    hw_commands_velocities_.resize(num_joints, 0.0);
    hw_commands_efforts_.resize(num_joints, 0.0);
    hw_commands_p_gains_.resize(num_joints, -1.0);
    hw_commands_d_gains_.resize(num_joints, -1.0);

    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "-r", "__node:=parol6_hardware_interface"});
    node_ = rclcpp::Node::make_shared("_", options);

    // Load per-joint PD gains from YAML file
    if (info_.hardware_parameters.count("gains_file_path")) {
        const auto gains_file = info_.hardware_parameters.at("gains_file_path");
        RCLCPP_INFO(logger_, "Loading gains from %s", gains_file.c_str());
        YAML::Node gains_yaml = YAML::LoadFile(gains_file);
        const std::string section = is_sim_ ? "sim" : "hardware";
        if (!gains_yaml[section]) {
            RCLCPP_ERROR(logger_, "Gains file missing '%s' section", section.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        for (auto it = gains_yaml[section].begin(); it != gains_yaml[section].end(); ++it) {
            const auto joint = it->first.as<std::string>();
            if (it->second["kp"]) gains_kp_[joint] = it->second["kp"].as<float>();
            if (it->second["kd"]) gains_kd_[joint] = it->second["kd"].as<float>();

            if (it->second["kp_min"] && it->second["kp_max"]) {
                gains_kp_min_[joint] = it->second["kp_min"].as<float>();
                gains_kp_max_[joint] = it->second["kp_max"].as<float>();
            }
            if (it->second["kd_min"] && it->second["kd_max"]) {
                gains_kd_min_[joint] = it->second["kd_min"].as<float>();
                gains_kd_max_[joint] = it->second["kd_max"].as<float>();
            }

            RCLCPP_DEBUG(logger_, " %s: kp=%.4f kd=%.4f", joint.c_str(),
                         gains_kp_.count(joint) ? gains_kp_[joint] : 0.f,
                         gains_kd_.count(joint) ? gains_kd_[joint] : 0.f);
        }
        RCLCPP_INFO(logger_, "Loaded %zu joint gains (%s)", gains_kp_.size(), section.c_str());
    }

    // Extract per-joint command limits from URDF (via ros2_control command_interfaces)
    for (const auto& joint : info_.joints) {
        for (const auto& ci : joint.command_interfaces) {
            if (ci.name == "position" && !ci.min.empty() && !ci.max.empty()) {
                joint_pos_min_[joint.name] = std::stod(ci.min);
                joint_pos_max_[joint.name] = std::stod(ci.max);
            } else if (ci.name == "velocity" && !ci.min.empty() && !ci.max.empty()) {
                joint_vel_min_[joint.name] = std::stod(ci.min);
                joint_vel_max_[joint.name] = std::stod(ci.max);
            } else if (ci.name == "effort" && !ci.min.empty() && !ci.max.empty()) {
                joint_eff_min_[joint.name] = std::stod(ci.min);
                joint_eff_max_[joint.name] = std::stod(ci.max);
            }
        }
        if (joint_pos_min_.count(joint.name)) {
            RCLCPP_DEBUG(logger_, "Joint %s limits: pos=[%.3f, %.3f]", joint.name.c_str(), joint_pos_min_[joint.name],
                         joint_pos_max_[joint.name]);
        }
    }
    RCLCPP_INFO(logger_, "Loaded URDF command limits for %zu joints", joint_pos_min_.size());

    // Initialize backend
    auto result = backend_->init(info_, node_);
    if (result != hardware_interface::CallbackReturn::SUCCESS) {
        return result;
    }

    debug_publisher_ = node_->create_publisher<sensor_msgs::msg::JointState>("debug/lowcmd", 10);

    RCLCPP_INFO(logger_, "Successfully initialized PAROL6 hardware interface");
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn Parol6HardwareInterface::on_activate(const rclcpp_lifecycle::State&) {
    // Backend blocks until first state is received (real) or sim is ready
    auto result = backend_->activate();
    if (result != hardware_interface::CallbackReturn::SUCCESS) {
        return result;
    }

    // Read initial state
    rclcpp::Time now = rclcpp::Clock().now();
    rclcpp::Duration period = rclcpp::Duration::from_nanoseconds(0);
    read(now, period);

    // Capture initial positions — hold activation pose until controller commands otherwise
    initial_positions_ = hw_positions_;

    for (size_t i = 0; i < joint_names_.size(); i++) {
        hw_commands_positions_[i] = initial_positions_[i];
        hw_commands_velocities_[i] = 0.0;
        hw_commands_efforts_[i] = 0.0;
        RCLCPP_DEBUG(logger_, "Joint %s (index %zu) - q_init: %.3f", joint_names_[i].c_str(), i, initial_positions_[i]);
    }

    // Perform an initial write so the backend has valid gains and setpoints
    write(now, rclcpp::Duration::from_nanoseconds(0));

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn Parol6HardwareInterface::on_deactivate(const rclcpp_lifecycle::State&) {
    backend_->deactivate();
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn Parol6HardwareInterface::on_error(const rclcpp_lifecycle::State&) {
    std::exit(-1);
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

hardware_interface::return_type Parol6HardwareInterface::read(const rclcpp::Time&, const rclcpp::Duration&) {
    std::vector<common::MotorState> states;
    if (!backend_->read(states)) {
        RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 5000, "Waiting for first state from robot...");
        return hardware_interface::return_type::OK;
    }

    state_received_ = true;

    // Map motor states into joint-order hw_* vectors
    for (size_t i = 0; i < joint_names_.size() && i < states.size(); i++) {
        hw_positions_[i] = states[i].q;
        hw_velocities_[i] = states[i].dq;
        hw_efforts_[i] = states[i].tau;
    }

    return hardware_interface::return_type::OK;
}

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

void Parol6HardwareInterface::populate_motor_command(common::MotorCommand& cmd, const std::string& joint_name, size_t i,
                                                     float default_kp, float default_kd) {
    cmd.enabled = true;
    cmd.q = hw_commands_positions_[i];
    cmd.dq = hw_commands_velocities_[i];
    cmd.tau = hw_commands_efforts_[i];

    // Clamp position/velocity/effort to URDF limits
    if (joint_pos_min_.count(joint_name)) {
        cmd.q = std::clamp(cmd.q, joint_pos_min_.at(joint_name), joint_pos_max_.at(joint_name));
    }
    if (joint_vel_min_.count(joint_name)) {
        cmd.dq = std::clamp(cmd.dq, joint_vel_min_.at(joint_name), joint_vel_max_.at(joint_name));
    }
    if (joint_eff_min_.count(joint_name)) {
        cmd.tau = std::clamp(cmd.tau, joint_eff_min_.at(joint_name), joint_eff_max_.at(joint_name));
    }

    // Resolve gains: use YAML defaults when controller sends -1
    cmd.kp = (hw_commands_p_gains_[i] == -1.0) ? default_kp : hw_commands_p_gains_[i];
    cmd.kd = (hw_commands_d_gains_[i] == -1.0) ? default_kd : hw_commands_d_gains_[i];

    // Clamp gains to configured ranges
    if (gains_kp_min_.count(joint_name)) {
        cmd.kp = std::clamp(cmd.kp, static_cast<double>(gains_kp_min_.at(joint_name)),
                            static_cast<double>(gains_kp_max_.at(joint_name)));
    }
    if (gains_kd_min_.count(joint_name)) {
        cmd.kd = std::clamp(cmd.kd, static_cast<double>(gains_kd_min_.at(joint_name)),
                            static_cast<double>(gains_kd_max_.at(joint_name)));
    }
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

hardware_interface::return_type Parol6HardwareInterface::write(const rclcpp::Time&, const rclcpp::Duration& period) {
    if (!state_received_) return hardware_interface::return_type::OK;

    // NaN safety check
    for (size_t i = 0; i < joint_names_.size(); i++) {
        if (std::isnan(hw_commands_positions_[i]) || std::isnan(hw_commands_velocities_[i]) ||
            std::isnan(hw_commands_efforts_[i]) || std::isnan(hw_commands_p_gains_[i]) ||
            std::isnan(hw_commands_d_gains_[i])) {
            RCLCPP_FATAL(logger_, "NaN detected in command for joint %s — shutting down!", joint_names_[i].c_str());
            return hardware_interface::return_type::ERROR;
        }
    }

    // Build motor command vector
    std::vector<common::MotorCommand> commands(joint_names_.size());

    for (size_t i = 0; i < joint_names_.size(); i++) {
        const auto& joint_name = joint_names_[i];
        float kp = gains_kp_.count(joint_name) ? gains_kp_[joint_name] : 0.0f;
        float kd = gains_kd_.count(joint_name) ? gains_kd_[joint_name] : 0.0f;
        populate_motor_command(commands[i], joint_name, i, kp, kd);
    }

    // Send to backend
    backend_->write(commands);

    // Step sim physics after all commands are stored
    backend_->step(period.seconds());

    // Activate backend now that the controller has had a cycle to run
    if (controller_pending_) {
        backend_->set_controller_active(true);
        controller_pending_ = false;
    }

    // Debug publishing
    sensor_msgs::msg::JointState debug_msg{};
    for (size_t j = 0; j < joint_names_.size(); j++) {
        debug_msg.name.push_back(joint_names_[j]);
        debug_msg.position.push_back(commands[j].q);
        debug_msg.velocity.push_back(commands[j].dq);
        debug_msg.effort.push_back(commands[j].tau);
    }
    debug_publisher_->publish(debug_msg);

    return hardware_interface::return_type::OK;
}

// ---------------------------------------------------------------------------
// perform_command_mode_switch — defer backend activation by one cycle
// ---------------------------------------------------------------------------

hardware_interface::return_type Parol6HardwareInterface::perform_command_mode_switch(
    const std::vector<std::string>& start_interfaces, const std::vector<std::string>& /*stop_interfaces*/) {
    if (!start_interfaces.empty()) {
        RCLCPP_INFO(logger_, "Controller claiming interfaces — will activate backend next cycle");
        controller_pending_ = true;
    }
    return hardware_interface::return_type::OK;
}

}  // namespace parol6_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(parol6_hardware_interface::Parol6HardwareInterface, hardware_interface::SystemInterface)
