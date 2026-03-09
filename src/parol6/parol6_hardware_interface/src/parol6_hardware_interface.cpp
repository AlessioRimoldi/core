#include "parol6_hardware_interface/parol6_hardware_interface.hpp"

#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace parol6_hardware_interface{

    hardware_interface::CallbackReturn Parol6HardwareInterface::on_init(
        const hardware_interface::HardwareInfo & info
    ) {
        if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS){
            return hardware_interface::CallbackReturn::ERROR;
        }
        //Setup backend
        if (backend_type_ == "mujoco"){
            RCLPP_INFO("Initializing Mujoco sim..")
            return hardware_interface::CallbackReturn::SUCCESS;
        }
        if (backend_type_ == "parol6" ){
            RCLCPP_INFO("Parol6 real hardware not yet supported!")
            return hardware_interface::CallbackReturn::ERROR;
        }
    }

    std::vector<hardware_interface::StateInterface> Parol6HardwareInterface::export_state_interfaces(){
         
        for (size_t i = 0; i < info_.joints.size(); i++){
            state_interfaces_.emplace_back(
                hardware_interface::StateInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
            state_interfaces_.emplace_back(
                hardware_interface::StateInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
            state_interfaces_.emplace_back(
                hardware_interface::StateInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]));
        }
        return state_interfaces_;
    }

    std::vector<hardware_interface::CommandInterface> Parol6HardwareInterface::export_command_interfaces(){

        for (size_t i = 0; i < info_.joints.size(); i++){
            command_interfaces_.emplace_back(
                hardware_interface::CommandInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
            command_interfaces_.emplace_back(
                hardware_interface::CommandInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
            command_interfaces_.emplace_back(
                hardware_interface::CommandInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_effort_[i]));
        }
        return command_interfaces_;
    }

    hardware_interface::CallbackReturn Parol6HardwareInterface::on_activate(){
        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn Parol6HardwareInterface::on_configure(){
        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn Parol6HardwareInterface::on_deactivate(){
        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::return_type Parol6HardwareInterface::read(){
        hardware_interface::return_type::OK;
    }

    hardware_interface::return_type Parol6HardwareInterface::write(){
        hardware_interface::return_type::OK;
    }
}

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(parol6_hardware_interface::Parol6HardwareInterface, hardware_interface::SystemInterface)