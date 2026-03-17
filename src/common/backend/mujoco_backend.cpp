#include "common/backend/mujoco_backend.hpp"

namespace common {

    // Static active-backend pointer used by the mjcb_control callback.
    static MujocoBackend* active_backend = nullptr;

    static void control_wrapper(const mjModel*, mjData*) {
        if (active_backend) {
            active_backend->control(active_backend->model(), active_backend->data());
        }
    }

    // Destructor 

    MujocoBackend::~MujocoBackend(){
        active_backend = nullptr;
        mjcb_control = nullptr;
        if (mj_data_) mj_deleteData(mj_data_);
        if (mj_model_) mj_deleteModel(mj_model_);
    }

    hardware_interface::CallbackReturn MujocoBackend::init(const  hardware_interface::HardwareInfo& info,rclcpp::Node::SharedPtr) {
        
        // Load MJCF model
        if (!info.hardware_parameters.contains("mjcf_file_path")) {
            RCLCPP_ERROR(logger_, "Missing 'mjcf_file_path' hardware parameter");
            return hardware_interface::CallbackReturn::ERROR;
        }
        const auto& mjcf_path = info.hardware_parameters.at("mjcf_file_path");
        RCLCPP_INFO(logger_, "Loading MuJoCo model from %s ...", mjcf_path.c_str());

        char error[20] = "Could not load model";
        mj_model_ = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));

        if (!mj_model_) {
            RCLCPP_ERROR(logger_, "Failed to load MJCF: %s", error);
            return hardware_interface::CallbackReturn::ERROR;
        }
        RCLCPP_INFO(logger_, " MuJoCo model loaded: nq=%d nv=%d nu=%d", mj_model_->nq, mj_model_->nv, mj_model_->nu);

        mj_data_ = mj_makeData(mj_model_);

        //Enforcing implicitfast integrator and 1ms timestep for stability
        if (mj_model_.->opt.integrator != mjINT_IMPLICITFAST) {
            RCLCPP_WARN(logger_, "Overriding MujoCO integrator to implicitfast");
            mj_model_->opt.integrator = mjINT_IMPLICITFAST;
        }
        if (mj_model_->opt.timestep >)

    }
}