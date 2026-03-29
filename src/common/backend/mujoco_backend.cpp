#include "common/backend/mujoco_backend.hpp"

#include <algorithm>
#include <cmath>
#include <string>

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

    hardware_interface::CallbackReturn MujocoBackend::init(const hardware_interface::HardwareInfo& info, rclcpp::Node::SharedPtr) {

        // Populate joint names from hardware info
        for (const auto& joint : info.joints) {
            joint_names_.push_back(joint.name);
        }

        // Load MJCF model
        if (info.hardware_parameters.count("mjcf_file_path") == 0) {
            RCLCPP_ERROR(logger_, "Missing 'mjcf_file_path' hardware parameter");
            return hardware_interface::CallbackReturn::ERROR;
        }
        const auto& mjcf_path = info.hardware_parameters.at("mjcf_file_path");
        RCLCPP_INFO(logger_, "Loading MuJoCo model from %s ...", mjcf_path.c_str());

        char error[1024] = "Could not load model";
        mj_model_ = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));

        if (!mj_model_) {
            RCLCPP_ERROR(logger_, "Failed to load MJCF: %s", error);
            return hardware_interface::CallbackReturn::ERROR;
        }
        RCLCPP_INFO(logger_, "MuJoCo model loaded: nq=%d nv=%d nu=%d", mj_model_->nq, mj_model_->nv, mj_model_->nu);

        mj_data_ = mj_makeData(mj_model_);

        // Enforcing implicitfast integrator and 1ms timestep for stability
        if (mj_model_->opt.integrator != mjINT_IMPLICITFAST) {
            RCLCPP_WARN(logger_, "Overriding MuJoCo integrator to implicitfast");
            mj_model_->opt.integrator = mjINT_IMPLICITFAST;
        }
        if (mj_model_->opt.timestep > 0.001) {
            RCLCPP_WARN(logger_, "Overriding MuJoCo timestep from %.4f to 0.001", mj_model_->opt.timestep);
            mj_model_->opt.timestep = 0.001;
        }

        // Add joint armature (reflected rotor inertia) to prevent discrete-time instability.
        constexpr double kDefaultArmature = 0.01; // kg*m^2 ~conservative motor inertia
        for (int i = 0; i < mj_model_->njnt; ++i) {
            if (mj_model_->jnt_type[i] == mjJNT_HINGE) {
                int dof = mj_model_->jnt_dofadr[i];
                if (mj_model_->dof_armature[dof] < kDefaultArmature) {
                    mj_model_->dof_armature[dof] = kDefaultArmature;
                }
            }
        }
        RCLCPP_INFO(logger_, "Applied default armature (%.4f) to %d hinge joints", kDefaultArmature, mj_model_->njnt);

        // Register the PD control callback via a static global pointer.
        active_backend = this;
        mjcb_control = control_wrapper;

        // Build joint name → MuJoCo qpos/dof index maps
        for (int i = 0; i < mj_model_->njnt; ++i) {
            auto name = std::string(mj_id2name(mj_model_, mjOBJ_JOINT, i));
            name_to_mj_q_index_[name] = mj_model_->jnt_qposadr[i];
            name_to_mj_dof_index_[name] = mj_model_->jnt_dofadr[i];
        }

        // Initialize per-joint PD state. Gains start at 0.
        for (const auto& name : joint_names_) {
            pos_setpoint_[name] = 0.0;
            vel_setpoint_[name] = 0.0;
            tau_feedforward_[name] = 0.0;
            kp_[name] = 0.0;
            kd_[name] = 0.0;
        }

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    // ----------------------------------------------------------------------------
    // activate - reset sim state, sync setpoints to initial qpos
    // ----------------------------------------------------------------------------

    hardware_interface::CallbackReturn MujocoBackend::activate() {
        mj_resetData(mj_model_, mj_data_);

        // Ensure the callback pointer is still set.
        active_backend = this;

        // Forward kinematics to get consistent derived quantities.
        mj_forward(mj_model_, mj_data_);

        // Sync setpoints to initial qpos so PD holds the initial pose.
        for (const auto& name : joint_names_) {
            auto it = name_to_mj_q_index_.find(name);
            if (it == name_to_mj_q_index_.end()) continue;
            pos_setpoint_[name] = mj_data_->qpos[it->second];
            vel_setpoint_[name] = 0.0;
            tau_feedforward_[name] = 0.0;
        }

        control_tick_ = 0;
        unstable_ = false;
        controller_active_ = false;
        RCLCPP_INFO(logger_, "MuJoCo backend activated - holding pose until controller claims interfaces");
        return hardware_interface::CallbackReturn::SUCCESS;
    }

    // -----------------------------------------------------------------------------
    // deactivate
    // -----------------------------------------------------------------------------
    hardware_interface::CallbackReturn MujocoBackend::deactivate() {
        RCLCPP_INFO(logger_, "MuJoCo backend deactivated");
        return hardware_interface::CallbackReturn::SUCCESS;
    }

    bool MujocoBackend::read(std::vector<MotorState>& states) {
        states.resize(joint_names_.size());
        for (size_t i = 0; i < joint_names_.size(); ++i) {
            const auto& name = joint_names_[i];
            auto q_it = name_to_mj_q_index_.find(name);
            auto dof_it = name_to_mj_dof_index_.find(name);
            if (q_it == name_to_mj_q_index_.end() || dof_it == name_to_mj_dof_index_.end()) continue;
            states[i].q = mj_data_->qpos[q_it->second];
            states[i].dq = mj_data_->qvel[dof_it->second];
            states[i].tau = mj_data_->qfrc_applied[dof_it->second];
            states[i].status = 0;
        }
        return true;
    }

    void MujocoBackend::write(const std::vector<MotorCommand>& commands) {
        absl::MutexLock lock(&control_mu_);
        for (size_t i = 0; i < joint_names_.size() && i < commands.size(); ++i) {
            const auto& name = joint_names_[i];
            if (name_to_mj_dof_index_.find(name) == name_to_mj_dof_index_.end()) continue;
            const auto& cmd = commands[i];
            if (!cmd.enabled) continue;
            pos_setpoint_[name] = cmd.q;
            vel_setpoint_[name] = cmd.dq;
            tau_feedforward_[name] = cmd.tau;
            kp_[name] = cmd.kp;
            kd_[name] = cmd.kd;
        }
    }

    // -----------------------------------------------------------------------------
    // step - advance physics by control_period_s (sub-stepping at mj_timestep)
    // -----------------------------------------------------------------------------

    void MujocoBackend::step(double control_period_s) {
        if (control_period_s <= 0.0) return;

        const int num_substeps = std::max(1, static_cast<int>(std::round(control_period_s / mj_model_->opt.timestep)));
        for (int i = 0; i < num_substeps; i++) {
            mj_step(mj_model_, mj_data_);
        }
    }

    // -----------------------------------------------------------------------------
    // set_controller_active
    // -----------------------------------------------------------------------------

    void MujocoBackend::set_controller_active(bool active) {
        controller_active_ = active;
        RCLCPP_INFO(logger_, "Controller active: %s", active ? "true" : "false");
    }

    // ------------------------------------------------------------------------------
    // control
    // ------------------------------------------------------------------------------

    void MujocoBackend::control(const mjModel* model, mjData* data) {
        // Zero out applied forces before accumulating
        for (int i = 0; i < model->nv; ++i) data->qfrc_applied[i] = 0.0;

        // Instability detection
        if (!unstable_) {
            for (const auto& name : joint_names_) {
                auto dof_it = name_to_mj_dof_index_.find(name);
                if (dof_it == name_to_mj_dof_index_.end()) continue;
                int dof_idx = dof_it->second;
                if (std::abs(data->qvel[dof_idx]) > 100.0 || std::isnan(data->qvel[dof_idx])) {
                    unstable_ = true;
                    RCLCPP_FATAL(logger_, "=== INSTABILITY at tick %d, t=%.6f, joint '%s' ===",
                                 control_tick_, data->time, name.c_str());
                    for (int j = 0; j < model->njnt; ++j) {
                        const char* jname = mj_id2name(model, mjOBJ_JOINT, j);
                        int q = model->jnt_qposadr[j];
                        int v = model->jnt_dofadr[j];
                        RCLCPP_FATAL(logger_, " joint '%s': qpos=%.6f qvel=%.6f",
                                     jname ? jname : "?", data->qpos[q], data->qvel[v]);
                    }
                    break;
                }
            }
        }
        if (unstable_) return;

        control_tick_++;

        // Hold pose via gravity compensation until a controller is active.
        if (!controller_active_) {
            for (const auto& name : joint_names_) {
                auto dof_it = name_to_mj_dof_index_.find(name);
                if (dof_it == name_to_mj_dof_index_.end()) continue;
                data->qfrc_applied[dof_it->second] = data->qfrc_bias[dof_it->second];
            }
            return;
        }

        // PD control — apply torques directly to joints via qfrc_applied
        absl::MutexLock lock(&control_mu_);

        for (const auto& name : joint_names_) {
            auto dof_it = name_to_mj_dof_index_.find(name);
            auto q_it = name_to_mj_q_index_.find(name);
            if (dof_it == name_to_mj_dof_index_.end() || q_it == name_to_mj_q_index_.end()) continue;

            int dof_idx = dof_it->second;
            int q_idx = q_it->second;

            double pos = data->qpos[q_idx];
            double vel = data->qvel[dof_idx];
            double tau = tau_feedforward_[name];
            double pos_d = pos_setpoint_[name];
            double vel_d = vel_setpoint_[name];
            double kp_val = kp_[name];
            double kd_val = kd_[name];

            data->qfrc_applied[dof_idx] = kp_val * (pos_d - pos) + kd_val * (vel_d - vel) + tau;
        }
    }

}  // namespace common