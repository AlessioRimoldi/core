#include "common/backend/backend.hpp"

#include <vector>
#include <unordered_map>

#include "absl/synchronization/mutex.hpp"
#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"
#include "hardware_interface/system_interface.hpp"

namespace common {
    class MujocoBackend : public Backend {
        public:
            MujocoBackend();
            ~MujocoBackend() override;

            MujocoBackend(): logger_(rclcpp::get_logger("mujoco_backend")) {}

            hardware_interface::CallbackReturn init(const hardware_interface::HardwareInfo& info, rclcpp::Node::SharedPtr node) override;
            hardware_interface::CallbackReturn activate() override;
            hardware_interface::CallbackReturn deactivate() override;

            void step(double control_period) override;
            void write(const std::array<MotorCommand, knumberOfMotors>& commands) override;

            // Mujoco control callback -- compute PD torques and writes to ctrl
            void control(const mjModel* model, mjData* data);

            // Accessors
            mjModel* model() const { return mj_model_; }
            mjData* data() const { return mj_data_; }

        private:
            rclcpp::Logger logger_;
            
            mjModel* mj_model_{nullptr};
            mjData* mj_data_{nullptr};

            // Joint ordering
            static const std::vector<std::string> kAllJoints;

            std::unordered_map<std::string, int> name_to_mj_q_index_;
            std::unordered_map<std::string, int> name_to_mj_ctrl_index_;

            absl::Mutex control_mu_;
            std::unordered_map<std::string, double> pos_setpoint_;
            std::unordered_map<std::string, double> vel_setpoint_;
            std::unordered_map<std::string, double> tau_feedforward;
            std::unordered_map<std::string, double> kp_;
            std::unordered_map<std::string, double> kd_;

            // Instability detection
            int control_tick_{0};
            bool unstable_{false};

    };
}