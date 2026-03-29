#pragma once

#include "common/backend/backend.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace parol6_hardware_interface {

static constexpr size_t kNumJoints = 6;

// Steps per motor revolution (200 full steps * 32 microstepping)
static constexpr double kStepsPerRev = 6400.0;

// Gear ratios per joint (motor revolutions to joint revolution)
static constexpr std::array<double, kNumJoints> kGearRatios = {
    6.4,         // J1: Belt
    20.0,        // J2: Planetary gearbox
    18.0952381,  // J3: Planetary gearbox x Belt (38:42)
    4.0,         // J4: Belt
    4.0,         // J5: Belt
    10.0         // J6: Planetary gearbox
};

// PAROL6 serial protocol command bytes
static constexpr uint8_t CMD_JOG = 123;
static constexpr uint8_t CMD_GO_TO_POSITION = 156;
static constexpr uint8_t CMD_HOME = 100;
static constexpr uint8_t CMD_ENABLE = 101;
static constexpr uint8_t CMD_DISABLE = 102;
static constexpr uint8_t CMD_CLEAR_ERROR = 103;
static constexpr uint8_t CMD_DUMMY = 255;

// Packet framing
static constexpr uint8_t START_BYTE = 0xFF;
static constexpr uint8_t END_BYTE_1 = 0x01;
static constexpr uint8_t END_BYTE_2 = 0x02;
static constexpr uint8_t INPUT_PACKET_LEN = 52;   // payload length PC -> robot
static constexpr uint8_t OUTPUT_PACKET_LEN = 56;  // payload length robot -> PC

class Parol6Backend : public common::Backend {
public:
    Parol6Backend();
    ~Parol6Backend() override;

    hardware_interface::CallbackReturn init(
        const hardware_interface::HardwareInfo& info,
        rclcpp::Node::SharedPtr node) override;
    hardware_interface::CallbackReturn activate() override;
    hardware_interface::CallbackReturn deactivate() override;

    bool read(std::vector<common::MotorState>& states) override;
    void write(const std::vector<common::MotorCommand>& commands) override;

private:
    rclcpp::Logger logger_;
    rclcpp::Node::SharedPtr node_;

    int serial_fd_{-1};
    std::string serial_port_;
    int baudrate_{3000000};

    // Joint names from hardware info
    std::vector<std::string> joint_names_;

    // Steps per radian for each joint: (steps_per_rev * gear_ratio) / (2*PI)
    std::array<double, kNumJoints> steps_per_rad_{};

    // Latest robot state (protected by mutex)
    std::mutex state_mu_;
    struct RobotState {
        std::array<int, kNumJoints> positions{};  // steps
        std::array<int, kNumJoints> speeds{};     // steps/s
        uint8_t homed_bits{0};
        uint8_t io_bits{0};
        uint8_t temp_error_bits{0};
        uint8_t position_error_bits{0};
        uint16_t timing_data{0};
        bool valid{false};
    };
    RobotState latest_state_;

    bool homed_{false};
    bool enabled_{false};

    // Serial port management
    bool open_serial(const std::string& port, int baudrate);
    void close_serial();

    // Send a full command packet and receive the response
    bool send_and_receive(
        const std::array<int, kNumJoints>& positions,
        const std::array<int, kNumJoints>& speeds,
        uint8_t command,
        RobotState& response);

    // Low-level serial I/O
    bool write_bytes(const uint8_t* data, size_t len);
    bool read_response_packet(std::vector<uint8_t>& payload, int timeout_ms);

    // Unit conversion
    int rad_to_steps(double rad, size_t joint_idx) const;
    double steps_to_rad(int steps, size_t joint_idx) const;
    int rad_per_s_to_steps_per_s(double rad_s, size_t joint_idx) const;
    double steps_per_s_to_rad_per_s(int steps_s, size_t joint_idx) const;

    // 3-byte signed integer encoding (big-endian, 24-bit)
    static void int_to_3bytes(int value, uint8_t* buf);
    static int bytes3_to_int(const uint8_t* buf);

    // 2-byte integer encoding (big-endian, unsigned)
    static void int_to_2bytes(int value, uint8_t* buf);
    static int bytes2_to_int(const uint8_t* buf);

    // Check if all 6 joints are homed from the homed bitfield
    static bool all_joints_homed(uint8_t homed_bits);
};

}  // namespace parol6_hardware_interface
