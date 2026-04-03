#include "parol6_hardware_interface/parol6_backend.hpp"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace parol6_hardware_interface {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

Parol6Backend::Parol6Backend() : logger_(rclcpp::get_logger("parol6_backend")) {
    // Precompute steps_per_radian for each joint
    for (size_t i = 0; i < kNumJoints; ++i) {
        steps_per_rad_[i] = (kStepsPerRev * kGearRatios[i]) / (2.0 * M_PI);
    }
}

Parol6Backend::~Parol6Backend() {
    close_serial();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

hardware_interface::CallbackReturn Parol6Backend::init(const hardware_interface::HardwareInfo& info,
                                                       rclcpp::Node::SharedPtr node) {
    node_ = node;

    // Get serial port from hardware parameters
    if (info.hardware_parameters.count("serial_port") == 0) {
        RCLCPP_ERROR(logger_, "Missing 'serial_port' hardware parameter");
        return hardware_interface::CallbackReturn::ERROR;
    }
    serial_port_ = info.hardware_parameters.at("serial_port");

    if (info.hardware_parameters.count("baudrate")) {
        baudrate_ = std::stoi(info.hardware_parameters.at("baudrate"));
    }

    // Store joint names from hardware info
    for (const auto& joint : info.joints) {
        joint_names_.push_back(joint.name);
    }
    if (joint_names_.size() != kNumJoints) {
        RCLCPP_ERROR(logger_, "Expected %zu joints, got %zu", kNumJoints, joint_names_.size());
        return hardware_interface::CallbackReturn::ERROR;
    }

    RCLCPP_INFO(logger_, "PAROL6 backend initialized (port=%s, baud=%d)", serial_port_.c_str(), baudrate_);
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn Parol6Backend::activate() {
    // Open serial port
    if (!open_serial(serial_port_, baudrate_)) {
        RCLCPP_ERROR(logger_, "Failed to open serial port %s", serial_port_.c_str());
        return hardware_interface::CallbackReturn::ERROR;
    }
    RCLCPP_INFO(logger_, "Serial port %s opened at %d baud", serial_port_.c_str(), baudrate_);

    // Send ENABLE command
    {
        std::array<int, kNumJoints> zeros{};
        RobotState response{};
        if (!send_and_receive(zeros, zeros, CMD_ENABLE, response)) {
            RCLCPP_WARN(logger_, "Failed to send ENABLE command, retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!send_and_receive(zeros, zeros, CMD_ENABLE, response)) {
                RCLCPP_ERROR(logger_, "Failed to send ENABLE command after retry");
                close_serial();
                return hardware_interface::CallbackReturn::ERROR;
            }
        }
        enabled_ = true;
        RCLCPP_INFO(logger_, "Robot enabled");
    }

    // Send HOME command and wait for homing to complete
    RCLCPP_INFO(logger_, "Starting homing sequence...");
    {
        std::array<int, kNumJoints> zeros{};
        RobotState response{};

        // Send HOME command once
        if (!send_and_receive(zeros, zeros, CMD_HOME, response)) {
            RCLCPP_ERROR(logger_, "Failed to send HOME command");
            close_serial();
            return hardware_interface::CallbackReturn::ERROR;
        }

        // Continue sending DUMMY while homing in progress
        auto start = std::chrono::steady_clock::now();
        constexpr auto kHomingTimeout = std::chrono::seconds(60);
        while (!all_joints_homed(response.homed_bits)) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > kHomingTimeout) {
                RCLCPP_ERROR(logger_, "Homing timed out after 60s (homed_bits=0x%02X)", response.homed_bits);
                close_serial();
                return hardware_interface::CallbackReturn::ERROR;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (!send_and_receive(zeros, zeros, CMD_DUMMY, response)) {
                RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 5000, "Serial communication error during homing");
            } else {
                RCLCPP_INFO_THROTTLE(logger_, *node_->get_clock(), 2000, "Homing in progress... (homed_bits=0x%02X)",
                                     response.homed_bits);
            }
        }

        homed_ = true;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            latest_state_ = response;
        }
        RCLCPP_INFO(logger_, "Homing complete - all joints homed");
    }

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn Parol6Backend::deactivate() {
    // Send DISABLE command
    if (serial_fd_ >= 0) {
        std::array<int, kNumJoints> zeros{};
        RobotState response{};
        send_and_receive(zeros, zeros, CMD_DISABLE, response);
    }
    enabled_ = false;
    homed_ = false;
    close_serial();
    RCLCPP_INFO(logger_, "PAROL6 backend deactivated");
    return hardware_interface::CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// Read / Write
// ---------------------------------------------------------------------------

bool Parol6Backend::read(std::vector<common::MotorState>& states) {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!latest_state_.valid) return false;

    states.resize(kNumJoints);
    for (size_t i = 0; i < kNumJoints; ++i) {
        states[i].q = steps_to_rad(latest_state_.positions[i], i);
        states[i].dq = steps_per_s_to_rad_per_s(latest_state_.speeds[i], i);
        states[i].tau = 0.0;  // PAROL6 uses stepper motors — no torque feedback
        states[i].status = 0;
    }
    return true;
}

void Parol6Backend::write(const std::vector<common::MotorCommand>& commands) {
    if (serial_fd_ < 0 || !enabled_) return;

    std::array<int, kNumJoints> pos_steps{};
    std::array<int, kNumJoints> speed_steps{};

    for (size_t i = 0; i < kNumJoints && i < commands.size(); ++i) {
        if (!commands[i].enabled) continue;
        pos_steps[i] = rad_to_steps(commands[i].q, i);
        speed_steps[i] = rad_per_s_to_steps_per_s(commands[i].dq, i);
    }

    RobotState response{};
    if (send_and_receive(pos_steps, speed_steps, CMD_GO_TO_POSITION, response)) {
        std::lock_guard<std::mutex> lock(state_mu_);
        latest_state_ = response;
    } else {
        RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 5000, "Serial communication error during write");
    }
}

// ---------------------------------------------------------------------------
// Serial port management
// ---------------------------------------------------------------------------

bool Parol6Backend::open_serial(const std::string& port, int baudrate) {
    serial_fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
        RCLCPP_ERROR(logger_, "Cannot open %s: %s", port.c_str(), strerror(errno));
        return false;
    }

    // Clear O_NONBLOCK after open (we'll use poll for timeouts)
    int flags = fcntl(serial_fd_, F_GETFL, 0);
    fcntl(serial_fd_, F_SETFL, flags & ~O_NONBLOCK);

    struct termios tty {};
    if (tcgetattr(serial_fd_, &tty) != 0) {
        RCLCPP_ERROR(logger_, "tcgetattr failed: %s", strerror(errno));
        close_serial();
        return false;
    }

    // Map baudrate to termios constant
    speed_t baud;
    switch (baudrate) {
        case 3000000:
            baud = B3000000;
            break;
        case 2000000:
            baud = B2000000;
            break;
        case 1000000:
            baud = B1000000;
            break;
        case 115200:
            baud = B115200;
            break;
        default:
            RCLCPP_ERROR(logger_, "Unsupported baudrate: %d", baudrate);
            close_serial();
            return false;
    }

    // Configure: 8N1, raw mode, no flow control
    cfmakeraw(&tty);
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    tty.c_cflag |= (CLOCAL | CREAD);  // Enable receiver, local mode
    tty.c_cflag &= ~CSTOPB;           // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;          // No hardware flow control
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;  // 100ms inter-character timeout

    tcflush(serial_fd_, TCIOFLUSH);
    if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0) {
        RCLCPP_ERROR(logger_, "tcsetattr failed: %s", strerror(errno));
        close_serial();
        return false;
    }

    return true;
}

void Parol6Backend::close_serial() {
    if (serial_fd_ >= 0) {
        ::close(serial_fd_);
        serial_fd_ = -1;
    }
}

// ---------------------------------------------------------------------------
// Protocol: send command + receive response
// ---------------------------------------------------------------------------

bool Parol6Backend::send_and_receive(const std::array<int, kNumJoints>& positions,
                                     const std::array<int, kNumJoints>& speeds, uint8_t command, RobotState& response) {
    // Build packet: start(3) + len(1) + payload(52) = 56 bytes total
    // Payload: joints(18) + speeds(18) + command(1) + affected(1) + inout(1) +
    //          timeout(1) + gripper(8) + crc(1) + end(2) = 52
    uint8_t packet[56];
    size_t idx = 0;

    // Start bytes
    packet[idx++] = START_BYTE;
    packet[idx++] = START_BYTE;
    packet[idx++] = START_BYTE;

    // Length
    packet[idx++] = INPUT_PACKET_LEN;

    // Joint positions (6 x 3 bytes)
    for (size_t j = 0; j < kNumJoints; ++j) {
        int_to_3bytes(positions[j], &packet[idx]);
        idx += 3;
    }

    // Joint speeds (6 x 3 bytes)
    for (size_t j = 0; j < kNumJoints; ++j) {
        int_to_3bytes(speeds[j], &packet[idx]);
        idx += 3;
    }

    // Command
    packet[idx++] = command;

    // Affected_joint (all joints affected)
    packet[idx++] = 0xFF;

    // InOut (no outputs set)
    packet[idx++] = 0x00;

    // Timeout
    packet[idx++] = 0x00;

    // Gripper position (2 bytes)
    packet[idx++] = 0x00;
    packet[idx++] = 0x00;

    // Gripper speed (2 bytes)
    packet[idx++] = 0x00;
    packet[idx++] = 0x00;

    // Gripper current (2 bytes)
    packet[idx++] = 0x00;
    packet[idx++] = 0x00;

    // Gripper command
    packet[idx++] = 0x00;

    // Gripper mode
    packet[idx++] = 0x00;

    // Gripper ID
    packet[idx++] = 0x00;

    // CRC byte (not validated by firmware)
    packet[idx++] = 0x00;

    // End bytes
    packet[idx++] = END_BYTE_1;
    packet[idx++] = END_BYTE_2;

    // Send
    if (!write_bytes(packet, idx)) {
        return false;
    }

    // Receive response
    std::vector<uint8_t> resp_payload;
    if (!read_response_packet(resp_payload, 100)) {
        return false;
    }

    // Parse response
    if (resp_payload.size() < OUTPUT_PACKET_LEN) {
        RCLCPP_WARN(logger_, "Response too short: %zu bytes", resp_payload.size());
        return false;
    }

    // Position data (6 x 3 bytes)
    for (size_t j = 0; j < kNumJoints; ++j) {
        response.positions[j] = bytes3_to_int(&resp_payload[j * 3]);
    }

    // Speed data (6 x 3 bytes, offset 18)
    for (size_t j = 0; j < kNumJoints; ++j) {
        response.speeds[j] = bytes3_to_int(&resp_payload[18 + j * 3]);
    }

    // Status bytes
    response.homed_bits = resp_payload[36];
    response.io_bits = resp_payload[37];
    response.temp_error_bits = resp_payload[38];
    response.position_error_bits = resp_payload[39];
    response.timing_data = bytes2_to_int(&resp_payload[40]);
    response.valid = true;

    return true;
}

// ---------------------------------------------------------------------------
// Low-level serial I/O
// ---------------------------------------------------------------------------

bool Parol6Backend::write_bytes(const uint8_t* data, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = ::write(serial_fd_, data + written, len - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            RCLCPP_ERROR(logger_, "Serial write error: %s", strerror(errno));
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

bool Parol6Backend::read_response_packet(std::vector<uint8_t>& payload, int timeout_ms) {
    payload.clear();

    // State machine to find start bytes and read payload
    enum class State { SYNC1, SYNC2, SYNC3, LENGTH, DATA };
    State state = State::SYNC1;
    uint8_t data_len = 0;
    size_t data_read = 0;

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        struct pollfd pfd {};
        pfd.fd = serial_fd_;
        pfd.events = POLLIN;

        auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) break;

        int ret = poll(&pfd, 1, static_cast<int>(remaining));
        if (ret <= 0) break;

        uint8_t byte;
        ssize_t n = ::read(serial_fd_, &byte, 1);
        if (n <= 0) continue;

        switch (state) {
            case State::SYNC1:
                if (byte == START_BYTE) state = State::SYNC2;
                break;
            case State::SYNC2:
                if (byte == START_BYTE)
                    state = State::SYNC3;
                else
                    state = State::SYNC1;
                break;
            case State::SYNC3:
                if (byte == START_BYTE)
                    state = State::LENGTH;
                else
                    state = State::SYNC1;
                break;
            case State::LENGTH:
                data_len = byte;
                payload.resize(data_len);
                data_read = 0;
                state = State::DATA;
                break;
            case State::DATA:
                payload[data_read++] = byte;
                if (data_read >= data_len) {
                    // Verify end bytes
                    if (data_len >= 2 && payload[data_len - 2] == END_BYTE_1 && payload[data_len - 1] == END_BYTE_2) {
                        return true;
                    }
                    // Bad end bytes, reset and try again
                    RCLCPP_WARN(logger_, "Bad end bytes in response, resync");
                    state = State::SYNC1;
                    payload.clear();
                }
                break;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------

int Parol6Backend::rad_to_steps(double rad, size_t joint_idx) const {
    if (joint_idx >= kNumJoints) return 0;
    return static_cast<int>(std::round(rad * steps_per_rad_[joint_idx]));
}

double Parol6Backend::steps_to_rad(int steps, size_t joint_idx) const {
    if (joint_idx >= kNumJoints) return 0.0;
    return static_cast<double>(steps) / steps_per_rad_[joint_idx];
}

int Parol6Backend::rad_per_s_to_steps_per_s(double rad_s, size_t joint_idx) const {
    if (joint_idx >= kNumJoints) return 0;
    return static_cast<int>(std::round(rad_s * steps_per_rad_[joint_idx]));
}

double Parol6Backend::steps_per_s_to_rad_per_s(int steps_s, size_t joint_idx) const {
    if (joint_idx >= kNumJoints) return 0.0;
    return static_cast<double>(steps_s) / steps_per_rad_[joint_idx];
}

// ---------------------------------------------------------------------------
// Byte encoding helpers
// ---------------------------------------------------------------------------

void Parol6Backend::int_to_3bytes(int value, uint8_t* buf) {
    buf[0] = static_cast<uint8_t>((value >> 16) & 0xFF);
    buf[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    buf[2] = static_cast<uint8_t>(value & 0xFF);
}

int Parol6Backend::bytes3_to_int(const uint8_t* buf) {
    int value = (static_cast<int>(buf[0]) << 16) | (static_cast<int>(buf[1]) << 8) | static_cast<int>(buf[2]);
    // Sign-extend from 24-bit
    if (value & 0x800000) {
        value |= ~0xFFFFFF;
    }
    return value;
}

void Parol6Backend::int_to_2bytes(int value, uint8_t* buf) {
    buf[0] = static_cast<uint8_t>((value >> 8) & 0xFF);
    buf[1] = static_cast<uint8_t>(value & 0xFF);
}

int Parol6Backend::bytes2_to_int(const uint8_t* buf) {
    return (static_cast<int>(buf[0]) << 8) | static_cast<int>(buf[1]);
}

bool Parol6Backend::all_joints_homed(uint8_t homed_bits) {
    // Bits 7..2 correspond to joints 1..6 (big-endian bit packing)
    // All 6 joints homed means bits 7,6,5,4,3,2 are all set = 0xFC
    return (homed_bits & 0xFC) == 0xFC;
}

}  // namespace parol6_hardware_interface
