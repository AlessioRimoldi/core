# CORE: Robotics Training & Simulation Stack

Infrastructure for training, evaluating, and deploying robot policies using **ROS2 Jazzy**, **MuJoCo** physics simulation, and **Docker Compose**.

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- X11 (for MuJoCo rendering)

## Quick Start

```bash
# Allow Docker to use the X11 display
xhost +local:docker

# Build and start all services (Redis, MLFlow, env)
cd docker
docker compose up --build -d
```

This starts three services:

| Service | Purpose | Port |
|---------|---------|------|
| `redis` | State/message store | 6379 |
| `mlflow` | Experiment tracking UI | 5000 |
| `env` | ROS2 + MuJoCo runtime | — |

On startup, the `env` container automatically builds all ROS2 packages in `src/`. To build only a specific package (and its dependencies):

```bash
ROBOT_PKG=parol6_launch docker compose up -d
```

## Connecting to the Container

**VS Code (recommended):** Open the Command Palette (`Ctrl+Shift+P`) → *Dev Containers: Attach to Running Container* → select `env`.

**Terminal:**
```bash
docker exec -it env bash
```

## Launching the PAROL6 Robot

Inside the container:

```bash
# Simulation (MuJoCo) — default
ros2 launch parol6_launch parol6.launch.py

# Real hardware
ros2 launch parol6_launch parol6.launch.py hardware_interface_type:=real

# Real hardware on a specific serial port
ros2 launch parol6_launch parol6.launch.py hardware_interface_type:=real serial_port:=/dev/ttyUSB0
```

The launch file starts:
- `robot_state_publisher` — publishes URDF to `/robot_description`
- `ros2_control_node` — controller manager with the hardware interface
- `joint_state_broadcaster` — publishes `/joint_states` at 100 Hz
- `joint_trajectory_controller` — accepts trajectory goals

Verify it's working:
```bash
ros2 topic echo /joint_states
```

## Rebuilding After Code Changes

C++ changes (hardware interface, backends) require a rebuild:
```bash
cd /ros2_ws && colcon build --symlink-install
source install/setup.bash
```

Python changes (launch files, configs) take effect immediately thanks to `--symlink-install`.

## Stack Management

```bash
# Stop all services
cd docker && docker compose down

# Rebuild the Docker image (after Dockerfile changes)
docker compose build

# View container logs
docker compose logs -f env
```

## Project Structure

```
src/
├── common/backend/              # Abstract Backend base class + MuJoCo backend
└── parol6/
    ├── parol6_description/      # URDF/xacro, robot-specific config (PD gains)
    ├── parol6_hardware_interface/  # ros2_control SystemInterface plugin
    └── parol6_launch/           # Launch files, controller config
```

## Adding a New Robot

1. Create `src/<robot>/<robot>_hardware_interface/` — implement a backend extending `common::Backend` and a `hardware_interface::SystemInterface`
2. Create `src/<robot>/<robot>_description/` — URDF and robot-specific config
3. Create `src/<robot>/<robot>_launch/` — launch files and controller config
4. The hardware interface selects the backend via the `hardware_interface_type` ROS parameter (`"sim"` or `"real"`)