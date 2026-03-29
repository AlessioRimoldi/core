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

# Simulation with scene objects (table, cubes, etc.)
ros2 launch parol6_launch parol6.launch.py scene_file:=scene.yaml

# Real hardware
ros2 launch parol6_launch parol6.launch.py hardware_interface_type:=real

# Real hardware on a specific serial port
ros2 launch parol6_launch parol6.launch.py hardware_interface_type:=real serial_port:=/dev/ttyUSB0

# Disable RViz auto-launch
ros2 launch parol6_launch parol6.launch.py rviz:=false
```

The launch file starts:
- `robot_state_publisher` — publishes URDF to `/robot_description`
- `ros2_control_node` — controller manager with the hardware interface
- `joint_state_broadcaster` — publishes `/joint_states` at 100 Hz
- `joint_trajectory_controller` — accepts trajectory goals
- `rviz2` — pre-configured 3D visualization (disable with `rviz:=false`)
- `scene_marker_publisher` — publishes `/scene_markers` MarkerArray (when `scene_file` is set)

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

## RL Training

A robot-agnostic RL pipeline using MuJoCo Python bindings directly (no ROS2 overhead) for training throughput, with ONNX export for deployment.

```bash
# Attach to the container
docker exec -it env bash

# Source the workspace (required for core_rl to be importable)
source /ros2_ws/install/setup.bash

# Train with PPO (8 parallel envs, streams to Redis + MLflow)
python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --num-envs 8

# Train with SAC, custom timesteps
python -m core_rl.train --robot parol6 --task joint_tracking --algo sac \
  --num-envs 16 --total-timesteps 2000000

# Offline training (no Redis/MLflow)
python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo \
  --no-redis --no-mlflow --total-timesteps 50000
```

Training outputs are saved to `/ros2_ws/core/models/` (mounted at `docker/data/models/` on the host) and include:
- SB3 model checkpoint
- VecNormalize statistics
- ONNX policy (with normalizer, policy MLP, gravity compensation, and PD controller baked in)

Monitor training at [http://localhost:5000](http://localhost:5000) (MLflow UI).

### Adding a New Robot for RL

1. Create `src/<robot>/<robot>_description/config/rl_config.yaml` declaring `urdf_path`, `mesh_dir`, `gains_path`, `joint_names`, and `uri_strip_patterns`
2. See `src/parol6/parol6_description/config/rl_config.yaml` as a reference

### Adding a New Task

Create a Python file in `src/common/rl/core_rl/tasks/` with a `@register_task("name")` decorated class extending `BaseTask`. See `joint_tracking.py` as a reference.

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
    └── parol6_launch/           # Launch files, controller config, scene objects
        ├── config/scene.yaml    # Example scene definition
        └── scripts/             # scene_marker_publisher, example_trajectory
```

## Scene Objects

Define objects in a YAML file to inject them into the MuJoCo simulation and visualize them in RViz:

```yaml
objects:
  - name: table
    type: box                    # box | sphere | cylinder
    size: [0.6, 0.4, 0.02]      # half-extents (box)
    position: [0.3, 0.0, -0.01]
    color: [0.6, 0.4, 0.2, 1.0] # rgba
    dynamic: false               # fixed in place (default)

  - name: red_cube
    type: box
    size: [0.025, 0.025, 0.025]
    position: [0.25, 0.1, 0.035]
    color: [0.9, 0.1, 0.1, 1.0]
    dynamic: true                # affected by gravity & contacts
    mass: 0.05                   # kg (optional)
```

Pass the filename (resolved from `parol6_launch/config/`) or an absolute path:
```bash
ros2 launch parol6_launch parol6.launch.py scene_file:=scene.yaml
```

## Adding a New Robot

1. Create `src/<robot>/<robot>_hardware_interface/` — implement a backend extending `common::Backend` and a `hardware_interface::SystemInterface`
2. Create `src/<robot>/<robot>_description/` — URDF and robot-specific config
3. Create `src/<robot>/<robot>_launch/` — launch files and controller config
4. The hardware interface selects the backend via the `hardware_interface_type` ROS parameter (`"sim"` or `"real"`)