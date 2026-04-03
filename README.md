# CORE: Robotics Training & Simulation Stack

Infrastructure for training, evaluating, and deploying robot policies using **ROS2 Jazzy**, **MuJoCo** physics simulation, and **Docker Compose**.

## Prerequisites

- Docker & Docker Compose
- X11 (for MuJoCo rendering)
- *(Optional)* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration

Make sure your user is in the `docker` group so you can run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
```

> **Note:** Log out and back in for the group change to take effect.

## Quick Start

After cloning the repo, run the setup script to create a virtual environment, install dependencies, and configure pre-commit hooks:

```bash
 bash scripts/setup.sh
```

This will:
1. Create a `.venv/` Python virtual environment
2. Install project dependencies from `pyproject.toml`
3. Install and activate [pre-commit](https://pre-commit.com/) hooks

Pre-commit runs automatically on every `git commit`. To run all checks manually:

```bash
source .venv/bin/activate
pre-commit run --all-files
```
After this run:

```bash
# Make scripts executable
chmod +x docker.sh scripts/setup.sh

# Allow Docker to use the X11 display
xhost +local:docker

# Start all services (Redis, MLflow, env)
./docker.sh start

# Start with GPU acceleration
./docker.sh start -g
```

This starts three services:

| Service | Purpose | Port |
|---------|---------|------|
| `redis` | State/message store | 6379 |
| `mlflow` | Experiment tracking UI | 5000 |
| `env` | ROS2 + MuJoCo runtime | — |

On first startup, the `env` container automatically builds **all** ROS2 packages in `src/`. To speed up the initial build, you can limit it to a specific package (and its dependencies) by setting `ROBOT_PKG`:

```bash
ROBOT_PKG=parol6_launch ./docker.sh start
```

For subsequent rebuilds after code changes, use `./docker.sh build` instead (see [Rebuilding After Code Changes](#rebuilding-after-code-changes)).

To stop the network and remove the containers run

```bash
# Stop all services
./docker.sh stop
```


## Connecting to the Container

**VS Code (recommended):** Open the Command Palette (`Ctrl+Shift+P`) → *Dev Containers: Attach to Running Container* → select `env`.

**Terminal:**
```bash
docker exec -it env bash
```

## Launching the a Robot

Inside the container:

```bash
# Simulation (MuJoCo) — default
ros2 launch [robot-name]_launch parol6.launch.py

# Simulation with scene objects (table, cubes, etc.)
ros2 launch [robot-name]_launch parol6.launch.py scene_file:=scene.yaml

# Real hardware
ros2 launch [robot-name]_launch [robot-name].launch.py hardware_interface_type:=real

# Real hardware on a specific serial port (if allowed, e.g. parol6)
ros2 launch  [robot-name].launch.py hardware_interface_type:=real serial_port:=/dev/ttyUSB0

# Disable RViz auto-launch
ros2 launch [robot-name]_launch [robot-name].launch.py rviz:=false
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
# Build all packages
./docker.sh build

# Build specific packages (and their dependencies)
./docker.sh build parol6_launch
./docker.sh build parol6_launch core_rl
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

### Code Quality Hooks

| Hook | Language | Purpose |
|------|----------|---------------------------------------------|
| `trailing-whitespace` | All | Remove trailing whitespace |
| `end-of-file-fixer` | All | Ensure files end with a newline |
| `black` | Python | Auto-format Python code (line length 120) |
| `ruff` | Python | Linting, import sorting, style enforcement |
| `clang-format` | C++ | Auto-format C/C++ (Google style, 120 cols) |
| `cpplint` | C++ | Google-style C++ linting |
| `cmake-format` | CMake | Auto-format CMakeLists.txt |
| `shellcheck` | Shell | Lint shell scripts |

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
