# CLAUDE.md — Repository Knowledge Base

## Overview

This repository is a **robotics infrastructure** for training, evaluating, and deploying robot policies. It uses **ROS2 Jazzy** (Ubuntu 24.04 Noble) with `ros2_control`, **MuJoCo** physics simulation, and Docker Compose orchestration. The first supported robot is the **PAROL6** 6-DOF desktop robotic arm.

---

## Architecture

### Docker Compose Stack (`docker/`)
| Service  | Purpose                    | Port  |
|----------|----------------------------|-------|
| `redis`  | State/message store        | 6379  |
| `mlflow` | Experiment tracking        | 5000  |
| `env`    | ROS2 + MuJoCo runtime     | —     |

The `env` container runs on `osrf/ros:jazzy-desktop-full-noble` with NVIDIA GPU support, X11 forwarding, and the full ROS2 desktop stack. It installs `mujoco_ros2_control`, `mujoco_vendor`, `libabsl-dev`, `yaml-cpp`, and clones the PAROL6 URDF from PCrnjak's GitHub repo.

### Backend Pattern (`src/common/backend/`)

All robot communication flows through an abstract **`common::Backend`** base class:

```
common::Backend (abstract)
├── common::MujocoBackend     — MuJoCo physics simulation (any robot)
└── parol6::Parol6Backend     — PAROL6 serial communication (real hardware)
```

**Key types:**
- `MotorCommand { q, dq, tau, kp, kd, enabled }` — commanded joint state
- `MotorState { q, dq, tau, status }` — measured joint state

**Interface:** `init()`, `activate()`, `deactivate()`, `read(vector<MotorState>&)`, `write(vector<MotorCommand>)`, `step(dt)`, `set_controller_active(bool)`

The hardware interface instantiates the correct backend based on the `hardware_interface_type` ROS parameter (`"sim"` or `"real"`).

### Scene Objects (`parol6_launch/config/scene.yaml`)

Scene objects (tables, cubes, etc.) can be injected into the MuJoCo simulation and visualized in RViz via a YAML file. The pipeline is robot-agnostic and runs entirely off the control loop:

1. **Launch file** reads the scene YAML, injects URDF `<link>` + `<joint>` elements into the MuJoCo-only URDF (not the RViz one). Fixed objects get `type="fixed"` joints; dynamic objects get `type="floating"` (6-DOF, affected by gravity/contacts).
2. **MujocoBackend** reads `scene_file_path` from hardware parameters, resolves body IDs via `mj_name2id()`, and publishes TF for each scene body at 10 Hz on a **dedicated background thread** (separate callback group + `SingleThreadedExecutor`). Uses existing `control_mu_` mutex to briefly read `xpos`/`xquat`.
3. **`scene_marker_publisher.py`** — standalone Python node that reads the YAML for shapes/colors, uses TF frames for positions, and publishes a `visualization_msgs/MarkerArray` on `/scene_markers` at 10 Hz.

**Scene YAML format:**
```yaml
objects:
  - name: table
    type: box              # box | sphere | cylinder
    size: [0.6, 0.4, 0.02] # half-extents (box only)
    position: [0.3, 0.0, -0.01]
    orientation: [0.0, 0.0, 0.0]  # rpy
    color: [0.6, 0.4, 0.2, 1.0]  # rgba
    dynamic: false          # true = free-floating, false = fixed (default)
    mass: 0.05              # kg (optional, auto-computed inertia)
```

**Key constraint:** `<mujoco><compiler fusestatic="false"/></mujoco>` is injected into the URDF to prevent MuJoCo from merging fixed-joint bodies into their parent, which would make them invisible to `mj_name2id()`.

### Hardware Interface (`src/parol6/parol6_hardware_interface/`)

A standard `ros2_control` **SystemInterface** plugin. Exports position/velocity/effort state interfaces and position/velocity/effort/p_gain/d_gain command interfaces per joint.

**Lifecycle:**
1. `on_init` — selects backend, loads YAML gains, extracts URDF limits
2. `on_activate` — activates backend (homes robot if real), reads initial state
3. `read()` — copies motor states into hw vectors
4. `write()` — builds `MotorCommand` array, handles gain resolution/clamping, sends to backend, steps physics
5. `perform_command_mode_switch` — defers backend activation by one cycle after controller startup

**CMake target:** `parol6_hardware_interface` (shared library), registered via pluginlib.

---

## PAROL6 Robot Details

### Specifications
- **Type:** 6-DOF desktop stepper motor arm (open-source, by PCrnjak)
- **Controller:** STM32F446RE with TMC5160 stepper drivers
- **Microstepping:** 200 steps/rev × 32 = **6400 steps/rev**
- **Gear ratios:** J1=6.4, J2=20.0, J3=18.0952381, J4=4.0, J5=4.0, J6=10.0

### Serial Protocol (UART over USB, 3 Mbaud)

**Packet framing:**
- Start: `0xFF 0xFF 0xFF`
- Length byte (payload size including end bytes)
- Payload data
- End: `0x01 0x02`

**PC → Robot (52 bytes payload):**
| Offset | Size | Field |
|--------|------|-------|
| 0–17   | 18   | Joint positions (6 × 3-byte signed big-endian, steps) |
| 18–35  | 18   | Joint speeds (6 × 3-byte signed big-endian, steps/s) |
| 36     | 1    | Command byte |
| 37     | 1    | Affected joint bitfield |
| 38     | 1    | I/O bitfield |
| 39     | 1    | Timeout |
| 40–49  | 10   | Gripper fields (position/speed/current/command/mode/ID) |
| 49     | 1    | CRC byte |
| 50–51  | 2    | End bytes (0x01, 0x02) |

**Robot → PC (56 bytes payload):**
| Offset | Size | Field |
|--------|------|-------|
| 0–17   | 18   | Position (6 × 3-byte signed big-endian, steps) |
| 18–35  | 18   | Speed (6 × 3-byte signed big-endian, steps/s) |
| 36     | 1    | Homed bitfield (bit7=J1 … bit2=J6) |
| 37     | 1    | I/O state bitfield |
| 38     | 1    | Temperature error bitfield |
| 39     | 1    | Position error bitfield |
| 40–43  | 4    | Timing data + timeout_error + extra |
| 44–52  | 9    | Gripper fields |
| 53     | 1    | CRC byte |
| 54–55  | 2    | End bytes (0x01, 0x02) |

**Command bytes (decimal):**
| Value | Command |
|-------|---------|
| 101   | ENABLE |
| 102   | DISABLE |
| 100   | HOME |
| 123   | JOG (velocity control) |
| 156   | GO_TO_POSITION (position + velocity tracking) |
| 103   | CLEAR_ERROR |
| 255   | DUMMY (keep-alive, no motion) |

### Unit Conversion
```
steps_per_radian[i] = (6400.0 × gear_ratio[i]) / (2π)
```

### Firmware Behavior
- Command 156 (GO_TO_POSITION): `speed = ((cmd_pos - cur_pos) / 0.01 + cmd_vel) / 2` — blends P-control on position error with velocity feedforward
- Command 123 (JOG): runs at `cmd_vel` directly
- Homing: send CMD_HOME once, then keep sending CMD_DUMMY; firmware continues homing in background

---

## File Map

```
src/
├── common/
│   ├── backend/
│   │   ├── backend.hpp               — Abstract Backend base class + BodyPose struct
│   │   ├── mujoco_backend.hpp        — MuJoCo simulation backend header
│   │   └── mujoco_backend.cpp        — MuJoCo backend implementation + scene TF publisher
│   └── rl/                           — Robot-agnostic RL training pipeline
│       ├── CMakeLists.txt, package.xml
│       ├── config/defaults.yaml
│       └── core_rl/                  — Python package (see RL section below)
└── parol6/
    ├── parol6_description/       — URDF/xacro, robot-specific config (PD gains)
    │   └── config/rl_config.yaml — RL robot config (URDF path, joints, gains)
    ├── parol6_launch/            — Launch files, controller configuration
    │   ├── launch/parol6.launch.py
    │   ├── config/controllers.yaml
    │   ├── config/scene.yaml     — Example scene objects definition
    │   ├── config/parol6.rviz    — RViz2 config (RobotModel + SceneMarkers)
    │   └── scripts/
    │       ├── example_trajectory.py
    │       └── scene_marker_publisher.py  — MarkerArray publisher for scene objects
    └── parol6_hardware_interface/
        ├── CMakeLists.txt
        ├── package.xml
        ├── parol6_hardware_interface_plugin.xml
        ├── include/parol6_hardware_interface/
        │   ├── parol6_hardware_interface.hpp
        │   └── parol6_backend.hpp
        └── src/
            ├── parol6_hardware_interface.cpp
            └── parol6_backend.cpp
```

---

## Build & Run

```bash
# Build and start all services
xhost +local:docker
cd docker && docker compose up --build -d

# ROS packages are built automatically on container startup.
# To rebuild manually inside the container:
docker exec -it env bash
cd /ros2_ws && colcon build --symlink-install
source install/setup.bash
```

---

## Adding a New Robot

1. Create `src/<robot>/<robot>_hardware_interface/` with the same structure as parol6
2. Implement a `<Robot>Backend` extending `common::Backend`
3. Create a hardware interface extending `hardware_interface::SystemInterface`
4. Register via pluginlib; the hardware interface selects backend via `hardware_interface_type` param
5. The `common::Backend` uses `std::vector` for joint state/commands, so it works with any joint count
6. Add `rl_config.yaml` to `<robot>_description/config/` (see PAROL6 as example)

---

## RL Training Pipeline (`src/common/rl/`)

A robot-agnostic RL training system. Uses MuJoCo Python bindings directly (no ROS2) for training throughput, then exports ONNX for deployment via `ros2_control`.

### Architecture

```
train.py CLI → Algorithm (PPO/SAC/...) → SubprocVecEnv (N parallel MujocoRobotEnvs)
                    ↓ callbacks                        ↓ MuJoCo Python (no ROS2)
              Redis Streams + MLflow            PD control + gravity comp (NumPy)

Export: DeployablePolicy(nn.Module) = Normalizer → Policy → GravCompNet → PDController → ONNX
```

### Usage

```bash
# Inside the Docker container (source workspace first)
source /ros2_ws/install/setup.bash

python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --num-envs 8

# With overrides
python -m core_rl.train --robot parol6 --task joint_tracking --algo sac \
  --num-envs 16 --total-timesteps 2000000 --config /path/to/override.yaml

# Disable streaming (offline training)
python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --no-redis --no-mlflow
```

### Robot Config Resolution

Each robot provides `rl_config.yaml` in its description package:
```
src/<robot>/<robot>_description/config/rl_config.yaml
```
Declares: `urdf_path`, `mesh_dir`, `gains_path`, `joint_names`, `uri_strip_patterns`.
`robot.py` reads this, patches the URDF for MuJoCo (strips URIs, injects `fusestatic="false"`), writes a MuJoCo-loadable URDF into the mesh directory, and returns a `RobotConfig` dataclass.

**Important:** Robot resolution is done **once in the parent process** and the `RobotConfig` is passed to `SubprocVecEnv` workers. This avoids race conditions where multiple subprocesses would write the same URDF file simultaneously.

### Task System

Tasks define observation/action spaces, reset logic, and reward functions:
- `BaseTask` ABC with `configure()`, `reset()`, `compute_observation()`, `compute_reward()`
- Registry: `@register_task("name")` decorator, looked up by string
- **`joint_tracking`**: track random target positions; obs = `[q, dq, q_target]`; reward = `-||q - q_target||² - α||dq||²`

### Algorithm System

- `BaseAlgorithm` ABC with `train()`, `save()`, `load()`, `get_policy()`
- Registry: `@register_algorithm("name")`
- **`ppo`**, **`sac`**: thin wrappers around Stable-Baselines3

### ONNX Export

All computation layers baked into a single `DeployablePolicy(nn.Module)`:
1. **ObservationNormalizer** — running mean/std from VecNormalize
2. **Policy MLP** — extracted from SB3 (supports both PPO's `mlp_extractor` + `action_net` and SAC's `actor.latent_pi` + `actor.mu`)
3. **GravityCompensationNet** — learned MLP `(q, dq) → qfrc_bias`
4. **PDController** — `tau = kp*(q_d - q) + kd*(0 - dq) + grav_comp`

GravComp data `(q, dq, qfrc_bias)` is collected during RL rollouts via `GravCompCollectorCallback` and the network is trained supervised post-hoc.

### Streaming

- **Redis Streams** (`rl:train:<experiment>:<run_id>:metrics`): step-level metrics, replay-able
- **MLflow** (`http://mlflow:5000`): experiment tracking, hyperparams, metric curves, ONNX artifacts

### File Map

```
src/common/rl/
├── CMakeLists.txt, package.xml      — Uses ament_cmake_python for correct install paths
├── config/defaults.yaml              — Default training hyperparameters
└── core_rl/
    ├── __init__.py, __main__.py      — Package init + `python -m core_rl` support
    ├── robot.py                      — Robot config resolution (name → URDF + gains + joints)
    ├── env.py                        — MujocoRobotEnv(gymnasium.Env) + make_env() factory
    ├── train.py                      — CLI entry point
    ├── export_onnx.py                — ONNX export with all layers
    ├── algorithms/
    │   ├── __init__.py               — BaseAlgorithm + registry
    │   ├── ppo.py                    — PPO (SB3 wrapper)
    │   └── sac.py                    — SAC (SB3 wrapper)
    ├── tasks/
    │   ├── __init__.py               — BaseTask + registry
    │   └── joint_tracking.py         — Joint position tracking task
    ├── modules/
    │   ├── gravity_comp.py           — Learned gravity compensation MLP
    │   ├── pd_controller.py          — PD controller as nn.Module
    │   ├── normalizer.py             — Observation normalizer
    │   └── deployable.py             — Full deployable pipeline
    └── callbacks/
        ├── redis_stream.py           — Redis Streams metric publishing
        ├── mlflow_logger.py          — MLflow experiment tracking
        └── grav_comp_collector.py    — Collects (q, dq, bias) data
```

---

## Key Design Decisions

- **Vectors over arrays** in the Backend interface: supports robots with different joint counts (PAROL6=6, future robots may differ)
- **Gain resolution with -1 sentinel**: controllers can override per-joint PD gains via command interfaces; -1 means "use YAML default"
- **Deferred backend activation**: `perform_command_mode_switch` sets a pending flag; the backend is activated on the next `write()` cycle so the controller has one cycle to populate commands before the robot starts moving
- **Synchronous serial I/O** for PAROL6: `write()` sends command + reads response; `read()` returns cached state from previous cycle
- **Scene objects off the control loop**: TF publishing for scene bodies uses a dedicated background thread with its own callback group and executor, locking `control_mu_` only briefly to read body poses. Zero impact on the 100 Hz control cycle.
- **`fusestatic="false"`**: Required so MuJoCo preserves fixed-joint scene bodies as separate bodies (default behavior fuses them into the parent, making them inaccessible via `mj_name2id()`)
- **Scene file resolution**: Bare filenames are resolved from `parol6_launch/config/`; absolute paths also accepted
- **`ament_cmake_python`**: The RL package uses `ament_python_install_package()` so `core_rl` is installed to the correct `site-packages` directory on `PYTHONPATH` after sourcing the workspace
- **Pre-resolved RobotConfig**: `resolve_robot()` runs once in the parent process; the `RobotConfig` object is passed to `SubprocVecEnv` workers to avoid file write race conditions
- **SB3 policy extraction for ONNX**: `_ExtractedPolicyNet` detects whether the SB3 policy is on-policy (PPO: `mlp_extractor` + `action_net`) or off-policy (SAC: `actor.latent_pi` + `actor.mu`) and handles both
