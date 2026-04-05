# CLAUDE.md — Repository Knowledge Base

## Overview

This repository is a **robotics infrastructure** for training, evaluating, and deploying robot policies. It uses **ROS2 Jazzy** (Ubuntu 24.04 Noble) with `ros2_control`, **MuJoCo MJX** (GPU-accelerated physics via JAX) + **Brax** (RL training), and Docker Compose orchestration. The first supported robot is the **PAROL6** 6-DOF desktop robotic arm.

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
# Allow Docker to use the X11 display
xhost +local:docker

# Start all services
./docker.sh start

# Start with GPU acceleration
./docker.sh start -g

# Stop all services
./docker.sh stop

# ROS packages are built automatically on container startup.
# To rebuild manually:
./docker.sh build

# Build specific packages (and their dependencies)
./docker.sh build parol6_launch
./docker.sh build parol6_launch core_rl
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

A robot-agnostic RL training system. Uses **Brax** (Google's physics-based RL library) with **MuJoCo MJX** (GPU-accelerated MuJoCo via JAX) for massively parallel training across thousands of environments on a single GPU. Exports ONNX for deployment via `ros2_control`.

### Architecture

```
train.py CLI → Algorithm (Brax PPO/SAC) → jax.vmap (4096+ parallel envs on GPU)
                    ↓ progress_fn hooks               ↓ MuJoCo MJX (JAX, no ROS2)
              Redis Streams + MLflow         PD control + gravity comp (pure JAX, JIT'd)

                    SimBackend abstraction
                    ├── MJXBackend (current)
                    └── IsaacSim / Newton (future)

Export: make_deployable_fn(JAX) = Normalizer → Policy → GravCompNet → PDController
        → NumPy → PyTorch bridge → ONNX
```

### Key Technologies
- **JAX** — Automatic differentiation, JIT compilation, GPU vectorisation via `jax.vmap`
- **Brax** — RL training loops (`brax.training.agents.ppo`, `brax.training.agents.sac`) with built-in observation normalisation, eval scheduling, and `progress_fn` hooks
- **MuJoCo MJX** — GPU-accelerated MuJoCo via `brax.mjx.pipeline` (replaces CPU-bound MuJoCo Python bindings)
- **Flax Linen** — Neural network definitions (`nn.Module` subclasses)
- **Optax** — JAX optimiser library (Adam, learning rate schedules)
- **PyTorch** — Used **only** for ONNX export (optional `[export]` dependency)

### Usage

```bash
# Inside the Docker container (source workspace first)
source /ros2_ws/install/setup.bash

# Train with 4096 parallel envs on GPU
python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --num-envs 4096

# With overrides
python -m core_rl.train --robot parol6 --task joint_tracking --algo sac \
  --num-envs 8192 --total-timesteps 10000000 --config /path/to/override.yaml

# Disable streaming (offline training)
python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --no-redis --no-mlflow

# Skip ONNX export
python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --no-export
```

### Robot Config Resolution

Each robot provides `rl_config.yaml` in its description package:
```
src/<robot>/<robot>_description/config/rl_config.yaml
```
Declares: `urdf_path`, `mesh_dir`, `gains_path`, `joint_names`, `uri_strip_patterns`.
`robot.py` reads this, patches the URDF for MuJoCo (strips URIs, injects `fusestatic="false"`), writes a MuJoCo-loadable URDF into the mesh directory, and returns a `RobotConfig` dataclass.

### Task System

Tasks define observation/action sizes, reset logic, and reward functions as **Brax `PipelineEnv` subclasses**:
- `BaseTask(PipelineEnv)` with JAX-pure `reset(rng)`, `step(state, action)`. All episode state lives in `State.info` (no mutable instance vars — required for JIT-traceability).
- Helper methods: `_get_joint_q()`, `_get_joint_qd()`, `_get_qfrc_bias()`, `_pd_control()`.
- Registry: `@register_task("name")` decorator, looked up by string.
- **`joint_tracking`**: track random target positions; obs = `[q, dq, q_target]`; reward = `-||q - q_target||² - α||dq||²`. Episode state (`q_target`, `step`, `qfrc_bias`) stored in `State.info`.

### Algorithm System

- `BaseAlgorithm` ABC with `train() → (MakePolicyFn, Params, Metrics)`, `save()`, `load()`, `make_inference_fn()`
- Registry: `@register_algorithm("name")`
- **`ppo`**, **`sac`**: thin wrappers around Brax's built-in training functions (`brax.training.agents.ppo.train()`, `brax.training.agents.sac.train()`)
- Config mapping from familiar names to Brax internals (e.g. `n_steps → unroll_length`, `gamma → discounting`, `clip_range → clipping_epsilon`)

### SimBackend Abstraction

Protocol-based backend system for future multi-simulator support:
- `SimBackend` Protocol: `init(model, q, qd) → State`, `step(model, state, action) → State`
- `@register_backend("name")` decorator with `get_backend()` lookup
- **`MJXBackend`**: wraps `brax.mjx.pipeline.init()` / `brax.mjx.pipeline.step()`
- Future: `IsaacSimBackend`, `NewtonBackend`

### ONNX Export

JAX parameters are extracted and bridged to PyTorch for ONNX export:
1. **Normalizer** — mean/std extracted from Brax's `RunningStatisticsState`
2. **Policy MLP** — weights/biases extracted from Flax params, mapped to `nn.Linear` layers
3. **GravityCompensationNet** — Flax `nn.Module` trained post-hoc with Optax, same extraction
4. **PDController** — `tau = kp*(q_d - q) + kd*(0 - dq) + grav_comp`

GravComp data `(q, dq, qfrc_bias)` is collected by running the trained policy for N steps post-training via `collect_grav_comp_data()`, then a Flax MLP is trained supervised.

### Callbacks / Hooks

Brax uses `progress_fn(step, metrics)` hooks instead of SB3 `BaseCallback` objects:
- `compose_progress_fn(*hooks)` combines multiple hooks with exception isolation
- **`MLflowHook`**: start/log/end lifecycle, artifact logging
- **`RedisStreamHook`**: Redis Streams publishing (`rl:train:<experiment>:<run_id>:metrics`)
- Simple print hook for console progress

### Streaming

- **Redis Streams** (`rl:train:<experiment>:<run_id>:metrics`): step-level metrics, replay-able
- **MLflow** (`http://mlflow:5000`): experiment tracking, hyperparams, metric curves, ONNX artifacts

### File Map

```
src/common/rl/
├── CMakeLists.txt, package.xml      — Uses ament_cmake_python for correct install paths
├── config/defaults.yaml              — Default training hyperparameters (Brax param names)
└── core_rl/
    ├── __init__.py, __main__.py      — Package init + `python -m core_rl` support
    ├── robot.py                      — Robot config resolution (name → URDF + gains + joints)
    ├── env.py                        — make_env() factory → BaseTask (PipelineEnv)
    ├── train.py                      — CLI entry point (Brax training loops)
    ├── export_onnx.py                — ONNX export: Flax → NumPy → PyTorch → ONNX
    ├── backends/
    │   ├── __init__.py               — SimBackend Protocol + registry
    │   └── mjx.py                    — MJX backend (brax.mjx.pipeline)
    ├── algorithms/
    │   ├── __init__.py               — BaseAlgorithm + registry
    │   ├── ppo.py                    — PPO (Brax wrapper)
    │   └── sac.py                    — SAC (Brax wrapper)
    ├── tasks/
    │   ├── __init__.py               — BaseTask(PipelineEnv) + registry
    │   └── joint_tracking.py         — Joint position tracking task (pure JAX)
    ├── modules/
    │   ├── gravity_comp.py           — Learned gravity compensation (Flax + Optax)
    │   ├── pd_controller.py          — PD controller (pure JAX function)
    │   ├── normalizer.py             — Observation normalizer (pure JAX)
    │   └── deployable.py             — Full deployable pipeline (JAX composition)
    └── callbacks/
        ├── __init__.py               — compose_progress_fn() combiner
        ├── redis_stream.py           — Redis Streams hook
        ├── mlflow_logger.py          — MLflow hook
        └── grav_comp_collector.py    — Post-training (q, dq, bias) collection
```

---

## Code Quality & Pre-Commit

The repo uses [pre-commit](https://pre-commit.com/) hooks configured in `.pre-commit-config.yaml`. Run `./scripts/setup.sh` to set up the environment and install hooks.

### Hooks
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-xml**, **check-merge-conflict**, **check-added-large-files** (max 500 KB)
- **black** — Python formatter (line length 120), scoped to `src/common/rl/`
- **ruff** — Python linter with `--fix`, scoped to `src/`. Rules: `E, F, W, I, N, UP, B, SIM, PLC0415` (enforces imports at top of file)
- **clang-format** — C/C++ formatter (Google base style, 120 col, indent 4). Config in `.clang-format`
- **cpplint** — C/C++ linter (120 col, filters: `-build/include_subdir`, `-legal/copyright`, `-build/c++11`, `-build/include_order`, `-runtime/references`)
- **cmake-format** — CMakeLists.txt formatter
- **shellcheck** — Shell linter (`--severity=warning`)

### Config Locations
- `.pre-commit-config.yaml` — hook definitions (repo root, required by pre-commit)
- `.clang-format` — C++ formatting rules (repo root, required by clang-format)
- `pyproject.toml` — Black, Ruff config (`[tool.ruff]`, `[tool.black]` sections)

### Conventions
- All Python imports must be at the top of the file (enforced by `PLC0415`)
- No lazy/deferred imports inside functions
- C++ line length: 120 characters
- Python line length: 120 characters (Black enforced)
- `#include <memory>` must be present when using `unique_ptr` or `make_shared`

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
- **Pre-resolved RobotConfig**: `resolve_robot()` runs once before training; the `RobotConfig` dataclass carries MJCF path, joint names, gains, and limits to all downstream code
- **Brax PipelineEnv**: Tasks extend `brax.envs.PipelineEnv`, which handles MJX model loading, GPU vectorisation via `jax.vmap`, and `pipeline_step()` physics integration
- **Pure JAX JIT-traceability**: All task `reset()` / `step()` functions, PD control, normalisation, and reward computation are pure functions operating on immutable `State` objects — required for `jax.jit` compilation across thousands of envs
- **Episode state in State.info**: Since JIT-compiled functions can't use mutable instance variables, per-episode state (`q_target`, `step_count`, `qfrc_bias`) is carried in the `State.info` dict
- **SimBackend Protocol**: Abstracts physics stepping behind `init()` / `step()` — currently only MJX, designed for future IsaacSim and Newton backends
- **Post-hoc gravity comp**: GravComp data is collected by running the trained policy for N steps after training completes, then a Flax MLP is trained supervised — avoids complicating the JIT'd training loop
- **PyTorch only for ONNX export**: JAX/Flax params are extracted to NumPy, loaded into thin PyTorch `nn.Module` wrappers, and exported via `torch.onnx.export()` — PyTorch is an optional dependency
- **Brax progress_fn hooks**: Replace SB3's `BaseCallback` pattern; `compose_progress_fn()` combines multiple hooks with exception isolation so a Redis failure doesn't crash training
