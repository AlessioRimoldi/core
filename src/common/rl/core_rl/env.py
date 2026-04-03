"""MuJoCo-based Gymnasium environment for robot RL training.

Uses MuJoCo Python bindings directly (no ROS2) for maximum throughput.
Mirrors the PD control + gravity compensation logic from the C++ MujocoBackend.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from core_rl.robot import RobotConfig, resolve_robot
from core_rl.tasks import BaseTask, get_task


class MujocoRobotEnv(gym.Env):
    """Gymnasium environment wrapping MuJoCo for any robot.

    Physics loop:
        1. Receive action from policy
        2. Apply PD control torques: tau = kp*(q_d - q) + kd*(dq_d - dq) + gravity_comp
        3. Step MuJoCo (sub-stepping at physics_dt within control_dt)
        4. Compute observation and reward via the Task

    The PD + gravity compensation logic mirrors ``mujoco_backend.cpp::control()``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: RobotConfig | str,
        task: BaseTask | str,
        control_dt: float = 0.01,
        physics_dt: float = 0.001,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        scene_file: str = "",
        task_kwargs: dict | None = None,
    ):
        super().__init__()

        # Resolve robot config if given as string
        if isinstance(robot, str):
            robot = resolve_robot(robot, scene_file=scene_file)
        self.robot = robot

        # Resolve task if given as string
        if isinstance(task, str):
            task = get_task(task, **(task_kwargs or {}))
        self.task = task

        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)

        # Enforce integrator and timestep (matching C++ backend)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.timestep = self.physics_dt

        # Add joint armature for stability (matching C++ backend)
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                dof = self.model.jnt_dofadr[i]
                if self.model.dof_armature[dof] < 0.01:
                    self.model.dof_armature[dof] = 0.01

        self.data = mujoco.MjData(self.model)

        # Build joint index maps
        self._joint_q_indices: list[int] = []
        self._joint_dof_indices: list[int] = []
        for name in self.robot.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            self._joint_q_indices.append(self.model.jnt_qposadr[jid])
            self._joint_dof_indices.append(self.model.jnt_dofadr[jid])

        # Configure task (sets obs/action spaces)
        self._task_spaces = self.task.configure(self.robot, self.model)
        self.observation_space = self._task_spaces.observation_space
        self.action_space = self._task_spaces.action_space

        # PD gains from robot config
        self._kp = np.array([self.robot.gains[n].kp for n in self.robot.joint_names], dtype=np.float64)
        self._kd = np.array([self.robot.gains[n].kd for n in self.robot.joint_names], dtype=np.float64)

        # Number of physics substeps per control step
        self._n_substeps = max(1, round(self.control_dt / self.physics_dt))

        # Gravity compensation data buffer (for training GravCompNet)
        self._grav_comp_buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

        # Renderer (lazy init)
        self._renderer = None

    def enable_grav_comp_collection(self, max_size: int = 500_000):
        """Enable collection of (q, dq, qfrc_bias) tuples for gravity comp training."""
        self._grav_comp_buffer = []
        self._grav_comp_max_size = max_size

    def get_grav_comp_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return collected gravity compensation data as (q, dq, qfrc_bias) arrays."""
        if not self._grav_comp_buffer:
            return None
        q_all = np.array([x[0] for x in self._grav_comp_buffer])
        dq_all = np.array([x[1] for x in self._grav_comp_buffer])
        bias_all = np.array([x[2] for x in self._grav_comp_buffer])
        return q_all, dq_all, bias_all

    def _apply_pd_control(self, action: np.ndarray):
        """Apply PD control + gravity compensation.

        Mirrors ``mujoco_backend.cpp::control()``:
            tau = kp * (q_d - q) + kd * (0 - dq) + qfrc_bias
        """
        q = np.array([self.data.qpos[i] for i in self._joint_q_indices])
        dq = np.array([self.data.qvel[i] for i in self._joint_dof_indices])

        # Gravity compensation: qfrc_bias includes gravity + Coriolis
        grav_comp = np.array([self.data.qfrc_bias[i] for i in self._joint_dof_indices])

        # Collect data for GravCompNet training
        if self._grav_comp_buffer is not None and len(self._grav_comp_buffer) < self._grav_comp_max_size:
            self._grav_comp_buffer.append(
                (
                    q.astype(np.float32).copy(),
                    dq.astype(np.float32).copy(),
                    grav_comp.astype(np.float32).copy(),
                )
            )

        # PD control based on action type
        if self._task_spaces.action_type == "position":
            q_target = action
            tau = self._kp * (q_target - q) + self._kd * (0.0 - dq) + grav_comp
        elif self._task_spaces.action_type == "velocity":
            dq_target = action
            tau = self._kp * (0.0 - q) + self._kd * (dq_target - dq) + grav_comp
        elif self._task_spaces.action_type == "torque":
            tau = action  # Direct torque control — no PD
        else:
            raise ValueError(f"Unknown action type: {self._task_spaces.action_type}")

        # Apply torques
        for i, dof_idx in enumerate(self._joint_dof_indices):
            self.data.qfrc_applied[dof_idx] = tau[i]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)

        # Apply PD control and sub-step physics
        self._apply_pd_control(action)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute observation and reward
        obs = self.task.compute_observation(self.robot, self.model, self.data)
        reward, terminated, info = self.task.compute_reward(self.robot, self.model, self.data, action)
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        obs = self.task.reset(self.robot, self.model, self.data, self.np_random)
        return obs, {}

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            if self.render_mode == "human":
                self._renderer = mujoco.viewer.launch_passive(self.model, self.data)
            elif self.render_mode == "rgb_array":
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        if self.render_mode == "human":
            self._renderer.sync()
            return None
        elif self.render_mode == "rgb_array":
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            if self.render_mode == "human":
                self._renderer.close()
            self._renderer = None


def make_env(
    robot: RobotConfig | str,
    task_name: str,
    seed: int = 0,
    control_dt: float = 0.01,
    physics_dt: float = 0.001,
    max_episode_steps: int = 500,
    scene_file: str = "",
    task_kwargs: dict | None = None,
):
    """Factory function for creating environments (used by SubprocVecEnv).

    Pass a pre-resolved ``RobotConfig`` when using ``SubprocVecEnv`` to avoid
    race conditions from multiple subprocesses writing the same URDF file.
    """

    def _init():
        env = MujocoRobotEnv(
            robot=robot,
            task=task_name,
            control_dt=control_dt,
            physics_dt=physics_dt,
            max_episode_steps=max_episode_steps,
            scene_file=scene_file,
            task_kwargs=task_kwargs,
        )
        env.reset(seed=seed)
        return env

    return _init
