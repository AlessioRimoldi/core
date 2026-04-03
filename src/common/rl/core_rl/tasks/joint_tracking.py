"""Joint position tracking task.

The agent must move joints to randomly sampled target positions.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from core_rl.robot import RobotConfig
from core_rl.tasks import BaseTask, TaskSpaces, register_task


@register_task("joint_tracking")
class JointTrackingTask(BaseTask):
    """Track random joint position targets.

    Observation: [q, dq, q_target]  (3 * num_joints)
    Action:      [delta_q] or [q_target] depending on action_type config
    Reward:      -||q - q_target||^2 - alpha * ||dq||^2 + bonus
    Done:        when error < threshold or max steps reached
    """

    def __init__(
        self,
        action_type: str = "position",
        reward_scale: float = 1.0,
        velocity_penalty: float = 0.01,
        success_threshold: float = 0.05,
        success_bonus: float = 10.0,
        target_range_fraction: float = 0.8,
    ):
        self.action_type = action_type
        self.reward_scale = reward_scale
        self.velocity_penalty = velocity_penalty
        self.success_threshold = success_threshold
        self.success_bonus = success_bonus
        self.target_range_fraction = target_range_fraction

        self._q_target: np.ndarray | None = None
        self._joint_q_indices: list[int] = []
        self._joint_dof_indices: list[int] = []

    def configure(self, robot: RobotConfig, model: mujoco.MjModel) -> TaskSpaces:
        n = robot.num_joints

        # Resolve MuJoCo joint indices
        self._joint_q_indices = []
        self._joint_dof_indices = []
        for name in robot.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            self._joint_q_indices.append(model.jnt_qposadr[jid])
            self._joint_dof_indices.append(model.jnt_dofadr[jid])

        # Observation: [q, dq, q_target]
        obs_low = np.full(3 * n, -np.inf, dtype=np.float32)
        obs_high = np.full(3 * n, np.inf, dtype=np.float32)
        observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: joint position targets (within limits)
        act_low = np.array(
            [robot.joint_limits[name].lower for name in robot.joint_names],
            dtype=np.float32,
        )
        act_high = np.array(
            [robot.joint_limits[name].upper for name in robot.joint_names],
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        return TaskSpaces(
            observation_space=observation_space,
            action_space=action_space,
            obs_components=["q", "dq", "q_target"],
            action_type=self.action_type,
        )

    def reset(
        self,
        robot: RobotConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        np_random: np.random.Generator,
    ) -> np.ndarray:
        # Sample random target within a fraction of joint limits
        frac = self.target_range_fraction
        targets = []
        for name in robot.joint_names:
            lim = robot.joint_limits[name]
            mid = (lim.lower + lim.upper) / 2.0
            half = (lim.upper - lim.lower) / 2.0 * frac
            targets.append(np_random.uniform(mid - half, mid + half))

        self._q_target = np.array(targets, dtype=np.float32)

        # Reset robot to a random initial position (small perturbation from zero)
        for i, idx in enumerate(self._joint_q_indices):
            lim = robot.joint_limits[robot.joint_names[i]]
            half = (lim.upper - lim.lower) / 2.0 * 0.3
            mid = (lim.lower + lim.upper) / 2.0
            data.qpos[idx] = np_random.uniform(mid - half, mid + half)

        for idx in self._joint_dof_indices:
            data.qvel[idx] = 0.0

        mujoco.mj_forward(model, data)
        return self.compute_observation(robot, model, data)

    def compute_observation(
        self,
        robot: RobotConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> np.ndarray:
        q = np.array([data.qpos[i] for i in self._joint_q_indices], dtype=np.float32)
        dq = np.array([data.qvel[i] for i in self._joint_dof_indices], dtype=np.float32)
        return np.concatenate([q, dq, self._q_target])

    def compute_reward(
        self,
        robot: RobotConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, Any]]:
        q = np.array([data.qpos[i] for i in self._joint_q_indices])
        dq = np.array([data.qvel[i] for i in self._joint_dof_indices])

        pos_error = np.linalg.norm(q - self._q_target)
        vel_norm = np.linalg.norm(dq)

        reward = -(pos_error**2) - self.velocity_penalty * (vel_norm**2)
        reward *= self.reward_scale

        success = pos_error < self.success_threshold
        if success:
            reward += self.success_bonus

        info = {
            "pos_error": float(pos_error),
            "vel_norm": float(vel_norm),
            "success": success,
        }

        return float(reward), False, info
