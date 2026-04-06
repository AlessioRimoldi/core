"""Reach-object task — pure JAX / Brax.

The agent must move its end-effector to a target object in the scene.
Target position can be randomized each episode via ``randomize_position``
in the scene YAML.

All methods are pure-functional and JAX-traceable for ``jax.vmap``
vectorisation across thousands of environments.
"""

from __future__ import annotations

from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp
from brax import base as brax_base

from core_rl.robot import RobotConfig
from core_rl.scene import SceneConfig
from core_rl.tasks import BaseTask, register_task


@register_task("reach_object")
class ReachObjectTask(BaseTask):
    """Reach a target object with the robot's end-effector.

    Observation: ``[q, dq, ee_pos, target_pos]``  (2×num_joints + 3 + 3)
    Action:      joint position targets (passed through PD control)
    Reward:      ``-||ee_pos - target_pos||² - α·||dq||² + bonus``

    Requires:
        - ``robot.ee_body`` set in ``rl_config.yaml``
        - A scene with at least one object with ``role: target``
    """

    def __init__(
        self,
        robot: RobotConfig,
        scene: SceneConfig | None = None,
        reward_scale: float = 1.0,
        velocity_penalty: float = 0.01,
        success_threshold: float = 0.03,
        success_bonus: float = 10.0,
        target_range_fraction: float = 0.8,
        max_episode_steps: int = 500,
        backend: str = "mjx",
        n_frames: int = 10,
        **kwargs: Any,
    ):
        if not robot.ee_body:
            raise ValueError("ReachObjectTask requires 'ee_body' in rl_config.yaml")
        if scene is None or not scene.get_by_role("target"):
            raise ValueError("ReachObjectTask requires a scene with at least one object with role='target'")

        super().__init__(robot=robot, backend=backend, n_frames=n_frames, scene=scene, **kwargs)

        self.reward_scale = reward_scale
        self.velocity_penalty = velocity_penalty
        self.success_threshold = success_threshold
        self.success_bonus = success_bonus
        self.max_episode_steps = max_episode_steps

        # Resolve target object (first object with role=target)
        target_obj = scene.get_by_role("target")[0]
        self._target_body_id = self._scene_body_ids[target_obj.name]
        self._target_name = target_obj.name
        self._target_nominal_pos = jnp.array(target_obj.position, dtype=jnp.float32)
        self._target_has_randomization = target_obj.randomize_position is not None

        # Pre-compute initial position sampling bounds (30% of joint range)
        mid = (self._q_lower + self._q_upper) / 2.0
        init_half = (self._q_upper - self._q_lower) / 2.0 * 0.3
        self._init_lo = mid - init_half
        self._init_hi = mid + init_half

    @property
    def observation_size(self) -> int:
        return 2 * self.robot.num_joints + 3 + 3

    @property
    def action_size(self) -> int:
        return self.robot.num_joints

    def reset(self, rng: jax.Array) -> brax_env.State:
        """Pure-functional episode reset."""
        rng, rng_init, rng_scene = jax.random.split(rng, 3)
        n = self.robot.num_joints

        # Sample random initial joint positions
        q_init = jax.random.uniform(rng_init, shape=(n,), minval=self._init_lo, maxval=self._init_hi)

        # Build full q / qd vectors
        q = jnp.zeros(self.sys.q_size())
        q = q.at[self._joint_q_indices].set(q_init)
        qd = jnp.zeros(self.sys.qd_size())

        # Randomize scene object positions (targets, etc.)
        q = self._randomize_scene_q(rng_scene, q)

        # Initialise physics state
        pipeline_state = self.pipeline_init(q, qd)

        # Read target position from physics state (after randomization)
        target_pos = self._get_body_pos(pipeline_state, self._target_body_id)
        ee_pos = self._get_ee_pos(pipeline_state)
        obs = self._compute_obs(pipeline_state, target_pos)

        info = {
            "target_pos": target_pos,
            "step": jnp.int32(0),
            "qfrc_bias": self._get_qfrc_bias(pipeline_state),
            "truncation": jnp.float32(0.0),
        }

        metrics = {
            "ee_dist": jnp.linalg.norm(ee_pos - target_pos),
            "vel_norm": jnp.float32(0.0),
            "success": jnp.float32(0.0),
            "reward": jnp.float32(0.0),
        }

        return brax_env.State(pipeline_state, obs, jnp.float32(0.0), jnp.float32(0.0), metrics, info)

    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        """Pure-functional environment step with PD control."""
        pipeline_state = state.pipeline_state
        target_pos = state.info["target_pos"]
        step_count = state.info["step"]

        # Current joint state
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        qfrc_bias = self._get_qfrc_bias(pipeline_state)

        # Clip action to joint limits
        action = jnp.clip(action, self._q_lower, self._q_upper)

        # PD control
        ctrl = self._pd_control(q_target=action, q=q, qd=qd, qfrc_bias=qfrc_bias)

        # Step physics
        next_pipeline_state = self.pipeline_step(pipeline_state, ctrl)

        # Compute EE position and distance to target
        ee_pos = self._get_ee_pos(next_pipeline_state)
        next_qd = self._get_joint_qd(next_pipeline_state)

        # Re-read target position from physics (for dynamic targets)
        target_pos = self._get_body_pos(next_pipeline_state, self._target_body_id)

        obs = self._compute_obs(next_pipeline_state, target_pos)

        ee_dist = jnp.linalg.norm(ee_pos - target_pos)
        vel_norm = jnp.linalg.norm(next_qd)

        reward = -(ee_dist**2) - self.velocity_penalty * (vel_norm**2)
        reward = reward * self.reward_scale

        success = ee_dist < self.success_threshold
        reward = reward + jnp.where(success, self.success_bonus, 0.0)

        next_step = step_count + 1
        done = jnp.where(next_step >= self.max_episode_steps, 1.0, 0.0)

        info = {**state.info}
        info.update(
            {
                "target_pos": target_pos,
                "step": next_step,
                "qfrc_bias": self._get_qfrc_bias(next_pipeline_state),
                "truncation": done,
            }
        )

        metrics = {**state.metrics}
        metrics.update(
            {
                "ee_dist": ee_dist,
                "vel_norm": vel_norm,
                "success": success.astype(jnp.float32),
                "reward": reward,
            }
        )

        return brax_env.State(next_pipeline_state, obs, reward, done, metrics, info)

    def _compute_obs(self, pipeline_state: brax_base.State, target_pos: jax.Array) -> jax.Array:
        """Build observation: [q, dq, ee_pos, target_pos]."""
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        ee_pos = self._get_ee_pos(pipeline_state)
        return jnp.concatenate([q, qd, ee_pos, target_pos])
