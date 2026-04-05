"""Joint position tracking task — pure JAX / Brax.

The agent must move joints to randomly sampled target positions.
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
from core_rl.tasks import BaseTask, register_task


@register_task("joint_tracking")
class JointTrackingTask(BaseTask):
    """Track random joint position targets.

    Observation: ``[q, dq, q_target]``  (3 × num_joints)
    Action:      joint position targets (passed through PD control)
    Reward:      ``-||q - q_target||² - α·||dq||² + bonus``

    Episode state (``q_target``, step counter) is stored in
    ``State.info`` so that everything is JIT-traceable.
    """

    def __init__(
        self,
        robot: RobotConfig,
        reward_scale: float = 1.0,
        velocity_penalty: float = 0.01,
        success_threshold: float = 0.05,
        success_bonus: float = 10.0,
        target_range_fraction: float = 0.8,
        max_episode_steps: int = 500,
        backend: str = "mjx",
        n_frames: int = 10,
        **kwargs: Any,
    ):
        super().__init__(robot=robot, backend=backend, n_frames=n_frames, **kwargs)

        self.reward_scale = reward_scale
        self.velocity_penalty = velocity_penalty
        self.success_threshold = success_threshold
        self.success_bonus = success_bonus
        self.target_range_fraction = target_range_fraction
        self.max_episode_steps = max_episode_steps

        # Pre-compute target sampling bounds (JAX arrays)
        mid = (self._q_lower + self._q_upper) / 2.0
        half = (self._q_upper - self._q_lower) / 2.0 * target_range_fraction
        self._target_lo = mid - half
        self._target_hi = mid + half

        # Pre-compute initial position sampling bounds (30% of range)
        init_half = (self._q_upper - self._q_lower) / 2.0 * 0.3
        self._init_lo = mid - init_half
        self._init_hi = mid + init_half

    # -- Brax Env interface -----------------------------------------------

    @property
    def observation_size(self) -> int:
        return 3 * self.robot.num_joints

    @property
    def action_size(self) -> int:
        return self.robot.num_joints

    def reset(self, rng: jax.Array) -> brax_env.State:
        """Pure-functional episode reset."""
        rng, rng_target, rng_init = jax.random.split(rng, 3)
        n = self.robot.num_joints

        # Sample random target within joint limits
        q_target = jax.random.uniform(rng_target, shape=(n,), minval=self._target_lo, maxval=self._target_hi)

        # Sample random initial position
        q_init = jax.random.uniform(rng_init, shape=(n,), minval=self._init_lo, maxval=self._init_hi)

        # Build full q / qd vectors (all DOFs, not just robot joints)
        q = jnp.zeros(self.sys.q_size())
        q = q.at[self._joint_q_indices].set(q_init)
        qd = jnp.zeros(self.sys.qd_size())

        # Initialise physics state via the pipeline
        pipeline_state = self.pipeline_init(q, qd)

        # Build observation
        obs = self._compute_obs(pipeline_state, q_target)

        # Compute initial reward (for State)
        reward = jnp.float32(0.0)
        done = jnp.float32(0.0)

        # Store episode-level state in info
        info = {
            "q_target": q_target,
            "step": jnp.int32(0),
            # Gravity-comp data for post-training collection
            "qfrc_bias": self._get_qfrc_bias(pipeline_state),
            # Brax wrappers expect truncation from the start
            "truncation": jnp.float32(0.0),
        }

        metrics = {
            "pos_error": jnp.float32(0.0),
            "vel_norm": jnp.float32(0.0),
            "success": jnp.float32(0.0),
            "reward": jnp.float32(0.0),
        }

        return brax_env.State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        """Pure-functional environment step with PD control."""
        pipeline_state = state.pipeline_state
        q_target_episode = state.info["q_target"]
        step_count = state.info["step"]

        # Get current joint state
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        qfrc_bias = self._get_qfrc_bias(pipeline_state)

        # Clip action to joint limits (policy outputs position targets)
        action = jnp.clip(action, self._q_lower, self._q_upper)

        # PD control: compute torques from position targets
        ctrl = self._pd_control(q_target=action, q=q, qd=qd, qfrc_bias=qfrc_bias)

        # Step physics (PipelineEnv handles n_frames sub-stepping)
        next_pipeline_state = self.pipeline_step(pipeline_state, ctrl)

        # Compute obs, reward, done
        next_q = self._get_joint_q(next_pipeline_state)
        next_qd = self._get_joint_qd(next_pipeline_state)

        obs = self._compute_obs(next_pipeline_state, q_target_episode)

        pos_error = jnp.linalg.norm(next_q - q_target_episode)
        vel_norm = jnp.linalg.norm(next_qd)

        reward = -(pos_error**2) - self.velocity_penalty * (vel_norm**2)
        reward = reward * self.reward_scale

        success = pos_error < self.success_threshold
        reward = reward + jnp.where(success, self.success_bonus, 0.0)

        next_step = step_count + 1
        done = jnp.where(next_step >= self.max_episode_steps, 1.0, 0.0)

        # Preserve wrapper-injected keys (EpisodeWrapper, AutoResetWrapper)
        info = {**state.info}
        info.update(
            {
                "q_target": q_target_episode,
                "step": next_step,
                "qfrc_bias": self._get_qfrc_bias(next_pipeline_state),
                "truncation": done,
            }
        )

        metrics = {**state.metrics}
        metrics.update(
            {
                "pos_error": pos_error,
                "vel_norm": vel_norm,
                "success": success.astype(jnp.float32),
                "reward": reward,
            }
        )

        return brax_env.State(next_pipeline_state, obs, reward, done, metrics, info)

    # -- internal helpers -------------------------------------------------

    def _compute_obs(self, pipeline_state: brax_base.State, q_target: jax.Array) -> jax.Array:
        """Build observation: [q, dq, q_target]."""
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        return jnp.concatenate([q, qd, q_target])
