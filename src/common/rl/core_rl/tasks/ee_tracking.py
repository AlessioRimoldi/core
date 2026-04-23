from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp
from brax import base as brax_base

from core_rl.robot import RobotConfig
from core_rl.tasks import BaseTask, register_task


@register_task("ee_tracking")
class EETrackingTask(BaseTask):
    """
    End effector (ee) tracking task

    Obs: [Joint angle, Joint velocity, ee_pos, ee_target]
        [q, dq, ee_error]

    Act: [q] Joint position targets for PD controller
    """

    def __init__(
        self,
        robot: RobotConfig,
        max_episode_steps: int = 500,
        success_threshold: float = 0.01,
        succes_bonus: float = 0.0,
        velocity_penalty: float = 1e-3,
        backend: str = "mjx",
        n_frames: int = 10,
        **kwargs: Any,
    ):
        super().__init__(robot=robot, backend=backend, n_frames=n_frames, **kwargs)

        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.velocity_penalty = velocity_penalty
        self.succes_bonus = succes_bonus

        self.mid_range = (self._q_lower + self._q_upper) / 2.0
        self.half_range = (self._q_upper - self._q_lower) / 2.0

    @property
    def observation_size(self) -> int:
        return 2 * self.robot.num_joints + 3

    @property
    def action_size(self) -> int:
        return self.robot.num_joints

    def reset(self, rng: jax.Array) -> brax_env.State:
        """
        Sample random ee_target within the possible range of targets
        Sample random init pos
        """
        rng, rng_init_q, rng_target_ee = jax.random.split(rng, 3)

        # Choose a valid ee_target by using a random inital position
        # and taking the ee pos a target pos
        q_target_joints = jax.random.uniform(
            rng_target_ee, shape=(self.robot.num_joints,), minval=self._q_lower, maxval=self._q_upper
        )

        q_target = jnp.zeros(self.sys.q_size())
        q_target = q_target.at[self._joint_q_indices].set(q_target_joints)

        ee_target_pipeline = self.pipeline_init(q_target, jnp.zeros(self.sys.qd_size()))
        ee_target = self._get_ee_pos(ee_target_pipeline)

        # Get initial robot arm pos
        q_init = jax.random.uniform(
            rng_init_q, shape=(self.robot.num_joints,), minval=self._q_lower, maxval=self._q_upper
        )

        q = jnp.zeros(self.sys.q_size())
        q = q.at[self._joint_q_indices].set(q_init)

        pipeline_state = self.pipeline_init(q, jnp.zeros(self.sys.qd_size()))

        obs = self._compute_obs(pipeline_state, ee_target)

        reward = jnp.float32(0.0)
        done = jnp.float32(0.0)

        # Store episode-level state in info
        info = {
            "ee_target": ee_target,
            "action": jnp.zeros(self.robot.num_joints),
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
        """
        Get current joint states
        Comput torques from position targets
        Physics step
        Compute obs, reward, done
        ?Preserve wrapper-injected keys?
        """
        pipeline_state = state.pipeline_state
        info = {**state.info}
        metrics = {**state.metrics}

        # Scale action to joint limits [-1,1] -> [q_lower,q_upper]
        action = self.mid_range + action * self.half_range

        next_pipeline_step = self.pipeline_step_pd(pipeline_state, action)

        next_ee_pos = self._get_ee_pos(next_pipeline_step)
        next_qd = self._get_joint_qd(next_pipeline_step)
        obs = self._compute_obs(next_pipeline_step, info["ee_target"])

        pos_error = jnp.linalg.norm(next_ee_pos - info["ee_target"])
        vel_norm = jnp.linalg.norm(next_qd)

        success = pos_error < self.success_threshold
        reward = self._compute_reward_2(next_pipeline_step, info["ee_target"])

        step_count = info["step"] + 1
        done = jnp.where(step_count >= self.max_episode_steps, 1.0, 0.0)

        info.update(
            {
                "action": action,
                "step": step_count,
                "qfrc_bias": self._get_qfrc_bias(next_pipeline_step),
                "truncation": done,
            }
        )

        metrics.update(
            {
                "pos_error": pos_error,
                "vel_norm": vel_norm,
                "success": success.astype(jnp.float32),
                "reward": reward,
            }
        )

        return brax_env.State(next_pipeline_step, obs, reward, done, metrics, info)

    def _compute_reward_1(self, pipeline_state: brax_base.State, ee_target: jax.Array) -> jax.Array:
        reward = jnp.linalg.norm(self._get_ee_pos(pipeline_state) - ee_target)
        return -(reward**2)

    def _compute_reward_2(self, pipeline_state: brax_base.State, ee_target: jax.Array) -> jax.Array:
        dist = jnp.linalg.norm(self._get_ee_pos(pipeline_state) - ee_target)
        dist_reward = jnp.exp(-dist / 0.05)

        qd = self._get_joint_qd(pipeline_state)
        vel_norm = jnp.linalg.norm(qd)
        vel_penalty = vel_norm * self.velocity_penalty

        return dist_reward - vel_penalty

    def _compute_reward_3(self, pipeline_state: brax_base.State, ee_target: jax.Array) -> jax.Array:
        pos_error = jnp.linalg.norm(self._get_ee_pos(pipeline_state) - ee_target)
        dist_penalty = pos_error**2

        qd = self._get_joint_qd(pipeline_state)
        vel_norm = (jnp.linalg.norm(qd)) ** 2
        vel_penalty = vel_norm * self.velocity_penalty

        success = pos_error < self.success_threshold
        reward = -dist_penalty - vel_penalty
        reward = reward + jnp.where(success, self.succes_bonus, 0.0)

        return reward

    def _compute_obs(self, pipeline_state: brax_base.State, ee_target: jax.Array) -> jax.Array:
        """Build observation: [q, dq, ee_error]."""
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        ee_pos = self._get_ee_pos(pipeline_state)
        ee_error = ee_target - ee_pos
        return jnp.concatenate([q, qd, ee_error])
