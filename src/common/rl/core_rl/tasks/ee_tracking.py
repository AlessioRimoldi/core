from collections.abc import Sequence
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

    Obs: [Joint angle, Joint velocity, ee_pos]   (15 dims for PAROL6)
    Act: [q] Joint position targets for PD controller

    Initial pose:
        Defaults to the joint mid-range (a safe neutral pose) plus a small
        per-joint uniform noise of ±``init_noise`` rad. This is required
        for DADS-style training: with fully random ``q_init`` (the older
        behavior) the policy has to first stabilise from a wildly
        different starting state before producing skill-specific motion —
        which usually consumes the entire episode budget on stabilisation
        and leads to mode collapse.

        Override ``init_pose`` to pin a custom home pose, or set
        ``init_noise=0.0`` for a strictly fixed start. Set
        ``random_init=True`` to instead sample ``q_init`` uniformly over the
        full joint range [q_lower, q_upper] (a *completely random* start —
        ``init_pose``/``init_noise`` are then ignored). Useful for probing
        skill robustness to arbitrary initial states.
    """

    def __init__(
        self,
        robot: RobotConfig,
        max_episode_steps: int = 500,
        success_threshold: float = 0.01,
        succes_bonus: float = 1.0,
        tracking_reward: float = 10,
        velocity_penalty: float = 1e-4,
        torque_penalty: float = 3e-5,
        action_rate_penalty: float = 5e-2,
        acceleration_penalty: float = 3e-6,
        init_pose: Sequence[float] | None = None,
        init_noise: float = 0.05,
        random_init: bool = False,
        coverage_grid_size: int = 10,
        coverage_lo: Sequence[float] = (-0.5, -0.5, -0.1),
        coverage_hi: Sequence[float] = (0.5, 0.5, 0.8),
        backend: str = "mjx",
        n_frames: int = 10,
        **kwargs: Any,
    ):
        super().__init__(robot=robot, backend=backend, n_frames=n_frames, **kwargs)

        # NOTE: `max_episode_steps` is accepted (make_env always passes it) but
        # intentionally unused — episode length / truncation is owned by brax's
        # EpisodeWrapper. ee_tracking is a continuing MDP with no terminal state.
        self.success_threshold = success_threshold
        self.tracking_reward = tracking_reward
        self.velocity_penalty = velocity_penalty
        self.succes_bonus = succes_bonus
        self.torque_penalty = torque_penalty
        self.action_rate_penalty = action_rate_penalty
        self.acceleration_penalty = acceleration_penalty

        self.mid_range = (self._q_lower + self._q_upper) / 2.0
        self.half_range = (self._q_upper - self._q_lower) / 2.0

        # Resolve initial pose: explicit `init_pose` or joint mid-range.
        if init_pose is None:
            self._init_q = jnp.asarray(self.mid_range, dtype=jnp.float32)
        else:
            init_q = jnp.asarray(init_pose, dtype=jnp.float32)
            if init_q.shape != (self.robot.num_joints,):
                raise ValueError(f"init_pose must have shape ({self.robot.num_joints},), " f"got {tuple(init_q.shape)}")
            self._init_q = init_q
        self._init_noise = float(init_noise)
        self._random_init = bool(random_init)

        # --- exploration coverage grid over the 3D end-effector workspace ---
        # Opt-in API read by the disagreement trainers to log coverage_cumulative
        # + state_entropy over where the EE actually goes (obs[..., -3:] = ee
        # world x, y, z). Bounds are a generous PAROL6 box and are edge-clipped,
        # so out-of-range positions fall into the boundary cells; tighten via
        # task_kwargs once you see the realised EE range for sharper resolution.
        self._cov_g = int(coverage_grid_size)
        self._cov_lo = jnp.asarray(coverage_lo, dtype=jnp.float32)
        self._cov_hi = jnp.asarray(coverage_hi, dtype=jnp.float32)
        self._cov_ncells = self._cov_g**3

    @property
    def observation_size(self) -> int:
        return 2 * self.robot.num_joints + 3

    @property
    def action_size(self) -> int:
        return self.robot.num_joints

    @property
    def coverage_num_cells(self) -> int:
        """Number of cells in the EE (x, y, z) coverage grid (G³)."""
        return self._cov_ncells

    def coverage_cell_from_obs(self, obs: jax.Array) -> jax.Array:
        """Flat grid-cell index from a (possibly batched) observation vector.

        obs = [q, dq, ee_pos]; the last three dims are the end-effector world
        (x, y, z). The disagreement trainers call this to accumulate cumulative
        coverage / state entropy straight from rollout observations. Batched obs
        of shape (N, obs_size) works element-wise and returns (N,).
        """
        ee = obs[..., -3:]
        frac = (ee - self._cov_lo) / (self._cov_hi - self._cov_lo)
        g = jnp.clip((frac * self._cov_g).astype(jnp.int32), 0, self._cov_g - 1)
        return (g[..., 0] * self._cov_g + g[..., 1]) * self._cov_g + g[..., 2]

    def reset(self, rng: jax.Array) -> brax_env.State:
        """
        Sample random ee_target within the possible range of targets.
        Start every episode from a (near-)fixed home pose — see class
        docstring for the rationale.
        """
        rng, rng_init_q, rng_target_ee = jax.random.split(rng, 3)

        # Choose a valid ee_target by using a random joint configuration
        # and taking its forward-kinematics ee pos as the target.
        q_target_joints = jax.random.uniform(
            rng_target_ee, shape=(self.robot.num_joints,), minval=self._q_lower, maxval=self._q_upper
        )

        q_target = jnp.zeros(self.sys.q_size())
        q_target = q_target.at[self._joint_q_indices].set(q_target_joints)

        ee_target_pipeline = self.pipeline_init(q_target, jnp.zeros(self.sys.qd_size()))
        ee_target = self._get_ee_pos(ee_target_pipeline)

        # Initial robot pose. Either:
        #  - completely random: q_init ~ U(q_lower, q_upper) per joint, or
        #  - fixed home pose (self._init_q) ± per-joint uniform noise of
        #    ±init_noise rad, clipped to the joint limits.
        if self._random_init:
            q_init = jax.random.uniform(
                rng_init_q,
                shape=(self.robot.num_joints,),
                minval=self._q_lower,
                maxval=self._q_upper,
            )
        else:
            noise = jax.random.uniform(
                rng_init_q,
                shape=(self.robot.num_joints,),
                minval=-self._init_noise,
                maxval=self._init_noise,
            )
            q_init = jnp.clip(self._init_q + noise, self._q_lower, self._q_upper)

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
            "prev_qd": jnp.zeros(self.robot.num_joints),
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
        reward = self._compute_reward_4(next_pipeline_step, info["ee_target"], info["action"], action, info["prev_qd"])

        # ee_tracking is a CONTINUING task — it has no true MDP terminal. The
        # time limit is a *truncation*, owned by brax's EpisodeWrapper (it sets
        # done + truncation=1 at episode_length). If the task also set done=1 at
        # its own max_episode_steps, EpisodeWrapper's `truncation = 1 - done`
        # would compute 0 at the boundary → the time-limit transition gets
        # mislabeled as a hard terminal (no value bootstrap) AND the q_φ
        # truncation mask never fires. So always return done=0 here.
        done = jnp.float32(0.0)

        info.update(
            {
                "action": action,
                "prev_qd": self._get_joint_qd(pipeline_state),
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
        dist_reward = jnp.exp(-dist / 0.15)

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

    def _compute_reward_4(
        self,
        pipeline_state: brax_base.State,
        ee_target: jax.Array,
        prev_action: jax.Array,
        action: jax.Array,
        prev_qd: jax.Array,
    ) -> jax.Array:
        dist = jnp.linalg.norm(self._get_ee_pos(pipeline_state) - ee_target)
        tracking = jnp.exp(-dist / 0.15)

        action_rate = jnp.linalg.norm(action - prev_action) ** 2

        qd = self._get_joint_qd(pipeline_state)
        qdd = (qd - prev_qd) / self.dt
        accel_penalty = jnp.linalg.norm(qdd) ** 2

        tau = pipeline_state.qfrc_actuator[self._joint_dof_indices]
        torque_penalty = jnp.linalg.norm(tau) ** 2

        reward = (
            self.tracking_reward * tracking
            - self.torque_penalty * torque_penalty
            - self.action_rate_penalty * action_rate
            - self.acceleration_penalty * accel_penalty
        )

        return reward

    def _compute_obs(self, pipeline_state: brax_base.State, ee_target: jax.Array) -> jax.Array:
        """Build observation: [q, dq, ee_pos]."""
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        ee_pos = self._get_ee_pos(pipeline_state)
        # ee_error = ee_target - ee_pos
        return jnp.concatenate([q, qd, ee_pos])
