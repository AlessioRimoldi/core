"""FetchPush task (GOAL-FREE) — pure JAX / Brax.

Inspired by FetchPush from the Fetch robotics benchmark (Plappert et al.
2018), but with the GOAL REMOVED: the arm just pushes a small block around
the table. This is a hard-exploration task for the disagreement experiment —
unlike a free-floating ball, the block only moves when the arm executes a
*coherent, committed push*, so naive high-entropy flailing barely explores
the block's state space. It is the "random exploration is bad" task that
distinguishes genuine directed multi-head exploration from "more action
noise".

Used with ``--algo disagreement_multi`` / ``disagreement`` and ``ext_coeff=0``
(pure exploration): the env reward is logged for eval only; the policy trains
on the ensemble-disagreement reward. The headline exploration signals come
from the task's opt-in hooks:

  * ``coverage_cumulative`` / ``state_entropy`` over the 2-D table area the
    BLOCK has been pushed across (block x, y — ``coverage_cell_from_obs``).
  * ``interaction_cumulative`` — lifetime count of timesteps the end-effector
    is in contact with the block (``interaction_from_obs``).

Observation layout: ``[q, dq, ee_pos, block_pos]``
For PAROL6 (num_joints=6) this is ``6 + 6 + 3 + 3 = 18`` dims, with
``block_pos`` at indices ``[15, 16, 17] = obs[..., -3:]`` and ``ee_pos`` at
``[12, 13, 14] = obs[..., -6:-3]``.

Action: per-step joint *deltas* in ``[-1, 1]^num_joints``, scaled by
``max_delta`` (radians) and added to the current joint pose. This avoids the
"absolute-target-jitter" failure mode where a stochastic policy commands
wildly different target positions at every control step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp
from brax import base as brax_base

from core_rl.robot import RobotConfig
from core_rl.scene import SceneConfig
from core_rl.tasks import BaseTask, register_task


@register_task("fetchpush")
class FetchPushTask(BaseTask):
    """Push a block around a table (goal-free FetchPush, for exploration)."""

    def __init__(
        self,
        robot: RobotConfig,
        scene: SceneConfig | None = None,
        # ── Action: delta-position in joint space ──────────────────────────
        max_delta: float = 0.20,  # rad per control step (≈ 11.4°/step)
        # ── Reward shaping (eval-only under ext_coeff=0) ───────────────────
        proximity_reward_scale: float = 1.0,
        proximity_sigma: float = 0.05,
        velocity_penalty: float = 0.01,
        action_rate_penalty: float = 0.05,
        # ── Block interaction (touch) proxy ────────────────────────────────
        interaction_distance: float = 0.06,  # m; ‖ee − block‖ below this = touch
        # ── Block (x, y) coverage grid over the table ──────────────────────
        coverage_grid_size: int = 16,
        coverage_lo: Sequence[float] = (-0.30, -0.40),
        coverage_hi: Sequence[float] = (0.90, 0.40),
        # ── Episode + init pose ────────────────────────────────────────────
        max_episode_steps: int = 200,
        init_pose: Sequence[float] | None = None,
        init_noise: float = 0.10,  # ± rad uniform jitter on init q
        # ── Richer block state (harder for a forward model to predict → more
        #    "curious" per Pathak 2019: a manipulated object should be a salient,
        #    complex source of prediction disagreement). Appends the block's
        #    orientation (quat, 4) + spatial velocity (cvel, 6) to the obs. ──
        rich_block_obs: bool = False,
        backend: str = "mjx",
        n_frames: int = 10,
        **kwargs: Any,
    ):
        if not robot.ee_body:
            raise ValueError("FetchPushTask requires 'ee_body' in rl_config.yaml")
        if scene is None or not scene.get_by_role("target"):
            raise ValueError(
                "FetchPushTask requires a scene with at least one object "
                "with role='target' (the block to push). Use "
                "fetchpush_scene.yaml or set role: target on the block in "
                "your scene yaml."
            )

        super().__init__(robot=robot, backend=backend, n_frames=n_frames, scene=scene, **kwargs)

        self.max_delta = float(max_delta)

        self.proximity_reward_scale = float(proximity_reward_scale)
        self.proximity_sigma = float(proximity_sigma)
        self.velocity_penalty = float(velocity_penalty)
        self.action_rate_penalty = float(action_rate_penalty)
        self.interaction_distance = float(interaction_distance)
        self.max_episode_steps = int(max_episode_steps)

        # The block object in the scene (same naming convention as PushBall:
        # role="target" = the manipulable object).
        block_obj = scene.get_by_role("target")[0]
        self._block_body_id = self._scene_body_ids[block_obj.name]
        self._block_name = block_obj.name

        # Init pose: midpoint of joint range (or user-supplied) + uniform jitter.
        mid = (self._q_lower + self._q_upper) / 2.0
        if init_pose is None:
            self._init_q = mid
        else:
            init_q = jnp.asarray(init_pose, dtype=jnp.float32)
            if init_q.shape != (self.robot.num_joints,):
                raise ValueError(f"init_pose must have shape ({self.robot.num_joints},), " f"got {tuple(init_q.shape)}")
            self._init_q = init_q
        self._init_noise = float(init_noise)

        # --- exploration coverage grid over the 2-D table area (block x, y) ---
        # Opt-in API read by the disagreement trainers to log coverage_cumulative
        # + state_entropy over where the BLOCK is pushed (obs[..., -3:-1] = block
        # world x, y). The block stays on the table plane, so a 2-D grid (G²) is
        # the right coverage space. Bounds default to the table extent and are
        # edge-clipped (out-of-range falls into the boundary cells).
        self._cov_g = int(coverage_grid_size)
        self._cov_lo = jnp.asarray(coverage_lo, dtype=jnp.float32)
        self._cov_hi = jnp.asarray(coverage_hi, dtype=jnp.float32)
        self._cov_ncells = self._cov_g**2

        self._rich_block_obs = bool(rich_block_obs)
        # Absolute obs indices (robust to the appended rich-block dims — the base
        # obs layout [q, dq, ee_pos, block_pos] is fixed at the front).
        n = self.robot.num_joints
        self._ee_lo = 2 * n  # ee_pos  = obs[ee_lo : ee_lo+3]
        self._block_lo = 2 * n + 3  # block_pos = obs[block_lo : block_lo+3]
        # Which obs dims describe the BLOCK (for object-focused disagreement).
        # pos(3) [+ quat(4) + cvel(6) when rich]. Read by the disagreement wrapper.
        n_block = 3 + (10 if self._rich_block_obs else 0)
        self._block_obs_indices = list(range(self._block_lo, self._block_lo + n_block))

    # ── Brax Env interface ─────────────────────────────────────────────
    @property
    def observation_size(self) -> int:
        # [q, dq, ee_pos, block_pos] (+ block_quat(4) + block_cvel(6) if rich)
        return 2 * self.robot.num_joints + 3 + 3 + (10 if self._rich_block_obs else 0)

    @property
    def block_obs_indices(self) -> list[int]:
        """Obs indices describing the block (pos [+ quat + cvel] when rich).

        Lets the disagreement trainer weight/restrict the novelty bonus to the
        cube's state so the agent gets curious about the OBJECT, not its own arm.
        """
        return list(self._block_obs_indices)

    @property
    def action_size(self) -> int:
        return self.robot.num_joints

    # ── exploration hooks (read by the disagreement trainers) ───────────
    @property
    def coverage_num_cells(self) -> int:
        """Number of cells in the block (x, y) table coverage grid (G²)."""
        return self._cov_ncells

    def coverage_cell_from_obs(self, obs: jax.Array) -> jax.Array:
        """Flat 2-D grid-cell index of the block from a (batched) observation.

        block (x, y) are ``obs[..., block_lo : block_lo+2]`` (absolute indices, so
        this is robust to the appended rich-block dims). Returns (N,) for (N, obs).
        """
        block_xy = obs[..., self._block_lo : self._block_lo + 2]
        frac = (block_xy - self._cov_lo) / (self._cov_hi - self._cov_lo)
        g = jnp.clip((frac * self._cov_g).astype(jnp.int32), 0, self._cov_g - 1)
        return g[..., 0] * self._cov_g + g[..., 1]

    def interaction_from_obs(self, obs: jax.Array) -> jax.Array:
        """1.0 where the end-effector is in contact with the block, else 0.0.

        Proximity proxy: ‖ee_pos − block_pos‖ < interaction_distance, using
        absolute obs indices (robust to the appended rich-block dims). The
        disagreement trainers sum this over all rollouts → interaction_cumulative.
        """
        ee = obs[..., self._ee_lo : self._ee_lo + 3]
        block = obs[..., self._block_lo : self._block_lo + 3]
        dist = jnp.linalg.norm(ee - block, axis=-1)
        return (dist < self.interaction_distance).astype(jnp.float32)

    def reset(self, rng: jax.Array) -> brax_env.State:
        rng, rng_init, rng_scene = jax.random.split(rng, 3)
        n = self.robot.num_joints

        # 1) Initial joint pose (midpoint + uniform noise, clipped to limits).
        noise = jax.random.uniform(rng_init, shape=(n,), minval=-self._init_noise, maxval=self._init_noise)
        q_init = jnp.clip(self._init_q + noise, self._q_lower, self._q_upper)

        q = jnp.zeros(self.sys.q_size())
        q = q.at[self._joint_q_indices].set(q_init)
        qd = jnp.zeros(self.sys.qd_size())

        # 2) Scene-level randomization (block start jitter via randomize_position).
        q = self._randomize_scene_q(rng_scene, q)

        pipeline_state = self.pipeline_init(q, qd)

        block_pos = self._get_body_pos(pipeline_state, self._block_body_id)
        ee_pos = self._get_ee_pos(pipeline_state)
        obs = self._compute_obs(pipeline_state, block_pos)

        ee_to_block_dist = jnp.linalg.norm(ee_pos - block_pos)
        touching = (ee_to_block_dist < self.interaction_distance).astype(jnp.float32)

        info = {
            "prev_q_target": q_init,  # for action-rate penalty
            "qfrc_bias": self._get_qfrc_bias(pipeline_state),
            "truncation": jnp.float32(0.0),
        }
        metrics = {
            "ee_to_block_dist": ee_to_block_dist,
            "touching": touching,
            "success": touching,  # "success" = first contact
            "reward": jnp.float32(0.0),
        }
        return brax_env.State(pipeline_state, obs, jnp.float32(0.0), jnp.float32(0.0), metrics, info)

    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        pipeline_state = state.pipeline_state
        prev_q_target = state.info["prev_q_target"]

        # ── Action: delta-position in joint space ──
        # Random initial actions ⇒ small local moves the PD can track ⇒
        # actual exploration (instead of jitter around mid_range).
        q_current = self._get_joint_q(pipeline_state)
        q_target = jnp.clip(q_current + self.max_delta * action, self._q_lower, self._q_upper)

        next_pipeline_state = self.pipeline_step_pd(pipeline_state, q_target)

        next_qd = self._get_joint_qd(next_pipeline_state)
        ee_pos = self._get_ee_pos(next_pipeline_state)
        block_pos = self._get_body_pos(next_pipeline_state, self._block_body_id)
        obs = self._compute_obs(next_pipeline_state, block_pos)

        # ── Dense reward (goal-free): stay near the block + smoothness. Used
        #    for EVAL logging only when ext_coeff=0 (pure exploration). ──
        ee_to_block_dist = jnp.linalg.norm(ee_pos - block_pos)
        proximity = jnp.exp(-(ee_to_block_dist**2) / (self.proximity_sigma**2))
        vel_pen = self.velocity_penalty * (jnp.linalg.norm(next_qd) ** 2)
        action_rate = jnp.linalg.norm(q_target - prev_q_target) ** 2
        rate_pen = self.action_rate_penalty * action_rate
        reward = self.proximity_reward_scale * proximity - vel_pen - rate_pen

        touching = (ee_to_block_dist < self.interaction_distance).astype(jnp.float32)

        # Continuing task: no MDP terminal. The time limit is a *truncation*
        # owned by brax's EpisodeWrapper (see ee_tracking for the rationale) —
        # always return done=0 so the time-limit transition bootstraps its value.
        done = jnp.float32(0.0)

        info = {**state.info}
        info.update(
            {
                "prev_q_target": q_target,
                "qfrc_bias": self._get_qfrc_bias(next_pipeline_state),
                "truncation": done,
            }
        )
        metrics = {**state.metrics}
        metrics.update(
            {
                "ee_to_block_dist": ee_to_block_dist,
                "touching": touching,
                "success": touching,
                "reward": reward,
            }
        )

        return brax_env.State(next_pipeline_state, obs, reward, done, metrics, info)

    # ── helpers ────────────────────────────────────────────────────────
    def _compute_obs(
        self,
        pipeline_state: brax_base.State,
        block_pos: jax.Array,
    ) -> jax.Array:
        """Observation = ``[q, dq, ee_pos, block_pos]`` (18 dims for PAROL6),
        with ``[block_quat(4), block_cvel(6)]`` appended when ``rich_block_obs``.
        """
        q = self._get_joint_q(pipeline_state)
        qd = self._get_joint_qd(pipeline_state)
        ee_pos = self._get_ee_pos(pipeline_state)
        parts = [q, qd, ee_pos, block_pos]
        if self._rich_block_obs:
            # Orientation (quat) + spatial velocity (cvel = [angular(3), linear(3)]),
            # both body-id-indexed like xpos. Off-center pushes tumble the cube →
            # these are genuinely hard to predict → a persistent disagreement source.
            block_quat = self._get_body_quat(pipeline_state, self._block_body_id)
            block_cvel = pipeline_state.cvel[self._block_body_id]
            parts += [block_quat, block_cvel]
        return jnp.concatenate(parts)
