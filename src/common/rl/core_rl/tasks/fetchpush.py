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

Action (``action_mode="joint"``, default): per-step joint *deltas* in
``[-1, 1]^num_joints``, scaled by ``max_delta`` (radians) and added to the
current joint pose. This avoids the "absolute-target-jitter" failure mode
where a stochastic policy commands wildly different target positions at every
control step.

Action (``action_mode="ee"``): per-step *end-effector* deltas in
``[-1, 1]^3``, scaled by ``max_ee_delta`` (meters) — the action space of the
original OpenAI FetchPush. A damped-least-squares differential-IK step
(resolved-rate control) maps the Cartesian delta to joint deltas each control
step; the existing PD then tracks the resulting joint target. Exploration in
EE space is geometrically aligned with pushing: a random walk sweeps the
workspace instead of flailing in joint space.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
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
        max_delta: float = 0.20,  # rad per control step (≈ 11.4°/step); also the dq rate cap in ee mode
        # ── Action mode: joint deltas (default) or end-effector deltas ─────
        action_mode: str = "joint",  # "joint" (Δq, num_joints dims) | "ee" (Δee, 3 dims, differential IK)
        max_ee_delta: float = 0.02,  # m per control step (ee mode)
        ik_damping: float = 0.05,  # DLS damping λ — bounds dq near singularities (ee mode)
        ik_posture_gain: float = 0.05,  # nullspace pull toward the init pose, rad/step (ee mode)
        ee_workspace_lo: Sequence[float] = (0.05, -0.40, 0.005),  # EE target box, world frame (ee mode)
        ee_workspace_hi: Sequence[float] = (0.55, 0.40, 0.40),
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
        # Start the EE near the block instead of at joint mid-range (which
        # parks the EE ~35-40cm away). Ignored if init_pose is set explicitly.
        # Solved ONCE at construction via CPU MuJoCo differential IK (see
        # _solve_ee_ik_numpy) — not part of the JAX training graph.
        init_near_target: bool = False,
        init_target_offset: Sequence[float] = (-0.08, 0.0, 0.05),  # world-frame offset from the block's nominal pos
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

        if action_mode not in ("joint", "ee"):
            raise ValueError(f"action_mode must be 'joint' or 'ee', got {action_mode!r}")
        self.action_mode = action_mode
        self.max_ee_delta = float(max_ee_delta)
        self.ik_damping = float(ik_damping)
        self.ik_posture_gain = float(ik_posture_gain)
        self._ws_lo = jnp.asarray(ee_workspace_lo, dtype=jnp.float32)
        self._ws_hi = jnp.asarray(ee_workspace_hi, dtype=jnp.float32)
        # MuJoCo joint ids of the arm joints — xanchor/xaxis are jnt-id-indexed
        # (used by the geometric Jacobian in ee mode). BaseTask already
        # validated that every robot joint exists in the model.
        self._arm_jnt_ids = jnp.array(
            [mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot.joint_names],
            dtype=jnp.int32,
        )

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

        # Init pose: midpoint of joint range (default), user-supplied, or
        # (init_near_target) solved so the EE starts near the block — a random
        # walk from mid-range starts ~35-40cm from the cube (see
        # EXPERIMENT_FETCHPUSH.md's "ignition" discussion: disagreement cannot
        # point toward an object that has never moved, so first contact is an
        # undirected search on a clock; starting closer shrinks that search).
        mid = (self._q_lower + self._q_upper) / 2.0
        if init_pose is not None:
            init_q = jnp.asarray(init_pose, dtype=jnp.float32)
            if init_q.shape != (self.robot.num_joints,):
                raise ValueError(f"init_pose must have shape ({self.robot.num_joints},), " f"got {tuple(init_q.shape)}")
            self._init_q = init_q
        elif init_near_target:
            target = np.asarray(block_obj.position, dtype=np.float64) + np.asarray(init_target_offset, dtype=np.float64)
            self._init_q = self._solve_ee_ik_numpy(target, seed_q=np.asarray(mid))
        else:
            self._init_q = mid
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
        # ee mode: 3-D Cartesian delta; joint mode: one delta per joint.
        return 3 if self.action_mode == "ee" else self.robot.num_joints

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

        # ── Action → joint-space delta ──
        # Both modes command small local moves the PD can track ⇒ actual
        # exploration (instead of jitter around mid_range). ee mode maps the
        # Cartesian delta through one differential-IK step first.
        q_current = self._get_joint_q(pipeline_state)
        if self.action_mode == "ee":
            dq = self._ee_action_to_dq(pipeline_state, q_current, action)
        else:
            dq = self.max_delta * action
        q_target = jnp.clip(q_current + dq, self._q_lower, self._q_upper)

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
    def _solve_ee_ik_numpy(
        self,
        target_pos: np.ndarray,
        seed_q: np.ndarray,
        iters: int = 200,
        damping: float = 0.1,
    ) -> jax.Array:
        """One-time DLS IK solve (CPU MuJoCo, plain NumPy) for a joint pose
        whose EE sits at ``target_pos``. Runs ONCE at task construction
        (``init_near_target``) — not JIT-traced, not part of the training
        graph. Same damped-least-squares formula as :meth:`_ee_action_to_dq`,
        iterated to convergence here instead of taken as one training step.
        """
        d = mujoco.MjData(self._mj_model)
        q = np.array(seed_q, dtype=np.float64)
        q_idx = np.asarray(self._joint_q_indices)
        jnt_ids = np.asarray(self._arm_jnt_ids)
        lower = np.asarray(self._q_lower, dtype=np.float64)
        upper = np.asarray(self._q_upper, dtype=np.float64)
        # Start from the model's default qpos (valid unit quaternions for any
        # free-jointed scene objects) so mj_kinematics never sees degenerate
        # state; only the arm's own slots are overwritten each iteration.
        full_q = np.array(self._mj_model.qpos0, dtype=np.float64)
        for _ in range(iters):
            full_q[q_idx] = q
            d.qpos[:] = full_q
            mujoco.mj_kinematics(self._mj_model, d)
            ee = d.xpos[self._ee_body_id].copy()
            err = target_pos - ee
            if np.linalg.norm(err) < 1e-4:
                break
            anchors = d.xanchor[jnt_ids]
            axes = d.xaxis[jnt_ids]
            jac = np.cross(axes, ee - anchors).T  # (3, n)
            a_inv = np.linalg.inv(jac @ jac.T + damping**2 * np.eye(3))
            dq = jac.T @ a_inv @ err
            q = np.clip(q + dq, lower, upper)
        return jnp.asarray(q, dtype=jnp.float32)

    def _ee_action_to_dq(
        self,
        pipeline_state: brax_base.State,
        q_current: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Resolved-rate (differential-IK) action: Cartesian Δee → joint deltas.

        ``action ∈ [-1, 1]³`` commands an end-effector step
        ``Δee = max_ee_delta · action``, clamped so the commanded EE target
        stays inside the workspace box. One damped-least-squares step maps it
        to joint space:

            dq = Jᵀ (J Jᵀ + λ² I₃)⁻¹ Δee  +  (I − J⁺J) · k_post (q_init − q)

        ``J`` is the geometric position Jacobian assembled from the CURRENT
        state — revolute joint i contributes column ``axis_i × (ee − anchor_i)``,
        and ``xaxis``/``xanchor`` are already computed by MJX kinematics, so
        this costs no extra FK pass. The damping bounds dq near singularities;
        the nullspace term keeps the 3 redundant DOFs near the init posture
        instead of drifting. dq is finally rate-capped at ±``max_delta`` — the
        same per-step joint budget as the joint mode.
        """
        ee = self._get_ee_pos(pipeline_state)
        ee_target = jnp.clip(ee + self.max_ee_delta * action, self._ws_lo, self._ws_hi)
        delta = ee_target - ee

        anchors = pipeline_state.xanchor[self._arm_jnt_ids]  # (n, 3) world joint anchors
        axes = pipeline_state.xaxis[self._arm_jnt_ids]  # (n, 3) world joint axes
        jac = jnp.cross(axes, ee - anchors).T  # (3, n)

        a_inv = jnp.linalg.inv(jac @ jac.T + (self.ik_damping**2) * jnp.eye(3))
        jac_pinv = jac.T @ a_inv  # damped pseudoinverse (n, 3)
        dq = jac_pinv @ delta
        nullspace = jnp.eye(q_current.shape[0]) - jac_pinv @ jac
        dq = dq + nullspace @ (self.ik_posture_gain * (self._init_q - q_current))
        return jnp.clip(dq, -self.max_delta, self.max_delta)

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
