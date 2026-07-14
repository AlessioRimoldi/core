"""Ball-in-cup task — pure JAX / Brax port of dm_control's ``ball_in_cup``.

This is the planar (2-D) variant: the cup is driven by two motors (``x``, ``z``)
and the ball hangs from a string (a spatial tendon). Unlike the other tasks in
this package, ball-in-cup is *not* a PAROL6-style robot — it ships its own MuJoCo
model — so it subclasses ``PipelineEnv`` directly instead of the robot-centric
``BaseTask``. The ``robot`` argument is accepted (``make_env`` always passes it)
but ignored.

Observation: ``[qpos(4), qvel(4)]`` = ``[cup_x, cup_z, ball_x, ball_z,
             cup_vx, cup_vz, ball_vx, ball_vz]``  (8 dims)
Action:      2 cup motors, applied directly as ``ctrl`` (ctrlrange [-1, 1]).
Reward:      1.0 while the ball sits inside the cup target, else 0.0. (DADS
             ignores this and recomputes the reward from q_φ; it matters only
             for non-DADS use of the task.)

Episode length / truncation is owned by brax's ``EpisodeWrapper`` (continuing
MDP, no terminal state) — same convention as ``ee_tracking``.
"""

from __future__ import annotations

from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp
import mujoco
from brax import base as brax_base
from brax.io import mjcf as brax_mjcf

from core_rl.tasks import register_task


@register_task("ball_in_cup")
class BallInCupTask(brax_env.PipelineEnv):
    def __init__(
        self,
        robot: Any = None,  # accepted for make_env compatibility; unused
        xml_path: str | None = None,
        backend: str = "mjx",
        n_frames: int = 10,
        hard_init: bool = False,
        **kwargs: Any,  # swallow scene / max_episode_steps / etc.
    ):
        if xml_path is None:
            raise ValueError(
                "ball_in_cup requires `xml_path` — the absolute path to "
                "ball_in_cup.xml (set it in the task_kwargs of your config)."
            )

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        sys = brax_mjcf.load_model(mj_model)
        super().__init__(sys=sys, backend=backend, n_frames=n_frames)

        self._mj_model = mj_model

        # Reward bodies/sites: dm_control's in_target compares the ball BODY to
        # the target SITE in (x, z), with a per-axis box threshold of
        # (target_half - ball_radius) on each axis.
        self._ball_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._target_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target")
        ball_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        ball_radius = float(mj_model.geom_size[ball_geom, 0])
        target_half_xz = mj_model.site_size[self._target_site][[0, 2]]  # box half-sizes (x, z)
        self._catch_thresh_xz = jnp.asarray(target_half_xz - ball_radius, dtype=jnp.float32)

        # ── exploration coverage grid ──
        # A G×G occupancy grid over the ball's (x, z) world position. Each
        # episode carries a visited-cells bitmap in state.info and emits a
        # per-step "ball_coverage" = 1 when the ball enters a not-yet-visited
        # cell, so the Brax evaluator's per-episode SUM equals the number of
        # distinct cells visited that episode (out of G²) — a direct measure of
        # how much of the reachable ball space the policy actually explores.
        # Bounds are generous over the string-limited reachable region; points
        # outside are clipped to the edge cells.
        self._cov_g = 16
        self._cov_xlo, self._cov_xhi = -0.45, 0.45
        self._cov_zlo, self._cov_zhi = 0.2, 0.7
        self._cov_ncells = self._cov_g * self._cov_g

        # Ball world position ↔ obs mapping. The ball body has a static base
        # offset (body pos) plus two slide joints (x, z); obs[2], obs[3] are the
        # joint displacements, so ball world (x, z) = base + displacement. Stored
        # so coverage_cell_from_obs() can reconstruct the ball's cell from a
        # plain observation vector (what the trainers have during a rollout).
        ball_body_pos = mj_model.body_pos[self._ball_body]
        self._ball_x0 = float(ball_body_pos[0])
        self._ball_z0 = float(ball_body_pos[2])

        # ── hard_init ──
        # When True, reset() is deterministic: cup centered, ball at rest hanging
        # straight down at the taut string length, zero velocity. A catch then
        # requires a deliberate pump-and-swing (a random policy essentially never
        # succeeds), making time-to-first-success a real exploration signal — see
        # MULTI_AGENT_DISAGREEMENT_EXPERIMENT.md §6.2. Derived from the model so
        # it stays correct if the geometry/string length is edited.
        self._hard_init = bool(hard_init)
        cup_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        cup_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "cup")
        ball_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        string_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_TENDON, "string")
        cup_site_world = mj_model.body_pos[cup_body] + mj_model.site_pos[cup_site]
        string_len = float(mj_model.tendon_range[string_id][1])
        ball_site_local = mj_model.site_pos[ball_site]
        # Ball site straight below the cup site at full string extension; back out
        # the ball body's slide-joint qpos (world = base + qpos).
        hang_ball_world_x = cup_site_world[0] - ball_site_local[0]
        hang_ball_world_z = cup_site_world[2] - string_len - ball_site_local[2]
        self._hang_qx = float(hang_ball_world_x - ball_body_pos[0])
        self._hang_qz = float(hang_ball_world_z - ball_body_pos[2])

    @property
    def observation_size(self) -> int:
        return self.sys.q_size() + self.sys.qd_size()  # 4 + 4 = 8

    @property
    def action_size(self) -> int:
        return self.sys.act_size()  # 2 cup motors

    @property
    def coverage_num_cells(self) -> int:
        """Number of cells in the ball (x, z) coverage grid (G²)."""
        return self._cov_ncells

    # -- Brax env API ------------------------------------------------------

    def reset(self, rng: jax.Array) -> brax_env.State:
        # qpos = [cup_x, cup_z, ball_x, ball_z].
        q = jnp.zeros(self.sys.q_size())
        if self._hard_init:
            # Deterministic hard start: cup centered, ball hanging straight down
            # at the taut string length (computed in __init__), zero velocity.
            q = q.at[2].set(self._hang_qx)
            q = q.at[3].set(self._hang_qz)
        else:
            # Cup starts at origin; random ball start matching dm_control
            # (ball_x ∈ [-.2, .2], ball_z ∈ [.2, .5]).
            rng, rng_x, rng_z = jax.random.split(rng, 3)
            q = q.at[2].set(jax.random.uniform(rng_x, (), minval=-0.2, maxval=0.2))
            q = q.at[3].set(jax.random.uniform(rng_z, (), minval=0.2, maxval=0.5))
        qd = jnp.zeros(self.sys.qd_size())

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._compute_obs(pipeline_state)

        # Exploration-quality metrics. The Brax evaluator SUMS per-step metrics
        # over an episode, so eval/episode_ball_speed ≈ total ball path and
        # eval/episode_ball_x_abs ≈ Σ|ball_x| (lateral swing) — both measure how
        # much the policy actually moves the (under-actuated) ball, independent
        # of whether it catches it. qpos = [cup_x, cup_z, ball_x, ball_z];
        # qvel index 2:4 is the ball's (vx, vz).
        # Coverage: mark the start cell as visited; the start counts as 1.
        cell = self._ball_cell(pipeline_state)
        visited = jnp.zeros(self._cov_ncells, dtype=bool).at[cell].set(True)

        metrics = {
            "caught": jnp.float32(0.0),
            "reward": jnp.float32(0.0),
            "ball_speed": jnp.float32(0.0),  # qd = 0 at reset
            "ball_x_abs": jnp.abs(pipeline_state.q[2]),
            "ball_coverage": jnp.float32(1.0),  # eval SUM ⇒ distinct cells / ep
        }
        info = {"truncation": jnp.float32(0.0), "coverage_visited": visited}
        return brax_env.State(pipeline_state, obs, jnp.float32(0.0), jnp.float32(0.0), metrics, info)

    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        # Motors expect ctrl in [-1, 1]; apply the action directly (no PD).
        ctrl = jnp.clip(action, -1.0, 1.0)
        next_pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        obs = self._compute_obs(next_pipeline_state)
        caught = self._is_caught(next_pipeline_state)
        reward = caught

        ball_speed = jnp.linalg.norm(next_pipeline_state.qd[2:4])
        ball_x_abs = jnp.abs(next_pipeline_state.q[2])

        # Coverage: 1.0 iff the ball entered a cell not yet visited this episode.
        cell = self._ball_cell(next_pipeline_state)
        visited = state.info["coverage_visited"]
        ball_coverage = (1.0 - visited[cell].astype(jnp.float32)).astype(jnp.float32)
        visited = visited.at[cell].set(True)

        metrics = {
            **state.metrics,
            "caught": caught,
            "reward": reward,
            "ball_speed": ball_speed,
            "ball_x_abs": ball_x_abs,
            "ball_coverage": ball_coverage,
        }
        info = {**state.info, "coverage_visited": visited}
        # Continuing MDP: EpisodeWrapper owns truncation/episode length.
        return state.replace(
            pipeline_state=next_pipeline_state,
            obs=obs,
            reward=reward,
            done=jnp.float32(0.0),
            metrics=metrics,
            info=info,
        )

    # -- helpers (JAX-traceable) -------------------------------------------

    def _compute_obs(self, pipeline_state: brax_base.State) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

    def _xz_to_cell(self, x: jax.Array, z: jax.Array) -> jax.Array:
        """Flat grid-cell index of a ball (x, z) world position (edge-clipped)."""
        fx = (x - self._cov_xlo) / (self._cov_xhi - self._cov_xlo)
        fz = (z - self._cov_zlo) / (self._cov_zhi - self._cov_zlo)
        gx = jnp.clip((fx * self._cov_g).astype(jnp.int32), 0, self._cov_g - 1)
        gz = jnp.clip((fz * self._cov_g).astype(jnp.int32), 0, self._cov_g - 1)
        return gx * self._cov_g + gz

    def _ball_cell(self, pipeline_state: brax_base.State) -> jax.Array:
        """Flat grid-cell index of the ball's (x, z) world position."""
        ball = pipeline_state.xpos[self._ball_body]
        return self._xz_to_cell(ball[0], ball[2])

    def coverage_cell_from_obs(self, obs: jax.Array) -> jax.Array:
        """Flat grid-cell index from a (possibly batched) observation vector.

        obs = [cup_x, cup_z, ball_x, ball_z, ...]; the ball's world (x, z) is the
        body base offset plus the slide-joint displacement. This lets the
        trainers accumulate cumulative coverage straight from rollout
        observations, sharing this task's grid as the single source of truth.
        Trailing dims beyond the first four are ignored, so batched obs of shape
        (N, obs_size) works element-wise and returns (N,).
        """
        world_x = self._ball_x0 + obs[..., 2]
        world_z = self._ball_z0 + obs[..., 3]
        return self._xz_to_cell(world_x, world_z)

    def _is_caught(self, pipeline_state: brax_base.State) -> jax.Array:
        # dm_control in_target: ball BODY vs target SITE, per-axis box test in x, z.
        xz = jnp.array([0, 2])
        ball_xz = pipeline_state.xpos[self._ball_body][xz]
        target_xz = pipeline_state.site_xpos[self._target_site][xz]
        in_box = jnp.all(jnp.abs(target_xz - ball_xz) < self._catch_thresh_xz)
        return in_box.astype(jnp.float32)
