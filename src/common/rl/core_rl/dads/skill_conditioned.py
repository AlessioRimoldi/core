"""Skill-conditioned wrapper task — required for DADS-style training.

Wraps a base task to:
- sample a skill z ~ U(-1, 1)^D on reset
- append z to the observation (so the policy sees it)
- expose two restricted views of state in info:
    * ``s_input``  — q_φ's conditioning input (what determines Δ_target)
    * ``s_target`` — q_φ's prediction target (what skills control)
  Decoupling them is "Option A": let q_φ predict a low-variance ee-space
  target from inputs that physically determine it (q, dq).
- emit reward = 0 (DADS recomputes the reward from q_φ in the trainer)

Within an episode, z is *fixed*. A fresh z is drawn on every reset (which
includes AutoResetWrapper-driven resets at episode end).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp

from core_rl.robot import RobotConfig
from core_rl.tasks import BaseTask, get_task, register_task


@register_task("skill_conditioned")
class SkillConditionedTask(BaseTask):
    def __init__(
        self,
        robot: RobotConfig,
        skill_size: int = 2,
        base_task=None,
        input_obs_indices: Sequence[int] | None = None,
        target_obs_indices: Sequence[int] | None = None,
        backend: str = "mjx",
        n_frames: int = 10,
        **base_task_kwargs: Any,
    ):
        self._base = get_task(base_task, robot=robot, n_frames=n_frames, backend=backend, **base_task_kwargs)
        self._skill_size = skill_size
        if input_obs_indices is None:
            raise ValueError(
                "skill_conditioned requires `input_obs_indices` — the base-obs "
                "indices that q_φ conditions on (e.g. [q, dq] = 0..11 for ee_tracking)."
            )
        if target_obs_indices is None:
            raise ValueError(
                "skill_conditioned requires `target_obs_indices` — the base-obs "
                "indices q_φ predicts the Δ of (e.g. ee_pos = [12,13,14])."
            )
        self._input_idx = jnp.array(input_obs_indices, dtype=jnp.int32)
        self._target_idx = jnp.array(target_obs_indices, dtype=jnp.int32)

        # ── exploration-hook delegation ─────────────────────────────────────
        # The disagreement trainers log coverage_cumulative / state_entropy /
        # interaction_cumulative from opt-in task hooks that read the obs with
        # negative indices (e.g. block xy = obs[..., -3:-1]). We append z to the
        # obs, which shifts those indices, so delegate to the BASE task's hooks
        # with z stripped first. Only bind a hook if the base defines it, so the
        # trainer's `callable(...)` detection stays honest for base tasks that
        # don't expose coverage/interaction (e.g. ee_tracking).
        if hasattr(self._base, "coverage_cell_from_obs"):
            self.coverage_cell_from_obs = functools.partial(self._delegate_hook, "coverage_cell_from_obs")
        if hasattr(self._base, "interaction_from_obs"):
            self.interaction_from_obs = functools.partial(self._delegate_hook, "interaction_from_obs")

    def _delegate_hook(self, name: str, obs: jax.Array) -> jax.Array:
        """Call ``base.<name>(obs)`` after stripping the appended skill z."""
        return getattr(self._base, name)(obs[..., : -self._skill_size])

    @property
    def coverage_num_cells(self) -> int:
        """Block (x, y) coverage grid size from the base task (0 ⇒ no coverage)."""
        return int(getattr(self._base, "coverage_num_cells", 0) or 0)

    @property
    def block_obs_indices(self) -> list[int] | None:
        """Object obs-dim indices from the base task (for object-focused novelty).

        z is appended AFTER the base obs, so the base indices are unchanged.
        Returns None if the base task doesn't expose object indices.
        """
        return getattr(self._base, "block_obs_indices", None)

    # Brax PipelineEnv interface — delegate to base
    @property
    def sys(self):
        return self._base.sys

    @property
    def backend(self):
        return self._base.backend

    @property
    def robot(self):
        return self._base.robot

    @property
    def scene(self):
        # Delegate the base task's scene so consumers that read it off the env
        # (e.g. the video recorder's skybox-background) see it through the wrapper.
        return getattr(self._base, "scene", None)

    @property
    def observation_size(self) -> int:
        return self._base.observation_size + self._skill_size

    @property
    def action_size(self) -> int:
        return self._base.action_size

    # Sizes consumed by the DADS algorithm to build q_φ.
    @property
    def input_obs_size(self) -> int:
        return int(self._input_idx.shape[0])

    @property
    def target_obs_size(self) -> int:
        return int(self._target_idx.shape[0])

    # helpers
    def _sample_skill(self, rng: jax.Array) -> jax.Array:
        return jax.random.uniform(rng, (self._skill_size,), minval=-1.0, maxval=1.0)

    def _select_input(self, base_obs: jax.Array) -> jax.Array:
        return base_obs[..., self._input_idx]

    def _select_target(self, base_obs: jax.Array) -> jax.Array:
        return base_obs[..., self._target_idx]

    # Brax env API
    def reset(self, rng: jax.Array) -> brax_env.State:
        rng_base, rng_z = jax.random.split(rng)
        base_state = self._base.reset(rng_base)

        z = self._sample_skill(rng_z)
        new_obs = jnp.concatenate([base_state.obs, z], axis=-1)
        s_input = self._select_input(base_state.obs)
        s_target = self._select_target(base_state.obs)

        # Everything DADS-specific lives in info so it survives the EpisodeWrapper
        # / AutoResetWrapper. Anything that goes through `extra_fields` in
        # actor_step MUST be present here as well (otherwise the very first
        # post-reset transition would miss the key).
        info = {
            **base_state.info,
            "z": z,
            "s_input": s_input,
            # First-step placeholders; overwritten on the very next step().
            "s_input_next": s_input,
            "s_target": s_target,
            "s_target_next": s_target,
        }

        return base_state.replace(
            obs=new_obs,
            reward=jnp.float32(0.0),
            info=info,
        )

    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        # Within an episode z is held fixed (DADS paper default).
        z = state.info["z"]

        # Strip the skill from obs before forwarding to the base task — the
        # base task's step expects only its own obs layout.
        base_state_in = state.replace(obs=state.obs[..., : -self._skill_size])
        next_base_state = self._base.step(base_state_in, action)

        # Rebuild the skill-conditioned obs.
        new_obs = jnp.concatenate([next_base_state.obs, z], axis=-1)

        # Plumb both PRE- and POST-step restricted views into info. Brax's
        # actor_step copies these from next_state.info into
        # transitions.extras['state_extras'] — see _dads_sac.py:get_experience.
        s_input = self._select_input(base_state_in.obs)
        s_input_next = self._select_input(next_base_state.obs)
        s_target = self._select_target(base_state_in.obs)
        s_target_next = self._select_target(next_base_state.obs)

        # Use next_base_state.info so wrapper-injected keys (truncation, step)
        # come from the post-step state, not pre-step.
        info = {
            **next_base_state.info,
            "z": z,
            "s_input": s_input,
            "s_input_next": s_input_next,
            "s_target": s_target,
            "s_target_next": s_target_next,
        }

        return next_base_state.replace(
            obs=new_obs,
            reward=jnp.float32(0.0),  # DADS reward is computed in the trainer
            info=info,
        )
