"""Resample-on-done auto-reset for DADS.

Brax's stock ``AutoResetWrapper`` caches the *first* reset state and, on
``done``, restores it — so the skill ``z`` and the initial pose are frozen for
the entire run (every episode replays the same start). DADS needs a fresh ``z``
(and a freshly randomized init pose) every episode.

``ResampleAutoResetWrapper`` keeps the same control flow as the stock wrapper
but, instead of a cached state, carries a per-env RNG in ``info['reset_rng']``
and, on ``done``, splices in a genuinely new ``reset()`` (new ``z``, new pose,
reset per-task step counter). The RNG is split forward every step, so each
episode gets a new draw rather than repeating the same one.

EpisodeWrapper's own bookkeeping keys (``steps``, ``truncation``,
``episode_done``, ``episode_metrics``) are left untouched — exactly as the stock
auto-reset does — so eval metric aggregation is unaffected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import (
    DomainRandomizationVmapWrapper,
    EpisodeWrapper,
    VmapWrapper,
)

# info keys owned by EpisodeWrapper / this wrapper — never spliced from a reset.
_RESERVED_INFO_KEYS = (
    "steps",
    "truncation",
    "episode_done",
    "episode_metrics",
    "reset_rng",
)


class ResampleAutoResetWrapper(Wrapper):
    """Auto-reset that re-runs ``reset()`` on done (fresh ``z`` + init pose)."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["reset_rng"] = rng
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Mirror AutoResetWrapper: zero the EpisodeWrapper step counter for envs
        # that finished last step, then clear done before stepping.
        if "steps" in state.info:
            steps = jnp.where(state.done, jnp.zeros_like(state.info["steps"]), state.info["steps"])
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        # Advance the per-env RNG and build a fresh reset for ALL envs; it is
        # only spliced where `done` (so the cost is one reset per step, the
        # price of correct per-episode randomization).
        rng = state.info["reset_rng"]
        split = jax.vmap(jax.random.split)(rng)  # (num_envs, 2, 2)
        next_rng, reset_rng = split[:, 0], split[:, 1]
        reset_state = self.env.reset(reset_rng)

        done = state.done

        def where_done(x, y):
            d = done
            if d.shape and d.shape[0] != x.shape[0]:
                return y
            if d.shape:
                d = jnp.reshape(d, [x.shape[0]] + [1] * (x.ndim - 1))
            return jnp.where(d, x, y)

        pipeline_state = jax.tree.map(where_done, reset_state.pipeline_state, state.pipeline_state)
        obs = where_done(reset_state.obs, state.obs)

        # Splice the inner env's reset info (z, pose-derived state, ee_target,
        # per-task step counter, …) where done; leave EpisodeWrapper's own keys.
        info = dict(state.info)
        for key, reset_val in reset_state.info.items():
            if key in _RESERVED_INFO_KEYS:
                continue
            info[key] = jax.tree.map(where_done, reset_val, info[key])
        info["reset_rng"] = next_rng

        return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)


def wrap_for_dads(
    env: Env,
    episode_length: int,
    action_repeat: int = 1,
    randomization_fn=None,
) -> Wrapper:
    """Like ``brax.envs.training.wrap`` but with resample-on-done auto-reset.

    Drop-in replacement for ``envs.training.wrap`` (same call signature, used
    via ``_dads_sac.train(wrap_env_fn=...)``). The only difference is the
    auto-reset: a fresh skill ``z`` and initial pose every episode.
    """
    env = VmapWrapper(env) if randomization_fn is None else DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    return ResampleAutoResetWrapper(env)
