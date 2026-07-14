"""FetchPush task WITH a goal — push the block to a target position.

Goal-conditioned variant of the goal-free :class:`FetchPushTask` (Plappert et
al. 2018 style): each episode samples a target (x, y) on the table and the
agent is rewarded for pushing the block there. Everything else — action space,
base observation layout, init/reset randomization, coverage/interaction hooks —
is inherited unchanged.

Observation: ``[q, dq, ee_pos, block_pos (, block_quat, block_cvel), goal_pos]``
— the parent's obs with the 3-D goal appended at the END, so all absolute
indices used elsewhere (``ee_lo``, ``block_lo``, ``block_obs_indices``) stay
valid.

Reward (dense, default):

    r = − goal_dist_scale · ‖block_xy − goal_xy‖     (pull block → goal)
        − reach_scale · ‖ee − block‖                 (shaping: find the block; 0 disables)
        + success_bonus · 1[‖block_xy − goal_xy‖ < success_threshold]

Reward (``sparse_reward: true``, Fetch-benchmark convention):

    r = −1[‖block_xy − goal_xy‖ ≥ success_threshold]   (0 at goal, −1 otherwise)

``metrics["success"]`` means "block within success_threshold of the goal"
(the parent's meaning was "ee touching block"), so ``eval_episode_success``
reads as task success.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp

from core_rl.tasks import register_task
from core_rl.tasks.fetchpush import FetchPushTask


@register_task("fetchpush_goal")
class FetchPushGoalTask(FetchPushTask):
    """Push the block to a per-episode random target position on the table."""

    def __init__(
        self,
        *args: Any,
        # Goal (x, y) sampled uniformly in this box. Defaults keep a margin to
        # the walled-scene walls (table interior x∈[0.11,0.49], y∈[−0.35,0.35]).
        goal_lo: Sequence[float] = (0.15, -0.30),
        goal_hi: Sequence[float] = (0.45, 0.30),
        success_threshold: float = 0.05,  # m; Fetch-benchmark standard
        # ── dense reward weights ────────────────────────────────────────────
        goal_dist_scale: float = 1.0,
        reach_scale: float = 0.1,
        success_bonus: float = 5.0,
        sparse_reward: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._goal_lo = jnp.asarray(goal_lo, dtype=jnp.float32)
        self._goal_hi = jnp.asarray(goal_hi, dtype=jnp.float32)
        self.success_threshold = float(success_threshold)
        self.goal_dist_scale = float(goal_dist_scale)
        self.reach_scale = float(reach_scale)
        self.success_bonus = float(success_bonus)
        self.sparse_reward = bool(sparse_reward)

    @property
    def observation_size(self) -> int:
        return super().observation_size + 3  # + goal_pos

    def _goal_reward(self, obs_base: jax.Array, goal: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(reward, block_to_goal_dist, success)`` from the parent obs."""
        ee = obs_base[self._ee_lo : self._ee_lo + 3]
        block = obs_base[self._block_lo : self._block_lo + 3]
        d_goal = jnp.linalg.norm(block[:2] - goal[:2])
        success = (d_goal < self.success_threshold).astype(jnp.float32)
        if self.sparse_reward:
            reward = success - 1.0
        else:
            reward = (
                -self.goal_dist_scale * d_goal
                - self.reach_scale * jnp.linalg.norm(ee - block)
                + self.success_bonus * success
            )
        return reward, d_goal, success

    def reset(self, rng: jax.Array) -> brax_env.State:
        rng, rng_goal = jax.random.split(rng)
        state = super().reset(rng)

        block_pos = state.obs[self._block_lo : self._block_lo + 3]
        goal_xy = jax.random.uniform(rng_goal, shape=(2,), minval=self._goal_lo, maxval=self._goal_hi)
        goal = jnp.concatenate([goal_xy, block_pos[2:3]])  # z = block height (viz only)

        reward, d_goal, success = self._goal_reward(state.obs, goal)
        return state.replace(
            obs=jnp.concatenate([state.obs, goal]),
            info={**state.info, "goal_pos": goal},
            metrics={**state.metrics, "block_to_goal_dist": d_goal, "success": success},
        )

    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        goal = state.info["goal_pos"]
        # Parent step reads only pipeline_state/info (never state.obs), and its
        # `info = {**state.info}` carries goal_pos through to the next step.
        inner = super().step(state, action)

        reward, d_goal, success = self._goal_reward(inner.obs, goal)
        return inner.replace(
            obs=jnp.concatenate([inner.obs, goal]),
            reward=reward,
            metrics={**inner.metrics, "block_to_goal_dist": d_goal, "success": success, "reward": reward},
        )
