"""Gravity compensation data collector for post-training supervised learning.

With Brax/MJX, ``qfrc_bias`` is stored in ``State.info["qfrc_bias"]`` at
every step for free (it's part of the JIT-compiled rollout).  This module
provides a ``policy_params_fn`` hook that collects ``(q, dq, qfrc_bias)``
data from eval rollouts after training completes, then trains the gravity
compensation MLP.

For the main training loop, data collection happens passively inside the
task's ``step()`` (which writes ``qfrc_bias`` into ``State.info``).
Post-training, we do a separate eval pass to collect a clean dataset.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from core_rl.tasks import BaseTask


def collect_grav_comp_data(
    env: BaseTask,
    make_policy_fn,
    params: Any,
    num_steps: int = 10_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the trained policy in the env and collect gravity comp data.

    Returns ``(q, dq, qfrc_bias)`` as NumPy arrays, each shape ``(N, num_joints)``.
    Data is collected across all vectorised environments.

    Args:
        env: The Brax task environment.
        make_policy_fn: Function returned by Brax ``train()`` — maps params to policy.
        params: Trained parameters (JAX pytree).
        num_steps: Number of env steps to collect.
        seed: Random seed.
    """
    policy = make_policy_fn(params, deterministic=True)

    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    state = env.reset(rng_reset)

    q_list = []
    dq_list = []
    bias_list = []

    q_idx = env._joint_q_indices
    dof_idx = env._joint_dof_indices

    for _ in range(num_steps):
        rng, rng_act = jax.random.split(rng)
        action = policy(state.obs, rng_act)
        state = env.step(state, action)

        # Extract from pipeline_state (on device)
        q = state.pipeline_state.q[q_idx]
        qd = state.pipeline_state.qd[dof_idx]
        qfrc_bias = state.info["qfrc_bias"]

        q_list.append(np.asarray(q))
        dq_list.append(np.asarray(qd))
        bias_list.append(np.asarray(qfrc_bias))

    return (
        np.stack(q_list).astype(np.float32),
        np.stack(dq_list).astype(np.float32),
        np.stack(bias_list).astype(np.float32),
    )
