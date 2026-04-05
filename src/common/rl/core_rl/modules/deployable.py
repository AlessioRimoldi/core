"""Deployable policy — composes all computation layers (JAX).

The forward pass: ``raw_obs → normalize → policy → PD control → torques``

Gravity compensation is **not** included — it should be computed analytically
at runtime (e.g. via Pinocchio RNEA in the C++ hardware interface).

At training time this is used for evaluation.  For real deployment, the
``export_onnx`` module converts these JAX functions + params into a single
PyTorch ``nn.Module`` for ``torch.onnx.export``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from core_rl.modules.normalizer import NormalizerParams, normalize
from core_rl.modules.pd_controller import PDGains, pd_control


class DeployableParams(NamedTuple):
    """All parameters needed for the full deployable pipeline."""

    normalizer: NormalizerParams
    # policy_fn is a closure over policy params (from Brax make_policy)
    pd_gains: PDGains
    num_joints: int


def make_deployable_fn(
    policy_fn: Callable[[jax.Array], jax.Array],
    params: DeployableParams,
    action_type: str = "position",
) -> Callable[[jax.Array], jax.Array]:
    """Build a full inference function: ``obs → torques``.

    Gravity compensation is excluded — set to zero here.  At deployment
    time, the C++ hardware interface computes it analytically via Pinocchio.

    Args:
        policy_fn: Deterministic policy function ``obs → action`` (from Brax).
        params: ``DeployableParams`` containing normalizer, gains, etc.
        action_type: ``"position"``, ``"velocity"``, or ``"torque"``.

    Returns:
        A JAX function ``obs → torques``.
    """
    n = params.num_joints

    def _forward(obs: jax.Array) -> jax.Array:
        # Extract q, dq from raw observation (first 2*n elements by convention)
        q_current = obs[..., :n]
        dq_current = obs[..., n : 2 * n]

        # 1. Normalize
        obs_norm = normalize(params.normalizer, obs)

        # 2. Policy
        action = policy_fn(obs_norm)

        # 3. Gravity compensation placeholder (zeros — computed at runtime)
        grav_comp = jnp.zeros_like(q_current)

        # 4. PD control
        if action_type == "position":
            torques = pd_control(
                gains=params.pd_gains,
                q_target=action,
                q_current=q_current,
                dq_current=dq_current,
                gravity_comp=grav_comp,
            )
        elif action_type == "torque":
            torques = action
        else:  # velocity
            torques = pd_control(
                gains=params.pd_gains,
                q_target=q_current,
                q_current=q_current,
                dq_current=dq_current,
                gravity_comp=grav_comp,
                dq_target=action,
            )

        return torques

    return _forward
