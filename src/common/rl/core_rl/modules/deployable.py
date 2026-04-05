"""Deployable policy — composes all computation layers (JAX).

The forward pass: ``raw_obs → normalize → policy → PD control (with gravity comp) → torques``

At training time this is used for evaluation.  For real deployment, the
``export_onnx`` module converts these JAX functions + params into a single
PyTorch ``nn.Module`` for ``torch.onnx.export``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from core_rl.modules.gravity_comp import GravityCompNet
from core_rl.modules.normalizer import NormalizerParams, normalize
from core_rl.modules.pd_controller import PDGains, pd_control


class DeployableParams(NamedTuple):
    """All parameters needed for the full deployable pipeline."""

    normalizer: NormalizerParams
    # policy_fn is a closure over policy params (from Brax make_policy)
    gravity_comp_params: Any  # Flax pytree (or None if untrained)
    pd_gains: PDGains
    num_joints: int


def make_deployable_fn(
    policy_fn: Callable[[jax.Array], jax.Array],
    gravity_comp_model: GravityCompNet | None,
    params: DeployableParams,
    action_type: str = "position",
) -> Callable[[jax.Array], jax.Array]:
    """Build a full inference function: ``obs → torques``.

    Args:
        policy_fn: Deterministic policy function ``obs → action`` (from Brax).
        gravity_comp_model: Flax ``GravityCompNet`` (or ``None`` for zeros).
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

        # 3. Gravity compensation
        if gravity_comp_model is not None and params.gravity_comp_params is not None:
            grav_comp = gravity_comp_model.apply(params.gravity_comp_params, q_current, dq_current)
        else:
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
            torques = action + grav_comp
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
