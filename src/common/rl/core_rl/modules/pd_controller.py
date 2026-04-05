"""PD controller — pure JAX function.

``tau = kp * (q_target - q) + kd * (dq_target - dq) + gravity_comp``

Gains are plain JAX arrays (not learnable parameters).
At ONNX export time, they are baked into a PyTorch ``nn.Module`` buffer.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class PDGains(NamedTuple):
    """PD controller gains."""

    kp: jax.Array  # (num_joints,)
    kd: jax.Array  # (num_joints,)


def pd_control(
    gains: PDGains,
    q_target: jax.Array,
    q_current: jax.Array,
    dq_current: jax.Array,
    gravity_comp: jax.Array,
    dq_target: jax.Array | None = None,
) -> jax.Array:
    """Compute PD + gravity compensation torques.

    Args:
        gains: ``PDGains(kp, kd)``.
        q_target: Desired joint positions, shape ``(..., num_joints)``.
        q_current: Current joint positions.
        dq_current: Current joint velocities.
        gravity_comp: Gravity compensation torques.
        dq_target: Desired joint velocities (default: zeros).

    Returns:
        Joint torques, shape ``(..., num_joints)``.
    """
    if dq_target is None:
        dq_target = jnp.zeros_like(dq_current)
    return gains.kp * (q_target - q_current) + gains.kd * (dq_target - dq_current) + gravity_comp


def to_numpy(gains: PDGains) -> tuple[np.ndarray, np.ndarray]:
    """Convert gains to NumPy for ONNX export."""
    return np.asarray(gains.kp), np.asarray(gains.kd)
