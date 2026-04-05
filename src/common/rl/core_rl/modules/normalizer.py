"""Observation normalizer — JAX / Flax.

At training time, Brax maintains a ``RunningStatisticsState`` inside the
training loop.  For export, we bake the final mean/var into a normalizer
that can be converted to a PyTorch ``nn.Module`` for ONNX.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class NormalizerParams(NamedTuple):
    """Baked observation normalizer parameters."""

    mean: jax.Array  # (obs_dim,)
    std: jax.Array  # (obs_dim,)
    clip: float = 10.0


def normalize(params: NormalizerParams, obs: jax.Array) -> jax.Array:
    """Normalize observations: ``clip((obs - mean) / (std + eps), -clip, clip)``."""
    normalized = (obs - params.mean) / (params.std + 1e-8)
    return jnp.clip(normalized, -params.clip, params.clip)


def from_brax_normalizer(normalizer_params, clip: float = 10.0) -> NormalizerParams:
    """Extract mean/std from Brax's ``RunningStatisticsState``.

    Brax stores ``(count, mean, variance, std)`` in its normalizer params.
    """
    mean = jnp.array(normalizer_params.mean)
    std = jnp.array(normalizer_params.std)
    return NormalizerParams(mean=mean, std=std, clip=clip)


def to_numpy(params: NormalizerParams) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert normalizer params to NumPy for ONNX export."""
    return np.asarray(params.mean), np.asarray(params.std), params.clip
