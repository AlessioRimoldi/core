"""
An ensemble of K deterministic MLPs f_i(s, a) → Δs, trained by MSE on the
agent's own rollouts, each on a bootstrap of the data (per-sample dropout
masks, as in the reference implementation). The intrinsic reward is the
variance of the K predictions, averaged over state dims, it does NOT
depend on the true next state.

The ensemble is built as ONE Linen module with K stacked parameter pytrees
(``jax.vmap`` over the parameter axis), wrapped in a Brax-style
``FeedForwardNetwork(init, apply)`` so the trainer treats it like every
other network:

- ``init(key) → params``        stacked pytree, leading axis K
- ``apply(params, s, a) → preds``  shape ``(K, ..., obs_size)``
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from brax.training import networks as brax_networks
from flax import linen as nn


class DynamicsModel(nn.Module):
    """A single deterministic forward model: (s, a) → predicted Δs."""

    obs_size: int
    hidden_layer_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, s: jax.Array, a: jax.Array) -> jax.Array:
        x = jnp.concatenate([s, a], axis=-1)
        # Final layer (obs_size) is linear: activate_final defaults to False.
        return brax_networks.MLP(layer_sizes=tuple(self.hidden_layer_sizes) + (self.obs_size,))(x)


def make_ensemble(
    obs_size: int,
    action_size: int,
    num_models: int = 5,
    hidden_layer_sizes: Sequence[int] = (256, 256),
) -> brax_networks.FeedForwardNetwork:
    """Build a ``FeedForwardNetwork`` over K stacked DynamicsModel params."""
    module = DynamicsModel(obs_size=obs_size, hidden_layer_sizes=hidden_layer_sizes)
    dummy_s = jnp.zeros((1, obs_size))
    dummy_a = jnp.zeros((1, action_size))

    def init(key: jax.Array):
        keys = jax.random.split(key, num_models)
        return jax.vmap(lambda k: module.init(k, dummy_s, dummy_a))(keys)

    def apply(params, s: jax.Array, a: jax.Array) -> jax.Array:
        return jax.vmap(lambda p: module.apply(p, s, a))(params)  # (K, ..., obs_size)

    return brax_networks.FeedForwardNetwork(init=init, apply=apply)


def ensemble_loss(
    network: brax_networks.FeedForwardNetwork,
    params,
    s: jax.Array,
    a: jax.Array,
    target: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    keep_prob: float = 0.8,
) -> jax.Array:
    """Bootstrap-masked MSE over all K models.

    Each model sees a random ``keep_prob`` fraction of the batch (fresh mask
    per call) — the cheap bootstrap approximation from the reference
    implementation (``dynamics.py:get_loss_partial``). ``params`` is the
    first argument after the bound network so this plugs straight into
    ``brax.training.gradients.gradient_update_fn``.

    Args:
        network: ``FeedForwardNetwork`` from :func:`make_ensemble`.
        params:  Stacked ensemble params (leading axis K).
        s:       Normalized current state, shape ``(N, obs_size)``.
        a:       Action, shape ``(N, action_size)``.
        target:  Normalized delta ``norm(s') − norm(s)``, shape ``(N, obs_size)``.
        mask:    1 = valid transition, 0 = autoreset boundary, shape ``(N,)``.
        key:     PRNG key for the bootstrap masks.
        keep_prob: Per-model per-sample keep probability.

    Returns:
        Scalar loss.
    """
    preds = network.apply(params, s, a)  # (K, N, obs)
    per_sample = jnp.mean((preds - target[None]) ** 2, axis=-1)  # (K, N)
    keep = jax.random.bernoulli(key, keep_prob, per_sample.shape).astype(jnp.float32)
    w = keep * mask[None]
    return jnp.sum(per_sample * w) / jnp.maximum(jnp.sum(w), 1.0)


def disagreement_reward(
    network: brax_networks.FeedForwardNetwork,
    params,
    s: jax.Array,
    a: jax.Array,
    weights: jax.Array | None = None,
) -> jax.Array:
    """Intrinsic reward: variance across models, (weighted) mean over state dims.

    Does not depend on the true next state.

    With ``weights`` (per-obs-dim, shape ``(obs_size,)``) the mean over dims is a
    weighted average — put most weight on the OBJECT dims so the agent gets
    curious about the object rather than its own body. In a low-dim proprioceptive
    obs the object is only a few of many dims, so an unweighted mean makes the
    agent ~proportionally curious about each dim (mostly the high-dim arm); the
    weighting is the low-dim analog of the object dominating an image observation.

    Args:
        network: ``FeedForwardNetwork`` from :func:`make_ensemble`.
        params:  Stacked ensemble params (leading axis K).
        s:       Normalized current state, shape ``(N, obs_size)``.
        a:       Action, shape ``(N, action_size)``.
        weights: Optional per-dim weights ``(obs_size,)``; ``None`` = uniform mean.

    Returns:
        Per-transition intrinsic reward, shape ``(N,)``.
    """
    preds = network.apply(params, s, a)  # (K, N, obs)
    var = jnp.var(preds, axis=0)  # (N, obs)
    if weights is None:
        return var.mean(axis=-1)  # (N,)
    return jnp.sum(var * weights, axis=-1) / jnp.sum(weights)  # (N,)


def build_reward_weights(
    obs_size: int,
    indices,
    bg_weight: float,
    env=None,
) -> jax.Array | None:
    """Per-obs-dim weight vector for :func:`disagreement_reward`, or None (uniform).

    Shared by every disagreement-bonus arm (single-agent, multi-head, and the
    dads_disagreement fusion) so cube-focused novelty means the SAME thing across
    arms and comparisons stay apples-to-apples.

    ``indices`` may be None (uniform), the string ``"block"`` (resolve via
    ``env.block_obs_indices``), or an explicit list of obs dims. Selected dims get
    weight 1.0; the rest get ``bg_weight``.
    """
    if indices is None:
        return None
    if indices == "block":
        indices = getattr(env, "block_obs_indices", None)
        if not indices:
            raise ValueError(
                "ensemble_reward_indices='block' but the task exposes no "
                "block_obs_indices — use an explicit index list or a task that "
                "defines the block dims (e.g. fetchpush)."
            )
    idx = [int(i) for i in indices]
    if any(i < 0 or i >= obs_size for i in idx):
        raise ValueError(f"ensemble_reward_indices {idx} out of range for obs_size {obs_size}.")
    weights = jnp.full((obs_size,), float(bg_weight), dtype=jnp.float32)
    return weights.at[jnp.asarray(idx, dtype=jnp.int32)].set(1.0)
