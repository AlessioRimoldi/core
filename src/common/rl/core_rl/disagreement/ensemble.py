"""
An ensemble of K deterministic MLPs f_i(s, a) → Δs, trained by MSE on the
agent's own rollouts, each on a bootstrap of the data (per-sample dropout
masks, as in the reference implementation). The intrinsic reward is the
variance of the K predictions, averaged over state dims, it does NOT
depend on the true next state.

Two optional extensions (both off by default, config-driven):
* per-dim ``weights`` — shared by the loss and the reward, so ensemble
  capacity concentrates on the dims the reward reads (object/position focus);
* ``make_position_features`` — frozen random-Fourier target dims φ(pos')
  appended to Δs, giving disagreement map content that cannot be inferred
  without visiting (the pixel recipe minus the renderer).

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
    """A single deterministic forward model: (s, a) → predicted target.

    The target is Δs by default (``target_size=None`` → ``obs_size`` outputs);
    with ``target_size`` set the output may be wider, e.g. Δs plus appended
    random position features (see :func:`make_position_features`).
    """

    obs_size: int
    hidden_layer_sizes: Sequence[int] = (256, 256)
    target_size: int | None = None

    @nn.compact
    def __call__(self, s: jax.Array, a: jax.Array) -> jax.Array:
        x = jnp.concatenate([s, a], axis=-1)
        # Final layer is linear: activate_final defaults to False.
        out = self.target_size or self.obs_size
        return brax_networks.MLP(layer_sizes=tuple(self.hidden_layer_sizes) + (out,))(x)


def make_ensemble(
    obs_size: int,
    action_size: int,
    num_models: int = 5,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    target_size: int | None = None,
) -> brax_networks.FeedForwardNetwork:
    """Build a ``FeedForwardNetwork`` over K stacked DynamicsModel params."""
    module = DynamicsModel(obs_size=obs_size, hidden_layer_sizes=hidden_layer_sizes, target_size=target_size)
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
    weights: jax.Array | None = None,
) -> jax.Array:
    """Bootstrap-masked MSE over all K models.

    Each model sees a random ``keep_prob`` fraction of the batch (fresh mask
    per call) — the cheap bootstrap approximation from the reference
    implementation (``dynamics.py:get_loss_partial``). ``params`` is the
    first argument after the bound network so this plugs straight into
    ``brax.training.gradients.gradient_update_fn``.

    ``weights`` (the SAME per-dim vector :func:`disagreement_reward` uses)
    turns the mean over target dims into a weighted mean, so ensemble capacity
    concentrates on the dims the reward actually reads. Without it, dims whose
    per-step normalized delta is small (e.g. torso x, y: ~0.1 m step vs a
    multi-meter position std → ~0.01-scale target) contribute ~nothing to the
    gradient, the ensemble fits them as an afterthought, and their prediction
    variance is fit-noise rather than novelty (the 20260717 xy-only run's
    failure mode).

    Args:
        network: ``FeedForwardNetwork`` from :func:`make_ensemble`.
        params:  Stacked ensemble params (leading axis K).
        s:       Normalized current state, shape ``(N, obs_size)``.
        a:       Action, shape ``(N, action_size)``.
        target:  Normalized delta ``norm(s') − norm(s)`` (optionally with
                 position-feature dims appended), shape ``(N, target_size)``.
        mask:    1 = valid transition, 0 = autoreset boundary, shape ``(N,)``.
        key:     PRNG key for the bootstrap masks.
        keep_prob: Per-model per-sample keep probability.
        weights: Optional per-dim weights ``(target_size,)``; None = uniform.

    Returns:
        Scalar loss.
    """
    preds = network.apply(params, s, a)  # (K, N, target)
    se = (preds - target[None]) ** 2  # (K, N, target)
    # (weighted) mean over target dims → (K, N)
    per_sample = jnp.mean(se, axis=-1) if weights is None else jnp.sum(se * weights, axis=-1) / jnp.sum(weights)
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
    idx = resolve_obs_indices(obs_size, indices, env=env)
    if idx is None:
        return None
    weights = jnp.full((obs_size,), float(bg_weight), dtype=jnp.float32)
    return weights.at[jnp.asarray(idx, dtype=jnp.int32)].set(1.0)


def resolve_obs_indices(obs_size: int, indices, env=None) -> list[int] | None:
    """``ensemble_reward_indices`` → validated obs-dim list (or None = unset).

    Accepts None, the string ``"block"`` (resolved via ``env.block_obs_indices``),
    or an explicit list of obs dims.
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
    return idx


def make_position_features(
    num_inputs: int,
    num_features: int,
    length_scale: float,
    seed: int = 0,
):
    """Frozen random-Fourier feature map φ: raw position (meters) → (num_features,).

    Appended to the ensemble's prediction TARGET as φ(position_{t+1}), this
    gives disagreement actual map content: unlike Δposition — which is
    ≈ velocity·dt, learnable from any small region and generalizing over the
    whole map — φ at an unvisited place is an independent random value that
    can only be learned by going there. It is the pixel-prediction recipe
    (predict features of the next frame) minus the renderer.

    Design notes:
    * sin features (random Fourier), NOT tanh — tanh saturates far from the
      origin, mapping all distant cells to the same feature vector; sin keeps
      a stationary ``length_scale`` everywhere in the maze.
    * Predict the ABSOLUTE φ(pos'), not its delta: Δφ over one control step is
      ~∇φ·v·dt ≈ 0.1-scale of the feature — which would re-create the small-
      target starvation this feature exists to fix. O(1) targets by design.
    * Fixed ``seed`` → the same φ for every run and both arms (single/multi),
      so comparisons stay apples-to-apples.

    Returns a pure function ``phi(pos: (..., num_inputs)) → (..., num_features)``.
    """
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    n_hidden = 64
    w1 = jax.random.normal(k1, (num_inputs, n_hidden))
    b1 = jax.random.uniform(k2, (n_hidden,), minval=-jnp.pi, maxval=jnp.pi)
    w2 = jax.random.normal(k3, (n_hidden, num_features)) / jnp.sqrt(n_hidden)

    def phi(pos: jax.Array) -> jax.Array:
        h = jnp.sin(pos / length_scale @ w1 + b1)
        return h @ w2  # ~unit-scale, non-saturating

    return phi
