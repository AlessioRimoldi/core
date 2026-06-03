from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from brax.training import networks
from flax import linen as nn

_NORM_EPS = 1e-8


class QPhiNorm(NamedTuple):
    """Running-stats normalization for q_φ I/O (DADS paper, Sec. 5).

    - ``s_*``     normalize the INPUT (q_φ's conditioning state), sized
      ``input_size`` (e.g. joint angles + velocities).
    - ``delta_*`` standardize the TARGET Δ (what q_φ predicts), sized
      ``target_size`` (e.g. Δee_pos). q_φ then operates with σ≈1 in
      normalized target space.

    Input and target can now live in different spaces (Option A): you
    feed q_φ the variables that determine Δ-target (`[q, dq]` for an arm)
    while keeping the skill space compact (`ee_pos`). The Δ Jacobian is
    a per-skill-independent constant so it cancels in r_dads and is
    omitted from log_prob (constant w.r.t. params for the training loss).

    Pass ``norm=None`` to disable (raw I/O, backward-compatible).
    """

    s_mean: jax.Array
    s_std: jax.Array
    delta_mean: jax.Array
    delta_std: jax.Array


def _standardize(x: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (x - mean) / (std + _NORM_EPS)


def _unstandardize(x: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return x * (std + _NORM_EPS) + mean


class SkillDynamics(nn.Module):

    target_size: int
    skill_size: int
    hidden_layer_sizes: Sequence[int]
    num_components: int = 4
    fixed_std: float = 1.0

    @nn.compact
    def __call__(self, s_input, z):
        x = jnp.concatenate([s_input, z], axis=-1)
        h = networks.MLP(layer_sizes=self.hidden_layer_sizes, activate_final=True)(x)

        # (batch_size, num_components)
        logits = nn.Dense(self.num_components, name="logits")(h)

        # (batch, num_comp * target_size) → (batch, num_comp, target_size)
        means_flat = nn.Dense(self.num_components * self.target_size, name="means")(h)
        means = means_flat.reshape(*h.shape[:-1], self.num_components, self.target_size)

        # Fixed (non-learned) isotropic std — see class docstring on choice.
        stds = jnp.full_like(means, self.fixed_std)

        return logits, means, stds


def make_skill_dynamics(
    input_size: int,
    target_size: int,
    skill_size: int,
    hidden_layer_sizes: Sequence[int],
    num_components: int = 4,
    fixed_std: float = 1.0,
) -> networks.FeedForwardNetwork:
    """Build a q_φ network with separate input and target dims.

    ``input_size`` is the dim of ``s_input`` (what q_φ conditions on,
    e.g. ``[q, dq]`` for an arm). ``target_size`` is the dim of the Δ
    vector q_φ predicts (e.g. ``Δee_pos``).
    """

    module = SkillDynamics(
        target_size=target_size,
        skill_size=skill_size,
        hidden_layer_sizes=hidden_layer_sizes,
        num_components=num_components,
        fixed_std=fixed_std,
    )

    dummy_s = jnp.zeros((1, input_size))
    dummy_z = jnp.zeros((1, skill_size))

    def init(key: jax.Array):
        return module.init(key, dummy_s, dummy_z)

    def apply(params, s_input: jax.Array, z: jax.Array):
        return module.apply(params, s_input, z)

    return networks.FeedForwardNetwork(init=init, apply=apply)


def log_prob(
    network: networks.FeedForwardNetwork,
    params,
    s_input: jax.Array,
    z: jax.Array,
    delta_target: jax.Array,
    norm: QPhiNorm | None = None,
) -> jax.Array:
    """
    Compute ``log q_φ(Δ_target | s_input, z)`` under the MoG distribution.

    If ``norm`` is given, ``s_input`` is standardized by ``s_*`` and
    ``delta_target`` by ``delta_*`` (the network is trained and evaluated
    in normalized space). The Δ Jacobian is omitted on purpose — it
    cancels in r_dads and is a param-independent constant for the loss.
    """
    if norm is not None:
        s_input = _standardize(s_input, norm.s_mean, norm.s_std)
        delta_target = _standardize(delta_target, norm.delta_mean, norm.delta_std)

    logits, means, stds = network.apply(params, s_input, z)

    # delta_target (batch, target_size) | means (batch, num_comp, target_size)
    # delta_target[..., None, :] → (batch, 1, target_size)
    diff = delta_target[..., None, :] - means

    # Per-component diagonal-Gaussian log-density, summed over target dims.
    # log N(x; μ, σ²·I) = -0.5 * Σ ((x-μ)/σ)² - Σ log σ - (D/2) log(2π)
    log_comp = -0.5 * jnp.sum((diff / stds) ** 2 + 2 * jnp.log(stds) + jnp.log(2.0 * jnp.pi), axis=-1)

    log_mix = jax.nn.log_softmax(logits, axis=-1)

    return jax.nn.logsumexp(log_mix + log_comp, axis=-1)


def skill_dynamics_loss(
    network: networks.FeedForwardNetwork,
    params,
    s_input: jax.Array,
    z: jax.Array,
    delta_target: jax.Array,
    norm: QPhiNorm | None = None,
) -> jax.Array:
    """Negative log-likelihood of observed Δ-targets — the q_φ training loss."""
    return -log_prob(network, params, s_input, z, delta_target, norm).mean()


def compute_dads_reward(
    network: networks.FeedForwardNetwork,
    params,
    s_input: jax.Array,
    z: jax.Array,
    delta_target: jax.Array,
    z_alts: jax.Array,
    clamp: float = 50.0,
    norm: QPhiNorm | None = None,
) -> jax.Array:
    """
    The DADS intrinsic reward ``r_z`` for a batch of transitions.
    r_z = log(L+1) - log(1 + Σ_i exp(clip(log q(Δ|s,z_i) - log q(Δ|s,z), -50, 50)))
    """
    logp = log_prob(network, params, s_input, z, delta_target, norm)

    num_alts = z_alts.shape[-2]

    # output shape (batch, L)
    logp_alts = jax.vmap(
        lambda zi: log_prob(network, params, s_input, zi, delta_target, norm), in_axes=-2, out_axes=-1
    )(z_alts)

    diff = jnp.clip(logp_alts - logp[..., None], -clamp, clamp)

    return jnp.log(num_alts + 1) - jnp.log1p(jnp.exp(diff).sum(axis=-1))


def modal_delta(
    network: networks.FeedForwardNetwork, params, s_input: jax.Array, z: jax.Array, norm: QPhiNorm | None = None
) -> jax.Array:
    """
    Predict the modal-component mean of the MoG and return it in PHYSICAL
    units of the target space. When ``norm`` is given, ``s_input`` is
    standardized before q_φ and the predicted (normalized) mean is mapped
    back to physical Δ via ``delta_*`` stats.
    """
    s_in = _standardize(s_input, norm.s_mean, norm.s_std) if norm is not None else s_input

    # logits (B, K), means (B, K, target_size)
    logits, means, _ = network.apply(params, s_in, z)

    # k_star (B,) index of the highest logit per batch
    k_star = jnp.argmax(logits, axis=-1)

    # gather the chosen component's mean per row → (B, target_size)
    modal = jnp.take_along_axis(means, k_star[..., None, None], axis=-2).squeeze(-2)

    if norm is not None:
        modal = _unstandardize(modal, norm.delta_mean, norm.delta_std)
    return modal
