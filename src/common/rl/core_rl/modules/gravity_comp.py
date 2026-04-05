"""Learned gravity compensation network — Flax / JAX.

An MLP that predicts ``qfrc_bias`` (gravity + Coriolis torques) from ``(q, dq)``.
Trained supervised on data collected during RL rollouts.

Uses Flax ``linen`` for the network definition and Optax for optimisation.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class GravityCompNet(nn.Module):
    """MLP: ``(q, dq) → qfrc_bias`` estimate."""

    num_joints: int
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, q: jax.Array, dq: jax.Array) -> jax.Array:
        x = jnp.concatenate([q, dq], axis=-1)
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_joints)(x)
        return x


def train_gravity_comp(
    num_joints: int,
    q_data: np.ndarray,
    dq_data: np.ndarray,
    bias_data: np.ndarray,
    hidden_dims: tuple[int, ...] = (256, 256),
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[GravityCompNet, Any, dict[str, list[float]]]:
    """Train a GravityCompNet on collected data.

    Args:
        num_joints: Number of robot joints.
        q_data: Joint positions, shape ``(N, num_joints)``.
        dq_data: Joint velocities, shape ``(N, num_joints)``.
        bias_data: Target ``qfrc_bias``, shape ``(N, num_joints)``.
        hidden_dims: Hidden layer sizes.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        ``(model, params, history)`` where ``params`` is a Flax pytree and
        ``history`` is a dict with ``"loss"`` key.
    """
    model = GravityCompNet(num_joints=num_joints, hidden_dims=hidden_dims)

    rng = jax.random.PRNGKey(seed)
    dummy_q = jnp.zeros((1, num_joints))
    dummy_dq = jnp.zeros((1, num_joints))
    params = model.init(rng, dummy_q, dummy_dq)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    q_jax = jnp.array(q_data)
    dq_jax = jnp.array(dq_data)
    bias_jax = jnp.array(bias_data)
    n_samples = len(q_data)

    @jax.jit
    def loss_fn(params, q_batch, dq_batch, bias_batch):
        pred = model.apply(params, q_batch, dq_batch)
        return jnp.mean((pred - bias_batch) ** 2)

    @jax.jit
    def train_step(params, opt_state, q_batch, dq_batch, bias_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, q_batch, dq_batch, bias_batch)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    history: dict[str, list[float]] = {"loss": []}

    for epoch in range(epochs):
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, n_samples)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            q_b = q_jax[idx]
            dq_b = dq_jax[idx]
            bias_b = bias_jax[idx]

            params, opt_state, loss = train_step(params, opt_state, q_b, dq_b, bias_b)
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  GravComp epoch {epoch+1}/{epochs} — loss: {avg_loss:.6f}")

    return model, params, history
