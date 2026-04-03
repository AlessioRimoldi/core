"""Learned gravity compensation network.

An MLP that predicts qfrc_bias (gravity + Coriolis torques) from (q, dq).
Trained supervised on data collected during RL rollouts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GravityCompensationNet(nn.Module):
    """MLP: (q, dq) -> qfrc_bias estimate."""

    def __init__(self, num_joints: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.num_joints = num_joints

        layers: list[nn.Module] = []
        in_dim = 2 * num_joints  # q + dq
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_joints))

        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Predict gravity compensation torques.

        Args:
            q: Joint positions, shape (..., num_joints)
            dq: Joint velocities, shape (..., num_joints)

        Returns:
            Estimated qfrc_bias, shape (..., num_joints)
        """
        x = torch.cat([q, dq], dim=-1)
        return self.net(x)


def train_gravity_comp(
    model: GravityCompensationNet,
    q_data: np.ndarray,
    dq_data: np.ndarray,
    bias_data: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the GravCompNet on collected data.

    Args:
        model: The GravityCompensationNet to train.
        q_data: Joint positions, shape (N, num_joints).
        dq_data: Joint velocities, shape (N, num_joints).
        bias_data: Target qfrc_bias, shape (N, num_joints).
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        device: Torch device.
        verbose: Print progress.

    Returns:
        Training history dict with "loss" key.
    """
    model = model.to(device)
    model.train()

    q_t = torch.tensor(q_data, dtype=torch.float32, device=device)
    dq_t = torch.tensor(dq_data, dtype=torch.float32, device=device)
    bias_t = torch.tensor(bias_data, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(q_t, dq_t, bias_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"loss": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for q_batch, dq_batch, bias_batch in loader:
            pred = model(q_batch, dq_batch)
            loss = loss_fn(pred, bias_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  GravComp epoch {epoch+1}/{epochs} — loss: {avg_loss:.6f}")

    model.eval()
    return history
