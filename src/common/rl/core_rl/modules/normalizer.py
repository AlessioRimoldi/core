"""Observation normalizer as a torch.nn.Module for ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class ObservationNormalizer(nn.Module):
    """Normalizes observations using running mean/std.

    Bakes VecNormalize statistics into a Module so they're included in ONNX.
    """

    def __init__(self, obs_dim: int, mean: np.ndarray | None = None, std: np.ndarray | None = None, clip: float = 10.0):
        super().__init__()
        if mean is None:
            mean = np.zeros(obs_dim, dtype=np.float32)
        if std is None:
            std = np.ones(obs_dim, dtype=np.float32)

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.clip = clip

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.mean) / (self.std + 1e-8)
        return torch.clamp(normalized, -self.clip, self.clip)

    @classmethod
    def from_vec_normalize(cls, vec_normalize) -> ObservationNormalizer:
        """Create from a Stable-Baselines3 VecNormalize wrapper."""
        obs_rms = vec_normalize.obs_rms
        mean = obs_rms.mean.astype(np.float32)
        std = np.sqrt(obs_rms.var + vec_normalize.epsilon).astype(np.float32)
        clip = vec_normalize.clip_obs
        return cls(obs_dim=len(mean), mean=mean, std=std, clip=clip)
