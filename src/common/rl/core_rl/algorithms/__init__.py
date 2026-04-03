"""Algorithm base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

_ALGORITHM_REGISTRY: dict[str, type] = {}


def register_algorithm(name: str):
    """Class decorator that registers an algorithm by name."""

    def decorator(cls):
        _ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm(name: str, **kwargs) -> BaseAlgorithm:
    """Instantiate a registered algorithm by name."""
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(f"Unknown algorithm '{name}'. Available: {available}")
    return _ALGORITHM_REGISTRY[name](**kwargs)


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHM_REGISTRY.keys())


class BaseAlgorithm(ABC):
    """Abstract base for RL algorithms.

    Wraps an underlying library (e.g. Stable-Baselines3) with a unified
    interface for training, saving/loading, and policy extraction.
    """

    @abstractmethod
    def __init__(self, env, config: dict[str, Any], callbacks: list | None = None): ...

    @abstractmethod
    def train(self, total_timesteps: int):
        """Run training for the given number of timesteps."""

    @abstractmethod
    def save(self, path: str):
        """Save the trained model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, env=None) -> BaseAlgorithm:
        """Load a trained model from disk."""

    @abstractmethod
    def get_policy_network(self) -> torch.nn.Module:
        """Extract the policy network as a PyTorch module for ONNX export."""

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Run inference on a single observation."""

    @property
    @abstractmethod
    def logger_data(self) -> dict[str, Any]:
        """Return latest logged training metrics."""


# Import concrete algorithms to trigger registration
from core_rl.algorithms import ppo as _  # noqa: F401, E402
from core_rl.algorithms import sac as _sac  # noqa: F401, E402
