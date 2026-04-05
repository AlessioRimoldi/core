"""Algorithm base class and registry.

Wraps Brax's training functions (``brax.training.agents.ppo.train``,
``brax.training.agents.sac.train``) behind a unified interface.

Brax ``train()`` returns ``(make_policy, params, metrics)`` where:
    - ``make_policy`` is a function ``params → (obs → action)``
    - ``params`` is a JAX pytree ``(normalizer_params, policy_params, ...)``
    - ``metrics`` is a dict of final eval metrics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import jax

from core_rl.tasks import BaseTask

# Type aliases for Brax training outputs
Metrics = dict[str, Any]
Params = Any  # JAX pytree
MakePolicyFn = Callable  # params → (obs → action)
ProgressFn = Callable[[int, Metrics], None]
PolicyParamsFn = Callable  # (step, make_policy, params) → None

_ALGORITHM_REGISTRY: dict[str, type] = {}


def register_algorithm(name: str):
    """Class decorator that registers an algorithm by name."""

    def decorator(cls: type) -> type:
        _ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm(name: str, **kwargs: Any) -> BaseAlgorithm:
    """Instantiate a registered algorithm by name."""
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(f"Unknown algorithm '{name}'. Available: {available}")
    return _ALGORITHM_REGISTRY[name](**kwargs)


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHM_REGISTRY.keys())


class BaseAlgorithm(ABC):
    """Abstract base for RL algorithms backed by Brax training loops.

    Each subclass wraps a specific ``brax.training.agents.*.train()``
    function, mapping config keys to the correct parameter names.
    """

    @abstractmethod
    def __init__(
        self,
        env: BaseTask,
        config: dict[str, Any],
        progress_fn: ProgressFn | None = None,
        policy_params_fn: PolicyParamsFn | None = None,
    ): ...

    @abstractmethod
    def train(self) -> tuple[MakePolicyFn, Params, Metrics]:
        """Run training.  Returns ``(make_policy, params, metrics)``."""

    @abstractmethod
    def save(self, path: str, params: Params):
        """Save trained params to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Params:
        """Load trained params from disk."""

    @abstractmethod
    def make_inference_fn(self, params: Params) -> Callable[[jax.Array], jax.Array]:
        """Build a deterministic inference function from params."""


# Import concrete algorithms to trigger registration
from core_rl.algorithms import ppo as _  # noqa: F401, E402
from core_rl.algorithms import sac as _sac  # noqa: F401, E402
