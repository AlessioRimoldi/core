"""Simulation backend protocol and registry.

A SimBackend is a thin abstraction over the physics engine used during
training.  Each backend owns its own vectorisation strategy (MJX via
``jax.vmap``, Isaac via its tensor API, etc.) but exposes the same
``init`` / ``step`` contract so the training loop is engine-agnostic.

Currently supported:
    - ``mjx`` — MuJoCo MJX via Brax's pipeline layer

Future backends (Isaac Sim, Newton) will implement the same protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import brax.base as brax_base
import jax
import mujoco

# ---------------------------------------------------------------------------
# Protocol — what every backend must provide
# ---------------------------------------------------------------------------


@runtime_checkable
class SimBackend(Protocol):
    """Minimal contract for a physics backend used in training."""

    def init(
        self,
        model: mujoco.MjModel,
        q: jax.Array,
        qd: jax.Array,
        *,
        ctrl: jax.Array | None = None,
    ) -> brax_base.State:
        """Initialise physics state from joint configuration.

        Args:
            model: MuJoCo model (used to create ``brax.System``).
            q: Generalised positions, shape ``(q_size,)``.
            qd: Generalised velocities, shape ``(qd_size,)``.
            ctrl: Optional control vector.

        Returns:
            A Brax-compatible physics ``State``.
        """
        ...

    def step(
        self,
        model: mujoco.MjModel,
        state: brax_base.State,
        action: jax.Array,
    ) -> brax_base.State:
        """Advance physics by one control step (may include sub-stepping).

        Args:
            model: MuJoCo model.
            state: Current physics state.
            action: Control action (e.g. ctrl vector).

        Returns:
            Next physics state.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str):
    """Class decorator that registers a backend by name."""

    def decorator(cls: type) -> type:
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str, **kwargs: Any) -> SimBackend:
    """Instantiate a registered backend by name."""
    if name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")
    return _BACKEND_REGISTRY[name](**kwargs)


def list_backends() -> list[str]:
    return sorted(_BACKEND_REGISTRY.keys())


# Import concrete backends to trigger registration
from core_rl.backends import mjx as _  # noqa: F401, E402
