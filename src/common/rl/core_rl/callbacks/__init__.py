"""Training hooks for streaming metrics.

Brax uses two callback signatures instead of SB3's ``BaseCallback`` lifecycle:
    - ``progress_fn(step: int, metrics: dict) → None`` — called at eval boundaries
    - ``policy_params_fn(step, make_policy, params) → None`` — called after each epoch

This module provides:
    - ``compose_progress_fn`` — combine multiple progress hooks into one
    - ``MLflowHook``     — experiment tracking (replaces MLflowCallback)
    - ``RedisStreamHook`` — real-time metric publishing (replaces RedisStreamCallback)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressFn = Callable[[int, dict[str, Any]], None]


def compose_progress_fn(*hooks: ProgressFn) -> ProgressFn:
    """Combine multiple progress hooks into a single ``progress_fn``.

    Each hook is called in order with ``(step, metrics)``.
    Exceptions in one hook do not prevent subsequent hooks from running.
    """

    def _composed(step: int, metrics: dict[str, Any]) -> None:
        for hook in hooks:
            try:
                hook(step, metrics)
            except Exception as e:
                print(f"Warning: progress hook {hook} raised: {e}")

    return _composed
