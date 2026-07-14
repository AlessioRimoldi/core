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


class FirstSuccessHook:
    """Track time-to-first-success and inject it into the metrics dict.

    Watches an eval success metric (default ``eval/episode_caught`` — the
    summed catch steps per eval episode) and, the first time it exceeds
    ``threshold``, records the env-step. From then on it adds
    ``eval/first_success_step`` to every metrics dict; it always adds
    ``eval/has_succeeded`` (0/1). Because :func:`compose_progress_fn` passes the
    *same* dict to each hook in order, placing this hook FIRST means the injected
    keys are seen (and logged) by the MLflow/Redis hooks downstream — no separate
    logging path needed. A no-op for tasks that don't emit the success metric.
    """

    def __init__(self, metric_key: str = "eval/episode_caught", threshold: float = 0.0):
        self._metric_key = metric_key
        self._threshold = threshold
        self._first_step: int | None = None

    def __call__(self, step: int, metrics: dict[str, Any]) -> None:
        value = metrics.get(self._metric_key)
        if value is not None:
            try:
                if self._first_step is None and float(value) > self._threshold:
                    self._first_step = int(step)
            except (TypeError, ValueError):
                pass
        metrics["eval/has_succeeded"] = 1.0 if self._first_step is not None else 0.0
        if self._first_step is not None:
            metrics["eval/first_success_step"] = float(self._first_step)
