"""MLflow logging hook for Brax training.

Replaces the SB3 ``BaseCallback`` with a simple callable that conforms to
Brax's ``progress_fn(step, metrics)`` signature.

Usage::

    hook = MLflowHook(tracking_uri="http://mlflow:5000", experiment_name="rl")
    hook.start(run_name="ppo_run_1", params={...})

    # Pass hook as progress_fn to Brax train()
    make_policy, params, metrics = ppo.train(..., progress_fn=hook)

    hook.end(artifact_paths=["/path/to/policy.onnx"])
"""

from __future__ import annotations

import os
import time
from typing import Any

import mlflow


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested dict for mlflow.log_params (dot-joined keys, str values)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            s = str(v)
            out[key] = s[:497] + "..." if len(s) > 500 else s
    return out


class MLflowHook:
    """Brax-compatible progress_fn that logs to MLflow.

    Call ``start()`` before training and ``end()`` after.  Between those
    calls, use the instance directly as a ``progress_fn``.
    """

    def __init__(
        self,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "rl_training",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._run = None
        self._start_time = 0.0

    def start(self, run_name: str = "", params: dict[str, Any] | None = None):
        """Begin an MLflow run and log hyperparameters.

        If `MLFLOW_SWEEP_RUN_NAME` is set in the environment, it overrides
        the `run_name` argument — used by multi_train.py so each child run
        gets the manifest's `name:` instead of an auto-generated timestamp.

        If `MLFLOW_PARENT_RUN_ID` is set, the new run is nested under that
        parent — used by multi_train.py to group a whole sweep into one
        collapsible block in the MLflow UI.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        sweep_run_name = os.environ.get("MLFLOW_SWEEP_RUN_NAME", "").strip()
        effective_run_name = sweep_run_name or run_name or f"run_{int(time.time())}"

        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID", "").strip()
        tags: dict[str, str] = {}
        if parent_run_id:
            tags["mlflow.parentRunId"] = parent_run_id
        if sweep_run_name:
            tags["sweep_run_name"] = sweep_run_name

        self._run = mlflow.start_run(run_name=effective_run_name, tags=tags or None)
        self._start_time = time.time()

        if params:
            flat_params = _flatten(params)
            # MLflow caps params at 100 per log_params call and 500 chars per value.
            items = list(flat_params.items())
            for i in range(0, len(items), 100):
                mlflow.log_params(dict(items[i : i + 100]))

    def __call__(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics at each Brax eval boundary."""
        if self._run is None:
            return

        elapsed = time.time() - self._start_time
        log_metrics: dict[str, float] = {
            "timesteps": float(step),
            "walltime": elapsed,
        }

        # Brax metrics include eval/episode_reward, eval/episode_length, etc.
        for key, value in metrics.items():
            safe_key = key.replace("/", "_")
            try:
                log_metrics[safe_key] = float(value)
            except (TypeError, ValueError):
                continue

        mlflow.log_metrics(log_metrics, step=step)

    def end(self, artifact_paths: list[str] | None = None):
        """End the MLflow run and optionally log artifacts."""
        if self._run is None:
            return
        if artifact_paths:
            for path in artifact_paths:
                mlflow.log_artifact(path)
        mlflow.end_run()
        self._run = None

    def log_artifact(self, path: str):
        """Log a single artifact to the active run."""
        if self._run:
            mlflow.log_artifact(path)
