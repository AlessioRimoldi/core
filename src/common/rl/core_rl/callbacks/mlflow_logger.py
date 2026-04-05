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

import time
from typing import Any

import mlflow


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
        """Begin an MLflow run and log hyperparameters."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=run_name or f"run_{int(time.time())}")
        self._start_time = time.time()

        if params:
            flat_params = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat_params[f"{k}.{kk}"] = str(vv)
                else:
                    flat_params[k] = str(v)
            mlflow.log_params(flat_params)

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
