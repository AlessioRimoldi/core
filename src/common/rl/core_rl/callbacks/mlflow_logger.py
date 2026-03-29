"""MLflow logging callback for experiment tracking."""

from __future__ import annotations

import os
import time
from typing import Any

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MLflowCallback(BaseCallback):
    """Logs training metrics, hyperparameters, and artifacts to MLflow.

    - on_training_start: creates run, logs hyperparams
    - on_rollout_end: logs metrics (mean reward, episode length, loss, FPS)
    - on_training_end: logs final model artifact, ends run
    """

    def __init__(
        self,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "rl_training",
        run_name: str = "",
        publish_freq: int = 1,
        log_params: dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.publish_freq = publish_freq
        self.log_params = log_params or {}

        self._run = None
        self._rollout_count = 0
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._start_time = 0.0
        self._last_step = 0

    def _init_callback(self) -> bool:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self._run = mlflow.start_run(run_name=self.run_name)
        self._start_time = time.time()
        self._last_step = 0

        # Log hyperparameters
        params = {
            "num_envs": self.training_env.num_envs,
            **self.log_params,
        }
        # MLflow params must be strings/numbers — flatten nested dicts
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat_params[f"{k}.{kk}"] = str(vv)
            else:
                flat_params[k] = str(v)

        mlflow.log_params(flat_params)

        if self.verbose:
            print(f"MLflow callback initialized — run: {self._run.info.run_id}")
        return True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1

        if self._rollout_count % self.publish_freq != 0:
            return

        step = self.num_timesteps
        elapsed = time.time() - self._start_time
        fps = (step - self._last_step) / max(elapsed, 1e-8) if self._last_step > 0 else step / max(elapsed, 1e-8)

        metrics: dict[str, float] = {
            "timesteps": float(step),
            "fps": fps,
        }

        if self._episode_rewards:
            metrics["mean_reward"] = float(np.mean(self._episode_rewards))
            metrics["mean_ep_length"] = float(np.mean(self._episode_lengths))
            metrics["num_episodes"] = float(len(self._episode_rewards))
            self._episode_rewards.clear()
            self._episode_lengths.clear()

        # Log training loss/diagnostics from SB3
        if hasattr(self.model, "logger") and self.model.logger is not None:
            for key, value in self.model.logger.name_to_value.items():
                safe_key = key.replace("/", "_")
                metrics[safe_key] = float(value)

        mlflow.log_metrics(metrics, step=step)
        self._last_step = step
        self._start_time = time.time()

    def _on_training_end(self) -> None:
        if self._run:
            mlflow.end_run()

    def log_artifact(self, path: str):
        """Log an artifact file (e.g. ONNX model) to the active run."""
        if self._run:
            mlflow.log_artifact(path)
