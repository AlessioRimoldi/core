"""Redis Streams callback for real-time metric publishing."""

from __future__ import annotations

import os
import time

import numpy as np
import redis
from stable_baselines3.common.callbacks import BaseCallback


class RedisStreamCallback(BaseCallback):
    """Publishes training metrics to Redis Streams.

    Uses XADD to append metrics to a stream keyed by experiment/run.
    Metrics are published every ``publish_freq`` steps.

    Stream key format: ``rl:train:<experiment>:<run_id>:metrics``
    """

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        password: str = "",
        experiment: str = "default",
        run_id: str = "",
        publish_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.host = host
        self.port = port
        self.password = password or os.environ.get("REDIS_PASS", "")
        self.experiment = experiment
        self.run_id = run_id or f"run_{int(time.time())}"
        self.publish_freq = publish_freq

        self._client = None
        self._stream_key = ""
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def _init_callback(self) -> bool:
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password if self.password else None,
            decode_responses=True,
        )
        self._stream_key = f"rl:train:{self.experiment}:{self.run_id}:metrics"

        # Publish run metadata
        meta_key = f"rl:train:{self.experiment}:{self.run_id}:meta"
        self._client.hset(
            meta_key,
            mapping={
                "experiment": self.experiment,
                "run_id": self.run_id,
                "start_time": str(time.time()),
                "num_envs": str(self.training_env.num_envs),
            },
        )

        if self.verbose:
            print(f"Redis callback initialized — stream: {self._stream_key}")
        return True

    def _on_step(self) -> bool:
        # Collect per-episode stats from info dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        if self.num_timesteps % self.publish_freq != 0:
            return True

        # Build metric payload
        metrics: dict[str, str] = {
            "step": str(self.num_timesteps),
            "time": str(time.time()),
        }

        if self._episode_rewards:
            metrics["mean_reward"] = f"{np.mean(self._episode_rewards):.4f}"
            metrics["mean_ep_length"] = f"{np.mean(self._episode_lengths):.1f}"
            metrics["num_episodes"] = str(len(self._episode_rewards))
            self._episode_rewards.clear()
            self._episode_lengths.clear()

        # Add training loss metrics if available
        if hasattr(self.model, "logger") and self.model.logger is not None:
            for key, value in self.model.logger.name_to_value.items():
                safe_key = key.replace("/", "_")
                metrics[safe_key] = f"{value:.6f}"

        try:
            self._client.xadd(self._stream_key, metrics, maxlen=100_000)
        except Exception as e:
            if self.verbose:
                print(f"Redis publish error: {e}")

        return True

    def _on_training_end(self) -> None:
        if self._client:
            end_key = f"rl:train:{self.experiment}:{self.run_id}:meta"
            self._client.hset(end_key, "end_time", str(time.time()))
            self._client.hset(end_key, "total_timesteps", str(self.num_timesteps))
