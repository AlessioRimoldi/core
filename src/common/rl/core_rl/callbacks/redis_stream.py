"""Redis Streams hook for real-time metric publishing.

Replaces the SB3 ``BaseCallback`` with a simple callable that conforms to
Brax's ``progress_fn(step, metrics)`` signature.

Usage::

    hook = RedisStreamHook(host="redis", port=6379)
    hook.start(experiment="rl", run_id="ppo_run_1", meta={...})

    # Pass hook as progress_fn to Brax train()
    make_policy, params, metrics = ppo.train(..., progress_fn=hook)

    hook.end(total_timesteps=1_000_000)
"""

from __future__ import annotations

import os
import time
from typing import Any

import redis


class RedisStreamHook:
    """Brax-compatible progress_fn that publishes metrics to Redis Streams.

    Call ``start()`` before training and ``end()`` after.
    """

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.password = password or os.environ.get("REDIS_PASS", "")

        self._client: redis.Redis | None = None
        self._stream_key = ""
        self._meta_key = ""

    def start(
        self,
        experiment: str = "default",
        run_id: str = "",
        meta: dict[str, Any] | None = None,
    ):
        """Connect to Redis and publish run metadata."""
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password if self.password else None,
            decode_responses=True,
        )

        run_id = run_id or f"run_{int(time.time())}"
        self._stream_key = f"rl:train:{experiment}:{run_id}:metrics"
        self._meta_key = f"rl:train:{experiment}:{run_id}:meta"

        meta_payload = {
            "experiment": experiment,
            "run_id": run_id,
            "start_time": str(time.time()),
        }
        if meta:
            meta_payload.update({k: str(v) for k, v in meta.items()})

        self._client.hset(self._meta_key, mapping=meta_payload)

    def __call__(self, step: int, metrics: dict[str, Any]) -> None:
        """Publish metrics via XADD at each Brax eval boundary."""
        if self._client is None:
            return

        payload: dict[str, str] = {
            "step": str(step),
            "time": str(time.time()),
        }

        for key, value in metrics.items():
            safe_key = key.replace("/", "_")
            try:
                payload[safe_key] = f"{float(value):.6f}"
            except (TypeError, ValueError):
                continue

        try:
            self._client.xadd(self._stream_key, payload, maxlen=100_000)
        except Exception as e:
            print(f"Redis publish error: {e}")

    def end(self, total_timesteps: int = 0):
        """Write end timestamp to Redis metadata."""
        if self._client is None:
            return
        self._client.hset(self._meta_key, "end_time", str(time.time()))
        if total_timesteps:
            self._client.hset(self._meta_key, "total_timesteps", str(total_timesteps))
