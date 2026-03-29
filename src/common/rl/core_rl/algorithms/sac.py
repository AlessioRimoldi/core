"""SAC algorithm — wraps Stable-Baselines3."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv

from core_rl.algorithms import BaseAlgorithm, register_algorithm


@register_algorithm("sac")
class SACAlgorithm(BaseAlgorithm):
    """SAC via Stable-Baselines3."""

    def __init__(self, env: VecEnv, config: dict[str, Any], callbacks: list | None = None):
        self._callbacks = callbacks or []

        policy = config.pop("policy", "MlpPolicy")
        policy_kwargs = config.pop("policy_kwargs", None)
        if policy_kwargs and "net_arch" in policy_kwargs:
            policy_kwargs["net_arch"] = [int(x) for x in policy_kwargs["net_arch"]]

        self._model = SAC(
            policy=policy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            **config,
        )

    def train(self, total_timesteps: int):
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=self._callbacks if self._callbacks else None,
        )

    def save(self, path: str):
        self._model.save(path)

    @classmethod
    def load(cls, path: str, env=None) -> SACAlgorithm:
        instance = object.__new__(cls)
        instance._model = SAC.load(path, env=env)
        instance._callbacks = []
        return instance

    def get_policy_network(self) -> torch.nn.Module:
        return self._model.policy

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return action

    @property
    def logger_data(self) -> dict[str, Any]:
        if self._model.logger is None:
            return {}
        return dict(self._model.logger.name_to_value)
