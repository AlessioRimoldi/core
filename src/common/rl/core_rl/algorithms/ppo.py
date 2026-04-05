"""PPO algorithm — wraps Brax's PPO training loop.

Maps config keys from defaults.yaml to ``brax.training.agents.ppo.train()``
parameter names.  The entire training loop (rollout + update) is JIT-compiled
and runs on GPU via ``jax.lax.scan``.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from typing import Any

import jax
from brax.training.agents.ppo import train as ppo_train

from core_rl.algorithms import (
    BaseAlgorithm,
    MakePolicyFn,
    Metrics,
    Params,
    PolicyParamsFn,
    ProgressFn,
    register_algorithm,
)
from core_rl.tasks import BaseTask


def _map_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map defaults.yaml PPO keys to Brax PPO parameter names."""
    mapping = {
        # SB3 name → Brax name
        "learning_rate": "learning_rate",
        "n_steps": "unroll_length",
        "batch_size": "batch_size",
        "n_epochs": "num_updates_per_batch",
        "gamma": "discounting",
        "gae_lambda": "gae_lambda",
        "clip_range": "clipping_epsilon",
        "ent_coef": "entropy_cost",
        "vf_coef": "vf_loss_coefficient",
        "max_grad_norm": "max_grad_norm",
        # Brax-native keys (passed through)
        "unroll_length": "unroll_length",
        "num_updates_per_batch": "num_updates_per_batch",
        "num_minibatches": "num_minibatches",
        "discounting": "discounting",
        "clipping_epsilon": "clipping_epsilon",
        "entropy_cost": "entropy_cost",
        "vf_loss_coefficient": "vf_loss_coefficient",
        "reward_scaling": "reward_scaling",
        "normalize_observations": "normalize_observations",
    }

    brax_cfg: dict[str, Any] = {}
    for key, value in config.items():
        # Skip keys that don't map to Brax params
        if key in ("policy", "policy_kwargs", "seed", "device"):
            continue
        brax_key = mapping.get(key, key)
        brax_cfg[brax_key] = value

    # Network architecture from policy_kwargs
    policy_kwargs = config.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        net_arch = tuple(int(x) for x in policy_kwargs["net_arch"])
        brax_cfg["network_factory_kwargs"] = {
            "policy_hidden_layer_sizes": net_arch,
            "value_hidden_layer_sizes": net_arch,
        }

    return brax_cfg


@register_algorithm("ppo")
class PPOAlgorithm(BaseAlgorithm):
    """PPO via Brax's JIT-compiled training loop."""

    def __init__(
        self,
        env: BaseTask,
        config: dict[str, Any],
        progress_fn: ProgressFn | None = None,
        policy_params_fn: PolicyParamsFn | None = None,
    ):
        self._env = env
        self._config = config.copy()
        self._progress_fn = progress_fn or (lambda *args: None)
        self._policy_params_fn = policy_params_fn or (lambda *args: None)

    def train(self) -> tuple[MakePolicyFn, Params, Metrics]:
        """Run PPO training via ``brax.training.agents.ppo.train``."""
        brax_cfg = _map_config(self._config)

        # Extract top-level training params
        num_timesteps = brax_cfg.pop("total_timesteps", brax_cfg.pop("num_timesteps", 1_000_000))
        num_envs = brax_cfg.pop("num_envs", 4096)
        seed = self._config.get("seed", 0)
        num_evals = brax_cfg.pop("num_evals", 20)
        episode_length = brax_cfg.pop("max_episode_steps", brax_cfg.pop("episode_length", 500))

        # Network factory kwargs are handled separately
        network_factory_kwargs = brax_cfg.pop("network_factory_kwargs", {})
        if network_factory_kwargs:
            from functools import partial

            from brax.training.agents.ppo import networks as ppo_networks

            # If user specified hidden_layer_sizes generically, split for PPO
            if "hidden_layer_sizes" in network_factory_kwargs:
                sizes = network_factory_kwargs.pop("hidden_layer_sizes")
                network_factory_kwargs.setdefault("policy_hidden_layer_sizes", sizes)
                network_factory_kwargs.setdefault("value_hidden_layer_sizes", sizes)

            network_factory = partial(ppo_networks.make_ppo_networks, **network_factory_kwargs)
            brax_cfg["network_factory"] = network_factory

        make_policy, params, metrics = ppo_train.train(
            environment=self._env,
            num_timesteps=num_timesteps,
            num_envs=num_envs,
            seed=seed,
            num_evals=num_evals,
            episode_length=episode_length,
            progress_fn=self._progress_fn,
            policy_params_fn=self._policy_params_fn,
            **brax_cfg,
        )

        return make_policy, params, metrics

    def save(self, path: str, params: Params):
        """Save params as a pickle file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, path: str) -> Params:
        """Load params from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def make_inference_fn(self, params: Params) -> Callable[[jax.Array], jax.Array]:
        """Build a deterministic inference function from trained params."""
        # This requires the make_policy function from training
        raise NotImplementedError("Use the make_policy function returned by train() directly.")
