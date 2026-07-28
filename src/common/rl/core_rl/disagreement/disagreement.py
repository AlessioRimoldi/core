"""Exploration via disagreement — PPO + a forward-dynamics ensemble."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks

from core_rl.algorithms import (
    BaseAlgorithm,
    MakePolicyFn,
    Metrics,
    Params,
    PolicyParamsFn,
    ProgressFn,
    register_algorithm,
)
from core_rl.algorithms.ppo import _map_config
from core_rl.dads.skill_autoreset import wrap_for_dads as wrap_resample_autoreset
from core_rl.disagreement import _disagreement_ppo
from core_rl.disagreement.ensemble import (
    build_reward_weights,
    make_ensemble,
    make_position_features,
    resolve_obs_indices,
)
from core_rl.tasks import BaseTask

# Keys consumed here, NOT passed through to the Brax-style trainer kwargs.
_DISAGREEMENT_KEYS = {
    "num_models",
    "ensemble_hidden_layer_sizes",
    "ensemble_lr",
    "ensemble_train_steps",
    "bootstrap_keep_prob",
    "int_coeff",
    "ext_coeff",
    "int_rew_ema_tau",
    "resample_on_reset",
    # Object-focused novelty (same knobs + semantics as disagreement_multi, via
    # the shared build_reward_weights — keeps arm comparisons apples-to-apples).
    # The weights are shared by the reward AND the ensemble loss.
    "ensemble_reward_indices",  # list[int] | "block" | None
    "ensemble_reward_bg_weight",  # weight on the OTHER dims (0 = restrict; >0 = keep bootstrap signal)
    # Frozen random position features appended to the ensemble TARGET (map-content
    # novelty — see ensemble.make_position_features). 0 = off.
    "ensemble_pos_features",  # int: number of φ target dims
    "ensemble_pos_feature_scale",  # φ length scale in meters (default 1.5)
}


@register_algorithm("disagreement")
class DisagreementAlgorithm(BaseAlgorithm):
    """Disagreement exploration via a vendored Brax PPO training loop."""

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
        """Run training via the vendored, disagreement-modified Brax PPO."""
        dis_cfg = {k: self._config[k] for k in _DISAGREEMENT_KEYS if k in self._config}
        ppo_cfg = {k: v for k, v in self._config.items() if k not in _DISAGREEMENT_KEYS}
        brax_cfg = _map_config(ppo_cfg)

        # Same top-level extraction / batch fixups as algorithms/ppo.py
        num_timesteps = brax_cfg.pop("total_timesteps", brax_cfg.pop("num_timesteps", 1_000_000))
        num_envs = brax_cfg.pop("num_envs", 4096)
        seed = ppo_cfg.get("seed", 0)
        num_evals = brax_cfg.pop("num_evals", 20)
        episode_length = brax_cfg.pop("max_episode_steps", brax_cfg.pop("episode_length", 500))

        # Brax requires: batch_size * num_minibatches % num_envs == 0
        num_minibatches = brax_cfg.get("num_minibatches", 32)
        if num_minibatches > num_envs:
            num_minibatches = num_envs
        batch_size = brax_cfg.get("batch_size", num_envs // num_minibatches)
        if batch_size == 0 or batch_size * num_minibatches % num_envs != 0:
            batch_size = num_envs // num_minibatches
        brax_cfg["batch_size"] = batch_size
        brax_cfg["num_minibatches"] = num_minibatches

        network_factory_kwargs = brax_cfg.pop("network_factory_kwargs", {})
        if network_factory_kwargs:
            if "hidden_layer_sizes" in network_factory_kwargs:
                sizes = network_factory_kwargs.pop("hidden_layer_sizes")
                network_factory_kwargs.setdefault("policy_hidden_layer_sizes", sizes)
                network_factory_kwargs.setdefault("value_hidden_layer_sizes", sizes)
            brax_cfg["network_factory"] = partial(ppo_networks.make_ppo_networks, **network_factory_kwargs)

        obs_size = self._env.observation_size
        if not isinstance(obs_size, int):
            raise ValueError(f"disagreement requires a flat observation vector, got {obs_size!r}")

        # Frozen random position features (map-content novelty; 0 = off). φ's
        # input dims are the same dims ensemble_reward_indices names.
        n_phi = int(dis_cfg.get("ensemble_pos_features", 0))
        pos_fn = pos_idx = None
        if n_phi > 0:
            pos_idx = resolve_obs_indices(obs_size, dis_cfg.get("ensemble_reward_indices"), env=self._env)
            if pos_idx is None:
                raise ValueError("ensemble_pos_features > 0 requires ensemble_reward_indices (the φ input dims)")
            pos_fn = make_position_features(len(pos_idx), n_phi, float(dis_cfg.get("ensemble_pos_feature_scale", 1.5)))

        ensemble_network = make_ensemble(
            obs_size=obs_size,
            action_size=self._env.action_size,
            num_models=int(dis_cfg.get("num_models", 5)),
            hidden_layer_sizes=tuple(dis_cfg.get("ensemble_hidden_layer_sizes", (256, 256))),
            target_size=obs_size + n_phi if n_phi else None,
        )

        # Object-focused novelty weights (None = uniform, unchanged behavior),
        # shared by the reward and the ensemble loss. φ dims get weight 1.0.
        ensemble_reward_weights = build_reward_weights(
            obs_size,
            dis_cfg.get("ensemble_reward_indices"),
            float(dis_cfg.get("ensemble_reward_bg_weight", 0.0)),
            env=self._env,
        )
        if ensemble_reward_weights is not None and n_phi > 0:
            ensemble_reward_weights = jnp.concatenate([ensemble_reward_weights, jnp.ones(n_phi, jnp.float32)])
        if ensemble_reward_weights is not None:
            print(
                f"  Disagreement focus:    {dis_cfg.get('ensemble_reward_indices')} "
                f"(bg_weight={dis_cfg.get('ensemble_reward_bg_weight', 0.0)}, pos_features={n_phi})"
            )

        # Auto-reset behaviour. Brax's stock AutoResetWrapper caches each env's
        # FIRST reset and replays it on every `done`, so initial states are
        # frozen per env for the whole run. With resample_on_reset=True we swap
        # in the resample-on-done wrapper (re-runs reset() → a fresh init pose
        # each episode), matching DADS's reset semantics. None → stock Brax.
        # Unlike DADS (where it is mandatory to refresh the skill z), this is
        # OPTIONAL for disagreement — see DISAGREEMENT_INTEGRATION.md §3.6.
        resample_on_reset = bool(dis_cfg.get("resample_on_reset", False))
        wrap_env_fn = wrap_resample_autoreset if resample_on_reset else None

        make_policy, params, metrics = _disagreement_ppo.train(
            environment=self._env,
            wrap_env_fn=wrap_env_fn,
            num_timesteps=num_timesteps,
            num_envs=num_envs,
            seed=seed,
            num_evals=num_evals,
            episode_length=episode_length,
            progress_fn=self._progress_fn,
            policy_params_fn=self._policy_params_fn,
            ensemble_network=ensemble_network,
            ensemble_lr=float(dis_cfg.get("ensemble_lr", 3.0e-4)),
            ensemble_train_steps=int(dis_cfg.get("ensemble_train_steps", 8)),
            bootstrap_keep_prob=float(dis_cfg.get("bootstrap_keep_prob", 0.8)),
            int_coeff=float(dis_cfg.get("int_coeff", 1.0)),
            ext_coeff=float(dis_cfg.get("ext_coeff", 0.0)),
            int_rew_ema_tau=float(dis_cfg.get("int_rew_ema_tau", 0.05)),
            ensemble_reward_weights=ensemble_reward_weights,
            ensemble_pos_feature_fn=pos_fn,
            ensemble_pos_indices=pos_idx,
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
        raise NotImplementedError("Use the make_policy function returned by train() directly.")
