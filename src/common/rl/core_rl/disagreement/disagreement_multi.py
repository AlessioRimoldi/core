"""Multi-agent disagreement — shared policy trunk + `m` policy heads + PPO.

Same exploration signal as the single-agent disagreement arm (one shared ensemble
of forward-dynamics models, intrinsic reward = ensemble variance), but the single
policy is replaced by a population: ONE shared trunk feeding `num_policy_heads`
heads. The `num_envs` parallel envs are split into contiguous blocks (n/m envs per
head); each head is trained on its own block's experience, while the shared trunk,
the shared value network, and the ensemble are trained on ALL envs' experience.
See multi-agent-ideas.md §9.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from typing import Any

import jax

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
from core_rl.disagreement import multi_agent_disagreement_ppo
from core_rl.disagreement.ensemble import build_reward_weights, make_ensemble
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
    "num_policy_heads",
    # Object-focused novelty (same knobs + semantics as the dads_disagreement arm,
    # via the shared build_reward_weights — keeps arm comparisons apples-to-apples).
    "ensemble_reward_indices",  # list[int] | "block" | None
    "ensemble_reward_bg_weight",  # weight on the OTHER dims (0 = restrict; >0 = keep bootstrap signal)
}


@register_algorithm("disagreement_multi")
class MultiAgentDisagreementAlgorithm(BaseAlgorithm):
    """Multi-head disagreement exploration via a vendored Brax PPO loop."""

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

        # Trunk/head sizes come from the same `network_factory_kwargs.hidden_layer_sizes`
        # the other arms use. The multi-head trainer builds its own networks, so we
        # do NOT construct a Brax network_factory — just forward the layer sizes.
        network_factory_kwargs = brax_cfg.pop("network_factory_kwargs", {}) or {}
        hidden = tuple(network_factory_kwargs.get("hidden_layer_sizes", (256, 256)))

        obs_size = self._env.observation_size
        if not isinstance(obs_size, int):
            raise ValueError(f"disagreement_multi requires a flat observation vector, got {obs_size!r}")
        ensemble_network = make_ensemble(
            obs_size=obs_size,
            action_size=self._env.action_size,
            num_models=int(dis_cfg.get("num_models", 5)),
            hidden_layer_sizes=tuple(dis_cfg.get("ensemble_hidden_layer_sizes", (256, 256))),
        )

        # Object-focused novelty weights (None = uniform, unchanged behavior).
        ensemble_reward_weights = build_reward_weights(
            obs_size,
            dis_cfg.get("ensemble_reward_indices"),
            float(dis_cfg.get("ensemble_reward_bg_weight", 0.0)),
            env=self._env,
        )
        if ensemble_reward_weights is not None:
            print(
                f"  Disagreement focus:    {dis_cfg.get('ensemble_reward_indices')} "
                f"(bg_weight={dis_cfg.get('ensemble_reward_bg_weight', 0.0)})"
            )

        resample_on_reset = bool(dis_cfg.get("resample_on_reset", False))
        wrap_env_fn = wrap_resample_autoreset if resample_on_reset else None

        make_policy, params, metrics = multi_agent_disagreement_ppo.train(
            environment=self._env,
            wrap_env_fn=wrap_env_fn,
            num_timesteps=num_timesteps,
            num_envs=num_envs,
            seed=seed,
            num_evals=num_evals,
            episode_length=episode_length,
            progress_fn=self._progress_fn,
            policy_params_fn=self._policy_params_fn,
            # --- multi-agent (shared trunk + m policy heads) ---
            num_policy_heads=int(dis_cfg.get("num_policy_heads", 4)),
            policy_hidden_layer_sizes=hidden,
            value_hidden_layer_sizes=hidden,
            # --- disagreement ---
            ensemble_network=ensemble_network,
            ensemble_lr=float(dis_cfg.get("ensemble_lr", 3.0e-4)),
            ensemble_train_steps=int(dis_cfg.get("ensemble_train_steps", 8)),
            bootstrap_keep_prob=float(dis_cfg.get("bootstrap_keep_prob", 0.8)),
            int_coeff=float(dis_cfg.get("int_coeff", 1.0)),
            ext_coeff=float(dis_cfg.get("ext_coeff", 0.0)),
            int_rew_ema_tau=float(dis_cfg.get("int_rew_ema_tau", 0.05)),
            ensemble_reward_weights=ensemble_reward_weights,
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
