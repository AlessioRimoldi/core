"""DADS algorithm — SAC + a learned skill-dynamics model q_φ.
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
from core_rl.dads import _dads_sac
from core_rl.dads.skill_autoreset import wrap_for_dads
from core_rl.dads.skill_dynamics import make_skill_dynamics
from core_rl.tasks import BaseTask


def _split_config(config: dict[str, Any]) -> tuple[dict, dict]:
    """Separate SAC keys (matching Brax param names) from DADS keys."""
    dads_keys = {
        "skill_size",
        "skill_dyn_lr",
        "skill_dyn_train_steps",
        "skill_dyn_hidden_layer_sizes",
        "num_mixture_components",
        "fixed_std",
        "prior_samples",
        "train_skill_dynamics_on_policy",
        "is_clip_eps",
    }
    dads_cfg = {k: config[k] for k in dads_keys if k in config}
    sac_cfg = {k: v for k, v in config.items() if k not in dads_keys}
    return sac_cfg, dads_cfg


def _map_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map defaults.yaml SAC keys to Brax SAC parameter names."""
    mapping = {
        "learning_rate": "learning_rate",
        "buffer_size": "max_replay_size",
        "learning_starts": "min_replay_size",
        "batch_size": "batch_size",
        "tau": "tau",
        "gamma": "discounting",
        # Brax-native keys
        "max_replay_size": "max_replay_size",
        "min_replay_size": "min_replay_size",
        "discounting": "discounting",
        "grad_updates_per_step": "grad_updates_per_step",
        "reward_scaling": "reward_scaling",
        "normalize_observations": "normalize_observations",
    }

    brax_cfg: dict[str, Any] = {}
    for key, value in config.items():
        if key in ("policy", "policy_kwargs", "seed", "device", "ent_coef"):
            continue
        brax_key = mapping.get(key, key)
        brax_cfg[brax_key] = value

    # Network architecture from policy_kwargs
    policy_kwargs = config.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        net_arch = tuple(int(x) for x in policy_kwargs["net_arch"])
        brax_cfg["network_factory_kwargs"] = {"hidden_layer_sizes": net_arch}

    return brax_cfg


def _infer_skill_dyn_sizes(env: BaseTask) -> tuple[int, int]:
    """Derive ``(input_obs_size, target_obs_size)`` from a SkillConditionedTask.

    Reads the wrapped env's ``input_obs_size`` and ``target_obs_size``
    properties, which come from the ``input_obs_indices`` /
    ``target_obs_indices`` config. Raises ValueError if the env doesn't
    look like a SkillConditionedTask.
    """
    _missing = object()
    input_size = getattr(env, "input_obs_size", _missing)
    target_size = getattr(env, "target_obs_size", _missing)
    if input_size is _missing or target_size is _missing:
        raise ValueError(
            "DADS requires a SkillConditionedTask-wrapped env exposing "
            f"`input_obs_size` and `target_obs_size`. Got {type(env).__name__}. "
            "Wrap your task with skill_conditioned and set "
            "`input_obs_indices` + `target_obs_indices` in the YAML."
        )
    return int(input_size), int(target_size)


@register_algorithm("dads")
class DADSAlgorithm(BaseAlgorithm):
    """DADS via Brax's JIT-compiled training loop."""

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

        sac_cfg, dads_cfg = _split_config(self._config)
        brax_cfg = _map_config(sac_cfg)

        num_timesteps = brax_cfg.pop("total_timesteps", brax_cfg.pop("num_timesteps", 1_000_000))
        num_envs = brax_cfg.pop("num_envs", 4096)
        seed = self._config.get("seed", 0)
        num_evals = brax_cfg.pop("num_evals", 20)
        episode_length = brax_cfg.pop("max_episode_steps", brax_cfg.pop("episode_length", 500))

        num_eval_envs = brax_cfg.pop("num_eval_envs", min(num_envs, 64))

        skill_size = dads_cfg.get("skill_size", 2)

        env_skill_size = getattr(self._env, "_skill_size", None)
        if env_skill_size is not None and int(env_skill_size) != int(skill_size):
            raise ValueError(
                f"skill_size mismatch: algorithms.dads.skill_size={skill_size} but "
                f"env.task_kwargs.skill_conditioned.skill_size={env_skill_size}. "
                "These MUST match — the env supplies z and DADS builds q_φ for that z. "
                "Set both to the same value (or rely on train.py's auto-sync)."
            )
        # Derive (input_size, target_size) from the wrapped env so they
        # can't drift out of sync with input_obs_indices / target_obs_indices.
        input_obs_size, target_obs_size = _infer_skill_dyn_sizes(self._env)
        skill_dyn_hidden_layer_sizes = tuple(dads_cfg.get("skill_dyn_hidden_layer_sizes", (256, 256)))
        num_mixture_components = dads_cfg.get("num_mixture_components", 4)
        fixed_std = float(dads_cfg.get("fixed_std", 1.0))

        print(f"  DADS skill_size:       {skill_size}")
        print(f"  DADS input_obs_size:   {input_obs_size}   (q_φ conditioning state)")
        print(f"  DADS target_obs_size:  {target_obs_size}   (q_φ Δ prediction target)")
        print(f"  DADS q_phi fixed_std:  {fixed_std}")
        skill_dynamics_network = make_skill_dynamics(
            input_obs_size,
            target_obs_size,
            skill_size,
            skill_dyn_hidden_layer_sizes,
            num_mixture_components,
            fixed_std=fixed_std,
        )

        network_factory_kwargs = brax_cfg.pop("network_factory_kwargs", {})
        if network_factory_kwargs:
            from functools import partial

            from brax.training.agents.sac import networks as sac_networks

            network_factory = partial(sac_networks.make_sac_networks, **network_factory_kwargs)
            brax_cfg["network_factory"] = network_factory

        make_policy, params, metrics = _dads_sac.train(
            environment=self._env,
            # DADS-faithful auto-reset: resample skill z + init pose each
            # episode (brax's default freezes them for the whole run).
            wrap_env_fn=wrap_for_dads,
            num_timesteps=num_timesteps,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            seed=seed,
            num_evals=num_evals,
            episode_length=episode_length,
            progress_fn=self._progress_fn,
            policy_params_fn=self._policy_params_fn,
            # --- DADS ---
            skill_dynamics_network=skill_dynamics_network,
            skill_size=skill_size,
            input_obs_size=input_obs_size,
            target_obs_size=target_obs_size,
            skill_dyn_lr=dads_cfg.get("skill_dyn_lr", 3e-4),
            skill_dyn_train_steps=dads_cfg.get("skill_dyn_train_steps", 8),
            prior_samples=dads_cfg.get("prior_samples", 100),
            train_skill_dynamics_on_policy=bool(dads_cfg.get("train_skill_dynamics_on_policy", True)),
            is_clip_eps=float(dads_cfg.get("is_clip_eps", 10.0)),
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
