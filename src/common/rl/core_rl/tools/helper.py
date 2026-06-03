from __future__ import annotations

import os
from pprint import pprint

import jax
import jax.numpy as jnp
import yaml

from core_rl.dads.dads import _infer_skill_dyn_sizes
from core_rl.dads.skill_dynamics import (
    make_skill_dynamics,
)

# Robot/config resolution — same path as train.py

# ---------------------------------------------------------------------------
# Config loading (mirrors train.py)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config(config_path: str | None = None) -> dict:
    """Load defaults.yaml and optionally merge with user config."""
    # Try ament share directory first (installed via colcon)
    try:
        from ament_index_python.packages import get_package_share_directory

        defaults_path = os.path.join(get_package_share_directory("core_rl"), "config", "defaults.yaml")
    except Exception:
        # Fallback: relative to source tree
        defaults_path = os.path.join(os.path.dirname(__file__), "..", "config", "defaults.yaml")

    with open(defaults_path) as f:
        config = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            override = yaml.safe_load(f)
        config = _deep_merge(config, override)

    return config


# ---------------------------------------------------------------------------
# Network reconstruction from config
# ---------------------------------------------------------------------------


def _build_sac_policy(env, dads_cfg: dict):
    """Recreate the SAC inference function matching what `_dads_sac.train` built."""
    from brax.training.acme import running_statistics
    from brax.training.agents.sac import networks as sac_networks

    normalize_obs = dads_cfg.get("normalize_observations", True)
    normalize_fn = running_statistics.normalize if normalize_obs else (lambda x, y: x)
    hidden = tuple(dads_cfg.get("network_factory_kwargs", {}).get("hidden_layer_sizes", (256, 256)))

    sac_network = sac_networks.make_sac_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=normalize_fn,
        hidden_layer_sizes=hidden,
    )
    return sac_networks.make_inference_fn(sac_network)


def _build_skill_dynamics(env, dads_cfg: dict):
    """returns q_phi_net, skill_size, input_obs_size, target_obs_size"""
    skill_size = dads_cfg.get("skill_size", 2)
    input_obs_size, target_obs_size = _infer_skill_dyn_sizes(env)
    skill_dyn_hidden = tuple(dads_cfg.get("skill_dyn_hidden_layer_sizes", (256, 256)))
    num_components = dads_cfg.get("num_mixture_components", 4)
    fixed_std = float(dads_cfg.get("fixed_std", 1.0))
    network = make_skill_dynamics(
        input_obs_size,
        target_obs_size,
        skill_size,
        skill_dyn_hidden,
        num_components,
        fixed_std=fixed_std,
    )
    return network, skill_size, input_obs_size, target_obs_size


# ---------------------------------------------------------------------------
# Skill sampling
# ---------------------------------------------------------------------------


def sample_eval_skills(skill_size: int, num_skills: int, rng: jax.Array) -> jax.Array:
    """Return ``(num_skills, skill_size)`` skills.

    For 2-D skill space with perfect-square count, returns a uniform grid
    on ``(-1, 1)²`` so visualizations are interpretable. Otherwise samples
    uniformly from the prior.
    """
    if skill_size == 2:
        side = int(round(num_skills**0.5))
        if side * side == num_skills:
            axis = jnp.linspace(-1.0, 1.0, side)
            xx, yy = jnp.meshgrid(axis, axis)
            return jnp.stack([xx.flatten(), yy.flatten()], axis=1)
    return jax.random.uniform(rng, (num_skills, skill_size), minval=-1.0, maxval=1.0)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_params(params: jax.Array):
    params_shape = jax.tree_util.tree_map(lambda x: x.shape, params)
    pprint(params_shape)
    print("======================")
