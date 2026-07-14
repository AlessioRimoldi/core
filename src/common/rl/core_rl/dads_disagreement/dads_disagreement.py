"""Disagreement-fuelled skill discovery — DADS + a forward-dynamics ensemble.

SAC optimizes

    r = r_dads(s, z, Δ) + β · d̂(s, a)

with ``d̂`` the EMA-RMS-normalized ensemble disagreement (epistemic novelty).
``β = 0`` recovers vanilla DADS, so the two reference arms (vanilla DADS and
single-agent disagreement) are a β-sweep / an ablation away.

This wrapper is ``dads/dads.py`` plus an ensemble: it reuses
:func:`core_rl.disagreement.ensemble.make_ensemble` for the side head and
:mod:`core_rl.dads_disagreement._dads_disagreement_sac` for the fused trainer.
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
from core_rl.dads.skill_autoreset import wrap_for_dads
from core_rl.dads.skill_dynamics import make_skill_dynamics
from core_rl.dads_disagreement import _dads_disagreement_sac
from core_rl.disagreement.ensemble import build_reward_weights, make_ensemble
from core_rl.tasks import BaseTask

# Keys consumed by this wrapper (DADS + disagreement), NOT passed through to the
# Brax SAC kwargs. DADS keys mirror dads.py; the rest are the ensemble knobs.
_DADS_KEYS = {
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
_DISAGREEMENT_KEYS = {
    "beta",
    "num_models",
    "ensemble_hidden_layer_sizes",
    "ensemble_lr",
    "ensemble_train_steps",
    "bootstrap_keep_prob",
    "int_rew_ema_tau",
    # Object-focused novelty: weight the disagreement bonus toward these obs dims.
    "ensemble_reward_indices",  # list[int] | "block" | None
    "ensemble_reward_bg_weight",  # weight on the OTHER dims (0 = restrict; >0 = keep bootstrap signal)
    # Hindsight skill relabeling (exploration data → skill supervision, HINDSIGHT.md).
    "hindsight_relabeling",  # master switch (bool)
    "relabel_mode",  # "direct" (z* = achieved outcome, HER-style) | "posterior" (EM)
    "contact_eps",  # ‖Δ_target‖ above which a row counts as contact
    "relabel_prob",  # HER-style mixture: fraction of eligible SAC rows relabeled
    "relabel_candidates",  # prior z' samples scored for the posterior
    "relabel_temperature",  # posterior temperature (>1 flattens, anti-collapse)
    "archive_size",  # contact-archive capacity
    "archive_insert_topk",  # rows archived per actor step (top-K by ‖Δ_target‖)
    "archive_sac_frac",  # fraction of each SAC minibatch drawn from the archive
    "archive_qphi_frac",  # fraction of each q_φ minibatch drawn from the archive (rest = fresh)
}


def _split_config(config: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Separate SAC keys (Brax param names) from DADS and disagreement keys."""
    dads_cfg = {k: config[k] for k in _DADS_KEYS if k in config}
    dis_cfg = {k: config[k] for k in _DISAGREEMENT_KEYS if k in config}
    sac_cfg = {k: v for k, v in config.items() if k not in _DADS_KEYS and k not in _DISAGREEMENT_KEYS}
    return sac_cfg, dads_cfg, dis_cfg


def _map_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map defaults.yaml SAC keys to Brax SAC parameter names (same as dads.py)."""
    mapping = {
        "learning_rate": "learning_rate",
        "buffer_size": "max_replay_size",
        "learning_starts": "min_replay_size",
        "batch_size": "batch_size",
        "tau": "tau",
        "gamma": "discounting",
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

    policy_kwargs = config.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        net_arch = tuple(int(x) for x in policy_kwargs["net_arch"])
        brax_cfg["network_factory_kwargs"] = {"hidden_layer_sizes": net_arch}

    return brax_cfg


def _infer_skill_dyn_sizes(env: BaseTask) -> tuple[int, int]:
    """Derive ``(input_obs_size, target_obs_size)`` from a SkillConditionedTask."""
    _missing = object()
    input_size = getattr(env, "input_obs_size", _missing)
    target_size = getattr(env, "target_obs_size", _missing)
    if input_size is _missing or target_size is _missing:
        raise ValueError(
            "dads_disagreement requires a SkillConditionedTask-wrapped env exposing "
            f"`input_obs_size` and `target_obs_size`. Got {type(env).__name__}. "
            "Wrap your task with skill_conditioned and set "
            "`input_obs_indices` + `target_obs_indices` in the YAML."
        )
    return int(input_size), int(target_size)


@register_algorithm("dads_disagreement")
class DADSDisagreementAlgorithm(BaseAlgorithm):
    """DADS + disagreement (``r = r_dads + β·d``) via the fused Brax SAC loop."""

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

        sac_cfg, dads_cfg, dis_cfg = _split_config(self._config)
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
                f"skill_size mismatch: algorithms.dads_disagreement.skill_size={skill_size} but "
                f"env.task_kwargs.skill_conditioned.skill_size={env_skill_size}. "
                "These MUST match — the env supplies z and DADS builds q_φ for that z. "
                "Set both to the same value (or rely on train.py's auto-sync)."
            )

        # q_φ I/O sizes (restricted skill space) — same as DADS.
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

        # ── disagreement: the forward ensemble side head. It works on the FULL
        #    SAC observation (incl. z) — the same input convention as the
        #    single-agent disagreement arm — predicting normalized Δobs.
        obs_size = self._env.observation_size
        if not isinstance(obs_size, int):
            raise ValueError(f"dads_disagreement requires a flat observation vector, got {obs_size!r}")
        beta = float(dis_cfg.get("beta", 0.0))
        ensemble_network = make_ensemble(
            obs_size=obs_size,
            action_size=self._env.action_size,
            num_models=int(dis_cfg.get("num_models", 5)),
            hidden_layer_sizes=tuple(dis_cfg.get("ensemble_hidden_layer_sizes", (256, 256))),
        )

        # ── object-focused novelty weights (Pathak-style saliency for low-dim obs) ─
        # `ensemble_reward_indices` selects the dims the novelty bonus focuses on:
        #   None      → uniform mean over all dims (default; unchanged behavior).
        #   "block"   → the env's object dims (env.block_obs_indices).
        #   list[int] → explicit dims.
        # `ensemble_reward_bg_weight` (default 0) weights the OTHER dims — keep a
        # small value (>0) so some arm/ee novelty survives to bootstrap reaching.
        ensemble_reward_weights = self._build_reward_weights(
            obs_size,
            dis_cfg.get("ensemble_reward_indices"),
            float(dis_cfg.get("ensemble_reward_bg_weight", 0.0)),
        )
        if ensemble_reward_weights is not None:
            focus = dis_cfg.get("ensemble_reward_indices")
            print(f"  Disagreement focus:    {focus} (bg_weight={dis_cfg.get('ensemble_reward_bg_weight', 0.0)})")
        print(f"  Disagreement beta:     {beta}   (0 ⇒ vanilla DADS)")
        print(f"  Disagreement K:        {int(dis_cfg.get('num_models', 5))}   (ensemble size)")
        if dis_cfg.get("hindsight_relabeling"):
            print(
                f"  Hindsight relabeling:  ON (mode={dis_cfg.get('relabel_mode', 'direct')}, "
                f"p={dis_cfg.get('relabel_prob', 0.5)}, "
                f"candidates={dis_cfg.get('relabel_candidates', 64)}, "
                f"T={dis_cfg.get('relabel_temperature', 1.0)}, "
                f"archive={dis_cfg.get('archive_size', 200_000)}"
                f"@top{dis_cfg.get('archive_insert_topk', 128)}/step, "
                f"sac_frac={dis_cfg.get('archive_sac_frac', 0.25)}, "
                f"qphi_frac={dis_cfg.get('archive_qphi_frac', 0.5)}, "
                f"contact_eps={dis_cfg.get('contact_eps', 1e-4)})"
            )

        network_factory_kwargs = brax_cfg.pop("network_factory_kwargs", {})
        if network_factory_kwargs:
            from functools import partial

            from brax.training.agents.sac import networks as sac_networks

            network_factory = partial(sac_networks.make_sac_networks, **network_factory_kwargs)
            brax_cfg["network_factory"] = network_factory

        # train.py injects `save_checkpoint_path` when --save-checkpoints is set;
        # the trainer's name for it is `checkpoint_logdir`.
        checkpoint_logdir = brax_cfg.pop("save_checkpoint_path", None)

        make_policy, params, metrics = _dads_disagreement_sac.train(
            environment=self._env,
            checkpoint_logdir=checkpoint_logdir,
            # DADS-faithful auto-reset: resample skill z + init pose each episode.
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
            # --- disagreement ---
            ensemble_network=ensemble_network,
            beta=beta,
            ensemble_lr=float(dis_cfg.get("ensemble_lr", 3e-4)),
            ensemble_train_steps=int(dis_cfg.get("ensemble_train_steps", 8)),
            bootstrap_keep_prob=float(dis_cfg.get("bootstrap_keep_prob", 0.8)),
            int_rew_ema_tau=float(dis_cfg.get("int_rew_ema_tau", 0.05)),
            ensemble_reward_weights=ensemble_reward_weights,
            # --- hindsight skill relabeling ---
            hindsight_relabeling=bool(dis_cfg.get("hindsight_relabeling", False)),
            relabel_mode=str(dis_cfg.get("relabel_mode", "direct")),
            contact_eps=float(dis_cfg.get("contact_eps", 1e-4)),
            relabel_prob=float(dis_cfg.get("relabel_prob", 0.5)),
            relabel_candidates=int(dis_cfg.get("relabel_candidates", 64)),
            relabel_temperature=float(dis_cfg.get("relabel_temperature", 1.0)),
            archive_size=int(dis_cfg.get("archive_size", 200_000)),
            archive_insert_topk=int(dis_cfg.get("archive_insert_topk", 128)),
            archive_sac_frac=float(dis_cfg.get("archive_sac_frac", 0.25)),
            archive_qphi_frac=float(dis_cfg.get("archive_qphi_frac", 0.5)),
            **brax_cfg,
        )

        return make_policy, params, metrics

    def _build_reward_weights(self, obs_size, indices, bg_weight):
        """Per-obs-dim weight vector for the disagreement bonus, or None (uniform).

        Delegates to the shared :func:`build_reward_weights` so the weighting
        semantics are identical across all disagreement arms (fair comparisons).
        """
        return build_reward_weights(obs_size, indices, bg_weight, env=self._env)

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
