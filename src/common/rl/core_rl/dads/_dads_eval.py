"""DADS evaluator.

A drop-in replacement for ``brax.training.acting.Evaluator`` that runs a
SINGLE eval rollout and reports BOTH the usual task-reward / episode metrics
AND the DADS skill-dynamics diagnostics (``r_dads``, ``q_phi_loss``, …)
computed on that same rollout.

Two problems with the naive approach (a second standalone eval rollout) that
this fixes:

* **Double work** — the brax ``Evaluator`` already rolls out the physics for
  ``num_eval_envs`` envs; a separate DADS eval ran it a second time. Here a
  single ``generate_unroll`` feeds both metric sets.
* **OOM** — ``r_dads`` evaluates ``q_φ`` on ``(B, L)`` alt-skill pairs, where
  the eval batch ``B = episode_length × num_eval_envs`` is far larger than the
  SAC minibatch used during training. With ``L = prior_samples`` that single
  ``vmap`` materializes a ``(B·L, hidden)`` activation tensor (multiple GB).
  We chunk over ``B`` — exactly as ``eval_dads.py`` does — so peak memory is
  bounded by ``reward_chunk × L`` and ``compute_dads_reward`` is reused as-is.
"""

import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training import acting, networks
from brax.training.types import Metrics, Params, PolicyParams, PRNGKey

from core_rl.dads.skill_dynamics import (
    QPhiNorm,
    _standardize,
    compute_dads_reward,
    log_prob,
    modal_delta,
    skill_dynamics_loss,
)


def dads_diagnostics(
    skill_dynamics_network: networks.FeedForwardNetwork,
    skill_dynamics_params: Params,
    s_input: jnp.ndarray,
    z: jnp.ndarray,
    s_target: jnp.ndarray,
    s_target_next: jnp.ndarray,
    r_dads: jnp.ndarray,
    prior_samples: int,
    norm: QPhiNorm | None = None,
    action: jnp.ndarray | None = None,
) -> Metrics:
    """q_φ skill-dynamics quality + DADS-reward summary metrics for a batch.

    Shared by the training ``sgd_step`` and the eval rollout so both report an
    identical metric set (train/eval parity is the whole point). ``r_dads`` is
    passed in (already computed by the caller) so the caller picks how to
    compute it — ``vmap`` over the small SAC minibatch during training, or
    chunked over the huge flattened batch during eval — without this helper
    having to care.

    Note: ``s_input`` and ``s_target`` may have different dimensions under
    Option A (input = [q, dq], target = ee_pos). The Δ that q_φ predicts
    lives in the target space.
    """
    delta_target = s_target_next - s_target  # (B, target_size)
    delta_norm = jnp.linalg.norm(delta_target, axis=-1)  # (B,)

    # log q_φ(Δ | s_input, z) — the numerator of r_dads (q_φ confidence on
    # the actually-executed skill). log_prob applies `norm` internally.
    logp = log_prob(
        skill_dynamics_network, skill_dynamics_params, s_input, z, delta_target, norm  # type: ignore[arg-type]
    )  # (B,)

    # Modal-component prediction in PHYSICAL target-space units (modal_delta
    # un-normalizes) + its L2 error vs the observed Δ.
    modal_pred = modal_delta(
        skill_dynamics_network, skill_dynamics_params, s_input, z, norm  # type: ignore[arg-type]
    )  # (B, target_size)
    pred_err = jnp.linalg.norm(modal_pred - delta_target, axis=-1)  # (B,)

    # MoG component usage: entropy of the mixture weights, computed on the
    # (normalized) input q_φ actually sees. Near log(K) → all components used;
    # near 0 → collapsed to a single component.
    s_in = _standardize(s_input, norm.s_mean, norm.s_std) if norm is not None else s_input
    logits, means, _ = skill_dynamics_network.apply(  # type: ignore[arg-type]
        skill_dynamics_params, s_in, z
    )  # logits (B,K), means (B,K,target)
    mix_w = jax.nn.softmax(logits, axis=-1)  # (B, K)
    mix_entropy = -jnp.sum(mix_w * jnp.log(mix_w + 1e-8), axis=-1)  # (B,)

    # ── QPhiNorm calibration + logp-explosion source ─────────────────────
    # q_φ trains/predicts on STANDARDIZED I/O. If QPhiNorm is healthy, the
    # standardized input AND Δ-target should each be ~mean 0 / std 1. A
    # standardized-Δ std >> 1 means delta_std is too small or lagging the data
    # → the Gaussian exponent (Δ_norm − μ)² blows up → logp → −∞ → r_dads pins
    # at the −50 floor. These metrics let you see that directly.
    delta_zscore = (
        _standardize(delta_target, norm.delta_mean, norm.delta_std) if norm is not None else delta_target
    )  # (B, target)
    # Modal-component mean in NORMALIZED space (no un-normalize), so sq_err is
    # the exponent q_φ actually optimizes. With fixed σ, sq_err ≈ −2·logp+const.
    k_star = jnp.argmax(logits, axis=-1)  # (B,)
    modal_mean_norm = jnp.take_along_axis(means, k_star[..., None, None], axis=-2).squeeze(-2)  # (B, target)
    sq_err_norm = jnp.sum((delta_zscore - modal_mean_norm) ** 2, axis=-1)  # (B,)
    # raw-batch σ vs the stored EMA σ: ~1 if QPhiNorm tracks the data; >1 means
    # the stored σ is too small (under-scales → explosion), <1 too large (flat).
    if norm is not None:
        delta_std_ratio = (delta_target.std(axis=0) / (norm.delta_std + 1e-8)).mean()
        delta_std_used = norm.delta_std.mean()
        # Smallest per-dim stored σ — a single near-constant target axis (e.g.
        # ee-z) drives σ→0, so Δ/σ explodes for that dim. The mean hides it.
        delta_std_min = norm.delta_std.min()
        s_std_used = norm.s_std.mean()
    else:
        delta_std_ratio = jnp.float32(1.0)
        delta_std_used = jnp.float32(0.0)
        delta_std_min = jnp.float32(0.0)
        s_std_used = jnp.float32(0.0)

    # r_dads ceiling is log(L+1); normalize for cross-config comparison.
    r_ceiling = jnp.log(prior_samples + 1.0)

    metrics = {
        # ── q_φ skill-dynamics quality ──
        "q_phi_logp_mean": logp.mean(),  # log q(Δ|s,z) — higher better
        "q_phi_logp_std": logp.std(),
        "q_phi_logp_min": logp.min(),  # worst transition; ≪0 = dead
        "q_phi_frac_logp_blown": (logp < -50.0).mean(),  # frac of near-zero-prob transitions
        "q_phi_pred_err_mean": pred_err.mean(),  # ‖Δ − μ_modal‖ (target units) — lower better
        "q_phi_pred_err_std": pred_err.std(),
        "q_phi_mixture_entropy": mix_entropy.mean(),  # MoG usage; ↓ near 0 = collapse
        # ── logp explosion source (normalized Gaussian exponent) ──
        "q_phi_sqerr_norm_mean": sq_err_norm.mean(),  # mean modal Mahalanobis term
        "q_phi_sqerr_norm_max": sq_err_norm.max(),  # worst case → drives logp_min
        # ── QPhiNorm calibration (both should be ~0 mean / ~1 std) ──
        "qphi_input_norm_mean": s_in.mean(),  # standardized s_input mean → 0
        "qphi_input_norm_std": s_in.std(),  # standardized s_input std → 1
        "qphi_delta_norm_mean": delta_zscore.mean(),  # standardized Δ mean → 0
        "qphi_delta_norm_std": delta_zscore.std(),  # standardized Δ std → 1; ≫1 = under-scaled
        "qphi_delta_std_ratio": delta_std_ratio,  # raw σ / stored σ; ~1 healthy, >1 lagging
        "qphi_delta_std_used": delta_std_used,  # mean stored Δ σ (EMA) — watch for collapse
        "qphi_delta_std_min": delta_std_min,  # smallest per-dim σ; →0 = a dim explodes
        "qphi_s_std_used": s_std_used,  # mean stored input σ (EMA)
        # ── DADS intrinsic reward ──
        "r_dads_mean": r_dads.mean(),
        "r_dads_std": r_dads.std(),  # spread of skill quality across batch
        "r_dads_min": r_dads.min(),  # worst skill in the batch
        "r_dads_max": r_dads.max(),  # best skill in the batch
        "r_dads_frac_at_floor": (r_dads <= -49.0).mean(),  # frac pinned at the −50 clamp
        "r_dads_normalized": r_dads.mean() / r_ceiling,  # fraction of log(L+1) ceiling (0→1)
        "r_dads_frac_positive": (r_dads > 0).mean(),  # frac of discriminable transitions
        # ── behavior sanity check ──
        "delta_target_norm_mean": delta_norm.mean(),  # is the target actually moving?
        "delta_target_norm_std": delta_norm.std(),
    }

    # ── policy action saturation (read-only; tanh-squashed a ∈ (−1, 1)) ──
    # A policy chasing a broken reward slams actions to the bounds, driving the
    # state out of q_φ's training distribution — a common partner to a QPhiNorm
    # blow-up. Purely diagnostic: `action` is the already-executed action, so
    # this never touches the reward/loss/buffer.
    if action is not None:
        a_abs = jnp.abs(action)
        metrics["action_abs_mean"] = a_abs.mean()  # overall magnitude
        metrics["action_frac_saturated"] = (a_abs > 0.99).mean()  # frac slammed to ±1
        metrics["action_max_abs"] = a_abs.max()

    return metrics


def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
    # Mirrors brax.training.acting._agg_fn (private), reimplemented to avoid
    # importing a private symbol.
    if not to_aggregate:
        return metric
    if to_normalize:
        return fn(metric / episode_lengths)
    return fn(metric)


class DadsEvaluator:
    """Like ``brax.training.acting.Evaluator`` but also reports DADS metrics."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[PolicyParams], Any],
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
        skill_dynamics_network: networks.FeedForwardNetwork,
        skill_size: int,
        input_obs_size: int,
        target_obs_size: int,
        prior_samples: int,
        reward_chunk: int = 256,
    ):
        self._key = key
        self._eval_walltime = 0.0
        self._steps_per_unroll = episode_length * num_eval_envs

        self._network = skill_dynamics_network
        self._skill_size = skill_size
        self._input_obs_size = input_obs_size
        self._target_obs_size = target_obs_size
        self._prior_samples = prior_samples
        self._reward_chunk = reward_chunk

        eval_env = envs.training.EvalWrapper(eval_env)

        def eval_unroll(policy_params, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            first_state = eval_env.reset(reset_keys)
            policy = eval_policy_fn(policy_params)
            final_state, transitions = acting.generate_unroll(
                eval_env,
                first_state,
                policy,
                key,
                unroll_length=episode_length // action_repeat,
                extra_fields=("z", "s_input", "s_input_next", "s_target", "s_target_next"),
            )
            # Flatten (unroll_length, num_eval_envs, …) → (B, …) on-device so
            # only the small restricted-state arrays cross to the host.
            se = transitions.extras["state_extras"]
            s_input = se["s_input"].reshape(-1, input_obs_size)
            z = se["z"].reshape(-1, skill_size)
            s_target = se["s_target"].reshape(-1, target_obs_size)
            s_target_next = se["s_target_next"].reshape(-1, target_obs_size)
            # action is already on the transition (no extra_fields plumbing).
            action = transitions.action.reshape(-1, transitions.action.shape[-1])
            # Per-env (= per-SKILL, z fixed for the episode) cube-trajectory
            # summaries, from the UNFLATTENED (L, envs, target) target = cube pos.
            # RESET-BOUNDARY CARE: actor_step reads extras from the POST-step
            # info, and ResampleAutoResetWrapper splices the RESET info into the
            # done step — so the final transition's s_target AND s_target_next
            # are both the post-reset (re-jittered) cube position. Using either
            # measures init jitter, not pushing (bit us twice: constant
            # push_dist ≈ E‖jitter−jitter‖ ≈ 0.031 across all runs). The last
            # valid in-episode state is stn[-2] (state after step L−2), so net
            # displacement and path both drop the final transition entirely.
            # Assumes episodes only end by truncation at episode_length (true
            # for fetchpush: no early termination) and L ≥ 2.
            st = se["s_target"]  # (L, envs, target)
            stn = se["s_target_next"]  # (L, envs, target)
            cube_net_disp = stn[-2] - st[0]  # (envs, target) — net push per skill
            cube_path_len = jnp.sum(jnp.linalg.norm(stn[:-1] - st[:-1], axis=-1), axis=0)  # (envs,)
            return final_state, s_input, z, s_target, s_target_next, action, cube_net_disp, cube_path_len

        self._eval_unroll = jax.jit(eval_unroll)
        # q_φ forward passes are cheap at full B (no L factor), so jit them once.
        # `norm` (QPhiNorm) is passed as a traced arg so the same compiled fn
        # serves any stats values.
        self._diagnostics = jax.jit(
            lambda params, s_in, z, s_t, s_tn, r, norm, act: dads_diagnostics(
                skill_dynamics_network, params, s_in, z, s_t, s_tn, r, prior_samples, norm, act
            )
        )
        self._q_phi_loss = jax.jit(
            lambda params, s_in, z, dt, norm: skill_dynamics_loss(skill_dynamics_network, params, s_in, z, dt, norm)
        )
        self._reward_chunk_fn = jax.jit(
            lambda params, s_in, z, dt, za, norm: compute_dads_reward(
                skill_dynamics_network, params, s_in, z, dt, za, norm=norm
            )
        )

    def _r_dads_chunked(self, params, s_input, z, delta_target, key, norm):
        """r_dads over the full batch, computed ``reward_chunk`` rows at a time.

        Each chunk's ``(chunk × L, hidden)`` intermediate is small; the full
        ``(B × L, hidden)`` tensor (which OOMs) is never materialized.
        """
        B = s_input.shape[0]  # noqa: N806  (batch dim, matches `B × L` notation in docstring)
        parts = []
        for start in range(0, B, self._reward_chunk):
            end = min(start + self._reward_chunk, B)
            z_alts = jax.random.uniform(
                jax.random.fold_in(key, start),
                (end - start, self._prior_samples, self._skill_size),
                minval=-1.0,
                maxval=1.0,
            )
            parts.append(
                self._reward_chunk_fn(params, s_input[start:end], z[start:end], delta_target[start:end], z_alts, norm)
            )
        return jnp.concatenate(parts, axis=0)

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        skill_dynamics_params: Params,
        qphi_norm: QPhiNorm,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation (task metrics + DADS diagnostics)."""
        self._key, unroll_key, metrics_key = jax.random.split(self._key, 3)

        t = time.time()
        eval_state, s_input, z, s_target, s_target_next, action, cube_net_disp, cube_path_len = self._eval_unroll(
            policy_params, unroll_key
        )
        delta_target = s_target_next - s_target

        # ── Standard brax episode metrics (identical to acting.Evaluator) ──
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        eval_metrics = jax.tree.map(np.asarray, eval_metrics)
        episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)

        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            for name, value in eval_metrics.episode_metrics.items():
                metrics[f"eval/episode_{name}{suffix}"] = _agg_fn(
                    value,
                    fn,
                    aggregate_episodes,
                    name.endswith("per_step"),
                    episode_lengths,
                )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/std_episode_length"] = np.std(eval_metrics.episode_steps)

        # ── DADS skill-dynamics diagnostics on the SAME rollout ──
        r_dads = self._r_dads_chunked(skill_dynamics_params, s_input, z, delta_target, metrics_key, qphi_norm)
        dads_metrics = self._diagnostics(
            skill_dynamics_params, s_input, z, s_target, s_target_next, r_dads, qphi_norm, action
        )
        # q_φ is frozen at eval, so q_phi_loss is a single forward-pass NLL
        # (no inner training steps as in sgd_step).
        dads_metrics["q_phi_loss"] = self._q_phi_loss(skill_dynamics_params, s_input, z, delta_target, qphi_norm)
        metrics.update({f"eval/{name}": np.asarray(value) for name, value in dads_metrics.items()})

        # ── cube-skill metrics (interpretable, in cube-position units) ──
        # cube_net_disp: (envs, target) net cube push of each skill's episode.
        #   push_dist_mean       — how far skills push the cube (0 ⇒ cube untouched).
        #   push_directedness    — ‖net push‖ / path length ∈ [0,1]; ~1 = clean push,
        #                          ~0 = jitter in place (contact but no transport).
        #   skill_cube_disp_spread — spread of the net-push VECTORS across skills:
        #                          the direct "do different skills move the cube to
        #                          different places" signal (upper bound — includes
        #                          intra-skill noise, but 0 ⇒ skills are cube-identical).
        #   push_dir_entropy     — magnitude-weighted entropy of push DIRECTIONS over
        #                          8 bins (0 ⇒ all one way; log8 ⇒ pushes every way).
        cube_net_disp = np.asarray(cube_net_disp)  # (envs, target)
        cube_path_len = np.asarray(cube_path_len)  # (envs,)
        push_dist = np.linalg.norm(cube_net_disp, axis=-1)  # (envs,)
        spread = float(np.sqrt(((cube_net_disp - cube_net_disp.mean(axis=0)) ** 2).sum(axis=-1).mean()))
        metrics["eval/cube_push_dist_mean"] = float(push_dist.mean())
        metrics["eval/cube_push_directedness"] = float((push_dist / (cube_path_len + 1e-8)).mean())
        metrics["eval/skill_cube_disp_spread"] = spread
        if cube_net_disp.shape[-1] >= 2:  # need x,y for a push direction
            ang = np.arctan2(cube_net_disp[:, 1], cube_net_disp[:, 0])  # (envs,)
            hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi), weights=push_dist)
            p = hist / (hist.sum() + 1e-8)
            metrics["eval/cube_push_dir_entropy"] = float(-(p * np.log(p + 1e-8)).sum())

        epoch_eval_time = time.time() - t
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }
        return metrics
