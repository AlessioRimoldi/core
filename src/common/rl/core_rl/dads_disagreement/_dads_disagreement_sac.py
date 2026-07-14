# Copyright 2026 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disagreement-fuelled skill discovery — DADS (SAC + q_φ) + a forward ensemble.

This is the DADS/SAC trainer (``dads/_dads_sac.py``) with one addition: a
forward-dynamics ensemble ``f_1..f_K`` supplies a novelty bonus, and the reward
SAC optimizes becomes

    r = r_dads(s, z, Δ) + β · d̂(s, a)

where ``r_dads`` is the usual DADS skill-distinguishability reward (a variational
estimate of ``I(s'; z | s)``) and ``d̂`` is the EMA-RMS-normalized ensemble
disagreement ``Var_i f_i(s, a)`` (epistemic novelty). ``β = 0`` recovers vanilla
DADS exactly. See ``multi-agent-ideas.md`` §8.2 / §10 and the doc in this folder.

Both halves are non-stationary, so — like ``r_dads`` — the bonus is recomputed
from the *current* ensemble at replay-sample time (never stored stale). The
ensemble trains on each fresh on-policy batch; its EMA-RMS scale is refreshed
once per training step and held fixed across the SAC grad scan, mirroring how the
q_φ I/O normalization is handled.

Base SAC: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import math
import os
import pickle
import time
from collections.abc import Callable
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import base, envs
from brax.training import acting, gradients, networks, pmap, replay_buffers, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import losses as sac_losses
from brax.training.agents.sac import networks as sac_networks
from brax.training.types import Params, PRNGKey

from core_rl.dads._dads_eval import DadsEvaluator, dads_diagnostics
from core_rl.dads.skill_dynamics import QPhiNorm, compute_dads_reward
from core_rl.dads.skill_dynamics import log_prob as skill_dynamics_log_prob
from core_rl.disagreement.ensemble import disagreement_reward, ensemble_loss

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"

# EMA momentum for q_φ I/O normalization stats. Matches the paper's batch-norm
# default (0.99): the stats track the CURRENT policy's Δs scale instead of
# freezing to early-exploration data (which a cumulative running average does).
QPHI_NORM_MOMENTUM = 0.99


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    # --- DADS ---
    skill_dynamics_params: Params
    skill_dynamics_optimizer_state: optax.OptState
    # EMA stats for q_φ I/O normalization (paper Sec. 5): standardize the
    # restricted-state input and the Δs target. EMA (not cumulative) so the
    # scale tracks the current policy. `qphi_initialized` is 0.0 until the first
    # update sets the stats directly (avoids a slow warm-up from std=1).
    qphi_norm: QPhiNorm
    qphi_initialized: jnp.ndarray
    # --- disagreement ---
    # Forward ensemble f_1..f_K (the novelty side head) + its optimizer, and the
    # EMA of mean(disagreement²) used to put the bonus on a ~unit scale before β
    # (the same EMA-RMS the single-agent disagreement arm uses).
    ensemble_params: Params
    ensemble_optimizer_state: optax.OptState
    int_rew_rms: jnp.ndarray
    # Lifetime exploration accumulators over training rollouts (optional; sized 1
    # when the env exposes no coverage grid). Same semantics as the disagreement
    # PPO arm so coverage_cumulative / state_entropy / interaction_cumulative are
    # directly comparable across arms.
    coverage_grid: jnp.ndarray  # (ncells,) lifetime-visited occupancy (0/1)
    visit_counts: jnp.ndarray  # (ncells,) lifetime per-cell visit counts → entropy
    interaction_count: jnp.ndarray  # () lifetime object-contact timesteps


def _unpmap(v):
    # Avoid degraded performance under the new jax.pmap.
    return jax.tree_util.tree_map(lambda x: x.addressable_shards[0].data.squeeze(0), v)


def _make_qphi_norm(training_state: "TrainingState") -> QPhiNorm:
    """The q_φ I/O normalization (EMA stats), already a QPhiNorm."""
    return training_state.qphi_norm


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    skill_dynamics_network: networks.FeedForwardNetwork,
    skill_dynamics_optimizer: optax.GradientTransformation,
    ensemble_network: networks.FeedForwardNetwork,
    ensemble_optimizer: optax.GradientTransformation,
    input_obs_size: int,
    target_obs_size: int,
    # Coverage-grid size (cells); 1 when the env exposes no coverage grid.
    cov_grid_size: int = 1,
    # If set, alpha is initialized to log(entropy_coef) and kept fixed by
    # sgd_step (which skips the alpha gradient step). Matches DADS paper:
    # agent_entropy=0.1, no auto-tuning.
    entropy_coef: float | None = None,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_skill, key_ensemble = jax.random.split(key, 4)
    if entropy_coef is not None:
        log_alpha = jnp.asarray(float(jnp.log(jnp.asarray(entropy_coef))), dtype=jnp.float32)
    else:
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    skill_dynamics_params = skill_dynamics_network.init(key_skill)
    skill_dynamics_optimizer_state = skill_dynamics_optimizer.init(skill_dynamics_params)

    # --- disagreement: forward ensemble side head ---
    ensemble_params = ensemble_network.init(key_ensemble)
    ensemble_optimizer_state = ensemble_optimizer.init(ensemble_params)

    # q_φ I/O normalization (EMA). Identity until the first update sets it.
    # Input (s) and target (Δ) live in different spaces under Option A.
    qphi_norm = QPhiNorm(
        s_mean=jnp.zeros((input_obs_size,), jnp.float32),
        s_std=jnp.ones((input_obs_size,), jnp.float32),
        delta_mean=jnp.zeros((target_obs_size,), jnp.float32),
        delta_std=jnp.ones((target_obs_size,), jnp.float32),
    )

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=types.UInt64(hi=0, lo=0),
        env_steps=types.UInt64(hi=0, lo=0),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        skill_dynamics_params=skill_dynamics_params,
        skill_dynamics_optimizer_state=skill_dynamics_optimizer_state,
        qphi_norm=qphi_norm,
        qphi_initialized=jnp.zeros((), jnp.float32),
        ensemble_params=ensemble_params,
        ensemble_optimizer_state=ensemble_optimizer_state,
        int_rew_rms=jnp.ones(()),
        coverage_grid=jnp.zeros((cov_grid_size,), jnp.float32),
        visit_counts=jnp.zeros((cov_grid_size,), jnp.float32),
        interaction_count=jnp.zeros((), jnp.float32),
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Callable[[Any], Any] | None = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: int | None = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: int | None = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[[int, Any, Any], None] = lambda *args: None,
    eval_env: envs.Env | None = None,
    randomization_fn: Callable[[base.System, jnp.ndarray], tuple[base.System, base.System]] | None = None,
    checkpoint_logdir: str | None = None,
    restore_checkpoint_path: str | None = None,
    # --- DADS ---
    final_checkpoint_path: str | None = None,  # pickle the full TrainingState here on finish
    entropy_coef: float | None = None,  # fix SAC alpha to this value (DADS paper: 0.1)
    skill_dynamics_network: networks.FeedForwardNetwork | None = None,
    skill_size: int = 2,
    input_obs_size: int = 0,  # |s_input|;  MUST be set by caller
    target_obs_size: int = 0,  # |s_target|; MUST be set by caller
    skill_dyn_lr: float = 3e-4,
    skill_dyn_train_steps: int = 8,
    prior_samples: int = 100,
    # If True, q_φ trains ONLY on the fresh actor_step batch (no IS needed;
    # weights = 1 by construction). If False, q_φ samples the off-policy
    # replay buffer and uses the IS correction below.
    train_skill_dynamics_on_policy: bool = True,
    # IS clip range α — w_i = clip(exp(log π_cur − log π_old), 1/α, α).
    # Paper uses 10; only consumed in the off-policy mode.
    is_clip_eps: float = 10.0,
    # --- disagreement (the §8.2 fusion) ---
    # Forward ensemble f_1..f_K. The novelty bonus is β · EMA-RMS-normalized
    # Var_i f_i(s, a). β = 0 recovers vanilla DADS exactly.
    ensemble_network: networks.FeedForwardNetwork | None = None,
    beta: float = 0.0,
    ensemble_lr: float = 3e-4,
    ensemble_train_steps: int = 8,
    bootstrap_keep_prob: float = 0.8,
    int_rew_ema_tau: float = 0.05,
    # Optional per-obs-dim weights for the disagreement bonus (shape obs_size).
    # None = uniform mean over dims. Set to focus novelty on the object dims so
    # the agent gets curious about the cube rather than its own arm.
    ensemble_reward_weights: jnp.ndarray | None = None,
    # --- hindsight skill relabeling (the exploration→skills interface) ---
    # Exploration data is z-UNCORRELATED (the novelty bonus ignores z), so
    # on-policy q_φ finds I(Δ;z)≈0 in it and skills never form (see HANDOVER.md).
    # This mode converts that data into skill supervision, EM-style:
    #   RETAIN    — a contact archive keeps the rare ‖Δ_target‖>0 transitions
    #               that the fast-turnover main buffer would evict.
    #   OVERSAMPLE— q_φ trains on archive_qphi_frac archive / rest fresh batches
    #               (default 50/50; also keeps qphi_norm.delta_std at push scale
    #               instead of collapsing to the stillness noise floor); SAC
    #               batches mix in archive rows.
    #   RELABEL   — replace the meaningless behavior-z with z* sampled from the
    #               posterior ∝ q_φ(Δ|s,z')^(1/T) over prior candidates, in BOTH
    #               q_φ batches (M-step) and SAC batches (HER-style credit: for
    #               every observed push there exists a z under which it is good).
    hindsight_relabeling: bool = False,
    relabel_mode: str = "direct",  # "direct" (z* = achieved outcome, HER-style) | "posterior" (EM)
    contact_eps: float = 1e-4,  # ‖Δ_target‖ (raw units) above which a row counts as contact
    relabel_prob: float = 0.5,  # fraction of eligible SAC-batch rows relabeled (HER-style mixture)
    relabel_candidates: int = 64,  # prior samples z' scored to form the posterior
    relabel_temperature: float = 1.0,  # >1 flattens the posterior (anti-collapse), <1 sharpens
    archive_size: int = 200_000,  # contact-archive capacity (transitions)
    archive_insert_topk: int = 128,  # rows kept per actor step, ranked by ‖Δ_target‖
    archive_sac_frac: float = 0.25,  # fraction of each SAC minibatch drawn from the archive
    archive_qphi_frac: float = 0.5,  # fraction of each q_φ minibatch drawn from the archive (rest = fresh)
):
    """SAC training."""
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        local_devices_to_use,
        device_count,
    )

    if min_replay_size >= num_timesteps:
        raise ValueError("No training will happen because min_replay_size >= num_timesteps")

    # q_φ shape guards differ by mode:
    #   hindsight : consumes (1−archive_qphi_frac) fresh batch + archive_qphi_frac contact-archive draws
    #   on-policy : consumes the fresh actor_step batch (size num_envs)
    #   off-policy: consumes the SAC-sized buffer draw (batch × grad_updates)
    if hindsight_relabeling:
        if relabel_mode not in ("direct", "posterior"):
            raise ValueError(f"relabel_mode must be 'direct' or 'posterior', got {relabel_mode!r}.")
        if relabel_mode == "direct" and skill_size != target_obs_size:
            raise ValueError(
                f"relabel_mode='direct' maps z* = tanh(zscore(Δ_target)) and therefore needs "
                f"skill_size ({skill_size}) == target_obs_size ({target_obs_size}). "
                f"Use relabel_mode='posterior' for mismatched sizes."
            )
        if not 0.0 < archive_qphi_frac < 1.0:
            raise ValueError(f"archive_qphi_frac must be in (0, 1), got {archive_qphi_frac}.")
        # q_φ minibatch split: archive_qphi_frac of each batch from the contact
        # archive, the rest from the fresh actor_step batch (0.5 ⇒ the balanced
        # 50/50 OVERSAMPLE default).
        qphi_arch_bs = max(min(int(round(batch_size * archive_qphi_frac)), batch_size - 1), 1)
        qphi_fresh_bs = batch_size - qphi_arch_bs
        if skill_dyn_train_steps * qphi_fresh_bs > num_envs:
            raise ValueError(
                f"Hindsight q_φ training needs skill_dyn_train_steps × fresh share "
                f"({skill_dyn_train_steps} × {qphi_fresh_bs}) ≤ num_envs ({num_envs}) "
                f"for the fresh share of each balanced batch."
            )
        if archive_insert_topk > num_envs:
            raise ValueError(f"archive_insert_topk ({archive_insert_topk}) must be ≤ num_envs ({num_envs}).")
    elif train_skill_dynamics_on_policy:
        if skill_dyn_train_steps * batch_size > num_envs:
            raise ValueError(
                f"On-policy q_φ training needs skill_dyn_train_steps × batch_size "
                f"({skill_dyn_train_steps} × {batch_size} = "
                f"{skill_dyn_train_steps * batch_size}) ≤ num_envs ({num_envs}). "
                f"Either lower skill_dyn_train_steps / batch_size or raise num_envs."
            )
    else:
        if skill_dyn_train_steps > grad_updates_per_step:
            raise ValueError(
                f"Off-policy q_φ training needs skill_dyn_train_steps "
                f"({skill_dyn_train_steps}) ≤ grad_updates_per_step "
                f"({grad_updates_per_step}) so the SAC-sized buffer draw can be "
                f"sliced into skill_dyn_train_steps batch_size-minibatches."
            )

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    if wrap_env:
        if wrap_env_fn is not None:
            wrap_for_training = wrap_env_fn
        elif isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            raise ValueError(f"Unsupported environment type: {type(env)}")

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(key, num_envs // jax.process_count() // local_devices_to_use),
            )
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )  # pytype: disable=wrong-keyword-args

    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        raise NotImplementedError("Dictionary observations not implemented in SAC")
    action_size = env.action_size

    # ── cumulative coverage / interaction (optional, task-provided) ──
    # Same opt-in API as the disagreement PPO arm: if the env exposes a coverage
    # grid (coverage_num_cells + coverage_cell_from_obs) and/or an interaction
    # hook (interaction_from_obs), accumulate lifetime occupancy / contact counts
    # over ALL training rollouts and log them. For skill_conditioned these hooks
    # delegate to the base task with the appended skill z stripped (see
    # skill_conditioned.py), so block-xy coverage is measured on the true state.
    _cov_env = getattr(environment, "unwrapped", environment)
    _cov_cells = int(getattr(_cov_env, "coverage_num_cells", 0) or 0)
    _cov_cell_fn = getattr(_cov_env, "coverage_cell_from_obs", None)
    _track_coverage = _cov_cells > 0 and callable(_cov_cell_fn)
    _cov_grid_size = max(_cov_cells, 1)
    _cov_log_ncells = math.log(max(_cov_grid_size, 2))  # state-entropy normalizer
    _int_fn = getattr(_cov_env, "interaction_from_obs", None)
    _track_interaction = callable(_int_fn)

    def normalize_fn(x, y):
        return x

    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
    )
    # ── DADS: wrap Brax's make_policy so it only sees (normalizer, policy),
    #     AND so that we expose the action log_prob in policy_extras (Brax's
    #     default inference fn drops it). The log_prob is what we need to
    #     compute importance-sampling weights when q_φ trains off-policy
    #     (paper Eq. for ∇_φ J(q_φ)):
    #         w_i = clip(exp(log π_cur(a|s,z) − log π_old(a|s,z)), 1/α, α)
    #     so π_old (the collection policy) MUST be stored at collection time.
    _parametric_dist = sac_network.parametric_action_distribution

    def make_policy(params, deterministic: bool = False):
        sac_params = params[:2]

        def policy(observations, key_sample):
            logits = sac_network.policy_network.apply(*sac_params, observations)
            if deterministic:
                return _parametric_dist.mode(logits), {}
            raw_action = _parametric_dist.sample_no_postprocessing(logits, key_sample)
            log_prob = _parametric_dist.log_prob(logits, raw_action)
            action = _parametric_dist.postprocess(raw_action)
            return action, {"log_prob": log_prob}

        return policy

    alpha_optimizer = optax.adam(learning_rate=3e-4)

    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)

    # --- DADS ---
    skill_dynamics_optimizer = optax.adam(skill_dyn_lr)

    # --- disagreement ---
    ensemble_optimizer = optax.adam(ensemble_lr)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    # --- DADS: dummy_transition determines the replay-buffer slot shape, so
    #          every key listed in `extra_fields` passed to acting.actor_step
    #          MUST appear here with the right shape — otherwise the buffer
    #          will be sized too small and `dynamic_update_slice` will fail.
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "z": jnp.zeros((skill_size,)),
                "s_input": jnp.zeros((input_obs_size,)),
                "s_input_next": jnp.zeros((input_obs_size,)),
                "s_target": jnp.zeros((target_obs_size,)),
                "s_target_next": jnp.zeros((target_obs_size,)),
            },
            # log_prob is the action log-prob under the COLLECTION policy
            # (π_old); needed for the IS correction at q_φ training time.
            "policy_extras": {"log_prob": jnp.zeros(())},
        },
    )
    # ── hindsight mode: SAC minibatches are a main/archive mix, so the main
    # queue's draw shrinks by the archive share and the two concat back to
    # batch_size per minibatch. buffer_state becomes the pytree (main, archive).
    sac_archive_bs = max(int(round(batch_size * archive_sac_frac)), 1) if hindsight_relabeling else 0
    sac_main_bs = batch_size - sac_archive_bs
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=sac_main_bs * grad_updates_per_step // device_count,
    )
    if hindsight_relabeling:
        # Contact archive: retains the top-`archive_insert_topk` rows per actor
        # step ranked by ‖Δ_target‖ — the rare "object actually moved" data that
        # the fast-turnover main buffer (~25 actor steps of history) would evict.
        # Two sampling "views" share ONE archive state (the queue object is
        # stateless; the state layout depends only on capacity + dummy sample):
        # one sized for the SAC mix-in, one for the q_φ balanced half-batches.
        archive_sac_view = replay_buffers.UniformSamplingQueue(
            max_replay_size=archive_size // device_count,
            dummy_data_sample=dummy_transition,
            sample_batch_size=sac_archive_bs * grad_updates_per_step // device_count,
        )
        archive_qphi_view = replay_buffers.UniformSamplingQueue(
            max_replay_size=archive_size // device_count,
            dummy_data_sample=dummy_transition,
            sample_batch_size=skill_dyn_train_steps * qphi_arch_bs,
        )

    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size,
    )
    alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )

    # --- DADS ---
    # Importance-sampling-weighted q_φ NLL. The reference (skill_dynamics.py
    # increase_prob_op) multiplies the per-sample log_prob by batch_weights
    # before reducing — same formula as the paper's Eq. for ∇_φ J(q_φ):
    #     L = − mean( w_i · log q_φ(Δ_i | s_i, z_i) )
    # In on-policy mode `is_weights` is jnp.ones, so this reduces to a plain
    # mean NLL; in off-policy mode it's the clipped π_cur/π_old ratio.
    def q_phi_loss_fn(params, s_input, z, delta_target, norm, is_weights):
        log_p = skill_dynamics_log_prob(  # type: ignore[arg-type]
            skill_dynamics_network, params, s_input, z, delta_target, norm
        )
        return -(log_p * is_weights).mean()

    q_phi_update = gradients.gradient_update_fn(
        q_phi_loss_fn, skill_dynamics_optimizer, _PMAP_AXIS_NAME  # type: ignore[arg-type]
    )

    def _hindsight_relabel(
        key: PRNGKey,
        s_input: jnp.ndarray,
        delta_target: jnp.ndarray,
        z_behavior: jnp.ndarray,
        sd_params: Params,
        norm: QPhiNorm,
        eligible: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """E-step of the hindsight relabeling: which skill would have wanted this outcome?

        Two modes (static ``relabel_mode``):

        * ``"direct"`` (recommended) — what HER actually does: the ACHIEVED
          outcome is the label, no model in the loop. ``z* = tanh(zscore(Δ))``
          via qphi_norm, so direction(z*) = push direction and ‖z*‖ = push
          strength (typical pushes land at ‖z‖≈tanh(1)≈0.76, big ones →1,
          filling the skill box radially). Self-calibrating — the balanced q_φ
          batches keep delta_std at push scale. Requires
          ``skill_size == target_obs_size`` (guarded at setup). Breaks symmetry
          at gradient step 1 by construction — the fix for sweep 1's finding
          that the posterior E-step never ignites (entropy pinned at ~1.0).

        * ``"posterior"`` — sample z* ∝ q_φ(Δ|s,z')^(1/T) over
          ``relabel_candidates`` prior draws (uniform prior ⇒ softmax over the
          candidate log-probs). Model-based EM; kept for comparison.

        Either way z* only replaces the behavior z on rows that are ``eligible``
        AND show real target motion (‖Δ‖ > contact_eps) — stillness rows carry
        no outcome to label.

        Returns ``(z_new, relabel_mask, contact_mask, posterior_entropy)``. The
        entropy is the normalized posterior entropy in ``"posterior"`` mode
        (→0 = collapse, ~1 = never ignited); constant 0.0 in ``"direct"`` mode
        (deterministic map — not applicable).
        """
        if relabel_mode == "direct":
            z_star = jnp.tanh((delta_target - norm.delta_mean) / (norm.delta_std + 1e-8))
            entropy = jnp.float32(0.0)
        else:  # "posterior"
            key_cand, key_pick = jax.random.split(key)
            b = s_input.shape[0]
            cand = jax.random.uniform(key_cand, (b, relabel_candidates, skill_size), minval=-1.0, maxval=1.0)
            # (B, K) candidate log-probs — same vmap pattern as compute_dads_reward's alt-z scoring.
            logp = jax.vmap(
                lambda zc: skill_dynamics_log_prob(  # type: ignore[arg-type]
                    skill_dynamics_network, sd_params, s_input, zc, delta_target, norm
                ),
                in_axes=-2,
                out_axes=-1,
            )(cand)
            logits = logp / relabel_temperature
            idx = jax.random.categorical(key_pick, logits, axis=-1)  # (B,)
            z_star = jnp.take_along_axis(cand, idx[:, None, None], axis=1)[:, 0, :]  # (B, skill)
            p = jax.nn.softmax(logits, axis=-1)
            entropy = -(p * jnp.log(p + 1e-8)).sum(axis=-1).mean() / math.log(relabel_candidates)
        contact = jnp.linalg.norm(delta_target, axis=-1) > contact_eps
        mask = jnp.logical_and(eligible, contact)
        z_new = jnp.where(mask[:, None], z_star, z_behavior)
        return z_new, mask, contact, entropy

    # --- disagreement ---
    # Bootstrap-masked MSE over the K forward models, same as the single-agent
    # disagreement arm: each model fits a random keep_prob fraction of the batch.
    # Plugs straight into gradient_update_fn (params is the first arg).
    ensemble_update = gradients.gradient_update_fn(
        functools.partial(ensemble_loss, ensemble_network, keep_prob=bootstrap_keep_prob),
        ensemble_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    def sgd_step(
        carry: tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> tuple[tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor, key_zalts, key_relabel = jax.random.split(key, 6)

        # --- DADS ---
        state_extras = transitions.extras["state_extras"]
        s_input = state_extras["s_input"]
        z = state_extras["z"]
        s_target = state_extras["s_target"]
        s_target_next = state_extras["s_target_next"]
        delta_target = s_target_next - s_target

        skill_dynamics_params = training_state.skill_dynamics_params
        norm = _make_qphi_norm(training_state)

        # --- hindsight relabeling (SAC side): HER-style credit assignment ---
        # On a relabel_prob coin-flip, contact rows get the z* that best explains
        # their observed Δ — so for every real push there exists a z-context in
        # which the critic sees it rewarded. z is swapped consistently into the
        # observation tail (the skill_conditioned wrapper appends z LAST), the
        # next_observation tail, and the r_dads computation below. Keeping
        # (1 − relabel_prob) of rows on their original z keeps Q honest for
        # wrong (s, a, z) combinations. Actions stay as stored — Q-learning's
        # bootstrap draws a′ ~ π(·|s′, z*), never the stored future actions.
        relabel_metrics = {}
        if hindsight_relabeling:
            key_coin, key_hs = jax.random.split(key_relabel)
            eligible = jax.random.bernoulli(key_coin, relabel_prob, (transitions.reward.shape[0],))
            z, relabel_mask, contact_mask, posterior_entropy = _hindsight_relabel(
                key_hs, s_input, delta_target, z, skill_dynamics_params, norm, eligible
            )
            transitions = transitions._replace(
                observation=transitions.observation.at[..., -skill_size:].set(z),
                next_observation=transitions.next_observation.at[..., -skill_size:].set(z),
                extras={**transitions.extras, "state_extras": {**state_extras, "z": z}},
            )
            relabel_metrics = {
                "relabel_fraction": relabel_mask.mean(),
                "relabel_posterior_entropy": posterior_entropy,
                "sac_batch_contact_frac": contact_mask.mean(),
            }

        # DADS reward + diagnostics on the SAC minibatch. The diagnostics metric
        # set is shared with the eval rollout (see DadsEvaluator) so eval/ mirrors
        # training/. Here B is the SAC minibatch, so the default vmap reward path
        # is fine; eval chunks it instead (much larger B).
        B = transitions.reward.shape[0]  # noqa: N806  (batch dim, see comment above)
        z_alts = jax.random.uniform(key_zalts, (B, prior_samples, skill_size), minval=-1.0, maxval=1.0)

        r_dads = compute_dads_reward(
            skill_dynamics_network,  # type: ignore[arg-type]
            skill_dynamics_params,
            s_input,
            z,
            delta_target,
            z_alts,
            norm=norm,
        )

        if hindsight_relabeling:
            # The handover health signal: relabeled rows are positive by
            # construction (z* was chosen to explain Δ); the KEPT rows catching
            # up means the policy is genuinely starting to condition on z.
            rel = relabel_mask.astype(jnp.float32)
            relabel_metrics["r_dads_relabeled_mean"] = (r_dads * rel).sum() / jnp.maximum(rel.sum(), 1.0)
            relabel_metrics["r_dads_kept_mean"] = (r_dads * (1.0 - rel)).sum() / jnp.maximum((1.0 - rel).sum(), 1.0)

        dads_metrics = dads_diagnostics(
            skill_dynamics_network,  # type: ignore[arg-type]
            skill_dynamics_params,
            s_input,
            z,
            s_target,
            s_target_next,
            r_dads,
            prior_samples,
            norm=norm,
        )

        # --- disagreement bonus: r = r_dads + β · d̂(s, a) (the §8.2 fusion) ---
        # Recompute the bonus from the CURRENT ensemble at replay-sample time
        # (it is non-stationary like r_dads, so we never store a stale value).
        # The EMA-RMS scale is refreshed once per training_step in train_ensemble
        # and held fixed across this sgd scan (mirrors qphi_norm). Uses the FULL
        # SAC observation (incl. z) normalized by the SAC running stats — the same
        # input convention as the single-agent disagreement reference arm.
        s_norm = running_statistics.normalize(transitions.observation, training_state.normalizer_params)
        d_raw = disagreement_reward(
            ensemble_network,  # type: ignore[arg-type]
            training_state.ensemble_params,
            s_norm,
            transitions.action,
            weights=ensemble_reward_weights,
        )
        # Disagreement is used RAW (un-normalized): the ensemble variance decays as
        # the ensemble learns, so beta * d_raw self-anneals and DADS takes over —
        # the handover the EMA normalization used to cancel. beta absorbs the raw
        # scale (peak d_raw ~0.03 std obs / ~0.1 rich obs) so that beta * d_raw is
        # ~O(r_dads) at peak and decays below it as novelty is exhausted. d_norm is
        # kept only as a scale-free monitoring diagnostic, NOT used in the reward.
        d_norm = d_raw / (jnp.sqrt(training_state.int_rew_rms) + 1e-8)
        r_combined = r_dads + beta * d_raw

        transitions = transitions._replace(reward=r_combined)

        # If entropy_coef is set, alpha is frozen at log(entropy_coef) — skip the
        # gradient step entirely. This is a Python-time conditional (entropy_coef
        # is a static value at trace time), so the resulting JIT graph either
        # includes alpha_update or it doesn't — no jax.lax.cond overhead.
        if entropy_coef is not None:
            alpha_loss = jnp.float32(0.0)
            alpha_params = training_state.alpha_params
            alpha_optimizer_state = training_state.alpha_optimizer_state
        else:
            alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params,
                training_state.policy_params,
                training_state.normalizer_params,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_q_params,
            q_params,
        )

        metrics = {
            # ── SAC ──
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            # ── shared q_φ quality + DADS reward + behavior diagnostics ──
            # (q_phi training loss is logged in training_step, where q_φ is trained)
            **dads_metrics,
            # ── disagreement bonus diagnostics ──
            # raw = Var_i f_i(s,a) (decays as the ensemble learns) — THIS is what
            # enters the reward now; normalized = raw / sqrt(EMA) (~O(1), diagnostic
            # only); beta_term = beta * d_raw = what is actually added to r_dads.
            # Compare beta_term vs r_dads_mean to read the balance.
            "disagreement_raw_mean": d_raw.mean(),
            "disagreement_normalized_mean": d_norm.mean(),
            "beta_disagreement_mean": (beta * d_raw).mean(),
            "r_combined_mean": r_combined.mean(),
            # ── which term drives the reward right now (the requested ratio) ──
            # Magnitudes of the two reward contributions and their balance.
            #   reward_dads_dominance ∈ [0,1]: |r_dads| / (|r_dads| + |β·d̂|).
            #     >0.5 ⇒ DADS (skill structure) dominates; <0.5 ⇒ novelty dominates;
            #     ~0.5 ⇒ balanced. Watch it rise toward 1 over training as the
            #     disagreement scaffold self-anneals (d̂ → 0).
            #   reward_ratio_dads_over_dis: |r_dads| / |β·d̂| (the same, unbounded).
            "reward_dads_abs_mean": jnp.abs(r_dads).mean(),
            "reward_disagreement_abs_mean": jnp.abs(beta * d_raw).mean(),
            "reward_dads_dominance": jnp.abs(r_dads).mean()
            / (jnp.abs(r_dads).mean() + jnp.abs(beta * d_raw).mean() + 1e-8),
            "reward_ratio_dads_over_dis": jnp.abs(r_dads).mean() / (jnp.abs(beta * d_raw).mean() + 1e-8),
            # ── hindsight relabeling diagnostics (empty dict when mode is off) ──
            **relabel_metrics,
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
            # q_φ + its norm stats are updated in train_skill_dynamics — pass through.
            skill_dynamics_params=training_state.skill_dynamics_params,
            skill_dynamics_optimizer_state=training_state.skill_dynamics_optimizer_state,
            qphi_norm=training_state.qphi_norm,
            qphi_initialized=training_state.qphi_initialized,
            # ensemble + reward-scale are updated in train_ensemble — pass through.
            ensemble_params=training_state.ensemble_params,
            ensemble_optimizer_state=training_state.ensemble_optimizer_state,
            int_rew_rms=training_state.int_rew_rms,
            # coverage accumulators are updated in training_step — pass through.
            coverage_grid=training_state.coverage_grid,
            visit_counts=training_state.visit_counts,
            interaction_count=training_state.interaction_count,
        )
        return (new_training_state, key), metrics

    def get_experience(
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
        Transition,
    ]:
        policy = make_policy((normalizer_params, policy_params))
        env_state, transitions = acting.actor_step(
            env,
            env_state,
            policy,
            key,
            extra_fields=("truncation", "z", "s_input", "s_input_next", "s_target", "s_target_next"),
        )

        normalizer_params = running_statistics.update(
            normalizer_params,
            transitions.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        if hindsight_relabeling:
            # RETAIN: main-buffer insert as usual, plus the top-K rows by
            # ‖Δ_target‖ into the contact archive (truncation rows score 0 —
            # their Δ crosses the auto-reset). top_k is rank-based, so before any
            # contact exists the archive holds stillness rows (harmless: they are
            # never relabeled and merely mirror the fresh distribution).
            main_state, arch_state = buffer_state
            main_state = replay_buffer.insert(main_state, transitions)
            se = transitions.extras["state_extras"]
            score = jnp.linalg.norm(se["s_target_next"] - se["s_target"], axis=-1) * (1.0 - se["truncation"])
            _, top_idx = jax.lax.top_k(score, archive_insert_topk)
            contact_rows = jax.tree_util.tree_map(lambda x: x[top_idx], transitions)
            arch_state = archive_sac_view.insert(arch_state, contact_rows)
            buffer_state = (main_state, arch_state)
        else:
            buffer_state = replay_buffer.insert(buffer_state, transitions)
        # Also return the freshly-collected transitions: q_φ trains on these
        # directly (on-policy) instead of resampling from the off-policy buffer.
        return normalizer_params, env_state, buffer_state, transitions

    def _compute_is_weights(
        mb_transitions: Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
    ) -> jnp.ndarray:
        """Per-sample IS weights ``w = clip(exp(log π_cur − log π_old), 1/α, α)``.

        Off-policy q_φ training only: ``log π_old`` was stored at collection time
        in ``policy_extras['log_prob']``; we recompute ``log π_cur`` with the
        current (post-EMA-update is fine) (normalizer, policy) params.
        """
        obs = mb_transitions.observation
        action = mb_transitions.action
        old_lp = mb_transitions.extras["policy_extras"]["log_prob"]

        logits = sac_network.policy_network.apply(normalizer_params, policy_params, obs)
        # action is post-tanh; invert (clip to avoid arctanh at ±1) to get the
        # pre-tanh raw action that parametric_action_distribution.log_prob expects.
        raw_action = _parametric_dist.inverse_postprocess(jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6))
        cur_lp = _parametric_dist.log_prob(logits, raw_action)

        ratio = jnp.exp(cur_lp - old_lp)
        return jnp.clip(ratio, 1.0 / is_clip_eps, is_clip_eps)

    def train_skill_dynamics(
        training_state: TrainingState,
        fresh_transitions: Transition,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> tuple[TrainingState, ReplayBufferState, jnp.ndarray]:
        """Train q_φ.

        Three modes:

        * **hindsight** (``hindsight_relabeling``, overrides the flag below) —
          each of the ``skill_dyn_train_steps`` minibatches is an
          ``archive_qphi_frac`` share of contact-archive draws, rest fresh batch
          (OVERSAMPLE; default 50/50). The archive share is relabeled with
          hindsight z* every gradient step (the EM M-step fits q_φ on its own
          E-step assignments; the fresh half keeps behavior z so genuine
          z-correlation enters as skills form). No IS — fitting a conditional
          density on relabeled data is not an on-policy expectation. Side
          benefit: qphi_norm's delta_std is computed on ~50%-push batches, so it
          settles at push scale instead of collapsing to the stillness floor.

        * **on-policy** (default) — q_φ trains on the actor_step's fresh batch
          sliced into ``skill_dyn_train_steps`` disjoint ``batch_size`` chunks.
          IS weights are 1 by construction (π_cur == π_old at collection time),
          so we skip the IS computation entirely.

        * **off-policy** — q_φ samples the SAC replay buffer like SAC does, and
          every sample's loss is reweighted by the clipped importance ratio
          ``w = clip(exp(log π_cur − log π_old), 1/α, α)`` (paper Eq. for
          ∇_φ J(q_φ)). ``buffer_state`` is threaded so the draw advances the
          buffer's RNG.
        """
        # ── 1. Pick the data q_φ will train on ─────────────────────────────
        if hindsight_relabeling:
            main_state, arch_state = buffer_state
            n_fresh = skill_dyn_train_steps * qphi_fresh_bs
            fresh_mb = jax.tree_util.tree_map(
                lambda x: x[:n_fresh].reshape((skill_dyn_train_steps, qphi_fresh_bs) + x.shape[1:]),
                fresh_transitions,
            )
            arch_state, arch_draw = archive_qphi_view.sample(arch_state)
            arch_mb = jax.tree_util.tree_map(
                lambda x: x.reshape((skill_dyn_train_steps, qphi_arch_bs) + x.shape[1:]),
                arch_draw,
            )
            # Row layout per minibatch: [fresh share | archive share] — the static
            # `qphi_eligible` mask below relies on this ordering.
            minibatches = jax.tree_util.tree_map(lambda f, a: jnp.concatenate([f, a], axis=1), fresh_mb, arch_mb)
            stats_source = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), minibatches)
            buffer_state = (main_state, arch_state)
        elif train_skill_dynamics_on_policy:
            n_needed = skill_dyn_train_steps * batch_size
            minibatches = jax.tree_util.tree_map(
                lambda x: x[:n_needed].reshape((skill_dyn_train_steps, batch_size) + x.shape[1:]),
                fresh_transitions,
            )
            stats_source = fresh_transitions
        else:
            buffer_state, draw = replay_buffer.sample(buffer_state)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((grad_updates_per_step, -1) + x.shape[1:])[:skill_dyn_train_steps],
                draw,
            )
            stats_source = draw

        # ── 2. Refresh I/O normalization (EMA) from the data we'll train on ─
        # On-policy: matches the current policy exactly (eval-time data IS this).
        # Off-policy: matches the buffer (which is the same data q_φ trains on).
        se0 = stats_source.extras["state_extras"]
        s_batch = se0["s_input"]
        d_batch = se0["s_target_next"] - se0["s_target"]

        def _batch_mean_std(x):
            m = jax.lax.pmean(jnp.mean(x, axis=0), _PMAP_AXIS_NAME)
            msq = jax.lax.pmean(jnp.mean(x * x, axis=0), _PMAP_AXIS_NAME)
            return m, jnp.sqrt(jnp.maximum(msq - m * m, 1e-12))

        s_m, s_s = _batch_mean_std(s_batch)
        d_m, d_s = _batch_mean_std(d_batch)

        prev = training_state.qphi_norm
        # Set directly on the very first update; EMA afterwards. Avoids the slow
        # warm-up that would otherwise leave normalization ≈ identity (std=1) for
        # hundreds of steps while q_φ trains on tiny mis-scaled targets.
        first = training_state.qphi_initialized < 0.5
        beta = QPHI_NORM_MOMENTUM

        def _ema(prev_v, new_v):
            return jnp.where(first, new_v, beta * prev_v + (1.0 - beta) * new_v)

        norm = QPhiNorm(
            s_mean=_ema(prev.s_mean, s_m),
            s_std=_ema(prev.s_std, s_s),
            delta_mean=_ema(prev.delta_mean, d_m),
            delta_std=_ema(prev.delta_std, d_s),
        )
        training_state = training_state.replace(
            qphi_norm=norm,
            qphi_initialized=jnp.ones((), jnp.float32),
        )

        # Capture (normalizer, policy) at q_φ-train start so the off-policy IS
        # ratio uses a fixed "current" policy across all skill_dyn_train_steps
        # gradient steps (matches the reference: log π_cur is computed once at
        # the start of q_φ training, not after each q_φ update).
        cur_normalizer = training_state.normalizer_params
        cur_policy = training_state.policy_params

        # ── 3. q_φ scan ────────────────────────────────────────────────────
        # Hindsight mode: rows in the archive share (the tail of each minibatch,
        # by construction above) are always eligible for relabeling — their
        # behavior z is meaningless exploration noise. Fresh rows keep z.
        if hindsight_relabeling:
            qphi_eligible = jnp.concatenate([jnp.zeros((qphi_fresh_bs,), bool), jnp.ones((qphi_arch_bs,), bool)])
        step_keys = jax.random.split(key, skill_dyn_train_steps)

        def q_phi_step(carry, xs):
            sd_params, sd_opt_state = carry
            mb, step_key = xs
            se = mb.extras["state_extras"]
            delta_target = se["s_target_next"] - se["s_target"]
            if hindsight_relabeling:
                # E-step with the CARRY's q_φ params — the assignments sharpen
                # across the skill_dyn_train_steps inner iterations (EM).
                z_train, _, _, _ = _hindsight_relabel(
                    step_key, se["s_input"], delta_target, se["z"], sd_params, norm, qphi_eligible
                )
                is_weights = jnp.ones((mb.observation.shape[0],), jnp.float32)
            elif train_skill_dynamics_on_policy:
                z_train = se["z"]
                is_weights = jnp.ones((mb.observation.shape[0],), jnp.float32)
            else:
                z_train = se["z"]
                is_weights = _compute_is_weights(mb, cur_normalizer, cur_policy)
            # Mask episode-boundary (truncation) transitions: with resample-on-done
            # auto-reset their Δ is the reset placeholder (Δ=0), not a real skill
            # transition, so q_φ must not fit them. (SAC already handles truncation
            # for the critic via the truncation flag.) Folded into the loss weights;
            # is_weights.mean() below stays the pure IS-drift sanity metric.
            weights = is_weights * (1.0 - se["truncation"])
            loss, sd_params, sd_opt_state = q_phi_update(
                sd_params,
                se["s_input"],
                z_train,
                delta_target,
                norm,
                weights,
                optimizer_state=sd_opt_state,
            )
            return (sd_params, sd_opt_state), (loss, is_weights.mean())

        (sd_params, sd_opt_state), (losses, is_weights_means) = jax.lax.scan(
            q_phi_step,
            (training_state.skill_dynamics_params, training_state.skill_dynamics_optimizer_state),
            (minibatches, step_keys),
        )
        training_state = training_state.replace(
            skill_dynamics_params=sd_params,
            skill_dynamics_optimizer_state=sd_opt_state,
        )
        # Returning is_weights_means alongside losses lets training_step log a
        # sanity-check metric — ≈1 means the cur/old policy ratio is tiny (which
        # is the on-policy case by construction; for off-policy, drift away from
        # 1 is the signature of IS doing meaningful work).
        return training_state, buffer_state, losses, is_weights_means

    def train_ensemble(
        training_state: TrainingState,
        fresh_transitions: Transition,
        key: PRNGKey,
    ) -> tuple[TrainingState, jnp.ndarray]:
        """Train the forward ensemble on the fresh on-policy batch and refresh
        the EMA-RMS scale used to normalize the disagreement bonus.

        Mirrors the single-agent disagreement arm (``ensemble_step`` + the
        ``int_rew_rms`` EMA), but on SAC's fresh actor_step batch instead of a
        PPO rollout. Off-policy data would also be fine for supervised dynamics
        fitting; the fresh batch is the cheapest source already in hand and keeps
        the EMA-RMS scale matched to the current policy (the scale is then held
        fixed across the sgd scan in :func:`sgd_step`, exactly like ``qphi_norm``).
        """
        normalizer_params = training_state.normalizer_params
        se = fresh_transitions.extras["state_extras"]
        # Mask episode-boundary transitions: after auto-reset next_observation is
        # the reset placeholder, so its Δ-target is invalid for the ensemble fit.
        mask = 1.0 - se["truncation"]

        s_n = running_statistics.normalize(fresh_transitions.observation, normalizer_params)
        target = running_statistics.normalize(fresh_transitions.next_observation, normalizer_params) - s_n
        act = fresh_transitions.action

        # Reward-scale normalizer: EMA of mean(disagreement²) on the fresh batch,
        # shared across devices. Computed from the PRE-update ensemble (reference
        # ordering) so the scale lags the params it normalizes by one step — fine.
        d_raw = (
            disagreement_reward(ensemble_network, training_state.ensemble_params, s_n, act, weights=ensemble_reward_weights)  # type: ignore[arg-type]
            * mask
        )
        batch_msq = jax.lax.pmean(jnp.mean(d_raw**2), _PMAP_AXIS_NAME)
        int_rew_rms = (1.0 - int_rew_ema_tau) * training_state.int_rew_rms + int_rew_ema_tau * batch_msq

        # K models, fresh bootstrap mask per gradient step (per-model keep_prob).
        def ensemble_step(carry, k):
            e_params, e_opt_state = carry
            loss, e_params, e_opt_state = ensemble_update(
                e_params, s_n, act, target, mask, k, optimizer_state=e_opt_state
            )
            return (e_params, e_opt_state), loss

        keys = jax.random.split(key, ensemble_train_steps)
        (ensemble_params, ensemble_optimizer_state), ens_losses = jax.lax.scan(
            ensemble_step,
            (training_state.ensemble_params, training_state.ensemble_optimizer_state),
            keys,
        )
        training_state = training_state.replace(
            ensemble_params=ensemble_params,
            ensemble_optimizer_state=ensemble_optimizer_state,
            int_rew_rms=int_rew_rms,
        )
        return training_state, ens_losses

        # Collects experiance and does a sgd_step

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> tuple[
        TrainingState,
        envs.State,
        ReplayBufferState,
        Metrics,
    ]:
        experience_key, training_key, ensemble_key, skill_key = jax.random.split(key, 4)
        normalizer_params, env_state, buffer_state, fresh_transitions = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        # ── cumulative coverage / interaction over THIS rollout's states ──
        # OR every visited cell into the lifetime occupancy grid (unioned across
        # devices so replicas stay identical) and accumulate visit counts +
        # object-contact timesteps. Uses the fresh actor-step observations.
        obs_f = fresh_transitions.observation
        if _track_coverage:
            cov_cells = _cov_cell_fn(obs_f)
            cov_seen = jnp.zeros((_cov_grid_size,), jnp.float32).at[cov_cells].set(1.0)
            coverage_grid = jax.lax.pmax(jnp.maximum(training_state.coverage_grid, cov_seen), _PMAP_AXIS_NAME)
            batch_counts = jnp.zeros((_cov_grid_size,), jnp.float32).at[cov_cells].add(1.0)
            visit_counts = training_state.visit_counts + jax.lax.psum(batch_counts, _PMAP_AXIS_NAME)
            training_state = training_state.replace(coverage_grid=coverage_grid, visit_counts=visit_counts)
        if _track_interaction:
            interaction_count = training_state.interaction_count + jax.lax.psum(
                jnp.sum(_int_fn(obs_f)), _PMAP_AXIS_NAME
            )
            training_state = training_state.replace(interaction_count=interaction_count)

        # ── DADS: train q_φ FIRST on the FRESH on-policy batch (or buffer +
        # IS in off-policy mode); SAC below relabels with the just-updated q_φ.
        training_state, buffer_state, q_phi_losses, q_phi_is_means = train_skill_dynamics(
            training_state,
            fresh_transitions,
            buffer_state,
            skill_key,
        )

        # ── disagreement: train the forward ensemble on the same fresh batch and
        # refresh the EMA-RMS scale; sgd_step below relabels with β·d̂ from it.
        training_state, ensemble_losses = train_ensemble(
            training_state,
            fresh_transitions,
            ensemble_key,
        )

        if hindsight_relabeling:
            # OVERSAMPLE (SAC side): each of the grad_updates_per_step minibatches
            # is a [main | archive] concat — archive rows put push data in front
            # of the critic at ~archive_sac_frac density instead of the ~1% it
            # has in the main buffer. Relabeling happens inside sgd_step.
            main_state, arch_state = buffer_state
            main_state, main_draw = replay_buffer.sample(main_state)
            arch_state, arch_draw = archive_sac_view.sample(arch_state)
            transitions = jax.tree_util.tree_map(
                lambda m, a: jnp.concatenate(
                    [
                        jnp.reshape(m, (grad_updates_per_step, -1) + m.shape[1:]),
                        jnp.reshape(a, (grad_updates_per_step, -1) + a.shape[1:]),
                    ],
                    axis=1,
                ),
                main_draw,
                arch_draw,
            )
            buffer_state = (main_state, arch_state)
        else:
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            # Change the front dimension of transitions so 'update_step' is called
            # grad_updates_per_step times by the scan.
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
                transitions,
            )
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        # q_φ training stats (computed once per training_step, not per sgd_step).
        metrics["q_phi_loss"] = q_phi_losses.mean()
        metrics["q_phi_loss_first"] = q_phi_losses[0]
        metrics["q_phi_loss_last"] = q_phi_losses[-1]
        # Ensemble training stats: mean fit error + the first/last step on the
        # fresh batch (last < first ⇒ the ensemble is learning the visited
        # dynamics, which is what self-anneals the novelty bonus toward 0).
        metrics["ensemble_loss"] = ensemble_losses.mean()
        metrics["ensemble_loss_first"] = ensemble_losses[0]
        metrics["ensemble_loss_last"] = ensemble_losses[-1]
        metrics["int_rew_rms"] = jnp.sqrt(training_state.int_rew_rms)
        # IS sanity: ≈1 in on-policy mode (always); drift away from 1 in
        # off-policy mode quantifies how much correction IS is applying.
        metrics["q_phi_is_weight_mean"] = q_phi_is_means.mean()
        # q_φ normalization scale — should TRACK the policy over training (drift,
        # not freeze). If these go flat early, the EMA isn't adapting.
        metrics["qphi_delta_std_mean"] = training_state.qphi_norm.delta_std.mean()
        metrics["qphi_s_std_mean"] = training_state.qphi_norm.s_std.mean()
        if hindsight_relabeling:
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state[0])
            metrics["archive_current_size"] = archive_sac_view.size(buffer_state[1])
        else:
            metrics["buffer_current_size"] = replay_buffer.size(
                buffer_state
            )  # pytype: disable=unsupported-operands  # lax-types
        # ── exploration headline (block-xy for fetchpush) ──
        if _track_coverage:
            # coverage_cumulative: fraction of grid cells ever visited (saturates
            # at 1). state_entropy: normalized entropy of the lifetime visit
            # histogram (keeps discriminating how EVENLY the space is explored).
            metrics["coverage_cumulative"] = jnp.mean(training_state.coverage_grid)
            p = training_state.visit_counts / jnp.maximum(jnp.sum(training_state.visit_counts), 1.0)
            metrics["state_entropy"] = -jnp.sum(p * jnp.log(jnp.where(p > 0, p, 1.0))) / _cov_log_ncells
        if _track_interaction:
            metrics["interaction_cumulative"] = training_state.interaction_count
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state, _ = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME, donate_argnums=(0, 1, 2))

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME, donate_argnums=(0, 1, 2))

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        # ── DADS additions ──
        skill_dynamics_network=skill_dynamics_network,
        skill_dynamics_optimizer=skill_dynamics_optimizer,
        # ── disagreement additions ──
        ensemble_network=ensemble_network,
        ensemble_optimizer=ensemble_optimizer,
        input_obs_size=input_obs_size,
        target_obs_size=target_obs_size,
        cov_grid_size=_cov_grid_size,
        entropy_coef=entropy_coef,
    )
    del global_key

    if restore_checkpoint_path is not None:
        # Load a previously-saved un-pmapped TrainingState pickle (produced by the
        # final_checkpoint_path save at the end of this function) and replicate it
        # to every local device. Step counters are reset to 0 so --total-timesteps
        # behaves as "train for this many additional steps".
        logging.info("Resuming from %s", restore_checkpoint_path)
        with open(restore_checkpoint_path, "rb") as f:
            saved_state = pickle.load(f)
        saved_state = saved_state.replace(
            env_steps=types.UInt64(hi=0, lo=0),
            gradient_steps=types.UInt64(hi=0, lo=0),
        )
        training_state = jax.device_put_replicated(saved_state, jax.local_devices()[:local_devices_to_use])

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    # Replay buffer init (hindsight mode: buffer_state = (main, contact archive))
    if hindsight_relabeling:
        rb_key, arch_key = jax.random.split(rb_key)
        buffer_state = (
            jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use)),
            jax.pmap(archive_sac_view.init)(jax.random.split(arch_key, local_devices_to_use)),
        )
    else:
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

    if not eval_env:
        eval_env = environment
    if wrap_env:
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(randomization_fn, rng=jax.random.split(eval_key, num_eval_envs))
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )  # pytype: disable=wrong-keyword-args

    # DadsEvaluator runs a single eval rollout and reports BOTH the usual
    # task/episode metrics AND the DADS skill-dynamics diagnostics (r_dads,
    # q_phi_loss, …) under `eval/` — so eval mirrors training without a second
    # physics rollout, and with r_dads chunked to stay within GPU memory.
    evaluator = DadsEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        skill_dynamics_network=skill_dynamics_network,
        skill_size=skill_size,
        input_obs_size=input_obs_size,
        target_obs_size=target_obs_size,
        prior_samples=prior_samples,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.policy_params)),
            training_metrics={},
            skill_dynamics_params=_unpmap(training_state.skill_dynamics_params),
            qphi_norm=_unpmap(_make_qphi_norm(training_state)),
        )
        logging.info(metrics)
        progress_fn(0, metrics)
        policy_params_fn(
            0,
            make_policy,
            _unpmap((training_state.normalizer_params, training_state.policy_params)),
        )

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    _main_buffer_state = buffer_state[0] if hindsight_relabeling else buffer_state
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(_main_buffer_state)) * jax.process_count()
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        training_state, env_state, buffer_state, training_metrics = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
        current_step = int(_unpmap(training_state.env_steps))

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                # Same 4-tuple format as the final params pickle, so any
                # checkpoint can be loaded by eval_dads / compare tools directly.
                # Skills on this task peak mid-training and can decay afterwards;
                # per-eval snapshots let us harvest the peak.
                ckpt_params = _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.policy_params,
                        training_state.skill_dynamics_params,
                        _make_qphi_norm(training_state),
                    )
                )
                os.makedirs(checkpoint_logdir, exist_ok=True)
                with open(os.path.join(checkpoint_logdir, f"params_step_{current_step}.pkl"), "wb") as f:
                    pickle.dump(ckpt_params, f)

            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.policy_params)),
                training_metrics,
                skill_dynamics_params=_unpmap(training_state.skill_dynamics_params),
                qphi_norm=_unpmap(_make_qphi_norm(training_state)),
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            policy_params_fn(
                current_step,
                make_policy,
                _unpmap((training_state.normalizer_params, training_state.policy_params)),
            )

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(f"Total steps {total_steps} is less than `num_timesteps`=" f" {num_timesteps}.")

    # --- DADS ---
    # 4-tuple: (sac_normalizer, policy, skill_dynamics, qphi_norm). The q_φ I/O
    # normalization stats must travel with the params so eval reconstructs the
    # exact same normalized space the model was trained in.
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.policy_params,
            training_state.skill_dynamics_params,
            _make_qphi_norm(training_state),
        )
    )

    # --- DADS: persist full TrainingState for resumeable training ---
    if final_checkpoint_path is not None:
        state_to_save = _unpmap(training_state)
        os.makedirs(os.path.dirname(final_checkpoint_path) or ".", exist_ok=True)
        with open(final_checkpoint_path, "wb") as f:
            pickle.dump(state_to_save, f)
        logging.info("Saved full training state to %s", final_checkpoint_path)

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
