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

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
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
from brax.training.agents.sac import checkpoint
from brax.training.agents.sac import losses as sac_losses
from brax.training.agents.sac import networks as sac_networks
from brax.training.types import Params, PRNGKey

from core_rl.dads._dads_eval import DadsEvaluator, dads_diagnostics
from core_rl.dads.skill_dynamics import QPhiNorm, compute_dads_reward
from core_rl.dads.skill_dynamics import log_prob as skill_dynamics_log_prob

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
    input_obs_size: int,
    target_obs_size: int,
    # If set, alpha is initialized to log(entropy_coef) and kept fixed by
    # sgd_step (which skips the alpha gradient step). Matches DADS paper:
    # agent_entropy=0.1, no auto-tuning.
    entropy_coef: float | None = None,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_skill = jax.random.split(key, 3)
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
    #   on-policy : consumes the fresh actor_step batch (size num_envs)
    #   off-policy: consumes the SAC-sized buffer draw (batch × grad_updates)
    if train_skill_dynamics_on_policy:
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
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count,
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

    def sgd_step(
        carry: tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> tuple[tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor, key_zalts = jax.random.split(key, 5)

        # --- DADS ---
        state_extras = transitions.extras["state_extras"]
        s_input = state_extras["s_input"]
        z = state_extras["z"]
        s_target = state_extras["s_target"]
        s_target_next = state_extras["s_target_next"]
        delta_target = s_target_next - s_target

        skill_dynamics_params = training_state.skill_dynamics_params
        norm = _make_qphi_norm(training_state)

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

        transitions = transitions._replace(reward=r_dads)

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
    ) -> tuple[TrainingState, ReplayBufferState, jnp.ndarray]:
        """Train q_φ.

        Two modes, selected by ``train_skill_dynamics_on_policy``:

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
        if train_skill_dynamics_on_policy:
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
        def q_phi_step(carry, mb):
            sd_params, sd_opt_state = carry
            se = mb.extras["state_extras"]
            delta_target = se["s_target_next"] - se["s_target"]
            if train_skill_dynamics_on_policy:
                is_weights = jnp.ones((mb.observation.shape[0],), jnp.float32)
            else:
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
                se["z"],
                delta_target,
                norm,
                weights,
                optimizer_state=sd_opt_state,
            )
            return (sd_params, sd_opt_state), (loss, is_weights.mean())

        (sd_params, sd_opt_state), (losses, is_weights_means) = jax.lax.scan(
            q_phi_step,
            (training_state.skill_dynamics_params, training_state.skill_dynamics_optimizer_state),
            minibatches,
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
        experience_key, training_key = jax.random.split(key, 2)
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

        # ── DADS: train q_φ FIRST on the FRESH on-policy batch (or buffer +
        # IS in off-policy mode); SAC below relabels with the just-updated q_φ.
        training_state, buffer_state, q_phi_losses, q_phi_is_means = train_skill_dynamics(
            training_state,
            fresh_transitions,
            buffer_state,
        )

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
        # IS sanity: ≈1 in on-policy mode (always); drift away from 1 in
        # off-policy mode quantifies how much correction IS is applying.
        metrics["q_phi_is_weight_mean"] = q_phi_is_means.mean()
        # q_φ normalization scale — should TRACK the policy over training (drift,
        # not freeze). If these go flat early, the EMA isn't adapting.
        metrics["qphi_delta_std_mean"] = training_state.qphi_norm.delta_std.mean()
        metrics["qphi_s_std_mean"] = training_state.qphi_norm.s_std.mean()
        metrics["buffer_current_size"] = replay_buffer.size(
            buffer_state
        )  # pytype: disable=unsupported-operands  # lax-types
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
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
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
        input_obs_size=input_obs_size,
        target_obs_size=target_obs_size,
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

    # Replay buffer init
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

    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
        current_step = int(_unpmap(training_state.env_steps))

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                ckpt_config = checkpoint.network_config(
                    observation_size=obs_size,
                    action_size=env.action_size,
                    normalize_observations=normalize_observations,
                    network_factory=network_factory,
                )
                checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

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
