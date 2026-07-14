"""DADS evaluation script.

Loads a trained DADS checkpoint (``params.pkl`` produced by ``train.py``)
and runs a battery of evaluations:

1. **Skill rollouts** — roll out N envs in parallel, each with a *fixed*
   skill ``z``. For 2-D skill space and ``num_skills = K²``, the K² skills
   are taken from a uniform grid over ``(-1, 1)²``; otherwise random.
2. **Skill diversity / discriminability**
     - Trajectory dispersion: ``std(s_target)`` across skills, per timestep.
     - Pairwise distance between skill trajectories at the final timestep.
3. **Skill-dynamics quality** (q_φ)
     - Mean log-probability of the observed deltas under ``q_φ``.
     - L2 error of the modal-mean Δs prediction vs the observed Δs.
4. **DADS intrinsic reward** at eval time (should be positive after training).
5. **Plots** — saved as PNGs.
6. **Video** (optional, ``--video``) — tiled MuJoCo render of all skills.

Usage::

    python -m core_rl.eval_dads \\
        --robot parol6 \\
        --task skill_conditioned \\
        --params /ros2_ws/core/models/<run_id>/dads_params.pkl \\
        --num-skills 16 \\
        --episode-length 200 \\
        --video
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from core_rl.dads.dads import _infer_skill_dyn_sizes
from core_rl.dads.mppi import (
    build_task_reward,
    mppi_plan,
)
from core_rl.dads.skill_dynamics import (
    compute_dads_reward,
    log_prob,
    make_skill_dynamics,
    modal_delta,
)

# Robot/config resolution — same path as train.py
from core_rl.env import make_env
from core_rl.robot import resolve_robot
from core_rl.scene import load_scene

# ---------------------------------------------------------------------------
# Config loading (mirrors train.py)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config(config_path: str | None) -> dict:
    import yaml

    try:
        from ament_index_python.packages import get_package_share_directory

        defaults_path = os.path.join(get_package_share_directory("core_rl"), "config", "defaults.yaml")
    except Exception:
        defaults_path = os.path.join(os.path.dirname(__file__), "..", "config", "defaults.yaml")

    with open(defaults_path) as f:
        cfg = yaml.safe_load(f)
    if config_path:
        with open(config_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    return cfg


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
    skill_size = dads_cfg.get("skill_size", 2)
    input_obs_size, target_obs_size = _infer_skill_dyn_sizes(env)
    skill_dyn_hidden = tuple(dads_cfg.get("skill_dyn_hidden_layer_sizes", (256, 256)))
    num_components = dads_cfg.get("num_mixture_components", 4)
    # MUST match the value used during training — see SkillDynamics docstring.
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
# Fixed-skill rollouts (jax.vmap over N envs in parallel)
# ---------------------------------------------------------------------------


def _build_rollout_fn(env, policy_fn, skill_size: int, episode_length: int, deterministic: bool = False):
    """Build a JIT-compiled function that rolls out N envs with fixed skills.

    Args:
        deterministic: If True, the policy uses the mode tanh(μ) of the
            tanh-Normal distribution. If False (DEFAULT), it samples
            tanh(μ + σ·ε) — same as during training. Use False to evaluate
            q_φ on data drawn from the same distribution it was trained on.

    Args expected on returned fn:
        params:  policy params (normalizer, policy)
        skills:  (N, skill_size)  — one skill per env (overrides the env's
                                     random sampling at reset)
        rng:     PRNGKey

    Returns dict of arrays of shape ``(N, T, ...)``.
    """

    def _rollout(params, skills, rng):
        n = skills.shape[0]
        rng, rng_reset = jax.random.split(rng)
        reset_keys = jax.random.split(rng_reset, n)

        # Reset N envs (their internal z sampling is overwritten below)
        states = jax.vmap(env.reset)(reset_keys)

        # Force fixed skills: replace z in both obs and info.
        # obs layout from SkillConditionedTask.reset:  [base_obs, z]
        new_obs = states.obs.at[..., -skill_size:].set(skills)
        new_info = dict(states.info)
        new_info["z"] = skills
        states = states.replace(obs=new_obs, info=new_info)

        # Match training: default to stochastic (tanh(μ + σ·ε)). q_φ was
        # trained against samples from this distribution, NOT against modes,
        # so deterministic eval can put trajectories outside q_φ's training
        # distribution and tank metrics.
        policy = policy_fn(params, deterministic=deterministic)

        def _step(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            # For stochastic mode: give each env an independent noise key
            # via vmap over (obs, key). For deterministic mode the key is
            # ignored, so vmap is still correct.
            sub_keys = jax.random.split(sub, n)
            action, _ = jax.vmap(policy)(state.obs, sub_keys)
            next_state = jax.vmap(env.step)(state, action)
            # IMPORTANT: SkillConditionedTask.step writes ALL restricted-view
            # keys (s_input, s_input_next, s_target, s_target_next) into the
            # RETURNED (next-state) info — *_input is restrict(pre-step base obs),
            # *_input_next is restrict(post-step). Pulling `s_input` from
            # `state.info` here would read the PREVIOUS step's value, producing
            # a 2-step delta and breaking q_φ's predictions. Match how training
            # does it via acting.actor_step: read every key from `next_state.info`.
            record = {
                "s_input": next_state.info["s_input"],
                "s_input_next": next_state.info["s_input_next"],
                "s_target": next_state.info["s_target"],
                "s_target_next": next_state.info["s_target_next"],
                "z": next_state.info["z"],
                "obs": state.obs,
                "action": action,
                "qpos": next_state.pipeline_state.q,
            }
            return (next_state, key), record

        (_, _), traj = jax.lax.scan(_step, (states, rng), None, length=episode_length)
        # traj fields: (T, N, ...). Transpose to (N, T, ...).
        traj = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return traj

    return jax.jit(_rollout)


# ---------------------------------------------------------------------------
# MPPI zero-shot planning eval
# ---------------------------------------------------------------------------


def _override_z(state, z: jax.Array, skill_size: int):
    """Replace the skill in a SkillConditionedTask state (both info and obs)."""
    new_obs = state.obs.at[..., -skill_size:].set(z)
    new_info = dict(state.info)
    new_info["z"] = z
    return state.replace(obs=new_obs, info=new_info)


def _build_executor(env, policy_fn, skill_size: int, primitive_horizon: int):
    """Build a JIT'd ``execute_skill(state, z, params, rng) → (next_state, traj)``.

    Holds the skill fixed and steps the env ``primitive_horizon`` times under
    the trained policy. Returns the trajectory (per env-step) of the
    restricted state — used by MPPI to score the rollout *retrospectively*
    against the task reward.
    """

    def _execute(state, z, params, rng):
        # Inject the planned skill (one env — no vmap here, we plan per-env)
        state = _override_z(state, z, skill_size)
        policy = policy_fn(params, deterministic=False)

        def step_fn(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            action, _ = policy(state.obs, sub)
            next_state = env.step(state, action)
            record = {
                "s_target": next_state.info["s_target"],
                "s_target_next": next_state.info["s_target_next"],
                "qpos": next_state.pipeline_state.q,
                "reward_env": next_state.reward,  # task reward (push_ball gives this)
            }
            return (next_state, key), record

        (final_state, _), traj = jax.lax.scan(step_fn, (state, rng), None, length=primitive_horizon)
        return final_state, traj

    return jax.jit(_execute)


def _build_planner(
    skill_dynamics_network,
    task_reward_fn,
    skill_size: int,
    planning_horizon: int,
    primitive_horizon: int,
    num_candidates: int,
    refine_steps: int,
    temperature: float,
    smoothing_beta: float,
    proposal_std: float,
    qphi_norm=None,
):
    """Build a JIT'd ``plan(q_phi_params, s0, proposal_mean, rng) → (first_skill, proposal)``."""

    def _plan(q_phi_params, s0, proposal_mean, rng):
        first_skill, proposal = mppi_plan(
            skill_dynamics_network,
            q_phi_params,
            s0,
            task_reward_fn,
            rng,
            planning_horizon=planning_horizon,
            primitive_horizon=primitive_horizon,
            num_candidates=num_candidates,
            refine_steps=refine_steps,
            temperature=temperature,
            smoothing_beta=smoothing_beta,
            proposal_std=proposal_std,
            skill_size=skill_size,
            initial_proposal=proposal_mean,
            norm=qphi_norm,
        )
        return first_skill, proposal

    return jax.jit(_plan)


def run_mppi_eval(
    env,
    make_policy,
    sac_params,
    skill_dynamics_network,
    q_phi_params,
    task_reward_fn,
    skill_size: int,
    *,
    num_outer_steps: int,
    input_obs_size: int,
    target_obs_size: int,
    planning_horizon: int = 4,
    primitive_horizon: int = 10,
    num_candidates: int = 64,
    refine_steps: int = 10,
    temperature: float = 10.0,
    smoothing_beta: float = 0.9,
    proposal_std: float = 0.5,
    rng: jax.Array = None,
    qphi_norm=None,
) -> dict:
    """Receding-horizon MPPI rollout in the real env.

    For each outer step:
        1. Read the env's current ``s_target``.
        2. Plan a skill sequence via MPPI (using ``q_φ`` as the world model).
        3. Execute the first skill for ``primitive_horizon`` env steps.
        4. Shift the proposal and continue.

    Requires ``input_obs_size == target_obs_size``: the MPPI rollout in
    :mod:`core_rl.dads.mppi` feeds q_φ's predicted Δ back as the next state,
    which only makes sense when q_φ's input and target live in the same
    space (the ant_xy-style setup, or any config with
    ``input_obs_indices == target_obs_indices``).
    """
    if input_obs_size != target_obs_size:
        raise ValueError(
            "MPPI requires input_obs_size == target_obs_size "
            f"(got {input_obs_size} vs {target_obs_size}). With decoupled "
            "input/target, q_φ predicts in target space but its input lives "
            "elsewhere, so the rollout `s + modal_delta(s, z)` can't loop "
            "back. Set input_obs_indices == target_obs_indices to use MPPI."
        )

    if rng is None:
        rng = jax.random.PRNGKey(0)

    rng, rng_reset = jax.random.split(rng)
    state = env.reset(rng_reset)

    # Build the JIT'd helpers
    executor = _build_executor(env, make_policy, skill_size, primitive_horizon)
    planner = _build_planner(
        skill_dynamics_network,
        task_reward_fn,
        skill_size,
        planning_horizon,
        primitive_horizon,
        num_candidates,
        refine_steps,
        temperature,
        smoothing_beta,
        proposal_std,
        qphi_norm=qphi_norm,
    )

    # Receding-horizon proposal (warm-started across outer steps)
    proposal_mean = jnp.zeros((planning_horizon, skill_size))

    s_target_log = []
    z_log = []
    qpos_log = []
    env_reward_log = []
    task_reward_log = []

    for _ in range(num_outer_steps):
        rng, rng_plan, rng_exec = jax.random.split(rng, 3)

        # 1) Plan from current s_target (== s_input under the input==target guard)
        s0 = state.info["s_target"]
        first_skill, proposal = planner(q_phi_params, s0, proposal_mean, rng_plan)

        # 2) Execute the first skill for primitive_horizon env steps
        state, sub_traj = executor(state, first_skill, sac_params, rng_exec)

        # 3) Warm-start next plan with shifted proposal
        proposal_mean = jnp.roll(proposal.mean, shift=-1, axis=0).at[-1].set(0.0)

        # Record
        s_target_log.append(np.asarray(sub_traj["s_target"]))
        z_log.append(np.asarray(first_skill))
        qpos_log.append(np.asarray(sub_traj["qpos"]))
        env_reward_log.append(np.asarray(sub_traj["reward_env"]))

        # Score the executed sub-trajectory under the same task reward
        # we planned with — this is the apples-to-apples MPPI metric.
        task_r = float(task_reward_fn(sub_traj["s_target"]))
        task_reward_log.append(task_r)

    s_target_arr = np.concatenate(s_target_log, axis=0)  # (T_total, target_size)
    qpos_arr = np.concatenate(qpos_log, axis=0)  # (T_total, nq)
    env_reward_arr = np.concatenate(env_reward_log, axis=0)  # (T_total,)
    z_arr = np.asarray(z_log)  # (num_outer_steps, D_z)
    task_reward_arr = np.asarray(task_reward_log)  # (num_outer_steps,)

    return {
        "s_target": s_target_arr,
        "qpos": qpos_arr,
        "skills": z_arr,
        "env_reward_per_step": env_reward_arr,
        "task_reward_per_segment": task_reward_arr,
        "env_reward_total": float(env_reward_arr.sum()),
        "task_reward_total": float(task_reward_arr.sum()),
        "final_s_target": s_target_arr[-1],
        "initial_s_target": s_target_arr[0] if len(s_target_arr) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    traj: dict,
    q_phi_net,
    q_phi_params,
    skill_size: int,
    prior_samples: int = 100,
    rng: jax.Array | None = None,
    qphi_norm=None,
) -> dict:
    """Compute the full DADS eval metric suite from a rollout trajectory."""
    if rng is None:
        rng = jax.random.PRNGKey(42)

    s_input = traj["s_input"]  # (N, T, input_size)  q_φ conditioning
    s_target = traj["s_target"]  # (N, T, target_size) skill space
    s_target_next = traj["s_target_next"]  # (N, T, target_size)
    z = traj["z"]  # (N, T, D_z)

    N, T = s_target.shape[:2]  # noqa: N806  (batch/time dims)
    deltas = s_target_next - s_target  # (N, T, target_size)

    # ── 1) Skill diversity (in TARGET space — what skills control) ────────
    # Std across skills (axis=0), at every timestep, then mean over (T, D).
    dispersion_per_t = jnp.std(s_target, axis=0).mean(axis=-1)  # (T,)
    mean_dispersion = float(dispersion_per_t.mean())
    end_dispersion = float(dispersion_per_t[-1])

    # Pairwise distance of final target states across skills
    final_s = s_target[:, -1]  # (N, target_size)
    pairwise = jnp.linalg.norm(final_s[:, None] - final_s[None, :], axis=-1)
    mean_pairwise = float((pairwise.sum() - jnp.trace(pairwise)) / (N * (N - 1)))

    # ── 2) q_φ log-probability of observed transitions ────────────────────
    s_input_flat = s_input.reshape(-1, s_input.shape[-1])
    delta_flat = deltas.reshape(-1, deltas.shape[-1])
    z_flat = z.reshape(-1, z.shape[-1])
    logp = log_prob(q_phi_net, q_phi_params, s_input_flat, z_flat, delta_flat, qphi_norm)
    mean_logp = float(logp.mean())

    # ── 3) Modal-mean prediction error (physical target units;
    #       modal_delta un-normalizes when qphi_norm is given) ─────────────
    modal_means = modal_delta(q_phi_net, q_phi_params, s_input_flat, z_flat, qphi_norm)
    pred_err = jnp.linalg.norm(modal_means - delta_flat, axis=-1)  # (N*T,)
    mean_pred_err = float(pred_err.mean())
    median_pred_err = float(jnp.median(pred_err))

    # ── 4) DADS intrinsic reward at eval time ─────────────────────────────
    # `compute_dads_reward` evaluates q_φ on (B, L) alt-skill pairs in a
    # single vmap — for B = num_skills × episode_length × L = 100 and a
    # [512, 512] q_φ, that's ~320k forward passes producing a 625 MB
    # intermediate tensor that fits poorly on small GPUs. Chunk over B.
    B = s_input_flat.shape[0]  # noqa: N806  (batch dim, chunked over below)
    metrics_chunk = 256
    r_chunks = []
    for start in range(0, B, metrics_chunk):
        end = min(start + metrics_chunk, B)
        chunk_rng = jax.random.fold_in(rng, start)
        z_alts_chunk = jax.random.uniform(
            chunk_rng,
            (end - start, prior_samples, skill_size),
            minval=-1.0,
            maxval=1.0,
        )
        r_chunk = compute_dads_reward(
            q_phi_net,
            q_phi_params,
            s_input_flat[start:end],
            z_flat[start:end],
            delta_flat[start:end],
            z_alts_chunk,
            norm=qphi_norm,
        )
        r_chunks.append(r_chunk)
    r_dads = jnp.concatenate(r_chunks, axis=0)
    mean_r = float(r_dads.mean())

    # ── Per-skill summaries (for plots) ───────────────────────────────────
    per_skill_logp = np.asarray(logp.reshape(N, T).mean(axis=1))
    per_skill_pred_err = np.asarray(pred_err.reshape(N, T).mean(axis=1))
    per_skill_r = np.asarray(r_dads.reshape(N, T).mean(axis=1))

    # ── 5) Behavioural magnitude / coverage (motion-collapse diagnostics) ──
    # r_dads can be maximized by tiny, hyper-predictable motions, so we report
    # how much the skills actually MOVE and how much workspace they COVER. A
    # high-r_dads / low-displacement run is a degenerate (near-stationary)
    # solution, not a good one.
    step_disp = jnp.linalg.norm(deltas, axis=-1)  # (N, T)
    mean_step_displacement = float(step_disp.mean())  # avg |Δ| per step
    mean_path_length = float(step_disp.sum(axis=1).mean())  # per-skill total, avg
    final_ranges = final_s.max(axis=0) - final_s.min(axis=0)  # (target_size,)
    final_bbox_volume = float(jnp.prod(final_ranges))  # workspace coverage proxy

    # Policy-health: tanh-squashed actions in (-1, 1); saturation = exploiting bounds.
    act = traj["action"]  # (N, T, A)
    action_abs_mean = float(jnp.abs(act).mean())
    action_frac_saturated = float((jnp.abs(act) > 0.99).mean())

    # The ceiling on r_dads is log(L+1) — useful for normalizing
    r_ceiling = float(np.log(prior_samples + 1))

    return {
        "scalars": {
            "mean_skill_dispersion": mean_dispersion,
            "end_skill_dispersion": end_dispersion,
            "mean_pairwise_distance_final_state": mean_pairwise,
            "mean_logp_q_phi": mean_logp,
            "mean_prediction_error_L2": mean_pred_err,
            "median_prediction_error_L2": median_pred_err,
            "mean_r_dads": mean_r,
            "r_dads_ceiling_log_L_plus_1": r_ceiling,
            "r_dads_normalized": mean_r / r_ceiling,
            "mean_step_displacement": mean_step_displacement,
            "mean_path_length": mean_path_length,
            "final_bbox_volume": final_bbox_volume,
            "action_abs_mean": action_abs_mean,
            "action_frac_saturated": action_frac_saturated,
        },
        "per_timestep_dispersion": np.asarray(dispersion_per_t),
        "per_skill": {
            "mean_logp": per_skill_logp,
            "mean_pred_err": per_skill_pred_err,
            "mean_r_dads": per_skill_r,
        },
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(traj: dict, metrics: dict, skills: np.ndarray, output_dir: str) -> list[str]:
    """Save diagnostic PNGs. Returns the list of saved file paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    saved = []
    os.makedirs(output_dir, exist_ok=True)

    # Plots are over q_φ's TARGET space (what skills are supposed to control).
    s_rest = np.asarray(traj["s_target"])  # (N, T, target_size)
    N, T, D = s_rest.shape  # noqa: N806  (batch/time/dim)

    # Color each skill — for 2-D z, color by angle; otherwise by index.
    if skills.shape[1] == 2:
        angles = np.arctan2(skills[:, 1], skills[:, 0])
        norm = (angles - angles.min()) / (angles.max() - angles.min() + 1e-8)
        colors = cm.hsv(norm)
    else:
        colors = cm.viridis(np.linspace(0, 1, N))

    # 1) Restricted-state trajectories — one subplot per dim
    fig, axes = plt.subplots(D, 1, figsize=(9, 2.0 * D + 1.0), squeeze=False, sharex=True)
    for d in range(D):
        ax = axes[d, 0]
        for n in range(N):
            ax.plot(s_rest[n, :, d], color=colors[n], alpha=0.85, linewidth=1.2)
        ax.set_ylabel(f"s_target[{d}]")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("step")
    fig.suptitle(f"Skill trajectories across {N} fixed skills", fontsize=11)
    path = os.path.join(output_dir, "skill_trajectories.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # 1b) If restricted state is the end-effector position (3 dims), plot the
    #     EE trajectories in 3-D space colored by skill. Start = black dot,
    #     end = same-color X marker, so direction is unambiguous.
    if D == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        for n in range(N):
            xs, ys, zs = s_rest[n, :, 0], s_rest[n, :, 1], s_rest[n, :, 2]
            ax.plot(xs, ys, zs, color=colors[n], alpha=0.85, linewidth=1.2)
            ax.scatter(xs[0], ys[0], zs[0], color="black", s=12, depthshade=False)
            ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[n], marker="x", s=35, depthshade=False)
        ax.set_xlabel("ee_x")
        ax.set_ylabel("ee_y")
        ax.set_zlabel("ee_z")
        ax.set_title(f"End-effector trajectories across {N} skills " "(● start, ✕ end)")
        path = os.path.join(output_dir, "ee_trajectories_3d.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

        # Top-down (xy) view too — easier to read than the 3-D plot for many
        # PAROL6 tasks where z barely changes.
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for n in range(N):
            ax.plot(s_rest[n, :, 0], s_rest[n, :, 1], color=colors[n], alpha=0.85, linewidth=1.2)
            ax.scatter(s_rest[n, 0, 0], s_rest[n, 0, 1], color="black", s=12, zorder=3)
            ax.scatter(s_rest[n, -1, 0], s_rest[n, -1, 1], color=colors[n], marker="x", s=40, zorder=3)
        ax.set_xlabel("ee_x")
        ax.set_ylabel("ee_y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"End-effector trajectories (top-down) — {N} skills")
        path = os.path.join(output_dir, "ee_trajectories_xy.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    # 2) Dispersion over time
    disp = metrics["per_timestep_dispersion"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(disp, linewidth=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("std(s_target) over skills")
    ax.set_title("Skill diversity over time (higher = skills produce more different trajectories)")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "skill_dispersion.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # 3) Per-skill r_dads, log-prob, prediction error — small multiples
    per_skill = metrics["per_skill"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, key, title in zip(
        axes,
        ["mean_logp", "mean_pred_err", "mean_r_dads"],
        ["mean log q_φ(Δs|s,z)", "mean ||Δs - μ_modal||₂", "mean r_dads"],
        strict=True,
    ):
        ax.bar(np.arange(N), per_skill[key], color=colors)
        ax.set_xlabel("skill index")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    # Add the DADS ceiling line on the r_dads panel
    axes[2].axhline(
        metrics["scalars"]["r_dads_ceiling_log_L_plus_1"],
        color="red",
        linestyle="--",
        linewidth=1,
        label="log(L+1) ceiling",
    )
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-skill quality breakdown", fontsize=11)
    path = os.path.join(output_dir, "per_skill_metrics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # 4) If 2-D z: 2-D skill space colored by reward — diagnostic of "which
    #    regions of skill space the policy can actually execute well"
    if skills.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sc = ax.scatter(
            skills[:, 0],
            skills[:, 1],
            c=per_skill["mean_r_dads"],
            cmap="viridis",
            s=200,
            edgecolors="black",
        )
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        ax.set_title("Per-skill DADS reward over 2-D z space")
        plt.colorbar(sc, ax=ax, label="mean r_dads")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "z_space_reward.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Optional MuJoCo video rendering
# ---------------------------------------------------------------------------


def render_video(
    env,
    traj: dict,
    output_path: str,
    grid_rows: int,
    grid_cols: int,
    fps: int = 30,
    resolution: tuple[int, int] = (256, 256),
) -> str:
    """Render a tiled video of the rollouts. Reuses the same offscreen
    MuJoCo pattern as VideoRecorderHook (EGL, full-mesh model)."""
    import imageio
    import mujoco

    from core_rl.callbacks.video_recorder import _add_skybox

    os.environ.setdefault("MUJOCO_GL", "egl")

    n_envs = grid_rows * grid_cols
    qpos_seq = np.asarray(traj["qpos"])  # (N, T, nq)
    n_avail, T, _ = qpos_seq.shape  # noqa: N806  (time dim)
    n_envs = min(n_envs, n_avail)

    # Build a render-quality model (full meshes), mirroring VideoRecorderHook —
    # including the scene's `background:` skybox (URDF models default to a
    # dark void; VideoRecorderHook already does this, eval videos didn't).
    spec = mujoco.MjSpec.from_file(env.robot.mjcf_path)
    for jname in env.robot.joint_names:
        act = spec.add_actuator()
        act.name = f"act_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = 1.0
    _add_skybox(spec, getattr(getattr(env, "scene", None), "background", None))
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    width, height = resolution
    renderer = mujoco.Renderer(mj_model, height, width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.15]
    camera.distance = 0.8
    camera.azimuth = 135.0
    camera.elevation = -25.0

    grid_h = height * grid_rows
    grid_w = width * grid_cols

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1)
    try:
        for t in range(T):
            frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for idx in range(n_envs):
                row, col = divmod(idx, grid_cols)
                mj_data.qpos[:] = qpos_seq[idx, t]
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera)
                pixels = renderer.render()
                y0, x0 = row * height, col * width
                frame[y0 : y0 + height, x0 : x0 + width] = pixels
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="DADS evaluation: roll out a trained policy with fixed skills.")
    p.add_argument("--robot", type=str, required=True)
    p.add_argument("--task", type=str, default="skill_conditioned")
    p.add_argument("--params", type=str, required=True, help="Path to dads_params.pkl")
    p.add_argument("--config", type=str, default=None, help="Optional config override YAML")
    p.add_argument(
        "--num-skills",
        type=int,
        default=16,
        help="Number of fixed skills to evaluate (perfect square ⇒ grid for 2-D z)",
    )
    p.add_argument("--episode-length", type=int, default=200)
    p.add_argument("--prior-samples", type=int, default=100, help="L for the DADS-reward denominator")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--video", action="store_true", help="Render tiled MP4 of rollouts")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use tanh(μ) actions instead of tanh(μ + σ·ε). "
        "Cleaner-looking videos, but q_φ metrics will be worse because "
        "q_φ was trained on stochastic-policy data.",
    )
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--scene-file", type=str, default="")

    # ── MPPI zero-shot planning eval (separate mode) ─────────────────
    p.add_argument(
        "--mppi",
        action="store_true",
        help="Run MPPI zero-shot planning eval in addition to the skill-diversity rollout.",
    )
    p.add_argument(
        "--mppi-reward",
        type=str,
        default="max_axis",
        choices=["max_axis", "axis_progress", "goal", "path_goal"],
        help="Task reward function for MPPI planning.",
    )
    p.add_argument(
        "--mppi-axis", type=int, default=0, help="For max_axis/axis_progress: which dim of s_target to maximise."
    )
    p.add_argument("--mppi-sign", type=float, default=1.0, help="For max_axis/axis_progress: +1 for max, -1 for min.")
    p.add_argument(
        "--mppi-goal",
        type=float,
        nargs="+",
        default=None,
        help="For goal/path_goal: goal coordinates in s_target space.",
    )
    p.add_argument("--mppi-planning-horizon", type=int, default=4, help="H_P: number of skills in each plan.")
    p.add_argument("--mppi-primitive-horizon", type=int, default=10, help="H_Z: env steps each skill is held for.")
    p.add_argument(
        "--mppi-num-candidates", type=int, default=64, help="N: candidate plans sampled per MPPI refine step."
    )
    p.add_argument("--mppi-refine-steps", type=int, default=10, help="R: refinement iterations per outer plan.")
    p.add_argument(
        "--mppi-temperature", type=float, default=10.0, help="γ: softmax sharpness in the importance-weighted average."
    )
    p.add_argument("--mppi-smoothing", type=float, default=0.9, help="EMA coefficient on the proposal mean.")
    p.add_argument(
        "--mppi-proposal-std",
        type=float,
        default=0.5,
        help="Sampling stddev around the proposal mean (in skill space).",
    )
    p.add_argument(
        "--mppi-outer-steps",
        type=int,
        default=20,
        help="Number of plan+execute cycles. Total env steps = outer × primitive_horizon.",
    )
    args = p.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.params) or ".", "eval")
    os.makedirs(output_dir, exist_ok=True)

    print("=== DADS Evaluation ===")
    print(f"  Robot:           {args.robot}")
    print(f"  Task:            {args.task}")
    print(f"  Params:          {args.params}")
    print(f"  Num skills:      {args.num_skills}")
    print(f"  Episode length:  {args.episode_length}")
    print(f"  Output dir:      {output_dir}")
    print()

    # ── 1. Config + env ────────────────────────────────────────────────
    cfg = _load_config(args.config)
    print(cfg)
    env_cfg = cfg["env"]
    dads_cfg = cfg["algorithms"]["dads"]

    all_task_kwargs = env_cfg.get("task_kwargs") or {}
    task_kwargs = dict(all_task_kwargs.get(args.task, {}) or {})

    # Auto-sync skill_size from algo config into env task_kwargs — must match
    # what the loaded params expect. (Same logic as train.py — otherwise the
    # env's default skill_size silently desyncs from the trained network.)
    if args.task == "skill_conditioned":
        algo_skill_size = dads_cfg.get("skill_size")
        env_skill_size = task_kwargs.get("skill_size")
        if algo_skill_size is not None:
            if env_skill_size is not None and int(env_skill_size) != int(algo_skill_size):
                raise SystemExit(
                    f"skill_size mismatch in eval config: "
                    f"algorithms.dads.skill_size={algo_skill_size} vs "
                    f"env.task_kwargs.skill_conditioned.skill_size={env_skill_size}."
                )
            if env_skill_size is None:
                task_kwargs["skill_size"] = int(algo_skill_size)
                print(f"  Auto-synced env.task_kwargs.skill_conditioned.skill_size = {algo_skill_size}")

    scene = load_scene(args.scene_file) if args.scene_file else None
    robot = resolve_robot(args.robot, scene=scene)

    env = make_env(
        robot=robot,
        task_name=args.task,
        backend=env_cfg.get("backend", "mjx"),
        control_dt=env_cfg["control_dt"],
        physics_dt=env_cfg["physics_dt"],
        max_episode_steps=max(env_cfg["max_episode_steps"], args.episode_length + 1),
        scene=scene,
        task_kwargs=task_kwargs,
    )

    # ── 2. Load params + rebuild networks ──────────────────────────────
    with open(args.params, "rb") as f:
        params = pickle.load(f)
    # New checkpoints are a 4-tuple (normalizer, policy, skill_dynamics,
    # qphi_norm); older ones are a 3-tuple with no q_φ I/O normalization.
    if isinstance(params, tuple) and len(params) == 4:
        norm_params, policy_params, q_phi_params, qphi_norm = params
    elif isinstance(params, tuple) and len(params) == 3:
        norm_params, policy_params, q_phi_params = params
        qphi_norm = None
        print(
            "  [warn] 3-tuple params (pre-normalization checkpoint) — q_φ "
            "evaluated WITHOUT input/target normalization."
        )
    else:
        raise ValueError(
            f"Expected params to be a 3- or 4-tuple; got type={type(params).__name__} "
            f"len={len(params) if hasattr(params, '__len__') else 'N/A'}"
        )
    sac_params = (norm_params, policy_params)

    make_policy = _build_sac_policy(env, dads_cfg)
    q_phi_net, skill_size, input_obs_size, target_obs_size = _build_skill_dynamics(env, dads_cfg)

    print(f"  Skill size:      {skill_size}")
    print(f"  Input obs:       {input_obs_size}   (q_φ conditioning)")
    print(f"  Target obs:      {target_obs_size}   (q_φ Δ prediction)")
    print()

    # ── 3. Skills + rollout ────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_skills, rng_rollout, rng_metrics = jax.random.split(rng, 4)
    skills = sample_eval_skills(skill_size, args.num_skills, rng_skills)
    print(f"  Skills (first 4):\n{np.asarray(skills[:4])}")

    print("\nCompiling + running rollouts...")
    t0 = time.time()
    print(f"  Policy mode:     {'deterministic (mode)' if args.deterministic else 'stochastic (sample)'}")
    rollout_fn = _build_rollout_fn(
        env,
        make_policy,
        skill_size,
        args.episode_length,
        deterministic=args.deterministic,
    )
    traj = rollout_fn(sac_params, skills, rng_rollout)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), traj)
    print(f"  Rollout done in {time.time() - t0:.1f}s " f"({args.num_skills} envs × {args.episode_length} steps).")

    # ── 4. Metrics ─────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics = compute_metrics(
        traj,
        q_phi_net,
        q_phi_params,
        skill_size=skill_size,
        prior_samples=args.prior_samples,
        rng=rng_metrics,
        qphi_norm=qphi_norm,
    )

    # ── 5. Plots ───────────────────────────────────────────────────────
    print("Generating plots...")
    plot_paths = make_plots(traj, metrics, np.asarray(skills), output_dir)
    for pth in plot_paths:
        print(f"  ↳ {pth}")

    # ── 6. Video (optional) ────────────────────────────────────────────
    if args.video:
        # Choose a grid that fits num_skills
        side = int(round(args.num_skills**0.5))
        rows = cols = side if side * side == args.num_skills else max(1, int(args.num_skills**0.5))
        rows = max(1, rows)
        cols = max(1, args.num_skills // rows)
        video_path = os.path.join(output_dir, "skills_rollout.mp4")
        print(f"\nRendering video ({rows}×{cols} grid) → {video_path}")
        t0 = time.time()
        render_video(env, traj, video_path, grid_rows=rows, grid_cols=cols, fps=args.video_fps)
        print(f"  Done in {time.time() - t0:.1f}s")

    # ── 7. Save metrics JSON ───────────────────────────────────────────
    json_path = os.path.join(output_dir, "metrics.json")
    payload = {
        "scalars": metrics["scalars"],
        "per_skill": {k: v.tolist() for k, v in metrics["per_skill"].items()},
        "skills": np.asarray(skills).tolist(),
        "config": {
            "robot": args.robot,
            "task": args.task,
            "num_skills": args.num_skills,
            "episode_length": args.episode_length,
            "prior_samples": args.prior_samples,
            "task_kwargs": task_kwargs,
            "skill_size": skill_size,
            "input_obs_size": input_obs_size,
            "target_obs_size": target_obs_size,
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nMetrics JSON saved → {json_path}")

    # ── 8. Console summary ─────────────────────────────────────────────
    s = metrics["scalars"]
    print("\n" + "=" * 60)
    print(" SUMMARY ".center(60, "="))
    print("=" * 60)
    print(f"  Skill diversity (mean dispersion):    {s['mean_skill_dispersion']:.4f}")
    print(f"  Skill diversity (final-step):         {s['end_skill_dispersion']:.4f}")
    print(f"  Mean pairwise final-state distance:   {s['mean_pairwise_distance_final_state']:.4f}")
    print(f"  q_φ mean log-prob (higher is better): {s['mean_logp_q_phi']:.4f}")
    print(f"  q_φ modal-mean L2 error (lower):      {s['mean_prediction_error_L2']:.4f}")
    print(f"  Mean r_dads:                          {s['mean_r_dads']:.4f}")
    print(f"  r_dads ceiling log(L+1):              {s['r_dads_ceiling_log_L_plus_1']:.4f}")
    print(f"  r_dads / ceiling (0→1 = excellent):   {s['r_dads_normalized']:.2%}")
    print("=" * 60)

    # ── 9. MPPI zero-shot planning (optional, --mppi) ──────────────────
    if args.mppi:
        print("\n" + "=" * 60)
        print(" MPPI ZERO-SHOT PLANNING ".center(60, "="))
        print("=" * 60)
        print(f"  Reward kind:           {args.mppi_reward}")
        if args.mppi_reward in ("max_axis", "axis_progress"):
            print(f"  Axis / sign:           {args.mppi_axis} / {args.mppi_sign:+g}")
        if args.mppi_reward in ("goal", "path_goal"):
            print(f"  Goal:                  {args.mppi_goal}")
        print(
            f"  Planning horizon:      {args.mppi_planning_horizon} skills "
            f"× {args.mppi_primitive_horizon} env steps each"
        )
        print(f"  Candidates × refines:  {args.mppi_num_candidates} × {args.mppi_refine_steps}")
        print(
            f"  Outer plan+exec loop:  {args.mppi_outer_steps} cycles "
            f"({args.mppi_outer_steps * args.mppi_primitive_horizon} total env steps)"
        )
        print()

        task_reward_fn = build_task_reward(
            args.mppi_reward,
            axis=args.mppi_axis,
            sign=args.mppi_sign,
            goal=args.mppi_goal,
        )

        rng_mppi = jax.random.PRNGKey(args.seed + 7919)  # decorrelate from rollout RNG

        print("Compiling MPPI + executor (first plan will take a moment)...")
        t_mppi = time.time()
        mppi_traj = run_mppi_eval(
            env=env,
            make_policy=make_policy,
            sac_params=sac_params,
            skill_dynamics_network=q_phi_net,
            q_phi_params=q_phi_params,
            task_reward_fn=task_reward_fn,
            skill_size=skill_size,
            num_outer_steps=args.mppi_outer_steps,
            input_obs_size=input_obs_size,
            target_obs_size=target_obs_size,
            planning_horizon=args.mppi_planning_horizon,
            primitive_horizon=args.mppi_primitive_horizon,
            num_candidates=args.mppi_num_candidates,
            refine_steps=args.mppi_refine_steps,
            temperature=args.mppi_temperature,
            smoothing_beta=args.mppi_smoothing,
            proposal_std=args.mppi_proposal_std,
            rng=rng_mppi,
            qphi_norm=qphi_norm,
        )
        print(f"  Done in {time.time() - t_mppi:.1f}s")
        print()
        print(f"  Env reward (sum over executed steps):   {mppi_traj['env_reward_total']:.4f}")
        print(f"  Task reward (sum over MPPI segments):   {mppi_traj['task_reward_total']:.4f}")
        s0_str = ", ".join(f"{v:.3f}" for v in mppi_traj["initial_s_target"])
        sf_str = ", ".join(f"{v:.3f}" for v in mppi_traj["final_s_target"])
        print(f"  Initial s_target:                       [{s0_str}]")
        print(f"  Final   s_target:                       [{sf_str}]")
        print("=" * 60)

        # Save per-segment skills + trajectory + metrics
        mppi_json = os.path.join(output_dir, "mppi_metrics.json")
        with open(mppi_json, "w") as f:
            json.dump(
                {
                    "config": {
                        "reward_kind": args.mppi_reward,
                        "axis": args.mppi_axis,
                        "sign": args.mppi_sign,
                        "goal": args.mppi_goal,
                        "planning_horizon": args.mppi_planning_horizon,
                        "primitive_horizon": args.mppi_primitive_horizon,
                        "num_candidates": args.mppi_num_candidates,
                        "refine_steps": args.mppi_refine_steps,
                        "temperature": args.mppi_temperature,
                        "smoothing_beta": args.mppi_smoothing,
                        "proposal_std": args.mppi_proposal_std,
                        "outer_steps": args.mppi_outer_steps,
                    },
                    "env_reward_total": mppi_traj["env_reward_total"],
                    "task_reward_total": mppi_traj["task_reward_total"],
                    "task_reward_per_segment": mppi_traj["task_reward_per_segment"].tolist(),
                    "env_reward_per_step": mppi_traj["env_reward_per_step"].tolist(),
                    "skills_per_segment": mppi_traj["skills"].tolist(),
                    "initial_s_target": mppi_traj["initial_s_target"].tolist(),
                    "final_s_target": mppi_traj["final_s_target"].tolist(),
                    "s_target_trajectory": mppi_traj["s_target"].tolist(),
                },
                f,
                indent=2,
            )
        print(f"MPPI metrics JSON saved → {mppi_json}")

        # Render the MPPI rollout as a single (1x1) video
        if args.video:
            mppi_video_path = os.path.join(output_dir, "mppi_rollout.mp4")
            print(f"\nRendering MPPI rollout → {mppi_video_path}")
            qpos = mppi_traj["qpos"][None]  # (1, T, nq)
            faux_traj = {"qpos": qpos}
            t_v = time.time()
            render_video(env, faux_traj, mppi_video_path, grid_rows=1, grid_cols=1, fps=args.video_fps)
            print(f"  Done in {time.time() - t_v:.1f}s")


if __name__ == "__main__":
    main()
