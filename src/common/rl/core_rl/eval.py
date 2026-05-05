"""Evaluate and visualize trained RL policies.

Usage::

    # Full debug run (all plots + console stats)
    python -m core_rl.eval --model-dir /ros2_ws/core/models/<run_id> \\
        --robot parol6 --task ee_tracking --algo ppo --debug

    # Video with ee_target marker + metric plots
    python -m core_rl.eval --model-dir /ros2_ws/core/models/<run_id> \\
        --robot parol6 --task ee_tracking --algo ppo --video --plot --plot-ee

    # 20-episode distribution stats
    python -m core_rl.eval --model-dir /ros2_ws/core/models/<run_id> \\
        --robot parol6 --task ee_tracking --algo ppo --episodes 20 --plot-debug

    # Compare two models side by side
    python -m core_rl.eval \\
        --compare /ros2_ws/core/models/run_A /ros2_ws/core/models/run_B \\
        --robot parol6 --task ee_tracking --algo ppo --plot
"""

from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402, I001
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from core_rl.env import make_env  # noqa: E402
from core_rl.robot import resolve_robot  # noqa: E402


# ---------------------------------------------------------------------------
# Network reconstruction
# ---------------------------------------------------------------------------


def reconstruct_make_policy(
    algo: str,
    env,
    hidden_layer_sizes: tuple[int, ...] = (256, 256),
    normalize_observations: bool = True,
):
    """Reconstruct ``make_policy`` from network architecture.

    Only the architecture matters here — trained weights come from the
    loaded ``params`` pickle.  The architecture must match what was used
    during training (hidden sizes, normalization, algorithm).
    """
    from brax.training.acme import running_statistics

    normalize = running_statistics.normalize if normalize_observations else lambda x, y: x

    if algo == "ppo":
        from brax.training.agents.ppo import networks as ppo_networks

        network = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=hidden_layer_sizes,
            value_hidden_layer_sizes=hidden_layer_sizes,
        )
        return ppo_networks.make_inference_fn(network)

    if algo == "sac":
        from brax.training.agents.sac import networks as sac_networks

        network = sac_networks.make_sac_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=normalize,
            hidden_layer_sizes=hidden_layer_sizes,
        )
        return sac_networks.make_inference_fn(network)

    raise ValueError(f"Unknown algorithm '{algo}'. Supported: ppo, sac")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout(env, make_policy, params, n_episodes=1, episode_length=500, seed=0):
    """Run deterministic rollouts and return per-step data as numpy arrays.

    Uses ``jax.lax.scan`` for a single GPU kernel — no per-step host
    transfers.  Each episode is JIT-compiled independently so different
    seeds produce different initial states / targets.
    """
    has_ee = hasattr(env, "_ee_body_id") and env._ee_body_id >= 0
    ee_body_id = env._ee_body_id if has_ee else 0

    def _run(params, rng):
        policy = make_policy(params, deterministic=True)
        rng, reset_rng = jax.random.split(rng)
        init_state = env.reset(reset_rng)

        def _step(carry, _):
            state, rng = carry
            rng, act_rng = jax.random.split(rng)
            action, _ = policy(state.obs, act_rng)
            ns = env.step(state, action)
            step_data = {
                "qpos": ns.pipeline_state.q,
                "reward": ns.reward,
                "pos_error": ns.metrics["pos_error"],
                "vel_norm": ns.metrics["vel_norm"],
                "success": ns.metrics["success"],
                "ee_pos": ns.pipeline_state.xpos[ee_body_id],
                "raw_action": action,
                "scaled_action": ns.info.get("action", action),
                # Capture the obs that generated this action (for NaN/range checks)
                "obs": state.obs,
            }
            return (ns, rng), step_data

        _, traj = jax.lax.scan(_step, (init_state, rng), None, length=episode_length)

        # Prepend initial frame for video (episode_length+1 qpos frames)
        traj["qpos"] = jnp.concatenate([init_state.pipeline_state.q[None], traj["qpos"]])
        traj["ee_target"] = init_state.info.get("ee_target", jnp.zeros(3))
        jax.debug.print("ee_traj: {ee}", ee=traj["ee_target"])
        traj["q_target"] = init_state.info.get("q_target", jnp.zeros(env.action_size))
        return traj

    jit_run = jax.jit(_run)

    trajectories = []
    for ep in range(n_episodes):
        rng = jax.random.PRNGKey(seed + ep)
        traj = jit_run(params, rng)
        traj_np = {k: np.asarray(v) for k, v in traj.items()}
        traj_np["has_ee"] = has_ee
        trajectories.append(traj_np)

    return trajectories


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------


def _build_render_model(env, add_target_marker=False):
    """Build a CPU MuJoCo model with full visual meshes for rendering.

    Optionally adds a small red sphere as an end-effector target marker.
    The marker is a free body so its position can be set via qpos.
    """
    spec = mujoco.MjSpec.from_file(env.robot.mjcf_path)

    for jname in env.robot.joint_names:
        act = spec.add_actuator()
        act.name = f"act_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = 1.0

    target_qpos_start = -1
    if add_target_marker:
        body = spec.worldbody.add_body()
        body.name = "ee_target_marker"
        body.add_freejoint().name = "ee_target_marker_joint"
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = np.array([1.015, 0.0, 0.0])
        geom.rgba = np.array([1.0, 0.2, 0.2, 0.7])
        geom.contype = 0
        geom.conaffinity = 0

    model = spec.compile()
    data = mujoco.MjData(model)

    if add_target_marker:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ee_target_marker_joint")
        target_qpos_start = model.jnt_qposadr[jid]

    return model, data, target_qpos_start


def render_video(env, traj, output_path, fps=30, resolution=(640, 480)):
    """Render a single trajectory to MP4 with optional ee_target marker."""
    has_target = traj["has_ee"] and traj.get("ee_target") is not None
    print(f"has_target: {has_target}")
    model, data, target_qpos_start = _build_render_model(env, add_target_marker=has_target)
    renderer = mujoco.Renderer(model, resolution[1], resolution[0])

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.15]
    camera.distance = 0.8
    camera.azimuth = 135.0
    camera.elevation = -25.0

    nq_robot = env._mj_model.nq
    qpos_traj = traj["qpos"]
    ee_target = traj.get("ee_target")

    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1)
    for t in range(len(qpos_traj)):
        data.qpos[:nq_robot] = qpos_traj[t]

        data.qpos[target_qpos_start : target_qpos_start + 3] = ee_target
        data.qpos[target_qpos_start + 3 : target_qpos_start + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        writer.append_data(renderer.render())

    writer.close()
    renderer.close()
    print(f"  Video saved: {output_path}")


# ---------------------------------------------------------------------------
# Plotting — standard
# ---------------------------------------------------------------------------


def plot_metrics(trajectories, output_path, labels=None):
    """Plot per-step reward, position error, velocity norm, and cumulative reward."""
    labels = labels or [f"Model {i}" for i in range(len(trajectories))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for traj, label in zip(trajectories, labels, strict=False):
        steps = np.arange(len(traj["reward"]))
        axes[0, 0].plot(steps, traj["reward"], label=label, alpha=0.8)
        axes[0, 1].plot(steps, traj["pos_error"], label=label, alpha=0.8)
        axes[1, 0].plot(steps, traj["vel_norm"], label=label, alpha=0.8)
        axes[1, 1].plot(steps, np.cumsum(traj["reward"]), label=label, alpha=0.8)

    titles = ["Reward", "Position Error", "Joint Velocity Norm", "Cumulative Reward"]
    for ax, title in zip(axes.flat, titles, strict=False):
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


def plot_ee_trajectory(traj, output_path):
    """3D scatter/line plot of EE path with start, end, and target markers."""
    if not traj["has_ee"]:
        return

    ee_pos = traj["ee_pos"]
    ee_target = traj["ee_target"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], "b-", alpha=0.6, label="EE path")
    ax.scatter(*ee_pos[0], c="green", s=60, marker="o", label="Start")
    ax.scatter(*ee_pos[-1], c="blue", s=60, marker="s", label="End")
    ax.scatter(*ee_target, c="red", s=100, marker="*", label="Target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Trajectory")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  EE trajectory plot saved: {output_path}")


def plot_actions(traj, output_path, joint_names=None):
    """Plot raw (policy output) and scaled (joint target) actions per joint."""
    raw = traj.get("raw_action")
    scaled = traj.get("scaled_action")
    if raw is None or scaled is None:
        print("  WARNING: action data not available, skipping action plot")
        return

    n_joints = raw.shape[1]
    joint_names = joint_names or [f"Joint {i + 1}" for i in range(n_joints)]
    steps = np.arange(len(raw))

    n_cols = 3
    n_rows = (n_joints + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)

    for j in range(n_joints):
        ax = axes[j // n_cols][j % n_cols]
        ax.plot(steps, raw[:, j], label="Raw (policy)", alpha=0.8, color="steelblue")
        ax.plot(steps, scaled[:, j], label="Scaled (joint target)", alpha=0.8, color="darkorange", linestyle="--")
        ax.set_title(joint_names[j])
        ax.set_xlabel("Step")
        ax.set_ylabel("Action")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_joints, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle("Raw vs Scaled Actions per Joint", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Action plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Plotting — debug diagnostics
# ---------------------------------------------------------------------------


def plot_debug(traj, output_path, env):
    """Comprehensive 6-panel diagnostic figure for debugging a non-learning policy.

    Panels:
        [0,0] Error over time      — does pos_error decrease and stay low?
        [0,1] Action distribution  — is the policy saturating at ±1?
        [0,2] Action smoothness    — |Δaction| per step; high = oscillation/instability
        [1,0] Obs validity         — L2 norm over time + NaN/Inf markers
        [1,1] Joint positions      — actual q per joint; erratic = PD instability
        [1,2] Reward breakdown     — reward/step vs cumulative, ee_dist on twin axis
    """
    steps = np.arange(len(traj["reward"]))
    raw = traj.get("raw_action")  # (T, n_joints)
    scaled = traj.get("scaled_action")
    obs = traj.get("obs")  # (T, obs_dim)
    joint_names = env.robot.joint_names
    n_joints = len(joint_names)
    q_indices = np.asarray(env._joint_q_indices)
    success_threshold = getattr(env, "success_threshold", 0.01)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── [0,0] Error over time ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(steps, traj["pos_error"], color="crimson", alpha=0.9, label="pos_error")
    ax.axhline(success_threshold, color="green", linestyle="--", linewidth=1.2, label=f"success ({success_threshold})")
    # Shade the region below success threshold to show where policy "wins"
    ax.fill_between(steps, 0, traj["pos_error"], where=traj["pos_error"] < success_threshold, color="green", alpha=0.2)
    ax.set_title("Position Error over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (m or rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── [0,1] Action distribution (saturation check) ──────────────────────
    ax = axes[0, 1]
    if raw is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
        for j in range(n_joints):
            ax.hist(raw[:, j], bins=40, alpha=0.5, color=colors[j], label=joint_names[j])
        ax.axvline(-0.9, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(0.9, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="±0.9 sat.")
        ax.set_title("Raw Action Distribution  (policy output ∈ [-1, 1])")
        ax.set_xlabel("Action value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No action data", ha="center", va="center", transform=ax.transAxes)

    # ── [0,2] Action smoothness ────────────────────────────────────────────
    ax = axes[0, 2]
    if raw is not None and len(raw) > 1:
        deltas = np.abs(np.diff(raw, axis=0))  # (T-1, n_joints)
        mean_delta = deltas.mean(axis=1)
        max_delta = deltas.max(axis=1)
        ax.plot(steps[1:], mean_delta, label="mean |Δaction|", color="steelblue", alpha=0.9)
        ax.plot(steps[1:], max_delta, label="max |Δaction|", color="darkorange", alpha=0.7)
        ax.set_title("Action Smoothness  (high = oscillation / instability)")
        ax.set_xlabel("Step")
        ax.set_ylabel("|action[t] − action[t−1]|")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No action data", ha="center", va="center", transform=ax.transAxes)

    # ── [1,0] Observation validity ─────────────────────────────────────────
    ax = axes[1, 0]
    if obs is not None:
        obs_norm = np.linalg.norm(obs, axis=1)
        bad_steps = np.where(~np.isfinite(obs).all(axis=1))[0]
        ax.plot(steps, obs_norm, color="purple", alpha=0.85, label="‖obs‖₂")
        if len(bad_steps) > 0:
            ax.scatter(
                bad_steps,
                obs_norm[np.clip(bad_steps, 0, len(obs_norm) - 1)],
                c="red",
                s=30,
                zorder=5,
                label=f"NaN/Inf ({len(bad_steps)} steps)",
            )
        # Mark each obs dimension's range as a thin band (useful for detecting exploding values)
        obs_max = np.abs(obs).max(axis=1)
        ax.plot(steps, obs_max, color="purple", alpha=0.3, linestyle="--", linewidth=0.8, label="max |obs dim|")
        ax.set_title("Observation Norm over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("‖obs‖₂")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No obs data", ha="center", va="center", transform=ax.transAxes)

    # ── [1,1] Joint positions over time ───────────────────────────────────
    ax = axes[1, 1]
    qpos = traj["qpos"]  # (T+1, nq_full) — has one extra frame from init
    q_joints = qpos[:, q_indices]  # (T+1, n_joints)
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    for j in range(n_joints):
        ax.plot(q_joints[:, j], color=colors[j], alpha=0.85, label=joint_names[j])
    if scaled is not None:
        for j in range(min(scaled.shape[1], n_joints)):
            ax.plot(steps, scaled[:, j], color=colors[j], linestyle="--", alpha=0.35, linewidth=1.0)
    # Add q_target (episode target for joint_tracking task) if available
    q_target = traj.get("q_target")
    if q_target is not None and q_target.ndim == 1 and np.any(q_target != 0):
        for j in range(min(len(q_target), n_joints)):
            ax.axhline(q_target[j], color=colors[j], linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_title("Joint Positions  (solid=actual, dashed=PD target, dotted=episode target)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position (rad)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── [1,2] Reward + EE distance ─────────────────────────────────────────
    ax = axes[1, 2]
    ax.plot(steps, traj["reward"], color="green", alpha=0.7, label="reward/step")
    ax.plot(steps, np.cumsum(traj["reward"]), color="darkgreen", linestyle="--", label="cumulative")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Step")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    if traj["has_ee"] and traj.get("ee_pos") is not None:
        ax2 = ax.twinx()
        ee_dist = np.linalg.norm(traj["ee_pos"] - traj["ee_target"], axis=1)
        ax2.plot(steps, ee_dist, color="crimson", alpha=0.5, linewidth=1.2, label="ee_dist")
        ax2.axhline(success_threshold, color="crimson", linestyle=":", linewidth=1.0, alpha=0.6)
        ax2.set_ylabel("EE dist (m)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Reward + EE Distance")

    fig.suptitle("Policy Debug Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Debug plot saved: {output_path}")


def plot_multi_episode_stats(trajs, output_path):
    """Distribution plots across multiple episodes — shows consistency of the policy.

    Useful for answering: does the policy fail on all targets, or just some?
    """
    if len(trajs) < 2:
        return

    final_errors = np.array([t["pos_error"][-1] for t in trajs])
    min_errors = np.array([t["pos_error"].min() for t in trajs])
    total_rewards = np.array([t["reward"].sum() for t in trajs])

    # Stack pos_error trajectories for box plot (may differ in length but shouldn't here)
    error_matrix = np.stack([t["pos_error"] for t in trajs], axis=0)  # (N, T)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Final error distribution
    axes[0].hist(final_errors, bins=min(20, len(trajs)), color="crimson", alpha=0.8, edgecolor="white")
    axes[0].axvline(final_errors.mean(), color="black", linestyle="--", label=f"mean={final_errors.mean():.4f}")
    axes[0].set_title("Final pos_error Distribution")
    axes[0].set_xlabel("pos_error at episode end")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Best-achievable error distribution (min over episode)
    axes[1].hist(min_errors, bins=min(20, len(trajs)), color="steelblue", alpha=0.8, edgecolor="white")
    axes[1].axvline(min_errors.mean(), color="black", linestyle="--", label=f"mean={min_errors.mean():.4f}")
    axes[1].set_title("Min pos_error Distribution\n(best the policy ever reaches per episode)")
    axes[1].set_xlabel("min pos_error per episode")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Episode return distribution
    axes[2].hist(total_rewards, bins=min(20, len(trajs)), color="green", alpha=0.8, edgecolor="white")
    axes[2].axvline(total_rewards.mean(), color="black", linestyle="--", label=f"mean={total_rewards.mean():.1f}")
    axes[2].set_title("Episode Return Distribution")
    axes[2].set_xlabel("Total reward per episode")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # pos_error trajectory: mean ± std across episodes
    n_steps = error_matrix.shape[1]
    steps = np.arange(n_steps)
    mean_err = error_matrix.mean(axis=0)
    std_err = error_matrix.std(axis=0)
    axes[3].plot(steps, mean_err, color="crimson", label="mean")
    axes[3].fill_between(steps, mean_err - std_err, mean_err + std_err, color="crimson", alpha=0.2, label="±1 std")
    axes[3].set_title("pos_error over Time (mean ± std)")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("pos_error")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"Multi-Episode Statistics  ({len(trajs)} episodes)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Multi-episode stats saved: {output_path}")


def print_debug_stats(traj, joint_names=None):
    """Print rich debugging statistics to the console.

    Covers the most common failure modes: reward signal too weak, policy
    saturation, observation explosion / NaN, action oscillation.
    """
    raw = traj.get("raw_action")
    obs = traj.get("obs")
    reward = traj["reward"]
    pos_error = traj["pos_error"]

    print("\n  ┌─ Debug Statistics ────────────────────────────────────────────")
    print(
        f"  │ Reward     : mean={reward.mean():.4f}  min={reward.min():.4f}  max={reward.max():.4f}  total={reward.sum():.2f}"
    )
    print(
        f"  │ pos_error  : init={pos_error[0]:.4f}  final={pos_error[-1]:.4f}"
        f"  min={pos_error.min():.4f}  mean={pos_error.mean():.4f}"
    )
    print(f"  │ Success    : {traj['success'].mean():.1%} of steps below threshold")
    print(f"  │ Vel norm   : mean={traj['vel_norm'].mean():.4f}  max={traj['vel_norm'].max():.4f}")

    if raw is not None:
        sat_pct = (np.abs(raw) > 0.9).mean() * 100
        mean_delta = np.abs(np.diff(raw, axis=0)).mean() if len(raw) > 1 else 0.0
        print(f"  │ Actions    : saturation(|a|>0.9)={sat_pct:.1f}%  mean|Δa|={mean_delta:.4f}")
        names = joint_names or [f"J{j}" for j in range(raw.shape[1])]
        for j, name in enumerate(names):
            jsat = (np.abs(raw[:, j]) > 0.9).mean() * 100
            jdelta = np.abs(np.diff(raw[:, j])).mean() if len(raw) > 1 else 0.0
            print(
                f"  │   {name:<12}: mean={raw[:, j].mean():+.3f}  std={raw[:, j].std():.3f}  sat={jsat:.0f}%  |Δa|={jdelta:.4f}"
            )

    if obs is not None:
        nan_count = int((~np.isfinite(obs)).sum())
        obs_norm = np.linalg.norm(obs, axis=1)
        if nan_count > 0:
            print(f"  │ Obs        : *** {nan_count} NaN/Inf values — check reward / physics stability ***")
        else:
            print(f"  │ Obs        : all finite  norm(mean={obs_norm.mean():.2f}  max={obs_norm.max():.2f})")

    print("  └───────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL policies")
    parser.add_argument("--model-dir", type=str, help="Path to a single model output directory")
    parser.add_argument("--compare", nargs="+", type=str, help="Compare multiple model directories")
    parser.add_argument("--robot", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1, help="Number of eval episodes")
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="Render MP4 video(s)")
    parser.add_argument("--plot", action="store_true", help="Plot per-step metrics (reward, error, vel)")
    parser.add_argument("--plot-ee", action="store_true", help="Plot 3D EE trajectory")
    parser.add_argument("--plot-actions", action="store_true", help="Plot raw vs scaled actions per joint")
    parser.add_argument("--plot-debug", action="store_true", help="6-panel diagnostic plot + console stats")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shorthand: enable --plot --plot-ee --plot-actions --plot-debug",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480], help="Video resolution (W H)")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Network hidden layer sizes (must match training config)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir (default: <model-dir>/eval)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific Brax checkpoint directory (e.g. <output-dir>/checkpoints/000002000000). "
        "Overrides the default <model-dir>/<algo>_params.pkl. "
        "Run `ls <output-dir>/checkpoints/` to see available steps.",
    )
    args = parser.parse_args()

    # --debug expands to all individual flags
    if args.debug:
        args.plot = True
        args.plot_ee = True
        args.plot_actions = True
        args.plot_debug = True

    model_dirs = args.compare or ([args.model_dir] if args.model_dir else [])
    if not model_dirs:
        parser.error("Provide --model-dir or --compare")

    # ── Env setup (same architecture as training, no actual training) ──
    robot = resolve_robot(args.robot)
    env = make_env(robot=robot, task_name=args.task, backend="mjx", max_episode_steps=args.episode_length)

    print("=== RL Policy Evaluation ===")
    print(f"  Robot: {args.robot}  Task: {args.task}  Algo: {args.algo}")
    print(f"  Obs: {env.observation_size}  Act: {env.action_size}")
    print(f"  Episodes: {args.episodes}  Steps: {args.episode_length}")
    print()

    # ── Reconstruct make_policy (architecture only) ──
    make_policy = reconstruct_make_policy(args.algo, env, tuple(args.hidden_sizes))

    all_trajs = []
    labels = []

    for model_dir in model_dirs:
        label = os.path.basename(model_dir)

        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            from brax.training.agents.ppo import checkpoint as ppo_checkpoint

            params = ppo_checkpoint.load(args.checkpoint)
        else:
            params_path = os.path.join(model_dir, f"{args.algo}_params.pkl")
            if not os.path.isfile(params_path):
                print(f"  WARNING: {params_path} not found, skipping")
                continue
            print(f"Loading {label}...")
            with open(params_path, "rb") as f:
                params = pickle.load(f)

        print(f"  Running {args.episodes} episode(s)...")
        trajs = rollout(
            env,
            make_policy,
            params,
            n_episodes=args.episodes,
            episode_length=args.episode_length,
            seed=args.seed,
        )

        # Print per-episode summary
        for i, traj in enumerate(trajs):
            total_reward = traj["reward"].sum()
            final_err = traj["pos_error"][-1]
            success_rate = traj["success"].mean()
            print(f"  Ep {i}: reward={total_reward:.2f}  final_err={final_err:.4f}  success={success_rate:.1%}")

        all_trajs.append(trajs[0])
        labels.append(label)

        # ── Per-model outputs ──
        out_dir = args.output_dir or os.path.join(model_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)

        if args.video:
            for i, traj in enumerate(trajs):
                render_video(env, traj, os.path.join(out_dir, f"eval_ep{i}.mp4"), args.fps, tuple(args.resolution))

        if args.plot_ee:
            for i, traj in enumerate(trajs):
                plot_ee_trajectory(traj, os.path.join(out_dir, f"ee_traj_ep{i}.png"))

        if args.plot_actions:
            for i, traj in enumerate(trajs):
                plot_actions(traj, os.path.join(out_dir, f"actions_ep{i}.png"), env.robot.joint_names)

        if args.plot_debug:
            # Single-episode diagnostic (use ep 0 as representative)
            print_debug_stats(trajs[0], env.robot.joint_names)
            plot_debug(trajs[0], os.path.join(out_dir, "debug_ep0.png"), env)

            # Multi-episode distribution (only meaningful with > 1 episode)
            if len(trajs) > 1:
                plot_multi_episode_stats(trajs, os.path.join(out_dir, "multi_episode_stats.png"))

    # ── Cross-model comparison plot ──
    if args.plot and all_trajs:
        out_dir = args.output_dir or os.path.join(model_dirs[0], "eval")
        os.makedirs(out_dir, exist_ok=True)
        plot_metrics(all_trajs, os.path.join(out_dir, "metrics.png"), labels)

    print("\nDone.")


if __name__ == "__main__":
    main()
