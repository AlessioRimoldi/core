"""Training CLI entry point — Brax / MJX.

Usage:
    python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --num-envs 4096

The entire rollout + PPO/SAC update is JIT-compiled by JAX and runs on GPU.
Brax handles vectorisation via ``jax.vmap`` — no ``SubprocVecEnv``.
"""

from __future__ import annotations

import argparse
import os
import time

import yaml

from core_rl.algorithms import get_algorithm
from core_rl.callbacks import compose_progress_fn
from core_rl.callbacks.mlflow_logger import MLflowHook
from core_rl.callbacks.redis_stream import RedisStreamHook
from core_rl.env import make_env
from core_rl.robot import resolve_robot


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


def main():
    parser = argparse.ArgumentParser(description="Robot-agnostic RL training pipeline (Brax/MJX)")
    parser.add_argument("--robot", type=str, required=True, help="Robot name (e.g. parol6)")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. joint_tracking)")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name (e.g. ppo, sac)")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config override YAML")
    parser.add_argument("--scene-file", type=str, default="", help="Scene YAML file for objects")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for models")
    parser.add_argument("--experiment", type=str, default="rl_training", help="MLflow experiment name")
    parser.add_argument("--no-redis", action="store_true", help="Disable Redis streaming")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--num-evals", type=int, default=20, help="Number of eval points during training")
    parser.add_argument("--backend", type=str, default="mjx", help="Simulation backend (mjx)")
    args = parser.parse_args()

    # Load config
    config = _load_config(args.config)
    training_cfg = config["training"]
    env_cfg = config["env"]
    export_cfg = config["export"]
    cb_cfg = config["callbacks"]

    # CLI overrides
    num_envs = args.num_envs or training_cfg["num_envs"]
    total_timesteps = args.total_timesteps or training_cfg["total_timesteps"]
    seed = args.seed if args.seed is not None else training_cfg["seed"]

    run_id = f"{args.robot}_{args.task}_{args.algo}_{int(time.time())}"
    output_dir = args.output_dir or os.path.join("/ros2_ws/core/models", run_id)
    os.makedirs(output_dir, exist_ok=True)

    print("=== RL Training Pipeline (Brax/MJX) ===")
    print(f"  Robot:      {args.robot}")
    print(f"  Task:       {args.task}")
    print(f"  Algorithm:  {args.algo}")
    print(f"  Backend:    {args.backend}")
    print(f"  Envs:       {num_envs}")
    print(f"  Timesteps:  {total_timesteps}")
    print(f"  Seed:       {seed}")
    print(f"  Output:     {output_dir}")
    print()

    # ── 1. Resolve robot ──
    print(f"Resolving robot '{args.robot}'...")
    robot = resolve_robot(args.robot, scene_file=args.scene_file)
    print(f"  Joints:     {robot.joint_names}")
    print(f"  MJCF:       {robot.mjcf_path}")
    print()

    # ── 2. Create environment (Brax PipelineEnv) ──
    print(f"Creating Brax environment ({args.backend} backend, {num_envs} vectorised envs)...")
    env = make_env(
        robot=robot,
        task_name=args.task,
        backend=args.backend,
        control_dt=env_cfg["control_dt"],
        physics_dt=env_cfg["physics_dt"],
        max_episode_steps=env_cfg["max_episode_steps"],
        scene_file=args.scene_file,
    )
    print(f"  Observation size: {env.observation_size}")
    print(f"  Action size:      {env.action_size}")
    print()

    # ── 3. Set up progress hooks ──
    hooks = []
    mlflow_hook = None
    redis_hook = None

    if not args.no_redis and cb_cfg["redis"]["enabled"]:
        redis_cfg = cb_cfg["redis"]
        redis_hook = RedisStreamHook(
            host=redis_cfg["host"],
            port=redis_cfg["port"],
            password=redis_cfg.get("password", ""),
        )
        redis_hook.start(
            experiment=args.experiment,
            run_id=run_id,
            meta={"num_envs": num_envs, "algo": args.algo, "robot": args.robot},
        )
        hooks.append(redis_hook)
        print("  Redis streaming:  enabled")

    if not args.no_mlflow and cb_cfg["mlflow"]["enabled"]:
        mlflow_cfg = cb_cfg["mlflow"]
        mlflow_hook = MLflowHook(
            tracking_uri=mlflow_cfg["tracking_uri"],
            experiment_name=args.experiment,
        )
        mlflow_hook.start(
            run_name=run_id,
            params={
                "robot": args.robot,
                "task": args.task,
                "algo": args.algo,
                "num_envs": num_envs,
                "total_timesteps": total_timesteps,
                "seed": seed,
                "backend": args.backend,
                "control_dt": env_cfg["control_dt"],
                "physics_dt": env_cfg["physics_dt"],
                "max_episode_steps": env_cfg["max_episode_steps"],
            },
        )
        hooks.append(mlflow_hook)
        print("  MLflow logging:   enabled")

    # Add a simple print hook for console progress
    def _print_progress(step: int, metrics: dict) -> None:
        reward = metrics.get("eval/episode_reward", metrics.get("eval/episode_reward_mean", "?"))
        sps = metrics.get("training/sps", "?")
        print(f"  Step {step:>10d} | reward={reward} | SPS={sps}")

    hooks.append(_print_progress)
    progress_fn = compose_progress_fn(*hooks)

    print()

    # ── 4. Build algorithm config ──
    algo_name = args.algo
    algo_cfg = config["algorithms"].get(algo_name, {}).copy()

    # Inject training-level params into algo config
    algo_cfg["seed"] = seed
    algo_cfg["num_envs"] = num_envs
    algo_cfg["total_timesteps"] = total_timesteps
    algo_cfg["num_evals"] = args.num_evals
    algo_cfg["max_episode_steps"] = env_cfg["max_episode_steps"]

    print(f"Initializing {algo_name.upper()}...")
    algorithm = get_algorithm(
        algo_name,
        env=env,
        config=algo_cfg,
        progress_fn=progress_fn,
    )

    # ── 5. Train ──
    print(f"\nTraining for {total_timesteps} timesteps ({num_envs} parallel envs)...\n")
    t0 = time.time()
    make_policy, params, metrics = algorithm.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({total_timesteps / elapsed:.0f} steps/s)")

    # ── 6. Save params ──
    params_path = os.path.join(output_dir, f"{algo_name}_params.pkl")
    algorithm.save(params_path, params)
    print(f"Params saved to {params_path}")

    # ── 7. Collect gravity comp data & ONNX export ──
    grav_comp_enabled = (
        export_cfg.get("onnx", False)
        and export_cfg.get("gravity_comp", {}).get("enabled", False)
        and not args.no_export
    )

    grav_comp_data = None
    if grav_comp_enabled:
        from core_rl.callbacks.grav_comp_collector import collect_grav_comp_data

        gc_cfg = export_cfg["gravity_comp"]
        print(f"\nCollecting gravity comp data ({gc_cfg.get('buffer_size', 10_000)} steps)...")
        grav_comp_data = collect_grav_comp_data(
            env=env,
            make_policy_fn=make_policy,
            params=params,
            num_steps=gc_cfg.get("buffer_size", 10_000),
            seed=seed + 1000,
        )
        print(f"  Collected {len(grav_comp_data[0])} samples")

    if export_cfg.get("onnx", False) and not args.no_export:
        from core_rl.export_onnx import export_onnx

        print("\nExporting ONNX model...")
        onnx_path = export_onnx(
            make_policy_fn=make_policy,
            params=params,
            robot=robot,
            output_dir=output_dir,
            grav_comp_config=export_cfg.get("gravity_comp", {}),
            grav_comp_data=grav_comp_data,
        )
        print(f"ONNX exported to {onnx_path}")

        if mlflow_hook:
            mlflow_hook.log_artifact(onnx_path)
            print("ONNX artifact logged to MLflow")

    # ── 8. Clean up hooks ──
    if mlflow_hook:
        artifacts = [params_path]
        mlflow_hook.end(artifact_paths=artifacts)

    if redis_hook:
        redis_hook.end(total_timesteps=total_timesteps)

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
