"""Training CLI entry point.

Usage:
    python -m core_rl.train --robot parol6 --task joint_tracking --algo ppo --num-envs 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from core_rl.algorithms import get_algorithm
from core_rl.callbacks.grav_comp_collector import GravCompCollectorCallback
from core_rl.callbacks.mlflow_logger import MLflowCallback
from core_rl.callbacks.redis_stream import RedisStreamCallback
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
        defaults_path = os.path.join(
            get_package_share_directory("core_rl"), "config", "defaults.yaml"
        )
    except Exception:
        # Fallback: relative to source tree
        defaults_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "defaults.yaml"
        )

    with open(defaults_path, "r") as f:
        config = yaml.safe_load(f)

    if config_path:
        with open(config_path, "r") as f:
            override = yaml.safe_load(f)
        config = _deep_merge(config, override)

    return config


def main():
    parser = argparse.ArgumentParser(description="Robot-agnostic RL training pipeline")
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

    print(f"=== RL Training Pipeline ===")
    print(f"  Robot:      {args.robot}")
    print(f"  Task:       {args.task}")
    print(f"  Algorithm:  {args.algo}")
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

    # ── 2. Create vectorized environment ──
    print(f"Creating {num_envs} parallel environments...")
    env_fns = [
        make_env(
            robot=robot,
            task_name=args.task,
            seed=seed + i,
            control_dt=env_cfg["control_dt"],
            physics_dt=env_cfg["physics_dt"],
            max_episode_steps=env_cfg["max_episode_steps"],
            scene_file=args.scene_file,
        )
        for i in range(num_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    print(f"  Observation space: {vec_env.observation_space}")
    print(f"  Action space:      {vec_env.action_space}")
    print()

    # ── 3. Set up callbacks ──
    callbacks = []

    if not args.no_redis and cb_cfg["redis"]["enabled"]:
        redis_cfg = cb_cfg["redis"]
        callbacks.append(RedisStreamCallback(
            host=redis_cfg["host"],
            port=redis_cfg["port"],
            password=redis_cfg.get("password", ""),
            experiment=args.experiment,
            run_id=run_id,
            publish_freq=redis_cfg["publish_freq"],
            verbose=1,
        ))
        print("  Redis streaming: enabled")

    mlflow_cb = None
    if not args.no_mlflow and cb_cfg["mlflow"]["enabled"]:
        mlflow_cfg = cb_cfg["mlflow"]
        mlflow_cb = MLflowCallback(
            tracking_uri=mlflow_cfg["tracking_uri"],
            experiment_name=args.experiment,
            run_name=run_id,
            publish_freq=mlflow_cfg["publish_freq"],
            log_params={
                "robot": args.robot,
                "task": args.task,
                "algo": args.algo,
                "num_envs": num_envs,
                "total_timesteps": total_timesteps,
                "seed": seed,
                "control_dt": env_cfg["control_dt"],
                "physics_dt": env_cfg["physics_dt"],
                "max_episode_steps": env_cfg["max_episode_steps"],
            },
            verbose=1,
        )
        callbacks.append(mlflow_cb)
        print("  MLflow logging:   enabled")

    # ── 4. Collect gravity comp data from first env ──
    # We enable grav_comp collection on the raw envs by signaling via
    # a wrapper. For SubprocVecEnv, we'll collect from rollout data instead.
    grav_comp_enabled = (
        export_cfg.get("onnx", False)
        and export_cfg.get("gravity_comp", {}).get("enabled", False)
        and not args.no_export
    )

    if grav_comp_enabled:
        gc_cfg = export_cfg["gravity_comp"]
        gc_callback = GravCompCollectorCallback(
            robot=robot,
            buffer_size=gc_cfg.get("buffer_size", 500_000),
        )
        callbacks.append(gc_callback)
        print("  GravComp data:    collecting")

    print()

    # ── 5. Create algorithm ──
    algo_name = args.algo
    algo_cfg = config["algorithms"].get(algo_name, {}).copy()

    # Inject seed and device
    algo_cfg["seed"] = seed
    algo_cfg["device"] = training_cfg.get("device", "auto")

    print(f"Initializing {algo_name.upper()}...")
    algorithm = get_algorithm(algo_name, env=vec_env, config=algo_cfg, callbacks=callbacks)

    # ── 6. Train ──
    print(f"\nTraining for {total_timesteps} timesteps...\n")
    t0 = time.time()
    algorithm.train(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({total_timesteps / elapsed:.0f} steps/s)")

    # ── 7. Save model ──
    model_path = os.path.join(output_dir, f"{algo_name}_model")
    algorithm.save(model_path)
    print(f"Model saved to {model_path}")

    # Save VecNormalize stats
    vec_norm_path = os.path.join(output_dir, "vec_normalize.pkl")
    vec_env.save(vec_norm_path)

    # ── 8. ONNX export ──
    if export_cfg.get("onnx", False) and not args.no_export:
        from core_rl.export_onnx import export_onnx

        print("\nExporting ONNX model...")

        grav_comp_data = None
        if grav_comp_enabled:
            grav_comp_data = gc_callback.get_data()

        onnx_path = export_onnx(
            algorithm=algorithm,
            robot=robot,
            vec_normalize=vec_env,
            output_dir=output_dir,
            grav_comp_config=export_cfg.get("gravity_comp", {}),
            grav_comp_data=grav_comp_data,
        )
        print(f"ONNX exported to {onnx_path}")

        if mlflow_cb:
            mlflow_cb.log_artifact(onnx_path)
            print("ONNX artifact logged to MLflow")

    vec_env.close()
    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
