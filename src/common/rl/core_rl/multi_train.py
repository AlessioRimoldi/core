"""Run multiple training jobs sequentially from a single manifest YAML.

Designed for hyperparameter sweeps: write a manifest listing N
runs, kick it off, come back in the morning with N trained models, N
sets of metrics, and a summary report.

Each run is launched as a subprocess invoking ``python -m core_rl.train``,
so a crash in one run does not bring down the others — the runner records
the failure and moves on. Per-run stdout/stderr is tee'd to console and
to a ``run.log`` file in that run's output directory.

Usage::

    python -m core_rl.multi_train --manifest sweep.yaml

Manifest format::

    # Defaults applied to every run (overridable per-run).
    defaults:
      robot: parol6
      task: skill_conditioned
      algo: dads
      num_envs: 2048
      total_timesteps: 5_000_000
      experiment: dads_sweep
      base_config: dads_joint_tracking.yaml   # optional, deep-merged onto defaults.yaml
      record_video: true
      video_interval: 5
      video_envs: 16
      save_checkpoints: false
      scene_file: ""
      num_evals: 20
      backend: mjx
      seed: 42
      # Optional config_overrides applied to *every* run (then overridden per-run).
      config_overrides: {}

    # The runs. Each entry inherits from `defaults`; any key may be
    # overridden, plus a `config_overrides` block is deep-merged onto the
    # effective config YAML passed to train.py via --config.
    runs:
      - name: skill_size_2
        config_overrides:
          algorithms:
            dads:
              skill_size: 2

      - name: skill_size_4
        config_overrides:
          algorithms:
            dads:
              skill_size: 4

      - name: low_lr
        seed: 7
        config_overrides:
          algorithms:
            dads:
              learning_rate: 1.0e-4
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (matches train.py:_deep_merge)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_defaults_yaml() -> str:
    """Same logic as train.py:_load_config — defaults.yaml location."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("core_rl"), "config", "defaults.yaml")
    except Exception:
        return os.path.join(os.path.dirname(__file__), "..", "config", "defaults.yaml")


def _build_effective_config(run_cfg: dict, defaults_block: dict) -> dict:
    """Return the merged config dict to write to a temp YAML for --config.

    Order: defaults.yaml → defaults_block.base_config → defaults_block.config_overrides
           → run_cfg.config_overrides.

    `defaults_block` is the manifest's `defaults` section (NOT defaults.yaml).
    """
    base = _load_yaml(_resolve_defaults_yaml())

    base_cfg_path = run_cfg.get("base_config", defaults_block.get("base_config"))
    if base_cfg_path:
        base = _deep_merge(base, _load_yaml(base_cfg_path))

    base = _deep_merge(base, defaults_block.get("config_overrides", {}) or {})
    base = _deep_merge(base, run_cfg.get("config_overrides", {}) or {})
    return base


def _resolved_run_settings(run_cfg: dict, defaults_block: dict) -> dict:
    """Merge top-level (non-config) keys: robot/task/algo/num_envs/..."""
    merged = copy.deepcopy(defaults_block)
    for k, v in run_cfg.items():
        if k in ("name", "config_overrides", "base_config"):
            continue
        merged[k] = v
    return merged


def _build_train_args(settings: dict, run_name: str, output_dir: str, config_path: str) -> list[str]:
    """Construct the argv list for `python -m core_rl.train`."""
    argv: list[str] = [
        sys.executable,
        "-m",
        "core_rl.train",
        "--robot",
        str(settings["robot"]),
        "--task",
        str(settings["task"]),
        "--algo",
        str(settings["algo"]),
        "--config",
        config_path,
        "--output-dir",
        output_dir,
        "--experiment",
        str(settings.get("experiment", "multi_train")),
    ]

    # Optional scalar flags — pass only when present
    def _opt(key: str, flag: str, cast=str):
        if settings.get(key) is not None and settings.get(key) != "":
            argv.extend([flag, str(cast(settings[key]))])

    _opt("num_envs", "--num-envs", int)
    _opt("total_timesteps", "--total-timesteps", int)
    _opt("seed", "--seed", int)
    _opt("num_evals", "--num-evals", int)
    _opt("backend", "--backend")
    _opt("scene_file", "--scene-file")
    _opt("video_interval", "--video-interval", int)
    _opt("video_envs", "--video-envs", int)

    # Boolean flags
    if settings.get("no_redis"):
        argv.append("--no-redis")
    if settings.get("no_mlflow"):
        argv.append("--no-mlflow")
    if settings.get("no_export"):
        argv.append("--no-export")
    if settings.get("save_checkpoints"):
        argv.append("--save-checkpoints")
    if settings.get("record_video"):
        argv.append("--record-video")

    return argv


def _resolve_mlflow_uri(defaults_block: dict) -> str | None:
    """Resolve the MLflow tracking URI used to host the sweep's parent run.

    Reads defaults.yaml as the base, then lets the manifest override via
    ``defaults.config_overrides.callbacks.mlflow.{enabled, tracking_uri}``.
    Returns None if MLflow is disabled in the merged config.
    """
    cfg = _load_yaml(_resolve_defaults_yaml())
    cfg = _deep_merge(cfg, defaults_block.get("config_overrides", {}) or {})
    mlflow_cfg = (cfg.get("callbacks") or {}).get("mlflow") or {}
    if not mlflow_cfg.get("enabled", True):
        return None
    return str(mlflow_cfg.get("tracking_uri", "http://mlflow:5000"))


def _maybe_start_mlflow_parent(defaults_block: dict, runs: list[dict], output_root: str) -> str | None:
    """Create one MLflow run that all children nest under.

    Skipped if MLflow is disabled at the manifest level, every run has
    ``no_mlflow: true``, or the import/connection fails. In any of those
    cases we just return None and the children log flat as before.
    """
    if defaults_block.get("no_mlflow"):
        return None
    if all(r.get("no_mlflow", defaults_block.get("no_mlflow")) for r in runs):
        return None

    tracking_uri = _resolve_mlflow_uri(defaults_block)
    if tracking_uri is None:
        return None
    experiment = defaults_block.get("experiment", "rl_training")

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        parent_name = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
        active = mlflow.start_run(run_name=parent_name)
        mlflow.set_tag("sweep_parent", "true")
        mlflow.set_tag("sweep_output_root", output_root)
        mlflow.set_tag("sweep_run_count", str(len(runs)))
        mlflow.set_tag("sweep_run_names", ", ".join(r.get("name", "?") for r in runs)[:500])
        manifest_path = os.path.join(output_root, "manifest.yaml")
        if os.path.exists(manifest_path):
            mlflow.log_artifact(manifest_path)
        # End the parent run *but keep its id* — children will reference it
        # via the MLFLOW_PARENT_RUN_ID tag. (MLflow nested runs do not need
        # the parent to remain "active" across processes — only its id.)
        parent_run_id = active.info.run_id
        mlflow.end_run()
        print(f"  [multi_train] MLflow parent run: {parent_name} ({parent_run_id})")
        return parent_run_id
    except Exception as e:  # noqa: BLE001
        print(f"  [multi_train] MLflow parent run setup failed ({e}); " "children will log flat.", file=sys.stderr)
        return None


def _maybe_end_mlflow_parent(parent_run_id: str | None, results: list[dict], summary_path: str) -> None:
    """Re-open the parent run to attach final status tags + summary artifact."""
    if not parent_run_id:
        return
    try:
        import mlflow

        n_ok = sum(1 for r in results if str(r.get("status", "")).startswith("ok"))
        n_fail = len(results) - n_ok
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.set_tag("sweep_status", "ok" if n_fail == 0 else f"{n_fail} failed")
            mlflow.set_tag("sweep_runs_succeeded", str(n_ok))
            mlflow.set_tag("sweep_runs_failed", str(n_fail))
            if os.path.exists(summary_path):
                mlflow.log_artifact(summary_path)
    except Exception as e:  # noqa: BLE001
        print(f"  [multi_train] MLflow parent finalize failed: {e}", file=sys.stderr)


def _run_one(
    run_name: str,
    settings: dict,
    effective_cfg: dict,
    output_root: str,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """Launch a single training subprocess, tee output to a log file."""
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg_path = os.path.join(run_dir, "_merged_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(effective_cfg, f, sort_keys=False)

    argv = _build_train_args(settings, run_name, run_dir, cfg_path)
    log_path = os.path.join(run_dir, "run.log")

    print("\n" + "=" * 70)
    print(f"  RUN: {run_name}")
    print(f"  out: {run_dir}")
    print(f"  cmd: {' '.join(argv)}")
    print("=" * 70 + "\n", flush=True)

    if dry_run:
        return {"name": run_name, "status": "dry_run", "output_dir": run_dir, "elapsed_s": 0.0}

    # Build child env: inherit parent + add MLflow grouping vars.
    child_env = os.environ.copy()
    if extra_env:
        child_env.update(extra_env)

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=child_env,
            )
            # Tee child stdout to both console and log file.
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            returncode = proc.wait()
    except KeyboardInterrupt:
        # Don't swallow Ctrl-C — let the user abort the whole sweep.
        with contextlib.suppress(Exception):
            proc.terminate()
        raise

    elapsed = time.time() - t0
    status = "ok" if returncode == 0 else f"failed (exit {returncode})"
    return {
        "name": run_name,
        "status": status,
        "returncode": returncode,
        "output_dir": run_dir,
        "log_path": log_path,
        "config_path": cfg_path,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser(description="Sequential multi-run training launcher (overnight sweeps).")
    p.add_argument("--manifest", type=str, required=True, help="Path to the sweep manifest YAML.")
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root dir for all run outputs. Default: /ros2_ws/core/models/sweeps/<timestamp>/",
    )
    p.add_argument(
        "--only", type=str, nargs="*", default=None, help="Run only the named entries (matches `runs[*].name`)."
    )
    p.add_argument("--skip", type=str, nargs="*", default=None, help="Skip the named entries.")
    p.add_argument(
        "--dry-run", action="store_true", help="Print what would be run + write merged configs; don't launch."
    )
    args = p.parse_args()

    manifest = _load_yaml(args.manifest)
    defaults_block = manifest.get("defaults", {}) or {}
    runs = manifest.get("runs", []) or []
    if not runs:
        print("Manifest has no `runs:` block — nothing to do.", file=sys.stderr)
        sys.exit(2)

    # Resolve output root
    output_root = args.output_root or os.path.join("/ros2_ws/core/models/sweeps", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)

    # Persist the original manifest next to the runs for reproducibility
    with open(os.path.join(output_root, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    # Filter runs
    if args.only:
        runs = [r for r in runs if r.get("name") in args.only]
    if args.skip:
        runs = [r for r in runs if r.get("name") not in args.skip]
    if not runs:
        print("No runs left after --only/--skip filtering.", file=sys.stderr)
        sys.exit(2)

    print(f"Multi-train: {len(runs)} run(s) → {output_root}\n")

    # ── Create an MLflow parent run so all sweep children collapse under
    # one block in the UI. The parent's name embeds the sweep timestamp so
    # successive sweeps in the same experiment don't pile up under the
    # same heading.
    parent_run_id = _maybe_start_mlflow_parent(defaults_block, runs, output_root)

    results: list[dict] = []
    summary_path = os.path.join(output_root, "summary.json")

    try:
        for i, run_cfg in enumerate(runs, start=1):
            run_name = run_cfg.get("name") or f"run_{i:02d}"
            print(f"\n[{i}/{len(runs)}] starting `{run_name}`")
            settings = _resolved_run_settings(run_cfg, defaults_block)
            effective_cfg = _build_effective_config(run_cfg, defaults_block)

            extra_env: dict[str, str] = {"MLFLOW_SWEEP_RUN_NAME": run_name}
            if parent_run_id:
                extra_env["MLFLOW_PARENT_RUN_ID"] = parent_run_id

            try:
                result = _run_one(
                    run_name=run_name,
                    settings=settings,
                    effective_cfg=effective_cfg,
                    output_root=output_root,
                    dry_run=args.dry_run,
                    extra_env=extra_env,
                )
            except KeyboardInterrupt:
                print("\nInterrupted by user — stopping sweep.")
                results.append({"name": run_name, "status": "interrupted"})
                break
            except Exception as e:  # noqa: BLE001
                # Subprocess crash shouldn't kill the sweep.
                print(f"  [multi_train] `{run_name}` crashed: {e}", file=sys.stderr)
                result = {"name": run_name, "status": f"error: {e}"}
            results.append(result)
            # Persist running summary so the user can peek mid-sweep.
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)
    finally:
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        _maybe_end_mlflow_parent(parent_run_id, results, summary_path)

    # Final summary
    print("\n" + "=" * 70)
    print(" SWEEP SUMMARY ".center(70, "="))
    print("=" * 70)
    for r in results:
        elapsed = r.get("elapsed_s", 0.0)
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s"
        print(f"  {r['name']:<32} {r['status']:<24} {elapsed_str:>10}")
    print("=" * 70)
    print(f"  Summary JSON: {summary_path}")

    n_failed = sum(1 for r in results if not str(r.get("status", "")).startswith("ok") and r.get("status") != "dry_run")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
