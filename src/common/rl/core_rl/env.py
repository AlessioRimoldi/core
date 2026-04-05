"""Environment factory for Brax/MJX training.

The old ``MujocoRobotEnv(gymnasium.Env)`` is replaced by the ``BaseTask``
(which extends ``brax.envs.PipelineEnv``).  This module provides a thin
factory function for constructing the task from config values.

Vectorisation is handled by Brax's ``VmapWrapper`` + ``AutoResetWrapper``
inside the training loop — there is no ``SubprocVecEnv``.
"""

from __future__ import annotations

from typing import Any

from core_rl.robot import RobotConfig, resolve_robot
from core_rl.tasks import BaseTask, get_task


def make_env(
    robot: RobotConfig | str,
    task_name: str,
    backend: str = "mjx",
    control_dt: float = 0.01,
    physics_dt: float = 0.001,
    max_episode_steps: int = 500,
    scene_file: str = "",
    task_kwargs: dict[str, Any] | None = None,
) -> BaseTask:
    """Create a Brax ``PipelineEnv`` (task) for the given robot.

    Args:
        robot: ``RobotConfig`` or robot name string.
        task_name: Registered task name (e.g. ``"joint_tracking"``).
        backend: Brax pipeline backend (``"mjx"``).
        control_dt: Control loop period (seconds).
        physics_dt: MuJoCo physics timestep (seconds).
        max_episode_steps: Max steps per episode.
        scene_file: Optional scene YAML path for objects.
        task_kwargs: Extra keyword arguments for the task constructor.

    Returns:
        A ``BaseTask`` instance (Brax ``PipelineEnv``).
    """
    if isinstance(robot, str):
        robot = resolve_robot(robot, scene_file=scene_file)

    n_frames = max(1, round(control_dt / physics_dt))

    kwargs: dict[str, Any] = {
        "robot": robot,
        "backend": backend,
        "n_frames": n_frames,
        "max_episode_steps": max_episode_steps,
    }
    if task_kwargs:
        kwargs.update(task_kwargs)

    return get_task(task_name, **kwargs)
