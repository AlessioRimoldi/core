"""Task base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from core_rl.robot import RobotConfig

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Class decorator that registers a task by name."""

    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_task(name: str, **kwargs) -> BaseTask:
    """Instantiate a registered task by name."""
    if name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return _TASK_REGISTRY[name](**kwargs)


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Task spaces — returned by configure() to define obs/action spaces
# ---------------------------------------------------------------------------


@dataclass
class TaskSpaces:
    observation_space: gym.Space
    action_space: gym.Space
    # Names of observation components for documentation / export
    obs_components: list[str]
    action_type: str  # "position", "velocity", "torque"


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------


class BaseTask(ABC):
    """Abstract base for RL tasks.

    A task defines:
    - How observation and action spaces are constructed for a given robot
    - How to reset the simulation to a new episode
    - How to compute observations from MuJoCo state
    - How to compute reward and termination
    """

    @abstractmethod
    def configure(self, robot: RobotConfig, model: mujoco.MjModel) -> TaskSpaces:
        """Configure spaces based on the robot.  Called once at env init."""

    @abstractmethod
    def reset(
        self,
        robot: RobotConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        np_random: np.random.Generator,
    ) -> np.ndarray:
        """Reset the simulation state for a new episode.

        Returns the initial observation.
        """

    @abstractmethod
    def compute_observation(
        self,
        robot: RobotConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """Compute the current observation vector."""

    @abstractmethod
    def compute_reward(
        self,
        robot: RobotConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Compute reward, terminated flag, and info dict."""


# Import concrete tasks to trigger registration
from core_rl.tasks import joint_tracking as _  # noqa: F401, E402
