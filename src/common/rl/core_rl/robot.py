"""Robot configuration resolution.

Given a robot name (e.g. "parol6"), resolves the URDF, gains, joint info, and
produces a MuJoCo-loadable XML string.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import yaml

if TYPE_CHECKING:
    from core_rl.scene import SceneConfig


@dataclass
class JointLimits:
    lower: float
    upper: float
    velocity: float
    effort: float


@dataclass
class JointGains:
    kp: float = 0.0
    kd: float = 0.0
    kp_min: float = 0.0
    kp_max: float = 500.0
    kd_min: float = 0.0
    kd_max: float = 50.0


@dataclass
class RobotConfig:
    """Resolved robot configuration ready for RL training."""

    name: str
    urdf_path: str
    mesh_dir: str
    joint_names: list[str]
    joint_limits: dict[str, JointLimits] = field(default_factory=dict)
    gains: dict[str, JointGains] = field(default_factory=dict)
    num_joints: int = 0
    # Paths written during resolution
    mjcf_path: str = ""
    # Optional end-effector body name (for tasks that need EE position)
    ee_body: str | None = None

    def __post_init__(self):
        self.num_joints = len(self.joint_names)


# ---------------------------------------------------------------------------
# Registry: each robot provides an rl_config.yaml in its description package.
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = "/ros2_ws/core/src"
_THIRDPARTY_ROOT = "/ros2_ws/core/thirdparty"


def _find_rl_config(robot_name: str) -> str:
    """Locate the rl_config.yaml for a robot."""
    # Convention: src/<robot>/<robot>_description/config/rl_config.yaml
    candidates = [
        os.path.join(_WORKSPACE_ROOT, robot_name, f"{robot_name}_description", "config", "rl_config.yaml"),
        os.path.join(_WORKSPACE_ROOT, robot_name, "config", "rl_config.yaml"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"No rl_config.yaml found for robot '{robot_name}'. " f"Searched: {candidates}")


# ---------------------------------------------------------------------------
# URDF patching (ported from parol6.launch.py — robot-agnostic)
# ---------------------------------------------------------------------------


def _strip_legacy_blocks(urdf_text: str) -> str:
    """Remove ROS1-era <transmission> and <gazebo> blocks."""
    urdf_text = re.sub(r"<transmission[\s\S]*?</transmission\s*>", "", urdf_text)
    urdf_text = re.sub(r"<gazebo[\s\S]*?</gazebo\s*>", "", urdf_text)
    return urdf_text


def _strip_package_uris(urdf_text: str, patterns: list[str]) -> str:
    """Replace package:// mesh URIs with bare filenames for MuJoCo."""
    for pat in patterns:
        urdf_text = urdf_text.replace(pat, "")
    return urdf_text


def _inject_mujoco_compiler(urdf_text: str) -> str:
    """Ensure fusestatic=false so scene bodies are individually addressable."""
    if "<mujoco>" not in urdf_text:
        closing = "</robot>"
        urdf_text = urdf_text.replace(
            closing,
            '  <mujoco><compiler fusestatic="false"/></mujoco>\n' + closing,
        )
    return urdf_text


def _parse_urdf_joints(urdf_text: str, joint_names: list[str]) -> dict[str, JointLimits]:
    """Extract joint limits from URDF XML for the specified joints."""
    root = ElementTree.fromstring(urdf_text)
    limits: dict[str, JointLimits] = {}

    for joint_el in root.findall("joint"):
        name = joint_el.get("name", "")
        if name not in joint_names:
            continue
        limit_el = joint_el.find("limit")
        if limit_el is not None:
            limits[name] = JointLimits(
                lower=float(limit_el.get("lower", "-3.14159")),
                upper=float(limit_el.get("upper", "3.14159")),
                velocity=float(limit_el.get("velocity", "3.14159")),
                effort=float(limit_el.get("effort", "100.0")),
            )
        else:
            limits[name] = JointLimits(
                lower=-math.pi,
                upper=math.pi,
                velocity=math.pi,
                effort=100.0,
            )

    return limits


def _load_gains(gains_path: str, joint_names: list[str], section: str = "sim") -> dict[str, JointGains]:
    """Load PD gains from a YAML file."""
    with open(gains_path) as f:
        gains_yaml = yaml.safe_load(f)
    gains: dict[str, JointGains] = {}

    if section not in gains_yaml:
        return {name: JointGains() for name in joint_names}

    for joint_name in joint_names:
        if joint_name in gains_yaml[section]:
            jg = gains_yaml[section][joint_name]
            gains[joint_name] = JointGains(
                kp=float(jg.get("kp", 0.0)),
                kd=float(jg.get("kd", 0.0)),
                kp_min=float(jg.get("kp_min", 0.0)),
                kp_max=float(jg.get("kp_max", 500.0)),
                kd_min=float(jg.get("kd_min", 0.0)),
                kd_max=float(jg.get("kd_max", 50.0)),
            )
        else:
            gains[joint_name] = JointGains()

    return gains


def resolve_robot(robot_name: str, scene: SceneConfig | None = None) -> RobotConfig:
    """Resolve a robot name into a fully-configured RobotConfig.

    Reads the robot's ``rl_config.yaml`` to find the URDF, mesh directory,
    joint names, gains, and produces a MuJoCo-loadable MJCF file.
    """
    from core_rl.scene import inject_scene_urdf

    rl_config_path = _find_rl_config(robot_name)
    with open(rl_config_path) as f:
        rl_config = yaml.safe_load(f)

    # Resolve paths relative to rl_config.yaml location
    config_dir = os.path.dirname(rl_config_path)

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(config_dir, p))

    urdf_path = _resolve(rl_config["urdf_path"])
    mesh_dir = _resolve(rl_config["mesh_dir"])
    gains_path = _resolve(rl_config["gains_path"])
    joint_names = rl_config["joint_names"]
    uri_patterns = rl_config.get("uri_strip_patterns", [])

    # Read and patch URDF
    with open(urdf_path) as f:
        urdf_text = f.read()

    urdf_text = _strip_legacy_blocks(urdf_text)
    urdf_text = _strip_package_uris(urdf_text, uri_patterns)
    urdf_text = _inject_mujoco_compiler(urdf_text)

    if scene is not None:
        urdf_text = inject_scene_urdf(urdf_text, scene)

    # Write MuJoCo-loadable URDF to mesh directory
    mjcf_path = os.path.join(mesh_dir, f"_{robot_name}_rl.urdf")
    with open(mjcf_path, "w") as f:
        f.write(urdf_text)

    # Parse joint limits from URDF
    joint_limits = _parse_urdf_joints(urdf_text, joint_names)

    # Load gains
    gains = _load_gains(gains_path, joint_names)

    ee_body = rl_config.get("ee_body")

    return RobotConfig(
        name=robot_name,
        urdf_path=urdf_path,
        mesh_dir=mesh_dir,
        joint_names=joint_names,
        joint_limits=joint_limits,
        gains=gains,
        mjcf_path=mjcf_path,
        ee_body=ee_body,
    )
