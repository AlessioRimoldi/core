"""Scene configuration for RL training environments.

Defines scene objects (tables, cubes, obstacles, targets) that are injected
into the MuJoCo simulation as URDF links/joints.  Tasks can query object
poses, randomize positions on reset, and detect contacts.

The YAML format is backward-compatible with the existing ROS2 ``scene.yaml``
used by ``parol6.launch.py``.  Optional RL-specific fields (``role``,
``randomize_position``, ``randomize_orientation``, ``friction``) are added
for training use-cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class SceneObject:
    """A single object in the scene."""

    name: str
    type: str  # box | sphere | cylinder
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    dynamic: bool = False
    mass: float | None = None

    # Shape-specific dimensions
    size: list[float] | None = None  # box half-extents [sx, sy, sz]
    radius: float | None = None  # sphere / cylinder
    length: float | None = None  # cylinder

    # RL extensions (all optional, backward-compatible)
    role: str | None = None  # target | obstacle | tool
    randomize_position: list[float] | None = None  # ±uniform range [dx, dy, dz]
    randomize_orientation: list[float] | None = None  # ±uniform range [dr, dp, dy]
    friction: list[float] | None = None  # MuJoCo [sliding, torsional, rolling]

    @property
    def effective_mass(self) -> float:
        if self.mass is not None:
            return self.mass
        return 0.1 if self.dynamic else 0.01


@dataclass
class SceneConfig:
    """Collection of scene objects with lookup helpers."""

    objects: list[SceneObject] = field(default_factory=list)

    def get_by_name(self, name: str) -> SceneObject | None:
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def get_by_role(self, role: str) -> list[SceneObject]:
        return [obj for obj in self.objects if obj.role == role]

    @property
    def dynamic_objects(self) -> list[SceneObject]:
        return [obj for obj in self.objects if obj.dynamic]

    @property
    def fixed_objects(self) -> list[SceneObject]:
        return [obj for obj in self.objects if not obj.dynamic]


def load_scene(path: str) -> SceneConfig:
    """Load a scene configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "objects" not in raw:
        return SceneConfig()

    objects = []
    for obj_dict in raw["objects"]:
        objects.append(
            SceneObject(
                name=obj_dict["name"],
                type=obj_dict["type"],
                position=obj_dict.get("position", [0.0, 0.0, 0.0]),
                orientation=obj_dict.get("orientation", [0.0, 0.0, 0.0]),
                color=obj_dict.get("color", [0.5, 0.5, 0.5, 1.0]),
                dynamic=obj_dict.get("dynamic", False),
                mass=obj_dict.get("mass"),
                size=obj_dict.get("size"),
                radius=obj_dict.get("radius"),
                length=obj_dict.get("length"),
                role=obj_dict.get("role"),
                randomize_position=obj_dict.get("randomize_position"),
                randomize_orientation=obj_dict.get("randomize_orientation"),
                friction=obj_dict.get("friction"),
            )
        )

    return SceneConfig(objects=objects)


def inject_scene_urdf(urdf_text: str, scene: SceneConfig) -> str:
    """Inject scene objects into URDF as links/joints for MuJoCo.

    Each object becomes a ``<link>`` + ``<joint>`` pair.  Dynamic objects
    get ``type="floating"`` (6-DOF, affected by gravity); static objects
    get ``type="fixed"``.

    The ``<mujoco><compiler fusestatic="false"/>`` directive is injected to
    prevent MuJoCo from merging fixed-joint bodies into their parent (which
    would make them invisible to ``mj_name2id``).
    """
    if not scene.objects:
        return urdf_text

    urdf_text = urdf_text.rstrip()
    if urdf_text.endswith("</robot>"):
        urdf_text = urdf_text[: -len("</robot>")]

    # Ensure fusestatic=false (may already be present from _inject_mujoco_compiler)
    if '<compiler fusestatic="false"/>' not in urdf_text:
        urdf_text += '\n  <mujoco><compiler fusestatic="false"/></mujoco>\n'

    for obj in scene.objects:
        geom = _build_geometry(obj)
        if geom is None:
            continue

        mass = obj.effective_mass
        ixx, iyy, izz = _compute_inertia(obj, mass)

        inertial = (
            f'<inertial><mass value="{mass}"/>'
            f'<inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>'
            f"</inertial>"
        )
        rgba = obj.color
        pos = obj.position
        rpy = obj.orientation
        joint_type = "floating" if obj.dynamic else "fixed"

        urdf_text += f"""
  <link name="{obj.name}">
    {inertial}
    <visual>
      {geom}
      <material name="{obj.name}_mat">
        <color rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
      </material>
    </visual>
    <collision>
      {geom}
    </collision>
  </link>
  <joint name="{obj.name}_joint" type="{joint_type}">
    <parent link="world"/>
    <child link="{obj.name}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
  </joint>
"""

    urdf_text += "</robot>\n"
    return urdf_text


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_geometry(obj: SceneObject) -> str | None:
    """Build a URDF ``<geometry>`` element string."""
    if obj.type == "box" and obj.size is not None:
        sx, sy, sz = obj.size
        return f'<geometry><box size="{sx * 2} {sy * 2} {sz * 2}"/></geometry>'
    elif obj.type == "sphere" and obj.radius is not None:
        return f'<geometry><sphere radius="{obj.radius}"/></geometry>'
    elif obj.type == "cylinder" and obj.radius is not None and obj.length is not None:
        return f'<geometry><cylinder radius="{obj.radius}" length="{obj.length}"/></geometry>'
    return None


def _compute_inertia(obj: SceneObject, mass: float) -> tuple[float, float, float]:
    """Compute diagonal inertia from shape and mass."""
    if obj.type == "box" and obj.size is not None:
        sx, sy, sz = (s * 2 for s in obj.size)  # full extents
        ixx = mass / 12.0 * (sy**2 + sz**2)
        iyy = mass / 12.0 * (sx**2 + sz**2)
        izz = mass / 12.0 * (sx**2 + sy**2)
    elif obj.type == "sphere" and obj.radius is not None:
        r = obj.radius
        ixx = iyy = izz = 2.0 / 5.0 * mass * r**2
    elif obj.type == "cylinder" and obj.radius is not None and obj.length is not None:
        r, h = obj.radius, obj.length
        ixx = iyy = mass / 12.0 * (3 * r**2 + h**2)
        izz = mass / 2.0 * r**2
    else:
        ixx = iyy = izz = mass * 0.001
    return ixx, iyy, izz
