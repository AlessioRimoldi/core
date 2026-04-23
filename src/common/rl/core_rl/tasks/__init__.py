"""Task base class and registry.

Tasks extend ``brax.envs.PipelineEnv`` so that the entire reset / step /
reward computation is JAX-traceable and can be vectorised with ``jax.vmap``
across thousands of environments on GPU.

Each task defines:
    - ``reset()``       — pure-functional episode reset (JAX key → State)
    - ``step()``        — pure-functional env step (State × action → State)
    - ``observation_size``, ``action_size`` properties
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import brax.envs.base as brax_env
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import base as brax_base
from brax.io import mjcf as brax_mjcf

from core_rl.robot import RobotConfig
from core_rl.scene import SceneConfig

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Class decorator that registers a task by name."""

    def decorator(cls: type) -> type:
        _TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_task(name: str, robot: RobotConfig, **kwargs: Any) -> BaseTask:
    """Instantiate a registered task by name.

    Args:
        name: Registered task name (e.g. ``"joint_tracking"``).
        robot: Resolved robot configuration.
        **kwargs: Extra keyword arguments forwarded to the task constructor.
    """
    if name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return _TASK_REGISTRY[name](robot=robot, **kwargs)


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Mesh → primitive collision helper
# ---------------------------------------------------------------------------


def _replace_mesh_collisions_with_capsules(spec: mujoco.MjSpec) -> None:
    """Replace mesh collision geoms with capsule primitives.

    URDFs typically reuse high-poly visual meshes for collision geometry.
    MJX pre-compiles convex decompositions of every collision mesh, which
    can consume tens of GB of memory for meshes with 10k+ vertices.

    This function computes an axis-aligned bounding capsule for each mesh
    (from the mesh vertex data) and swaps the geom type in-place.  The
    visual appearance is irrelevant for RL — only contact forces matter,
    and capsules are a standard approximation for robot link collisions.
    """
    # Compile once to read mesh vertex data (the spec itself doesn't expose it)
    m_tmp = spec.compile()

    # Map body name → (center, radius, half_length) from mesh AABB
    capsule_params: dict[str, tuple[np.ndarray, float, float]] = {}
    for i in range(m_tmp.ngeom):
        if m_tmp.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_id = m_tmp.geom_bodyid[i]
        body_name = mujoco.mj_id2name(m_tmp, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mesh_id = m_tmp.geom_dataid[i]
        vert_start = m_tmp.mesh_vertadr[mesh_id]
        vert_count = m_tmp.mesh_vertnum[mesh_id]
        verts = m_tmp.mesh_vert[vert_start : vert_start + vert_count]

        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        bbox = maxs - mins
        center = (mins + maxs) / 2.0

        # Capsule: orient along longest axis (z in local frame)
        radius = float(max(bbox[0], bbox[1])) / 2.0
        half_len = max(float(bbox[2]) / 2.0 - radius, 0.001)
        capsule_params[body_name] = (center, radius, half_len)

    # Now mutate the spec geoms in-place
    for body in spec.bodies:
        if body.name not in capsule_params:
            continue
        center, radius, half_len = capsule_params[body.name]
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                geom.size = np.array([radius, half_len, 0.0])
                geom.pos = center
                geom.meshname = ""


# ---------------------------------------------------------------------------
# Base task — extends Brax's PipelineEnv
# ---------------------------------------------------------------------------


class BaseTask(brax_env.PipelineEnv):
    """Abstract base for RL tasks running on Brax / MJX.

    Subclasses must implement ``reset`` and ``step``.  The ``sys``
    (``brax.System``) is built from the robot MJCF at construction time;
    sub-stepping (``n_frames``) maps ``control_dt / physics_dt``.

    All methods must be pure-functional and JAX-traceable.
    """

    def __init__(
        self,
        robot: RobotConfig,
        backend: str = "mjx",
        n_frames: int = 10,
        scene: SceneConfig | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            robot: Resolved ``RobotConfig`` (MJCF path must exist).
            backend: Brax pipeline backend name (``"mjx"``).
            n_frames: Number of physics sub-steps per control step.
            scene: Optional scene configuration for object interaction.
            **kwargs: Forwarded to ``PipelineEnv.__init__``.
        """
        self.robot = robot
        self.scene = scene

        # Load MuJoCo model via MjSpec so we can add actuators (URDFs have none)
        spec = mujoco.MjSpec.from_file(robot.mjcf_path)

        # Apply per-object friction overrides from scene config
        if scene is not None:
            for obj in scene.objects:
                if obj.friction is not None:
                    for body in spec.bodies:
                        if body.name == obj.name:
                            for geom in body.geoms:
                                geom.friction = np.array(obj.friction[:3] + [0.0] * max(0, 3 - len(obj.friction)))

        # Add a direct-drive torque actuator for every controlled joint
        for jname in robot.joint_names:
            act = spec.add_actuator()
            act.name = f"act_{jname}"
            act.target = jname
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gainprm[0] = 1.0  # direct torque: ctrl → force (1:1)

        # Replace high-poly mesh collision geoms with capsule primitives.
        # URDFs often reuse detailed visual meshes for collision, which causes
        # MJX to build huge convex decompositions (OOM).  Capsule primitives
        # keep collisions functional at trivial cost — standard sim-to-real
        # practice since visual fidelity is irrelevant for RL contact forces.
        _replace_mesh_collisions_with_capsules(spec)

        mj_model = spec.compile()

        # Enforce integrator and timestep (matching C++ backend)
        mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Add joint armature for stability (matching C++ backend)
        for i in range(mj_model.njnt):
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                dof = mj_model.jnt_dofadr[i]
                if mj_model.dof_armature[dof] < 0.01:
                    mj_model.dof_armature[dof] = 0.01

        sys = brax_mjcf.load_model(mj_model)
        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        # Build joint index arrays (JAX-friendly, computed once on CPU)
        q_indices = []
        dof_indices = []
        for name in robot.joint_names:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            q_indices.append(mj_model.jnt_qposadr[jid])
            dof_indices.append(mj_model.jnt_dofadr[jid])

        self._joint_q_indices = jnp.array(q_indices, dtype=jnp.int32)
        self._joint_dof_indices = jnp.array(dof_indices, dtype=jnp.int32)

        # PD gains as JAX arrays
        self._kp = jnp.array([robot.gains[n].kp for n in robot.joint_names], dtype=jnp.float32)
        self._kd = jnp.array([robot.gains[n].kd for n in robot.joint_names], dtype=jnp.float32)

        # Joint limits as JAX arrays
        self._q_lower = jnp.array([robot.joint_limits[n].lower for n in robot.joint_names], dtype=jnp.float32)
        self._q_upper = jnp.array([robot.joint_limits[n].upper for n in robot.joint_names], dtype=jnp.float32)

        # Keep reference to the MjModel (for gravity-comp data access in State.info)
        self._mj_model = mj_model

        # -- scene body / geom / qpos index resolution -----------------------
        self._scene_body_ids: dict[str, int] = {}
        self._scene_geom_ids: dict[str, int] = {}
        self._scene_qpos_slices: dict[str, tuple[int, int]] = {}
        self._scene_nominal_qpos: dict[str, np.ndarray] = {}
        self._ee_body_id: int = -1

        if scene is not None:
            for obj in scene.objects:
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, obj.name)
                if body_id < 0:
                    continue
                self._scene_body_ids[obj.name] = body_id

                # Resolve geom ID (first geom belonging to this body)
                for gi in range(mj_model.ngeom):
                    if mj_model.geom_bodyid[gi] == body_id:
                        self._scene_geom_ids[obj.name] = gi
                        break

                # For dynamic (floating-joint) objects, resolve qpos slice
                if obj.dynamic:
                    jnt_name = f"{obj.name}_joint"
                    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                    if jid >= 0:
                        qpos_start = mj_model.jnt_qposadr[jid]
                        # floating joint: 3 pos + 4 quat = 7 DOF in qpos
                        self._scene_qpos_slices[obj.name] = (qpos_start, qpos_start + 7)
                        # Store nominal qpos (position xyz + identity quat)
                        pos = np.array(obj.position, dtype=np.float32)
                        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz
                        self._scene_nominal_qpos[obj.name] = np.concatenate([pos, quat])

        # Resolve end-effector body ID
        if robot.ee_body:
            self._ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, robot.ee_body)
            if self._ee_body_id < 0:
                raise ValueError(f"End-effector body '{robot.ee_body}' not found in MuJoCo model")

    # -- helpers (JAX-traceable) ------------------------------------------

    def _get_joint_q(self, pipeline_state: brax_base.State) -> jax.Array:
        """Extract joint positions for the robot's joints."""
        return pipeline_state.q[self._joint_q_indices]

    def _get_joint_qd(self, pipeline_state: brax_base.State) -> jax.Array:
        """Extract joint velocities for the robot's joints."""
        return pipeline_state.qd[self._joint_dof_indices]

    def _get_qfrc_bias(self, pipeline_state: brax_base.State) -> jax.Array:
        """Extract gravity + Coriolis torques (``qfrc_bias``) for robot joints.

        Available because MJX ``State`` is also ``mjx.Data``.
        """
        return pipeline_state.qfrc_bias[self._joint_dof_indices]

    def _pd_control(
        self,
        q_target: jax.Array,
        q: jax.Array,
        qd: jax.Array,
        qfrc_bias: jax.Array,
    ) -> jax.Array:
        """Compute PD + gravity compensation torques (mirrors C++ backend)."""
        return self._kp * (q_target - q) + self._kd * (0.0 - qd) + qfrc_bias

    def pipeline_step_pd(self, pipeline_state: brax_base.State, q_target: jax.Array) -> brax_base.State:

        def f(state, _):
            q = state.q[self._joint_q_indices]
            qd = state.qd[self._joint_dof_indices]
            qfrc_bias = state.qfrc_bias[self._joint_dof_indices]
            action = self._kp * (q_target - q) + self._kd * (0.0 - qd) + qfrc_bias
            return (
                self._pipeline.step(self.sys, state, action, self._debug),
                None,
            )

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    # -- scene helpers (JAX-traceable) ------------------------------------

    def _get_body_pos(self, pipeline_state: brax_base.State, body_id: int) -> jax.Array:
        """Get world position of a body by its MuJoCo body ID."""
        return pipeline_state.xpos[body_id]

    def _get_body_quat(self, pipeline_state: brax_base.State, body_id: int) -> jax.Array:
        """Get world quaternion (wxyz) of a body by its MuJoCo body ID."""
        return pipeline_state.xquat[body_id]

    def _get_ee_pos(self, pipeline_state: brax_base.State) -> jax.Array:
        """Get end-effector world position. Requires ``robot.ee_body`` to be set."""
        return pipeline_state.xpos[self._ee_body_id]

    def _randomize_scene_q(self, rng: jax.Array, q: jax.Array) -> jax.Array:
        """Randomize floating-joint qpos for scene objects with randomization ranges.

        For each dynamic scene object that has ``randomize_position`` or
        ``randomize_orientation``, samples uniformly within ±range of the
        nominal position/orientation and sets the corresponding qpos slots.
        """
        if self.scene is None:
            return q

        for obj in self.scene.objects:
            if obj.name not in self._scene_qpos_slices:
                continue
            start, end = self._scene_qpos_slices[obj.name]
            nominal = jnp.array(self._scene_nominal_qpos[obj.name])

            if obj.randomize_position is not None:
                rng, sub_key = jax.random.split(rng)
                half_range = jnp.array(obj.randomize_position, dtype=jnp.float32)
                pos_offset = jax.random.uniform(sub_key, shape=(3,), minval=-half_range, maxval=half_range)
                nominal = nominal.at[:3].set(nominal[:3] + pos_offset)

            if obj.randomize_orientation is not None:
                # Randomize via small Euler angle perturbations converted to quaternion
                rng, sub_key = jax.random.split(rng)
                half_range = jnp.array(obj.randomize_orientation, dtype=jnp.float32)
                rpy = jax.random.uniform(sub_key, shape=(3,), minval=-half_range, maxval=half_range)
                # Approximate: for small angles, quat ≈ [1, rx/2, ry/2, rz/2] (normalized)
                half_rpy = rpy / 2.0
                quat = jnp.array([1.0, half_rpy[0], half_rpy[1], half_rpy[2]])
                quat = quat / jnp.linalg.norm(quat)
                nominal = nominal.at[3:7].set(quat)

            q = q.at[start:end].set(nominal)

        return q

    def _get_contact_dist(self, pipeline_state: brax_base.State, geom_id: int) -> jax.Array:
        """Get minimum contact distance for a scene geom (experimental).

        Returns the minimum signed distance across all contact pairs
        involving the given geom.  Negative = penetration.
        Uses MJX pre-allocated contact arrays.
        """
        contact = pipeline_state.contact
        # contact.geom is (ncon, 2), contact.dist is (ncon,)
        involved = jnp.any(contact.geom == geom_id, axis=-1)
        # Where not involved, set distance to a large value
        dist = jnp.where(involved, contact.dist, jnp.float32(1e6))
        return jnp.min(dist)

    # -- abstract interface -----------------------------------------------

    @abstractmethod
    def reset(self, rng: jax.Array) -> brax_env.State:
        """Pure-functional episode reset.  Must return a ``brax.envs.State``."""

    @abstractmethod
    def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
        """Pure-functional environment step."""

    # Subclasses must also define ``observation_size`` and ``action_size``
    # as properties (inherited from PipelineEnv / Env).


# Import concrete tasks to trigger registration
from core_rl.tasks import ee_tracking as _ee  # noqa: F401, E402
from core_rl.tasks import joint_tracking as _joint_tracking  # noqa: F401, E402
from core_rl.tasks import reach_object as _reach  # noqa: F401, E402
