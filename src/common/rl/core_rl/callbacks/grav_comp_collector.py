"""Callback to collect (q, dq, qfrc_bias) data for gravity compensation training.

Because SubprocVecEnv runs each env in a separate process, we can't directly
access the MuJoCo state. Instead, we extract q and dq from observations
(which always start with [q, dq, ...]) and compute qfrc_bias would require
MuJoCo access. For SubprocVecEnv, we collect (q, dq) from observations and
rely on a secondary pass through a single env to get qfrc_bias.

Alternative approach used here: run a single non-vectorized env on the side
to collect the full (q, dq, qfrc_bias) tuples.
"""

from __future__ import annotations

import mujoco
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from core_rl.robot import RobotConfig


class GravCompCollectorCallback(BaseCallback):
    """Collects gravity compensation training data during RL rollouts.

    Maintains a separate single MuJoCo env instance to query qfrc_bias
    for observed (q, dq) pairs.
    """

    def __init__(self, robot: RobotConfig, buffer_size: int = 500_000, verbose: int = 0):
        super().__init__(verbose)
        self.robot = robot
        self.buffer_size = buffer_size
        self.n = robot.num_joints

        self._q_buf = np.zeros((buffer_size, self.n), dtype=np.float32)
        self._dq_buf = np.zeros((buffer_size, self.n), dtype=np.float32)
        self._bias_buf = np.zeros((buffer_size, self.n), dtype=np.float32)
        self._count = 0

        # Lazy-init a single MuJoCo model for qfrc_bias queries
        self._model = None
        self._data = None
        self._joint_q_indices: list[int] = []
        self._joint_dof_indices: list[int] = []

    def _ensure_model(self):
        if self._model is not None:
            return

        self._model = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
        self._model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._model.opt.timestep = 0.001
        self._data = mujoco.MjData(self._model)

        for name in self.robot.joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_q_indices.append(self._model.jnt_qposadr[jid])
            self._joint_dof_indices.append(self._model.jnt_dofadr[jid])

    def _query_bias(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Set joint state and run mj_forward to get qfrc_bias."""
        self._ensure_model()

        for i, idx in enumerate(self._joint_q_indices):
            self._data.qpos[idx] = q[i]
        for i, idx in enumerate(self._joint_dof_indices):
            self._data.qvel[idx] = dq[i]

        mujoco.mj_forward(self._model, self._data)

        return np.array(
            [self._data.qfrc_bias[idx] for idx in self._joint_dof_indices],
            dtype=np.float32,
        )

    def _on_step(self) -> bool:
        if self._count >= self.buffer_size:
            return True

        # Extract q, dq from the observations of env 0
        # Observations are [q(n), dq(n), ...] by convention
        obs = self.locals.get("new_obs")
        if obs is None:
            obs = self.locals.get("obs_tensor")
        if obs is None:
            return True

        if hasattr(obs, "cpu"):
            obs = obs.cpu().numpy()

        # Take first env's observation
        if obs.ndim > 1:
            obs = obs[0]

        q = obs[:self.n]
        dq = obs[self.n:2 * self.n]

        bias = self._query_bias(q, dq)

        idx = self._count
        self._q_buf[idx] = q
        self._dq_buf[idx] = dq
        self._bias_buf[idx] = bias
        self._count += 1

        return True

    def get_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return collected data as (q, dq, qfrc_bias) arrays."""
        if self._count == 0:
            return None
        n = self._count
        return self._q_buf[:n].copy(), self._dq_buf[:n].copy(), self._bias_buf[:n].copy()
