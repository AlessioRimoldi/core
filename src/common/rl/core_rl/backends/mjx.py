"""MJX simulation backend — MuJoCo MJX via Brax's pipeline layer.

This is the default training backend.  It JIT-compiles the entire physics
step via ``mujoco.mjx`` and exposes the result as Brax ``State`` objects
so that ``jax.vmap`` can vectorise over thousands of environments on GPU.
"""

from __future__ import annotations

import jax
import mujoco
from brax.mjx import pipeline as mjx_pipeline

from core_rl.backends import SimBackend, register_backend


@register_backend("mjx")
class MJXBackend(SimBackend):
    """MuJoCo MJX backend — thin wrapper around ``brax.mjx.pipeline``."""

    def init(
        self,
        model: mujoco.MjModel,
        q: jax.Array,
        qd: jax.Array,
        *,
        ctrl: jax.Array | None = None,
    ) -> mjx_pipeline.State:
        """Create an initial physics state on device.

        Uses ``brax.mjx.pipeline.init`` which:
        1. Creates ``mjx.Data`` via ``mjx.make_data``
        2. Replaces q / qd (and optionally ctrl)
        3. Runs ``mjx.forward`` for kinematics
        4. Returns a ``State`` that is both ``brax.base.State`` **and**
           ``mjx.Data`` (so you have access to ``qfrc_bias``, ``xpos``, etc.)
        """
        return mjx_pipeline.init(model, q, qd, ctrl=ctrl)

    def step(
        self,
        model: mujoco.MjModel,
        state: mjx_pipeline.State,
        action: jax.Array,
    ) -> mjx_pipeline.State:
        """Advance physics by one step.

        Sets ``ctrl = action`` on the state and calls ``mjx.step``.
        Sub-stepping (``n_frames``) is handled by the ``PipelineEnv``
        wrapper in the task layer, not here.
        """
        return mjx_pipeline.step(model, state, action)
