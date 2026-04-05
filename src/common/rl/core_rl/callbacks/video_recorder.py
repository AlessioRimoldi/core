"""Eval video recording for RL training visualization.

Records periodic rollout videos as tiled MP4 grids showing multiple
environments running the current policy.  Uses MuJoCo EGL offscreen
rendering — no display required.

Designed as a ``policy_params_fn`` callback for Brax's training loop.
"""

from __future__ import annotations

import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from core_rl.tasks import BaseTask


class VideoRecorderHook:
    """Records tiled eval rollout videos during Brax training.

    Every ``record_interval`` invocations of ``policy_params_fn``, this hook:

    1. Runs a short rollout of ``grid_cols × grid_rows`` environments on GPU
       using the current policy (deterministic mode).
    2. Transfers joint positions (qpos) to CPU.
    3. Replays each trajectory in CPU MuJoCo for rendering.
    4. Tiles the renders into a grid and saves as MP4.

    The GPU rollout is JIT-compiled on first use; subsequent calls reuse the
    cached trace and only vary the policy parameters.

    Args:
        env: The Brax ``BaseTask`` (must expose ``_mj_model``).
        output_dir: Base directory; videos saved to ``output_dir/videos/``.
        record_interval: Record every N ``policy_params_fn`` invocations.
        grid_cols: Columns in the tiled video grid.
        grid_rows: Rows in the tiled video grid.
        resolution: ``(width, height)`` per environment tile in pixels.
        episode_length: Steps per eval rollout.
        fps: Video frame rate.
    """

    def __init__(
        self,
        env: BaseTask,
        output_dir: str,
        record_interval: int = 5,
        grid_cols: int = 4,
        grid_rows: int = 4,
        resolution: tuple[int, int] = (320, 320),
        episode_length: int = 200,
        fps: int = 30,
    ):
        self._env = env
        self._output_dir = output_dir
        self._record_interval = record_interval
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows
        self._n_envs = grid_cols * grid_rows
        self._width, self._height = resolution
        self._episode_length = episode_length
        self._fps = fps
        self._call_count = 0
        self._rollout_fn = None

        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

        # Use EGL for offscreen rendering (no display needed)
        os.environ.setdefault("MUJOCO_GL", "egl")

        # Load a separate model with full visual meshes for rendering.
        # The training model (env._mj_model) has capsule-approximated geoms
        # for MJX performance — we want the original meshes for video quality.
        spec = mujoco.MjSpec.from_file(env.robot.mjcf_path)
        for jname in env.robot.joint_names:
            act = spec.add_actuator()
            act.name = f"act_{jname}"
            act.target = jname
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gainprm[0] = 1.0
        self._mj_model = spec.compile()
        self._mj_data = mujoco.MjData(self._mj_model)
        self._renderer = mujoco.Renderer(self._mj_model, self._height, self._width)

        # Default camera: front-right, slightly above
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.lookat[:] = [0.0, 0.0, 0.15]
        self._camera.distance = 0.8
        self._camera.azimuth = 135.0
        self._camera.elevation = -25.0

    def __call__(self, step: int, make_policy: Any, params: Any) -> None:
        """``policy_params_fn(step, make_policy, params)`` callback."""
        self._call_count += 1
        if self._call_count % self._record_interval != 0:
            return
        try:
            self._record(step, make_policy, params)
        except Exception as e:
            print(f"  [VideoRecorder] Failed at step {step}: {e}")

    def _build_rollout_fn(self, make_policy):
        """JIT-compile a batched rollout that returns qpos trajectories.

        Shape: ``(episode_length + 1, n_envs, nq)``.
        Compiled once; subsequent calls reuse the JAX trace cache.
        """
        env = self._env
        n_envs = self._n_envs
        ep_len = self._episode_length

        def _rollout(params, rng):
            policy = make_policy(params, deterministic=True)
            keys = jax.random.split(rng, n_envs)
            init_state = jax.vmap(env.reset)(keys)
            q0 = init_state.pipeline_state.q  # (n_envs, nq)

            def _step(carry, _):
                state, rng = carry
                rng, act_rng = jax.random.split(rng)
                # Policy handles batched obs natively (Flax MLP)
                action, _ = policy(state.obs, act_rng)
                next_state = jax.vmap(env.step)(state, action)
                return (next_state, rng), next_state.pipeline_state.q

            (_, _), qpos_seq = jax.lax.scan(_step, (init_state, rng), None, length=ep_len)
            # qpos_seq: (ep_len, n_envs, nq)
            return jnp.concatenate([q0[None], qpos_seq], axis=0)

        return jax.jit(_rollout)

    def _record(self, step: int, make_policy, params):
        """Run GPU rollout → CPU render → save MP4."""
        import imageio

        t0 = time.time()

        if self._rollout_fn is None:
            print("  [VideoRecorder] JIT-compiling rollout (first call, may take a minute)...")
            self._rollout_fn = self._build_rollout_fn(make_policy)

        rng = jax.random.PRNGKey(step % (2**31))
        qpos_traj = np.asarray(self._rollout_fn(params, rng))
        # qpos_traj: (n_steps, n_envs, nq)

        n_steps = qpos_traj.shape[0]
        grid_h = self._height * self._grid_rows
        grid_w = self._width * self._grid_cols

        video_path = os.path.join(self._output_dir, "videos", f"eval_{step:08d}.mp4")
        writer = imageio.get_writer(video_path, fps=self._fps, macro_block_size=1)

        for t in range(n_steps):
            grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for idx in range(self._n_envs):
                row, col = divmod(idx, self._grid_cols)
                self._mj_data.qpos[:] = qpos_traj[t, idx]
                mujoco.mj_forward(self._mj_model, self._mj_data)
                self._renderer.update_scene(self._mj_data, self._camera)
                pixels = self._renderer.render()
                y0, x0 = row * self._height, col * self._width
                grid_frame[y0 : y0 + self._height, x0 : x0 + self._width] = pixels
            writer.append_data(grid_frame)

        writer.close()
        elapsed = time.time() - t0
        print(
            f"  [VideoRecorder] Step {step}: {video_path} " f"({n_steps} frames, {self._n_envs} envs, {elapsed:.1f}s)"
        )

    def close(self):
        """Release rendering resources."""
        if hasattr(self, "_renderer"):
            self._renderer.close()
