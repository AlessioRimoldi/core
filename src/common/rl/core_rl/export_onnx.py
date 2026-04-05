"""ONNX export — convert JAX/Flax params to a PyTorch Module for ONNX.

Composes: ObservationNormalizer → Policy MLP → GravityCompensationNet → PDController
into a single ``torch.nn.Module`` and exports via ``torch.onnx.export``.

PyTorch is only needed at export time — it is an optional ``[export]`` dependency.
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from core_rl.modules.gravity_comp import train_gravity_comp
from core_rl.modules.normalizer import NormalizerParams, from_brax_normalizer
from core_rl.modules.normalizer import to_numpy as norm_to_numpy
from core_rl.robot import RobotConfig

# ---------------------------------------------------------------------------
# PyTorch wrappers (import torch only here — optional dependency)
# ---------------------------------------------------------------------------


def _build_torch_deployable(
    normalizer_mean: np.ndarray,
    normalizer_std: np.ndarray,
    normalizer_clip: float,
    policy_layers: list[tuple[np.ndarray, np.ndarray]],
    grav_comp_layers: list[tuple[np.ndarray, np.ndarray]] | None,
    kp: np.ndarray,
    kd: np.ndarray,
    num_joints: int,
    action_type: str,
):
    """Build a PyTorch nn.Module that mirrors the JAX deployable pipeline.

    Args:
        policy_layers: List of (weight, bias) tuples for the policy MLP.
        grav_comp_layers: List of (weight, bias) tuples for the GravComp MLP, or None.
    """
    import torch
    import torch.nn as nn

    class _OnnxNormalizer(nn.Module):
        def __init__(self, mean, std, clip):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
            self.clip = clip

        def forward(self, obs):
            return torch.clamp((obs - self.mean) / (self.std + 1e-8), -self.clip, self.clip)

    class _OnnxMLP(nn.Module):
        def __init__(self, layers):
            super().__init__()
            modules = []
            for i, (w, b) in enumerate(layers):
                linear = nn.Linear(w.shape[1], w.shape[0])
                linear.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=False)
                linear.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False)
                modules.append(linear)
                if i < len(layers) - 1:
                    modules.append(nn.ReLU())
            self.net = nn.Sequential(*modules)

        def forward(self, x):
            return self.net(x)

    class _OnnxPD(nn.Module):
        def __init__(self, kp, kd):
            super().__init__()
            self.register_buffer("kp", torch.tensor(kp, dtype=torch.float32))
            self.register_buffer("kd", torch.tensor(kd, dtype=torch.float32))

        def forward(self, q_target, q_current, dq_current, gravity_comp):
            return self.kp * (q_target - q_current) + self.kd * (0.0 - dq_current) + gravity_comp

    class _OnnxDeployable(nn.Module):
        def __init__(self, normalizer, policy, grav_comp, pd, num_joints, action_type):
            super().__init__()
            self.normalizer = normalizer
            self.policy = policy
            self.grav_comp = grav_comp
            self.pd = pd
            self.num_joints = num_joints
            self.action_type = action_type

        def forward(self, obs):
            n = self.num_joints
            q_current = obs[..., :n]
            dq_current = obs[..., n : 2 * n]

            obs_norm = self.normalizer(obs)
            action = self.policy(obs_norm)

            if self.grav_comp is not None:
                gc_input = torch.cat([q_current, dq_current], dim=-1)
                grav = self.grav_comp(gc_input)
            else:
                grav = torch.zeros_like(q_current)

            if self.action_type == "position":
                return self.pd(action, q_current, dq_current, grav)
            elif self.action_type == "torque":
                return action + grav
            else:
                return self.pd(q_current, q_current, dq_current, grav)

    normalizer = _OnnxNormalizer(normalizer_mean, normalizer_std, normalizer_clip)
    policy = _OnnxMLP(policy_layers)
    grav_comp = _OnnxMLP(grav_comp_layers) if grav_comp_layers else None
    pd = _OnnxPD(kp, kd)

    deployable = _OnnxDeployable(normalizer, policy, grav_comp, pd, num_joints, action_type)
    deployable.eval()
    return deployable


def _extract_flax_mlp_layers(params: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract (weight, bias) pairs from a Flax Dense-stack parameter dict.

    Flax stores params as nested dicts like:
    ``{'params': {'Dense_0': {'kernel': ..., 'bias': ...}, 'Dense_1': ...}}``
    """
    p = params.get("params", params)
    layers = []
    i = 0
    while f"Dense_{i}" in p:
        kernel = np.asarray(p[f"Dense_{i}"]["kernel"])  # (in, out)
        bias = np.asarray(p[f"Dense_{i}"]["bias"])  # (out,)
        # PyTorch Linear expects (out, in) weight
        layers.append((kernel.T, bias))
        i += 1
    return layers


def _extract_brax_policy_layers(params, make_policy_fn) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract policy MLP layers from Brax training params.

    Brax PPO params structure: ``(normalizer_params, policy_params)`` or
    ``(normalizer_params, policy_params, value_params)``.
    The policy_params are Flax params for the policy network.
    """
    # Brax stores params as a tuple; policy params are the second element
    policy_params = (params[1] if len(params) > 1 else params[0]) if isinstance(params, tuple | list) else params

    return _extract_flax_mlp_layers(policy_params)


def export_onnx(
    make_policy_fn,
    params,
    robot: RobotConfig,
    output_dir: str,
    grav_comp_config: dict[str, Any] | None = None,
    grav_comp_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    action_type: str = "position",
) -> str:
    """Export the trained policy as a deployable ONNX model.

    Args:
        make_policy_fn: ``make_policy`` function returned by Brax ``train()``.
        params: Trained parameters (JAX pytree from Brax).
        robot: RobotConfig for the robot.
        output_dir: Directory to write output files.
        grav_comp_config: Config for gravity compensation training.
        grav_comp_data: Optional pre-collected ``(q, dq, bias)`` data.
        action_type: ``"position"``, ``"velocity"``, or ``"torque"``.

    Returns:
        Path to the exported ONNX file.
    """
    import torch

    gc_cfg = grav_comp_config or {}
    n_joints = robot.num_joints
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Normalizer params from Brax ──
    if isinstance(params, tuple | list) and len(params) >= 1:
        normalizer_params = params[0]
        norm_params = from_brax_normalizer(normalizer_params)
    else:
        norm_params = NormalizerParams(
            mean=jnp.zeros(3 * n_joints),
            std=jnp.ones(3 * n_joints),
        )

    norm_mean, norm_std, norm_clip = norm_to_numpy(norm_params)
    obs_dim = len(norm_mean)
    print(f"  Normalizer: obs_dim={obs_dim}")

    # ── 2. Policy MLP layers ──
    policy_layers = _extract_brax_policy_layers(params, make_policy_fn)
    print(f"  Policy: {len(policy_layers)} layers extracted")

    # ── 3. Gravity compensation network ──
    grav_comp_layers = None
    if gc_cfg.get("enabled", False) and grav_comp_data is not None:
        q_data, dq_data, bias_data = grav_comp_data
        hidden_dims = tuple(gc_cfg.get("hidden_dims", [256, 256]))
        print(f"  Training GravCompNet on {len(q_data)} samples...")

        gc_model, gc_params, gc_history = train_gravity_comp(
            num_joints=n_joints,
            q_data=q_data,
            dq_data=dq_data,
            bias_data=bias_data,
            hidden_dims=hidden_dims,
            epochs=gc_cfg.get("train_epochs", 50),
            lr=gc_cfg.get("train_lr", 1e-3),
        )
        final_loss = gc_history["loss"][-1]
        print(f"  GravComp trained — final loss: {final_loss:.6f}")

        grav_comp_layers = _extract_flax_mlp_layers(gc_params)
    else:
        print("  GravComp: skipped (no data or disabled)")

    # ── 4. PD gains ──
    kp = np.array([robot.gains[n].kp for n in robot.joint_names], dtype=np.float32)
    kd = np.array([robot.gains[n].kd for n in robot.joint_names], dtype=np.float32)
    print(f"  PD gains: kp={kp.tolist()}, kd={kd.tolist()}")

    # ── 5. Build PyTorch deployable and export ONNX ──
    deployable = _build_torch_deployable(
        normalizer_mean=norm_mean,
        normalizer_std=norm_std,
        normalizer_clip=norm_clip,
        policy_layers=policy_layers,
        grav_comp_layers=grav_comp_layers,
        kp=kp,
        kd=kd,
        num_joints=n_joints,
        action_type=action_type,
    )

    dummy_input = torch.randn(1, obs_dim)
    onnx_path = os.path.join(output_dir, "policy.onnx")

    torch.onnx.export(
        deployable,
        dummy_input,
        onnx_path,
        input_names=["observation"],
        output_names=["torques"],
        dynamic_axes={
            "observation": {0: "batch"},
            "torques": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  ONNX exported: {onnx_path}")

    # ── 6. Validate with ONNX Runtime ──
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, obs_dim).astype(np.float32)
        outputs = session.run(None, {"observation": test_input})
        assert outputs[0].shape == (1, n_joints), f"Expected (1, {n_joints}), got {outputs[0].shape}"
        print(f"  ONNX validation: OK — output shape {outputs[0].shape}")
    except ImportError:
        print("  ONNX validation: skipped (onnxruntime not installed)")

    return onnx_path
