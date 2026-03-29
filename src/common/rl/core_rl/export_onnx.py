"""ONNX export with all computation layers baked in.

Composes: ObservationNormalizer → Policy → GravityCompensationNet → PDController
into a single ``DeployablePolicy`` module, then exports via ``torch.onnx.export``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from core_rl.modules.deployable import DeployablePolicy
from core_rl.modules.gravity_comp import GravityCompensationNet, train_gravity_comp
from core_rl.modules.normalizer import ObservationNormalizer
from core_rl.modules.pd_controller import PDController
from core_rl.robot import RobotConfig


class _ExtractedPolicyNet(torch.nn.Module):
    """Extracts the MLP from an SB3 policy for clean ONNX export.

    SB3 policies have complex forward() signatures. This wrapper
    extracts the raw MLP layers and provides a simple tensor→tensor forward.
    Supports both PPO (MlpPolicy) and SAC (Actor) architectures.
    """

    def __init__(self, sb3_policy, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

        if hasattr(sb3_policy, "mlp_extractor"):
            # PPO / on-policy: mlp_extractor.policy_net + action_net
            self.mlp_extractor = sb3_policy.mlp_extractor.policy_net
            self.action_net = sb3_policy.action_net
            self._arch = "on_policy"
        elif hasattr(sb3_policy, "actor"):
            # SAC / off-policy: actor.latent_pi + actor.mu (deterministic)
            actor = sb3_policy.actor
            self.latent_pi = actor.latent_pi
            self.mu = actor.mu
            self._arch = "sac"
        else:
            raise ValueError(
                f"Unsupported SB3 policy type: {type(sb3_policy)}. "
                "Expected PPO MlpPolicy or SAC policy."
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._arch == "on_policy":
            latent = self.mlp_extractor(obs)
            return self.action_net(latent)
        else:
            latent = self.latent_pi(obs)
            return self.mu(latent)


def export_onnx(
    algorithm,
    robot: RobotConfig,
    vec_normalize,
    output_dir: str,
    grav_comp_config: dict[str, Any] | None = None,
    grav_comp_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> str:
    """Export the trained policy as a deployable ONNX model.

    Args:
        algorithm: Trained BaseAlgorithm instance.
        robot: RobotConfig for the robot.
        vec_normalize: VecNormalize wrapper (for obs normalization stats).
        output_dir: Directory to write output files.
        grav_comp_config: Config for gravity compensation training.
        grav_comp_data: Optional pre-collected (q, dq, bias) data.

    Returns:
        Path to the exported ONNX file.
    """
    gc_cfg = grav_comp_config or {}
    n_joints = robot.num_joints
    device = "cpu"  # ONNX export on CPU

    # ── 1. Observation normalizer ──
    normalizer = ObservationNormalizer.from_vec_normalize(vec_normalize)
    print(f"  Normalizer: obs_dim={normalizer.mean.shape[0]}")

    # ── 2. Extract policy network ──
    sb3_policy = algorithm.get_policy_network()
    sb3_policy.eval()

    action_dim = n_joints  # Task dependent, but position targets = num_joints
    policy_net = _ExtractedPolicyNet(sb3_policy, action_dim)
    policy_net.eval()
    print(f"  Policy: extracted MLP + action net")

    # ── 3. Gravity compensation network ──
    grav_comp = GravityCompensationNet(
        num_joints=n_joints,
        hidden_dims=gc_cfg.get("hidden_dims", [256, 256]),
    )

    if gc_cfg.get("enabled", False) and grav_comp_data is not None:
        q_data, dq_data, bias_data = grav_comp_data
        print(f"  Training GravCompNet on {len(q_data)} samples...")
        history = train_gravity_comp(
            model=grav_comp,
            q_data=q_data,
            dq_data=dq_data,
            bias_data=bias_data,
            epochs=gc_cfg.get("train_epochs", 50),
            lr=gc_cfg.get("train_lr", 1e-3),
            device=device,
        )
        final_loss = history["loss"][-1]
        print(f"  GravComp trained — final loss: {final_loss:.6f}")

        # Save standalone GravComp model
        gc_path = os.path.join(output_dir, "gravity_comp.pt")
        torch.save(grav_comp.state_dict(), gc_path)
    else:
        print("  GravComp: using untrained (zeros) — no data available")

    grav_comp.eval()

    # ── 4. PD controller ──
    kp = torch.tensor([robot.gains[n].kp for n in robot.joint_names], dtype=torch.float32)
    kd = torch.tensor([robot.gains[n].kd for n in robot.joint_names], dtype=torch.float32)
    pd_controller = PDController(kp=kp, kd=kd)
    print(f"  PD gains: kp={kp.tolist()}, kd={kd.tolist()}")

    # ── 5. Compose into DeployablePolicy ──
    deployable = DeployablePolicy(
        normalizer=normalizer,
        policy_net=policy_net,
        gravity_comp=grav_comp,
        pd_controller=pd_controller,
        num_joints=n_joints,
        action_type="position",
    )
    deployable.eval()

    # ── 6. Export to ONNX ──
    obs_dim = normalizer.mean.shape[0]
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

    # ── 7. Validate with ONNX Runtime ──
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, obs_dim).astype(np.float32)
        outputs = session.run(None, {"observation": test_input})
        assert outputs[0].shape == (1, n_joints), (
            f"Expected output shape (1, {n_joints}), got {outputs[0].shape}"
        )
        print(f"  ONNX validation: OK — output shape {outputs[0].shape}")
    except ImportError:
        print("  ONNX validation: skipped (onnxruntime not installed)")

    return onnx_path
