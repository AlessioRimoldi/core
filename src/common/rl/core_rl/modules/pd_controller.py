"""PD controller as a torch.nn.Module for ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn


class PDController(nn.Module):
    """PD controller: tau = kp * (q_target - q) + kd * (dq_target - dq) + gravity_comp.

    Gains are stored as buffers (not parameters) so they're baked into ONNX
    but not updated by optimizers.
    """

    def __init__(self, kp: torch.Tensor, kd: torch.Tensor):
        """
        Args:
            kp: Proportional gains, shape (num_joints,).
            kd: Derivative gains, shape (num_joints,).
        """
        super().__init__()
        self.register_buffer("kp", kp.float())
        self.register_buffer("kd", kd.float())

    def forward(
        self,
        q_target: torch.Tensor,
        q_current: torch.Tensor,
        dq_current: torch.Tensor,
        gravity_comp: torch.Tensor,
        dq_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute PD + gravity compensation torques.

        Args:
            q_target: Desired joint positions, shape (..., num_joints).
            q_current: Current joint positions, shape (..., num_joints).
            dq_current: Current joint velocities, shape (..., num_joints).
            gravity_comp: Gravity compensation torques, shape (..., num_joints).
            dq_target: Desired joint velocities (default: zeros).

        Returns:
            Joint torques, shape (..., num_joints).
        """
        if dq_target is None:
            dq_target = torch.zeros_like(dq_current)

        tau = (
            self.kp * (q_target - q_current)
            + self.kd * (dq_target - dq_current)
            + gravity_comp
        )
        return tau
