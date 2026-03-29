"""Deployable policy — composes all computation layers into a single Module.

The forward pass: raw_obs → normalize → policy → PD control (with gravity comp) → torques

This is the module exported to ONNX for sim-to-real deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core_rl.modules.gravity_comp import GravityCompensationNet
from core_rl.modules.normalizer import ObservationNormalizer
from core_rl.modules.pd_controller import PDController


class DeployablePolicy(nn.Module):
    """Full deployment pipeline as a single nn.Module.

    Input:  raw observation [q, dq, q_target, ...]  (depends on task)
    Output: joint torques to apply

    Layers:
        1. ObservationNormalizer — standardize inputs
        2. Policy network — map observation to action (joint position targets)
        3. GravityCompensationNet — predict gravity/Coriolis torques from (q, dq)
        4. PDController — convert position targets + grav comp to torques
    """

    def __init__(
        self,
        normalizer: ObservationNormalizer,
        policy_net: nn.Module,
        gravity_comp: GravityCompensationNet,
        pd_controller: PDController,
        num_joints: int,
        action_type: str = "position",
    ):
        super().__init__()
        self.normalizer = normalizer
        self.policy_net = policy_net
        self.gravity_comp = gravity_comp
        self.pd_controller = pd_controller
        self.num_joints = num_joints
        self.action_type = action_type

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Full inference pipeline.

        Args:
            obs: Raw observation tensor, shape (batch, obs_dim).
                 Expected to contain [q, dq, ...] as first 2*num_joints elements.

        Returns:
            Joint torques, shape (batch, num_joints).
        """
        n = self.num_joints

        # Extract q, dq from raw observation (always the first 2*n elements)
        q_current = obs[..., :n]
        dq_current = obs[..., n:2*n]

        # 1. Normalize observation
        obs_norm = self.normalizer(obs)

        # 2. Policy inference → action
        action = self.policy_net(obs_norm)

        # 3. Gravity compensation
        grav_comp = self.gravity_comp(q_current, dq_current)

        # 4. PD control to produce torques
        if self.action_type == "position":
            torques = self.pd_controller(
                q_target=action,
                q_current=q_current,
                dq_current=dq_current,
                gravity_comp=grav_comp,
            )
        elif self.action_type == "torque":
            torques = action + grav_comp
        else:
            torques = self.pd_controller(
                q_target=q_current,  # Hold position
                q_current=q_current,
                dq_current=dq_current,
                gravity_comp=grav_comp,
                dq_target=action,
            )

        return torques
