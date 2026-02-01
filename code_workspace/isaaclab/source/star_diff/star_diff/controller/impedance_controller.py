# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Impedance Controller implementation for STAR-Diff.

This module provides:
1. JointImpedanceController: Simple joint-space impedance control
2. ImpedanceController: IK + Joint Impedance cascade controller
"""

from __future__ import annotations

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from ..configs.controller_cfg import JointImpedanceControllerCfg


class JointImpedanceController:
    """Custom Joint Impedance Controller.

    Implements joint-space impedance control:
        τ = K_p * (q_des - q) - K_d * q_dot + g(q)

    where:
        - K_p: Joint stiffness matrix (diagonal)
        - K_d: Joint damping matrix (computed from damping ratio)
        - g(q): Gravity compensation torques
    """

    def __init__(
        self,
        cfg: JointImpedanceControllerCfg,
        num_envs: int,
        num_joints: int,
        device: str,
    ):
        """Initialize the joint impedance controller.

        Args:
            cfg: Controller configuration.
            num_envs: Number of parallel environments.
            num_joints: Number of joints to control.
            device: Device to run computations on.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device

        # Compute stiffness and damping matrices
        self._kp = torch.full((num_envs, num_joints), cfg.stiffness, device=device)
        # K_d = 2 * damping_ratio * sqrt(K_p) for critically damped response
        self._kd = 2.0 * cfg.damping_ratio * torch.sqrt(self._kp)

        # Command buffer
        self._joint_pos_des = torch.zeros(num_envs, num_joints, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset controller state for specified environments."""
        if env_ids is None:
            self._joint_pos_des.zero_()
        else:
            self._joint_pos_des[env_ids] = 0.0

    def set_command(self, joint_pos_des: torch.Tensor):
        """Set desired joint positions.

        Args:
            joint_pos_des: Desired joint positions. Shape: (num_envs, num_joints).
        """
        self._joint_pos_des[:] = joint_pos_des

    def compute(
        self,
        current_joint_pos: torch.Tensor,
        current_joint_vel: torch.Tensor,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint torques using impedance control.

        Args:
            current_joint_pos: Current joint positions. Shape: (num_envs, num_joints).
            current_joint_vel: Current joint velocities. Shape: (num_envs, num_joints).
            gravity: Gravity compensation torques. Shape: (num_envs, num_joints).

        Returns:
            Joint torques. Shape: (num_envs, num_joints).
        """
        # Position error
        pos_error = self._joint_pos_des - current_joint_pos

        # PD control: τ = K_p * e - K_d * q_dot
        torques = self._kp * pos_error - self._kd * current_joint_vel

        # Add gravity compensation if enabled and provided
        if self.cfg.use_gravity_compensation and gravity is not None:
            torques = torques + gravity

        return torques


class ImpedanceController:
    """Impedance Controller (SE(3) -> IK -> Joint Impedance).

    Takes SE(3) pose command, converts to joint targets using IK,
    then tracks with Joint Impedance.
    """

    def __init__(
        self,
        ik_cfg: DifferentialIKControllerCfg,
        imp_cfg: JointImpedanceControllerCfg,
        num_envs: int,
        device: str,
        dof_pos_limits: torch.Tensor,
    ):
        """Initialize the impedance controller.

        Args:
            ik_cfg: Configuration for the IK controller.
            imp_cfg: Configuration for the joint impedance controller.
            num_envs: Number of parallel environments.
            device: Device to run computations on.
            dof_pos_limits: Joint position limits. Shape: (num_envs, num_joints, 2).
        """
        self.num_envs = num_envs
        self.device = device

        # Infer number of joints from limits
        num_joints = dof_pos_limits.shape[1]

        # Initialize controllers
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs, device)
        self.imp_controller = JointImpedanceController(imp_cfg, num_envs, num_joints, device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset both controllers for specified environments."""
        self.ik_controller.reset(env_ids)
        self.imp_controller.reset(env_ids)

    def compute(
        self,
        ee_pos_des: torch.Tensor,
        ee_quat_des: torch.Tensor,
        jacobian: torch.Tensor,
        current_ee_pose: torch.Tensor,
        current_ee_vel: torch.Tensor,
        mass_matrix: torch.Tensor,
        current_joint_pos: torch.Tensor,
        current_joint_vel: torch.Tensor,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint torques.

        Args:
            ee_pos_des: Desired end-effector position. Shape: (num_envs, 3).
            ee_quat_des: Desired end-effector quaternion (w, x, y, z). Shape: (num_envs, 4).
            jacobian: End-effector Jacobian. Shape: (num_envs, 6, num_joints).
            current_ee_pose: Current EE pose [pos, quat]. Shape: (num_envs, 7).
            current_ee_vel: Current EE velocity [lin, ang]. Shape: (num_envs, 6).
            mass_matrix: Joint-space mass matrix. Shape: (num_envs, num_joints, num_joints).
            current_joint_pos: Current joint positions. Shape: (num_envs, num_joints).
            current_joint_vel: Current joint velocities. Shape: (num_envs, num_joints).
            gravity: Gravity compensation torques. Shape: (num_envs, num_joints).

        Returns:
            Joint torques. Shape: (num_envs, num_joints).
        """
        # Extract current EE state
        current_ee_pos = current_ee_pose[:, :3]
        current_ee_quat = current_ee_pose[:, 3:7]

        # 1. Run IK to get joint targets
        ik_command = torch.cat([ee_pos_des, ee_quat_des], dim=-1)
        self.ik_controller.set_command(ik_command, current_ee_pos, current_ee_quat)
        joint_pos_des = self.ik_controller.compute(
            current_ee_pos, current_ee_quat, jacobian, current_joint_pos
        )

        # 2. Run Joint Impedance
        self.imp_controller.set_command(joint_pos_des)
        joint_torques = self.imp_controller.compute(
            current_joint_pos=current_joint_pos,
            current_joint_vel=current_joint_vel,
            gravity=gravity,
        )

        return joint_torques
