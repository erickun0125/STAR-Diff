# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for STAR-Diff Peg Transfer task.

These functions compute rewards for the surgical peg transfer task with RCM constraints.
All functions return tensors with shape (num_envs,) for batch processing.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_peg(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    peg_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for reaching towards the peg.

    Uses exponential kernel: r = 1 - tanh(distance / std)
    Closer to peg = higher reward.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        peg_cfg: Configuration for the peg asset.
        body_name: Name of the end-effector body.
        std: Standard deviation for exponential kernel.

    Returns:
        Reaching reward with shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    peg_pos = peg.data.root_pos_w

    distance = torch.norm(ee_pos - peg_pos, dim=-1)
    return 1 - torch.tanh(distance / std)


def grasping_peg(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    peg_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
    grasp_threshold: float = 0.03,
) -> torch.Tensor:
    """Reward for grasping the peg (EE close to peg position).

    Binary reward when end-effector is within grasp threshold of peg.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        peg_cfg: Configuration for the peg asset.
        body_name: Name of the end-effector body.
        grasp_threshold: Distance threshold for successful grasp.

    Returns:
        Grasping reward with shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    peg_pos = peg.data.root_pos_w

    distance = torch.norm(ee_pos - peg_pos, dim=-1)
    return (distance < grasp_threshold).float()


def lifting_peg(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    min_height: float = 0.08,
) -> torch.Tensor:
    """Reward for lifting the peg above minimum height.

    Linear reward based on height above minimum:
        r = max(0, z - min_height) / min_height

    Args:
        env: The environment instance.
        peg_cfg: Configuration for the peg asset.
        min_height: Minimum height for reward.

    Returns:
        Lifting reward with shape (num_envs,).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_height = peg.data.root_pos_w[:, 2]

    # Reward for height above min_height, capped at 1.0
    height_reward = torch.clamp((peg_height - min_height) / min_height, 0.0, 1.0)
    return height_reward


def reaching_target(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for moving the peg towards the target position.

    Uses exponential kernel: r = 1 - tanh(distance / std)

    Args:
        env: The environment instance.
        peg_cfg: Configuration for the peg asset.
        target_cfg: Configuration for the target asset.
        std: Standard deviation for exponential kernel.

    Returns:
        Target reaching reward with shape (num_envs,).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    peg_pos = peg.data.root_pos_w
    target_pos = target.data.root_pos_w

    distance = torch.norm(peg_pos - target_pos, dim=-1)
    return 1 - torch.tanh(distance / std)


def placing_peg(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    success_threshold: float = 0.02,
) -> torch.Tensor:
    """Reward for successfully placing the peg at target position.

    Binary reward when peg is within success threshold of target.

    Args:
        env: The environment instance.
        peg_cfg: Configuration for the peg asset.
        target_cfg: Configuration for the target asset.
        success_threshold: Distance threshold for successful placement.

    Returns:
        Placement reward with shape (num_envs,).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    peg_pos = peg.data.root_pos_w
    target_pos = target.data.root_pos_w

    distance = torch.norm(peg_pos - target_pos, dim=-1)
    return (distance < success_threshold).float()


def rcm_violation_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    trocar_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
    tolerance: float = 0.01,
) -> torch.Tensor:
    """Penalty for violating the RCM constraint.

    The RCM constraint requires the instrument's longitudinal axis to pass through
    the trocar point. This penalty increases when the deviation exceeds tolerance.

    Penalty = max(0, deviation - tolerance) / tolerance

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        trocar_cfg: Configuration for the trocar asset.
        body_name: Name of the end-effector body.
        tolerance: Acceptable deviation distance.

    Returns:
        RCM violation penalty (positive value) with shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Trocar can be RigidObject or Articulation
    trocar = env.scene[trocar_cfg.name]

    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    ee_quat = robot.data.body_quat_w[:, body_idx]
    trocar_pos = trocar.data.root_pos_w

    # Get the z-axis direction of the end-effector (instrument axis)
    w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
    z_axis = torch.stack(
        [
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )

    # Vector from EE to trocar
    ee_to_trocar = trocar_pos - ee_pos

    # Project onto z_axis and compute perpendicular distance
    proj_length = torch.sum(ee_to_trocar * z_axis, dim=-1, keepdim=True)
    proj_vec = proj_length * z_axis
    perp_vec = ee_to_trocar - proj_vec
    deviation = torch.norm(perp_vec, dim=-1)

    # Penalty only when deviation exceeds tolerance
    penalty = torch.clamp((deviation - tolerance) / tolerance, min=0.0)
    return penalty


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high joint velocities (encourages smooth motion).

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.

    Returns:
        Joint velocity L2 penalty with shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_vel = robot.data.joint_vel
    return torch.sum(joint_vel**2, dim=-1)


def joint_torque_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high joint torques (encourages efficient motion).

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.

    Returns:
        Joint torque L2 penalty with shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # applied_torque is the actual torque applied by the actuators
    joint_torque = robot.data.applied_torque
    return torch.sum(joint_torque**2, dim=-1)
