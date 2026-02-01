# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for STAR-Diff Peg Transfer task.

These functions determine when episodes should terminate.
All functions return boolean tensors with shape (num_envs,).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_success(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    success_threshold: float = 0.02,
) -> torch.Tensor:
    """Terminate when peg is successfully placed at target position.

    Args:
        env: The environment instance.
        peg_cfg: Configuration for the peg asset.
        target_cfg: Configuration for the target asset.
        success_threshold: Distance threshold for successful placement.

    Returns:
        Boolean tensor indicating task success with shape (num_envs,).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    peg_pos = peg.data.root_pos_w
    target_pos = target.data.root_pos_w

    distance = torch.norm(peg_pos - target_pos, dim=-1)
    return distance < success_threshold


def rcm_violation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    trocar_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
    max_deviation: float = 0.05,
) -> torch.Tensor:
    """Terminate when RCM constraint is severely violated.

    The RCM constraint requires the instrument's longitudinal axis to pass through
    the trocar point. Terminate if deviation exceeds max_deviation.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        trocar_cfg: Configuration for the trocar asset.
        body_name: Name of the end-effector body.
        max_deviation: Maximum allowable deviation from RCM constraint.

    Returns:
        Boolean tensor indicating RCM violation with shape (num_envs,).
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

    return deviation > max_deviation


def out_of_bounds(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
    bounds: dict[str, tuple[float, float]] = None,
) -> torch.Tensor:
    """Terminate when end-effector goes out of workspace bounds.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        body_name: Name of the end-effector body.
        bounds: Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples.
                Default bounds: x=[0.1, 0.8], y=[-0.4, 0.4], z=[0.0, 0.5]

    Returns:
        Boolean tensor indicating out of bounds with shape (num_envs,).
    """
    if bounds is None:
        bounds = {
            "x": (0.1, 0.8),
            "y": (-0.4, 0.4),
            "z": (0.0, 0.5),
        }

    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]

    x_min, x_max = bounds.get("x", (-float("inf"), float("inf")))
    y_min, y_max = bounds.get("y", (-float("inf"), float("inf")))
    z_min, z_max = bounds.get("z", (-float("inf"), float("inf")))

    out_of_x = (ee_pos[:, 0] < x_min) | (ee_pos[:, 0] > x_max)
    out_of_y = (ee_pos[:, 1] < y_min) | (ee_pos[:, 1] > y_max)
    out_of_z = (ee_pos[:, 2] < z_min) | (ee_pos[:, 2] > z_max)

    return out_of_x | out_of_y | out_of_z


def peg_dropped(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    min_height: float = 0.0,
) -> torch.Tensor:
    """Terminate when peg falls below minimum height (dropped).

    Args:
        env: The environment instance.
        peg_cfg: Configuration for the peg asset.
        min_height: Minimum acceptable height for the peg.

    Returns:
        Boolean tensor indicating peg dropped with shape (num_envs,).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_height = peg.data.root_pos_w[:, 2]
    return peg_height < min_height


def joint_limit_violation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.01,
) -> torch.Tensor:
    """Terminate when robot joints exceed their limits.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        margin: Safety margin from joint limits (in radians).

    Returns:
        Boolean tensor indicating joint limit violation with shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos
    joint_limits = robot.data.soft_joint_pos_limits  # (num_envs, num_joints, 2)

    # Check if any joint exceeds limits with margin
    lower_violation = joint_pos < (joint_limits[:, :, 0] + margin)
    upper_violation = joint_pos > (joint_limits[:, :, 1] - margin)

    # Any joint violating limits terminates the episode
    return (lower_violation | upper_violation).any(dim=-1)
