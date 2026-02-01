# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for STAR-Diff surgical environments.

These functions compute observations for the RCM-aware surgical manipulation task.
All functions return tensors with shape (num_envs, ...) for batch processing.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
) -> torch.Tensor:
    """End-effector position in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        body_name: Name of the end-effector body.

    Returns:
        End-effector position with shape (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    return asset.data.body_pos_w[:, body_idx]


def ee_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
) -> torch.Tensor:
    """End-effector quaternion (wxyz) in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        body_name: Name of the end-effector body.

    Returns:
        End-effector quaternion with shape (num_envs, 4).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    return asset.data.body_quat_w[:, body_idx]


def ee_vel_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
) -> torch.Tensor:
    """End-effector linear and angular velocity in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        body_name: Name of the end-effector body.

    Returns:
        End-effector velocity (linear + angular) with shape (num_envs, 6).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    return asset.data.body_vel_w[:, body_idx]


def trocar_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Trocar position in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the trocar asset.

    Returns:
        Trocar position with shape (num_envs, 3).
    """
    # Trocar can be either RigidObject or Articulation
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def trocar_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Trocar quaternion (wxyz) in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the trocar asset.

    Returns:
        Trocar quaternion with shape (num_envs, 4).
    """
    # Trocar can be either RigidObject or Articulation
    asset = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def trocar_spherical_params(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ref_point: tuple[float, float, float] = (0.5, 0.0, 0.0),
) -> torch.Tensor:
    """Trocar position as spherical coordinates (r, theta, phi) relative to reference point.

    The spherical coordinates are defined as:
        - r: radial distance from reference point
        - theta: polar angle (from +z axis)
        - phi: azimuthal angle (from +x axis in xy plane)

    Formula: p_T = O_ref + [r sin(θ) cos(φ), r sin(θ) sin(φ), r cos(θ)]

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the trocar asset.
        ref_point: Reference point for spherical coordinates.

    Returns:
        Spherical parameters (r, theta, phi) with shape (num_envs, 3).
    """
    # Trocar can be either RigidObject or Articulation
    asset = env.scene[asset_cfg.name]
    trocar_pos = asset.data.root_pos_w  # (num_envs, 3)

    # Reference point as tensor
    ref = torch.tensor(ref_point, device=env.device).unsqueeze(0)  # (1, 3)

    # Relative position from reference
    rel_pos = trocar_pos - ref  # (num_envs, 3)

    # Compute spherical coordinates
    r = torch.norm(rel_pos, dim=-1, keepdim=True)  # (num_envs, 1)

    # Avoid division by zero
    r_safe = torch.clamp(r, min=1e-6)

    # theta = arccos(z / r), in range [0, pi]
    theta = torch.acos(torch.clamp(rel_pos[:, 2:3] / r_safe, -1.0, 1.0))

    # phi = atan2(y, x), in range [-pi, pi]
    phi = torch.atan2(rel_pos[:, 1:2], rel_pos[:, 0:1])

    return torch.cat([r, theta, phi], dim=-1)  # (num_envs, 3)


def ee_to_trocar_vec(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    trocar_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
) -> torch.Tensor:
    """Vector from end-effector to trocar position.

    This is useful for computing the RCM constraint direction.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        trocar_cfg: Configuration for the trocar asset.
        body_name: Name of the end-effector body.

    Returns:
        Vector from EE to trocar with shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trocar: Articulation = env.scene[trocar_cfg.name]

    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    trocar_pos = trocar.data.root_pos_w

    return trocar_pos - ee_pos


def rcm_deviation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    trocar_cfg: SceneEntityCfg,
    body_name: str = "ee_link",
) -> torch.Tensor:
    """Distance from the end-effector's axis to the trocar point (RCM deviation).

    The RCM constraint requires the instrument's longitudinal axis to pass through
    the trocar point. This function computes how far the axis deviates from the trocar.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot asset.
        trocar_cfg: Configuration for the trocar asset.
        body_name: Name of the end-effector body.

    Returns:
        RCM deviation distance with shape (num_envs, 1).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trocar: Articulation = env.scene[trocar_cfg.name]

    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    ee_quat = robot.data.body_quat_w[:, body_idx]
    trocar_pos = trocar.data.root_pos_w

    # Get the z-axis direction of the end-effector (instrument axis)
    # Quaternion to rotation matrix's z-column
    # For quaternion q = (w, x, y, z), z-axis is:
    # z_axis = [2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)]
    w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
    z_axis = torch.stack(
        [
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )  # (num_envs, 3)

    # Vector from EE to trocar
    ee_to_trocar = trocar_pos - ee_pos  # (num_envs, 3)

    # Project ee_to_trocar onto z_axis
    proj_length = torch.sum(ee_to_trocar * z_axis, dim=-1, keepdim=True)  # (num_envs, 1)
    proj_vec = proj_length * z_axis  # (num_envs, 3)

    # Perpendicular component is the deviation
    perp_vec = ee_to_trocar - proj_vec  # (num_envs, 3)
    deviation = torch.norm(perp_vec, dim=-1, keepdim=True)  # (num_envs, 1)

    return deviation


def tiled_camera_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """RGB image from tiled camera.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the camera sensor.

    Returns:
        RGB images with shape (num_envs, H, W, 3) normalized to [0, 1].
    """
    camera: TiledCamera = env.scene[sensor_cfg.name]
    # Get RGB data and normalize to [0, 1]
    rgb = camera.data.output["rgb"]

    # If data is uint8, convert to float and normalize
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0

    return rgb
