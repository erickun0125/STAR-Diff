# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory generation utilities for demonstration collection.

This module provides trajectory generation functions for smooth motion
between waypoints during surgical demonstration collection.
"""

from __future__ import annotations

import torch
from typing import Literal


class TrajectoryGenerator:
    """Trajectory generator for smooth motion between waypoints.

    This class provides methods to generate smooth trajectories between
    start and end poses using various interpolation methods.

    Args:
        num_envs: Number of parallel environments.
        device: Device for tensor operations (e.g., "cuda:0").
        default_duration: Default duration for trajectories in seconds.
        dt: Time step for trajectory sampling.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        default_duration: float = 1.0,
        dt: float = 0.01,
    ):
        self._num_envs = num_envs
        self._device = device
        self._default_duration = default_duration
        self._dt = dt

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def device(self) -> str:
        """Device for tensor operations."""
        return self._device

    def generate_linear(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate linear trajectory between start and end positions.

        Args:
            start: Start positions. Shape: (num_envs, dim) or (dim,).
            end: End positions. Shape: (num_envs, dim) or (dim,).
            num_steps: Number of trajectory steps. If None, computed from duration/dt.

        Returns:
            Trajectory tensor of shape (num_envs, num_steps, dim).
        """
        if num_steps is None:
            num_steps = int(self._default_duration / self._dt)

        # Ensure tensors are on correct device
        start = start.to(self._device)
        end = end.to(self._device)

        # Handle broadcasting for single position input
        if start.dim() == 1:
            start = start.unsqueeze(0).expand(self._num_envs, -1)
        if end.dim() == 1:
            end = end.unsqueeze(0).expand(self._num_envs, -1)

        dim = start.shape[-1]

        # Linear interpolation parameter t ∈ [0, 1]
        t = torch.linspace(0, 1, num_steps, device=self._device)  # (num_steps,)
        t = t.view(1, num_steps, 1)  # (1, num_steps, 1)

        # Expand start and end for broadcasting
        start = start.unsqueeze(1)  # (num_envs, 1, dim)
        end = end.unsqueeze(1)  # (num_envs, 1, dim)

        # Linear interpolation: p(t) = start + t * (end - start)
        trajectory = start + t * (end - start)  # (num_envs, num_steps, dim)

        return trajectory

    def generate_min_jerk(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate minimum-jerk trajectory between start and end positions.

        Minimum-jerk trajectory provides smooth motion with zero velocity
        and acceleration at endpoints. Uses the polynomial:
            p(t) = start + (end - start) * (10t³ - 15t⁴ + 6t⁵)

        Args:
            start: Start positions. Shape: (num_envs, dim) or (dim,).
            end: End positions. Shape: (num_envs, dim) or (dim,).
            num_steps: Number of trajectory steps. If None, computed from duration/dt.

        Returns:
            Trajectory tensor of shape (num_envs, num_steps, dim).
        """
        if num_steps is None:
            num_steps = int(self._default_duration / self._dt)

        # Ensure tensors are on correct device
        start = start.to(self._device)
        end = end.to(self._device)

        # Handle broadcasting for single position input
        if start.dim() == 1:
            start = start.unsqueeze(0).expand(self._num_envs, -1)
        if end.dim() == 1:
            end = end.unsqueeze(0).expand(self._num_envs, -1)

        dim = start.shape[-1]

        # Interpolation parameter t ∈ [0, 1]
        t = torch.linspace(0, 1, num_steps, device=self._device)  # (num_steps,)

        # Minimum-jerk smooth step function: s(t) = 10t³ - 15t⁴ + 6t⁵
        t3 = t**3
        t4 = t**4
        t5 = t**5
        s = 10 * t3 - 15 * t4 + 6 * t5  # (num_steps,)

        s = s.view(1, num_steps, 1)  # (1, num_steps, 1)

        # Expand start and end for broadcasting
        start = start.unsqueeze(1)  # (num_envs, 1, dim)
        end = end.unsqueeze(1)  # (num_envs, 1, dim)

        # Minimum-jerk interpolation
        trajectory = start + s * (end - start)  # (num_envs, num_steps, dim)

        return trajectory

    def generate_via_points(
        self,
        waypoints: torch.Tensor,
        num_steps_per_segment: int | None = None,
        method: Literal["linear", "min_jerk"] = "min_jerk",
    ) -> torch.Tensor:
        """Generate trajectory through multiple waypoints.

        Args:
            waypoints: Waypoint positions. Shape: (num_envs, num_waypoints, dim).
            num_steps_per_segment: Steps per segment. If None, uses default duration.
            method: Interpolation method ("linear" or "min_jerk").

        Returns:
            Trajectory tensor of shape (num_envs, total_steps, dim).
        """
        waypoints = waypoints.to(self._device)

        if num_steps_per_segment is None:
            num_steps_per_segment = int(self._default_duration / self._dt)

        num_waypoints = waypoints.shape[1]
        num_segments = num_waypoints - 1

        if num_segments <= 0:
            # Single waypoint, just return it
            return waypoints

        # Generate trajectory for each segment
        trajectories = []
        for i in range(num_segments):
            start = waypoints[:, i]
            end = waypoints[:, i + 1]

            if method == "linear":
                segment = self.generate_linear(start, end, num_steps_per_segment)
            else:  # min_jerk
                segment = self.generate_min_jerk(start, end, num_steps_per_segment)

            # Exclude last point of each segment (except final) to avoid duplicates
            if i < num_segments - 1:
                segment = segment[:, :-1]

            trajectories.append(segment)

        # Concatenate all segments
        return torch.cat(trajectories, dim=1)

    def generate_circular_arc(
        self,
        center: torch.Tensor,
        radius: float,
        start_angle: torch.Tensor,
        end_angle: torch.Tensor,
        height: torch.Tensor | float = 0.0,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate circular arc trajectory in the XY plane.

        Args:
            center: Center of the arc. Shape: (num_envs, 2) or (2,).
            radius: Radius of the arc.
            start_angle: Start angle in radians. Shape: (num_envs,) or scalar.
            end_angle: End angle in radians. Shape: (num_envs,) or scalar.
            height: Z-coordinate. Shape: (num_envs,) or scalar.
            num_steps: Number of trajectory steps.

        Returns:
            3D trajectory tensor of shape (num_envs, num_steps, 3).
        """
        if num_steps is None:
            num_steps = int(self._default_duration / self._dt)

        # Ensure tensors are on correct device
        center = center.to(self._device)

        # Handle scalar inputs
        if isinstance(start_angle, (int, float)):
            start_angle = torch.full((self._num_envs,), start_angle, device=self._device)
        else:
            start_angle = start_angle.to(self._device)

        if isinstance(end_angle, (int, float)):
            end_angle = torch.full((self._num_envs,), end_angle, device=self._device)
        else:
            end_angle = end_angle.to(self._device)

        if isinstance(height, (int, float)):
            height = torch.full((self._num_envs,), height, device=self._device)
        else:
            height = height.to(self._device)

        # Handle broadcasting for center
        if center.dim() == 1:
            center = center.unsqueeze(0).expand(self._num_envs, -1)

        # Interpolation parameter
        t = torch.linspace(0, 1, num_steps, device=self._device)  # (num_steps,)

        # Angle interpolation
        angle = start_angle.unsqueeze(1) + t.unsqueeze(0) * (
            end_angle - start_angle
        ).unsqueeze(1)  # (num_envs, num_steps)

        # Compute XY coordinates
        x = center[:, 0:1] + radius * torch.cos(angle)  # (num_envs, num_steps)
        y = center[:, 1:2] + radius * torch.sin(angle)  # (num_envs, num_steps)
        z = height.unsqueeze(1).expand(-1, num_steps)  # (num_envs, num_steps)

        # Stack to form trajectory
        trajectory = torch.stack([x, y, z], dim=-1)  # (num_envs, num_steps, 3)

        return trajectory

    def get_current_target(
        self,
        trajectory: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Get current target position from trajectory based on step index.

        Args:
            trajectory: Pre-computed trajectory. Shape: (num_envs, num_steps, dim).
            step_idx: Current step index for each environment. Shape: (num_envs,).

        Returns:
            Current target positions. Shape: (num_envs, dim).
        """
        # Clamp step indices to valid range
        max_steps = trajectory.shape[1] - 1
        step_idx = torch.clamp(step_idx, 0, max_steps)

        # Gather current targets
        batch_indices = torch.arange(self._num_envs, device=self._device)
        targets = trajectory[batch_indices, step_idx]

        return targets

    def compute_velocity(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity profile from trajectory using finite differences.

        Args:
            trajectory: Position trajectory. Shape: (num_envs, num_steps, dim).

        Returns:
            Velocity trajectory. Shape: (num_envs, num_steps, dim).
            Note: Last velocity is copied from second-to-last.
        """
        # Finite difference: v[i] = (p[i+1] - p[i]) / dt
        velocity = (trajectory[:, 1:] - trajectory[:, :-1]) / self._dt

        # Pad to maintain shape (copy last velocity)
        velocity = torch.cat([velocity, velocity[:, -1:]], dim=1)

        return velocity

    def compute_acceleration(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute acceleration profile from trajectory using finite differences.

        Args:
            trajectory: Position trajectory. Shape: (num_envs, num_steps, dim).

        Returns:
            Acceleration trajectory. Shape: (num_envs, num_steps, dim).
            Note: First and last accelerations are copied from adjacent values.
        """
        velocity = self.compute_velocity(trajectory)

        # Finite difference: a[i] = (v[i+1] - v[i]) / dt
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / self._dt

        # Pad to maintain shape
        acceleration = torch.cat(
            [acceleration[:, :1], acceleration, acceleration[:, -1:]], dim=1
        )
        acceleration = acceleration[:, :trajectory.shape[1]]  # Trim to original size

        return acceleration
