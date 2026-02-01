# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surgical FSM for Peg Transfer demonstration collection.

This FSM implements an 8-state machine for the peg transfer task:
IDLE → APPROACH_PEG → GRASP → LIFT → MOVE_TO_TARGET → LOWER → RELEASE → RETRACT
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import torch

from .base_fsm import BaseFSM


class SurgicalState(Enum):
    """States for the surgical peg transfer FSM."""

    IDLE = 0
    """Initial state, waiting to begin."""

    APPROACH_PEG = auto()
    """Moving towards the peg position."""

    GRASP = auto()
    """Grasping the peg (closing gripper)."""

    LIFT = auto()
    """Lifting the peg to transport height."""

    MOVE_TO_TARGET = auto()
    """Moving the peg towards target position."""

    LOWER = auto()
    """Lowering the peg to placement position."""

    RELEASE = auto()
    """Releasing the peg (opening gripper)."""

    RETRACT = auto()
    """Retracting to safe position after placement."""


class SurgicalFSM(BaseFSM[SurgicalState]):
    """Finite State Machine for surgical peg transfer demonstration.

    This FSM generates waypoints for the peg transfer task, transitioning
    through 8 states to complete the pick-and-place operation.

    Args:
        num_envs: Number of parallel environments.
        device: Device for tensor operations.
        approach_height: Height offset for approaching the peg.
        lift_height: Height to lift the peg during transport.
        grasp_threshold: Distance threshold to consider peg grasped.
        position_threshold: Position tolerance for waypoint completion.
        min_state_steps: Minimum steps to spend in each state.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        approach_height: float = 0.05,
        lift_height: float = 0.15,
        grasp_threshold: float = 0.02,
        position_threshold: float = 0.01,
        min_state_steps: int = 10,
    ):
        super().__init__(num_envs, device, SurgicalState.IDLE)

        self._approach_height = approach_height
        self._lift_height = lift_height
        self._grasp_threshold = grasp_threshold
        self._position_threshold = position_threshold
        self._min_state_steps = min_state_steps

        # Cache for target positions
        self._target_pos = torch.zeros(num_envs, 3, device=device)

        # Gripper state (0 = open, 1 = closed)
        self._gripper_state = torch.zeros(num_envs, 1, device=device)

    @property
    def gripper_state(self) -> torch.Tensor:
        """Current gripper state for each environment. Shape: (num_envs, 1)."""
        return self._gripper_state

    def update(self, obs: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Update FSM states and generate targets.

        Expected observation keys:
            - "ee_pos": End-effector position (num_envs, 3)
            - "peg_pos": Peg position (num_envs, 3)
            - "target_pos": Target position (num_envs, 3)

        Args:
            obs: Dictionary of observation tensors.

        Returns:
            Dictionary containing:
                - "target_pos": Target positions for RCM action (num_envs, 3)
                - "gripper": Gripper commands (num_envs, 1)
                - "state_changed": Boolean mask for state transitions (num_envs,)
                - "current_state": Current state indices (num_envs,)
        """
        # Increment step counters
        self._state_steps += 1
        self._total_steps += 1

        # Track state changes
        state_changed = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Check transitions for each state
        for state in SurgicalState:
            state_mask = self.get_state_mask(state)
            if state_mask.any():
                should_transition, next_states = self.check_transition(state, obs)

                # Only transition for environments in this state
                transition_mask = state_mask & should_transition

                if transition_mask.any():
                    # Get indices of environments that should transition
                    transition_ids = torch.where(transition_mask)[0]

                    # Update states
                    self._current_states[transition_ids] = next_states[transition_ids]
                    self._state_steps[transition_ids] = 0
                    state_changed[transition_ids] = True

        # Generate targets for each state
        self._update_targets(obs)

        return {
            "target_pos": self._target_pos.clone(),
            "gripper": self._gripper_state.clone(),
            "state_changed": state_changed,
            "current_state": self._current_states.clone(),
        }

    def check_transition(
        self, current_state: SurgicalState, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if state transition should occur.

        Args:
            current_state: Current state to check transitions from.
            obs: Current observations.

        Returns:
            Tuple of (should_transition, next_state_values).
        """
        ee_pos = obs.get("ee_pos", torch.zeros(self._num_envs, 3, device=self._device))
        peg_pos = obs.get("peg_pos", torch.zeros(self._num_envs, 3, device=self._device))
        target_pos = obs.get("target_pos", torch.zeros(self._num_envs, 3, device=self._device))

        # Initialize outputs
        should_transition = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        next_states = self._current_states.clone()

        # Minimum steps requirement
        min_steps_met = self._state_steps >= self._min_state_steps

        if current_state == SurgicalState.IDLE:
            # Transition to APPROACH_PEG after minimum steps
            should_transition = min_steps_met
            next_states[should_transition] = SurgicalState.APPROACH_PEG.value

        elif current_state == SurgicalState.APPROACH_PEG:
            # Transition when EE is close to peg approach position
            approach_pos = peg_pos.clone()
            approach_pos[:, 2] += self._approach_height
            dist = torch.norm(ee_pos - approach_pos, dim=-1)
            should_transition = min_steps_met & (dist < self._position_threshold)
            next_states[should_transition] = SurgicalState.GRASP.value

        elif current_state == SurgicalState.GRASP:
            # Transition when EE is close enough to peg (gripper closed)
            dist = torch.norm(ee_pos - peg_pos, dim=-1)
            should_transition = min_steps_met & (dist < self._grasp_threshold)
            next_states[should_transition] = SurgicalState.LIFT.value

        elif current_state == SurgicalState.LIFT:
            # Transition when peg is lifted to transport height
            lift_target_z = peg_pos[:, 2] + self._lift_height
            should_transition = min_steps_met & (ee_pos[:, 2] > lift_target_z - self._position_threshold)
            next_states[should_transition] = SurgicalState.MOVE_TO_TARGET.value

        elif current_state == SurgicalState.MOVE_TO_TARGET:
            # Transition when EE is above target position
            target_above = target_pos.clone()
            target_above[:, 2] += self._lift_height
            dist_xy = torch.norm(ee_pos[:, :2] - target_above[:, :2], dim=-1)
            should_transition = min_steps_met & (dist_xy < self._position_threshold)
            next_states[should_transition] = SurgicalState.LOWER.value

        elif current_state == SurgicalState.LOWER:
            # Transition when EE is close to target placement position
            place_pos = target_pos.clone()
            place_pos[:, 2] += self._approach_height
            dist = torch.norm(ee_pos - place_pos, dim=-1)
            should_transition = min_steps_met & (dist < self._position_threshold)
            next_states[should_transition] = SurgicalState.RELEASE.value

        elif current_state == SurgicalState.RELEASE:
            # Transition after gripper opens (time-based)
            should_transition = self._state_steps >= self._min_state_steps * 2
            next_states[should_transition] = SurgicalState.RETRACT.value

        elif current_state == SurgicalState.RETRACT:
            # Episode complete - stay in RETRACT (external reset needed)
            should_transition = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        return should_transition, next_states

    def get_target(
        self, state: SurgicalState, obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get target position for a given state.

        Args:
            state: Current state.
            obs: Current observations.

        Returns:
            Target position tensor of shape (num_envs, 3).
        """
        peg_pos = obs.get("peg_pos", torch.zeros(self._num_envs, 3, device=self._device))
        target_pos = obs.get("target_pos", torch.zeros(self._num_envs, 3, device=self._device))
        ee_pos = obs.get("ee_pos", torch.zeros(self._num_envs, 3, device=self._device))

        if state == SurgicalState.IDLE:
            # Stay at current position
            return ee_pos.clone()

        elif state == SurgicalState.APPROACH_PEG:
            # Move above the peg
            target = peg_pos.clone()
            target[:, 2] += self._approach_height
            return target

        elif state == SurgicalState.GRASP:
            # Move down to peg position
            return peg_pos.clone()

        elif state == SurgicalState.LIFT:
            # Lift the peg
            target = peg_pos.clone()
            target[:, 2] += self._lift_height
            return target

        elif state == SurgicalState.MOVE_TO_TARGET:
            # Move above target position
            target = target_pos.clone()
            target[:, 2] += self._lift_height
            return target

        elif state == SurgicalState.LOWER:
            # Lower to placement position
            target = target_pos.clone()
            target[:, 2] += self._approach_height
            return target

        elif state == SurgicalState.RELEASE:
            # Stay at current position while releasing
            return ee_pos.clone()

        elif state == SurgicalState.RETRACT:
            # Retract upward
            target = ee_pos.clone()
            target[:, 2] += self._approach_height
            return target

        else:
            return ee_pos.clone()

    def _update_targets(self, obs: dict[str, torch.Tensor]) -> None:
        """Update target positions and gripper states for all environments.

        Args:
            obs: Current observations.
        """
        # Update target positions for each state
        for state in SurgicalState:
            state_mask = self.get_state_mask(state)
            if state_mask.any():
                state_ids = torch.where(state_mask)[0]
                targets = self.get_target(state, obs)
                self._target_pos[state_ids] = targets[state_ids]

        # Update gripper states
        # Gripper closed during GRASP, LIFT, MOVE_TO_TARGET, LOWER
        close_states = [
            SurgicalState.GRASP.value,
            SurgicalState.LIFT.value,
            SurgicalState.MOVE_TO_TARGET.value,
            SurgicalState.LOWER.value,
        ]
        close_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        for state_val in close_states:
            close_mask = close_mask | (self._current_states == state_val)

        self._gripper_state[close_mask] = 1.0
        self._gripper_state[~close_mask] = 0.0

    def is_episode_complete(self) -> torch.Tensor:
        """Check if episodes are complete (reached RETRACT state).

        Returns:
            Boolean tensor indicating complete episodes. Shape: (num_envs,).
        """
        return self.get_state_mask(SurgicalState.RETRACT)
