# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base Finite State Machine for demonstration collection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Generic, TypeVar

import torch

# Type variable for state enum
StateT = TypeVar("StateT", bound=Enum)


class BaseFSM(ABC, Generic[StateT]):
    """Abstract base class for Finite State Machines.

    This class provides a generic FSM implementation that can be specialized
    for different demonstration collection tasks. The FSM operates on batched
    environments, maintaining separate states for each environment.

    Args:
        num_envs: Number of parallel environments.
        device: Device for tensor operations (e.g., "cuda:0").
        initial_state: Initial state for all environments.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        initial_state: StateT,
    ):
        self._num_envs = num_envs
        self._device = device
        self._initial_state = initial_state

        # State tracking for each environment
        self._current_states: torch.Tensor = torch.full(
            (num_envs,), initial_state.value, dtype=torch.int32, device=device
        )

        # Step counter for each environment (within current state)
        self._state_steps: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )

        # Total step counter for each environment
        self._total_steps: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def device(self) -> str:
        """Device for tensor operations."""
        return self._device

    @property
    def current_states(self) -> torch.Tensor:
        """Current state indices for all environments. Shape: (num_envs,)."""
        return self._current_states

    @property
    def state_steps(self) -> torch.Tensor:
        """Steps spent in current state for each environment. Shape: (num_envs,)."""
        return self._state_steps

    @property
    def total_steps(self) -> torch.Tensor:
        """Total steps since last reset for each environment. Shape: (num_envs,)."""
        return self._total_steps

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset FSM states for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self._current_states[env_ids] = self._initial_state.value
        self._state_steps[env_ids] = 0
        self._total_steps[env_ids] = 0

    def set_state(self, state: StateT, env_ids: torch.Tensor | None = None) -> None:
        """Manually set FSM state for specified environments.

        Args:
            state: Target state to set.
            env_ids: Environment indices to update. If None, updates all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self._current_states[env_ids] = state.value
        self._state_steps[env_ids] = 0

    def get_state_mask(self, state: StateT) -> torch.Tensor:
        """Get boolean mask for environments in a specific state.

        Args:
            state: State to check.

        Returns:
            Boolean tensor of shape (num_envs,) indicating which environments
            are in the specified state.
        """
        return self._current_states == state.value

    @abstractmethod
    def update(self, obs: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Update FSM states based on current observations.

        This method should:
        1. Check transition conditions for each environment
        2. Update states as needed
        3. Generate appropriate waypoints/targets for each state

        Args:
            obs: Dictionary of observation tensors from the environment.

        Returns:
            Dictionary containing:
                - "targets": Target poses/positions for the current step
                - "state_changed": Boolean mask indicating state transitions
                - Any other relevant information
        """
        pass

    @abstractmethod
    def check_transition(
        self, current_state: StateT, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if state transition should occur.

        Args:
            current_state: Current state to check transitions from.
            obs: Current observations.

        Returns:
            Tuple of:
                - should_transition: Boolean tensor (num_envs,) indicating
                  which environments should transition
                - next_state_values: Integer tensor (num_envs,) with next state
                  values for environments that should transition
        """
        pass

    @abstractmethod
    def get_target(
        self, state: StateT, obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get target pose for a given state.

        Args:
            state: Current state.
            obs: Current observations.

        Returns:
            Target pose tensor appropriate for the action space.
        """
        pass
