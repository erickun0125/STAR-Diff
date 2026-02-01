# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for STAR-Diff surgical environments.

This module contains the action, observation, reward, termination, and event
definitions for the surgical manipulation tasks with RCM constraints.
"""

from . import actions
from . import observations
from . import rewards
from . import terminations
from . import events

# Re-export action classes
from .actions import RCMAwareAction, RCMAwareActionCfg

# Re-export observation functions
from .observations import (
    ee_pos_w,
    ee_quat_w,
    ee_vel_w,
    trocar_pos_w,
    trocar_quat_w,
    trocar_spherical_params,
    ee_to_trocar_vec,
    rcm_deviation,
    tiled_camera_rgb,
)

# Re-export reward functions
from .rewards import (
    reaching_peg,
    grasping_peg,
    lifting_peg,
    reaching_target,
    placing_peg,
    rcm_violation_penalty,
    joint_velocity_penalty,
    joint_torque_penalty,
)

# Re-export termination functions
from .terminations import (
    task_success,
    rcm_violation,
    out_of_bounds,
    peg_dropped,
    joint_limit_violation,
)

# Re-export event functions
from .events import randomize_trocar_pose

__all__ = [
    # Submodules
    "actions",
    "observations",
    "rewards",
    "terminations",
    "events",
    # Action classes
    "RCMAwareAction",
    "RCMAwareActionCfg",
    # Observation functions
    "ee_pos_w",
    "ee_quat_w",
    "ee_vel_w",
    "trocar_pos_w",
    "trocar_quat_w",
    "trocar_spherical_params",
    "ee_to_trocar_vec",
    "rcm_deviation",
    "tiled_camera_rgb",
    # Reward functions
    "reaching_peg",
    "grasping_peg",
    "lifting_peg",
    "reaching_target",
    "placing_peg",
    "rcm_violation_penalty",
    "joint_velocity_penalty",
    "joint_torque_penalty",
    # Termination functions
    "task_success",
    "rcm_violation",
    "out_of_bounds",
    "peg_dropped",
    "joint_limit_violation",
    # Event functions
    "randomize_trocar_pose",
]
