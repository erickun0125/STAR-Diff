# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
STAR-Diff: Surgical Trocar-Adaptive, RCM-aware Diffusion Policy

A custom IsaacLab extension for laparoscopic surgery simulation and
diffusion policy learning with Remote Center of Motion (RCM) constraints.
"""

import os

import toml

# Conveniences to other module directories via relative paths
STAR_DIFF_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

STAR_DIFF_ASSETS_DIR = os.path.join(STAR_DIFF_EXT_DIR, "assets")
"""Path to the extension assets directory (USD files, etc.)."""

STAR_DIFF_METADATA = toml.load(os.path.join(STAR_DIFF_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = STAR_DIFF_METADATA["package"]["version"]

# Import submodules
from . import assets
from . import configs
from . import controller
from . import envs
from . import policy
from . import demo_collector
from . import utils

# Re-export commonly used items
from .assets.configs.fr5_cfg import FR5_CFG
from .assets.configs.trocar_cfg import TROCAR_CFG
from .configs.controller_cfg import (
    JointImpedanceControllerCfg,
    IKWithJointSpaceControllerCfg,
    ImpedanceControllerCfg,
    OperationalSpaceControllerWrapperCfg,
)
from .configs.policy_cfg import DiffusionPolicyCfg
from .policy.base_policy import BasePolicy
from .policy.diffusion_wrapper import DiffusionPolicyWrapper

__all__ = [
    # Version
    "__version__",
    # Paths
    "STAR_DIFF_EXT_DIR",
    "STAR_DIFF_ASSETS_DIR",
    # Submodules
    "assets",
    "configs",
    "controller",
    "envs",
    "policy",
    "demo_collector",
    "utils",
    # Asset Configs
    "FR5_CFG",
    "TROCAR_CFG",
    # Controller Configs
    "JointImpedanceControllerCfg",
    "IKWithJointSpaceControllerCfg",
    "ImpedanceControllerCfg",
    "OperationalSpaceControllerWrapperCfg",
    # Policy
    "DiffusionPolicyCfg",
    "BasePolicy",
    "DiffusionPolicyWrapper",
]
