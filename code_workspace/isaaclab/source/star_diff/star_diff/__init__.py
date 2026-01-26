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

STAR_DIFF_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
"""Path to the extension assets directory."""

STAR_DIFF_METADATA = toml.load(os.path.join(STAR_DIFF_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = STAR_DIFF_METADATA["package"]["version"]

# Import submodules
from . import assets, envs
