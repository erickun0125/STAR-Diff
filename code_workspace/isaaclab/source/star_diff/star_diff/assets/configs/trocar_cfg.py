# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trocar Configuration for STAR-Diff surgical manipulation.

The trocar is the entry port for surgical instruments. It is modeled as
a simple rigid body that represents the Remote Center of Motion (RCM) point.

Note: If no USD file is available, the trocar is spawned as a small cylinder.
"""

from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils


# Trocar Configuration (as a simple cylinder marker)
# The trocar represents the RCM point that the instrument must pass through
TROCAR_CFG = RigidObjectCfg(
    spawn=sim_utils.CylinderCfg(
        radius=0.01,
        height=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,  # Static, not affected by physics
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.5, 1.0),  # Blue color for visibility
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.3),  # Default position (will be randomized in env)
    ),
)
