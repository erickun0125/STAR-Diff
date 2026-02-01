# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR5 Articulation Configuration for STAR-Diff surgical manipulation.

This module defines the articulation configuration for the Fairino FR5 robot
with end-effector attached, suitable for RCM-constrained surgical tasks.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

from star_diff import STAR_DIFF_ASSETS_DIR


# FR5 Robot with End-Effector Configuration
FR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{STAR_DIFF_ASSETS_DIR}/fr5_with_ee_usda.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "j1": 1.5,
            "j2": -1.0,
            "j3": 0.0,
            "j4": -1.0,
            "j5": 0.5,
            "j6": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["j[1-6]"],
            stiffness=4000.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# High PD gains variant for precise position control
FR5_HIGH_PD_CFG = FR5_CFG.copy()
FR5_HIGH_PD_CFG.actuators["arm"].stiffness = 8000.0
FR5_HIGH_PD_CFG.actuators["arm"].damping = 80.0
