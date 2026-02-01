# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR5 Surgical Environment Configuration for Peg Transfer Task."""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from ..mdp.actions.actions_cfg import RCMAwareActionCfg
from ...configs.controller_cfg import IKWithJointSpaceControllerCfg
from ...assets.configs.fr5_cfg import FR5_CFG
from ...assets.configs.trocar_cfg import TROCAR_CFG
from ..mdp.events import randomize_trocar_pose
from ..mdp import rewards as star_rewards
from ..mdp import terminations as star_terminations
from ..mdp import observations as star_observations


##
# Scene configuration
##


@configclass
class FR5SurgicalSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with FR5, Trocar, Peg, and Target."""

    # Robots
    robot = FR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Surgical Setup - Trocar
    trocar = TROCAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Trocar")

    # Task Objects - Peg (Red Cube)
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.06),  # Peg dimensions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.1, 0.03)),
    )

    # Target Position (Green Cube - Visual Marker)
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # Static marker
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.03)),
    )

    # Ground Plane
    ground = sim_utils.GroundPlaneCfg()

    # Camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.8, 0.0, 0.6), rot=(0.8776, 0.0, 0.4794, 0.0)),
        data_types=["rgb"],
        width=256,
        height=256,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
    )

    # Lights
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))


##
# Action configuration
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # RCM-aware action with Variant 4 (gamma_rel, p_abs) and IK controller
    arm_action: RCMAwareActionCfg = RCMAwareActionCfg(
        asset_name="robot",
        trocar_asset_name="trocar",
        rcm_variant=4,
        controller=IKWithJointSpaceControllerCfg(
            command_type="pose",
            ik_method="pinv",
        ),
        body_name="psm_tool_tip_link",
        scale=1.0,
        debug_vis=False,
    )


##
# Observation configuration
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy network."""

        # Robot Joint State
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # End-Effector State (World Frame)
        ee_pos = ObsTerm(
            func=star_observations.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "psm_tool_tip_link"},
        )
        ee_quat = ObsTerm(
            func=star_observations.ee_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "psm_tool_tip_link"},
        )

        # Trocar State
        trocar_pos = ObsTerm(
            func=star_observations.trocar_pos_w,
            params={"asset_cfg": SceneEntityCfg("trocar")},
        )
        trocar_spherical = ObsTerm(
            func=star_observations.trocar_spherical_params,
            params={
                "asset_cfg": SceneEntityCfg("trocar"),
                "ref_point": (0.5, 0.0, 0.0),
            },
        )

        # Peg State
        peg_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("peg")},
        )

        # Target State
        target_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("target")},
        )

        # Last Action
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CameraCfg(ObsGroup):
        """Observations for camera images."""

        rgb = ObsTerm(
            func=star_observations.tiled_camera_rgb,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    camera: CameraCfg = CameraCfg()


##
# Reward configuration
##


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task rewards
    reaching_peg = RewTerm(
        func=star_rewards.reaching_peg,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "peg_cfg": SceneEntityCfg("peg"),
            "body_name": "psm_tool_tip_link",
            "std": 0.1,
        },
    )

    grasping_peg = RewTerm(
        func=star_rewards.grasping_peg,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "peg_cfg": SceneEntityCfg("peg"),
            "body_name": "psm_tool_tip_link",
            "grasp_threshold": 0.03,
        },
    )

    lifting_peg = RewTerm(
        func=star_rewards.lifting_peg,
        weight=2.0,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "min_height": 0.08,
        },
    )

    reaching_target = RewTerm(
        func=star_rewards.reaching_target,
        weight=1.5,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "target_cfg": SceneEntityCfg("target"),
            "std": 0.1,
        },
    )

    placing_peg = RewTerm(
        func=star_rewards.placing_peg,
        weight=10.0,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "target_cfg": SceneEntityCfg("target"),
            "success_threshold": 0.02,
        },
    )

    # Penalty rewards
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    rcm_violation = RewTerm(
        func=star_rewards.rcm_violation_penalty,
        weight=-5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "trocar_cfg": SceneEntityCfg("trocar"),
            "body_name": "psm_tool_tip_link",
            "tolerance": 0.01,
        },
    )


##
# Termination configuration
##


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Task success
    task_success = DoneTerm(
        func=star_terminations.task_success,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "target_cfg": SceneEntityCfg("target"),
            "success_threshold": 0.02,
        },
    )

    # Safety terminations
    rcm_violation = DoneTerm(
        func=star_terminations.rcm_violation,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "trocar_cfg": SceneEntityCfg("trocar"),
            "body_name": "psm_tool_tip_link",
            "max_deviation": 0.05,
        },
    )

    out_of_bounds = DoneTerm(
        func=star_terminations.out_of_bounds,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "body_name": "psm_tool_tip_link",
            "bounds": {
                "x": (0.1, 0.8),
                "y": (-0.4, 0.4),
                "z": (0.0, 0.5),
            },
        },
    )


##
# Event configuration
##


@configclass
class EventCfg:
    """Configuration for environment events."""

    # Reset events
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.01, 0.01),
        },
    )

    reset_peg = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("peg"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

    randomize_trocar = EventTerm(
        func=randomize_trocar_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("trocar"),
            # Specification: r ∈ [0, 5]cm, θ ∈ [60°, 120°], φ ∈ [0°, 360°]
            "r_range": (0.0, 0.05),  # 0-5cm (meters)
            "theta_range": (math.pi / 3, 2 * math.pi / 3),  # 60°-120° (radians)
            "phi_range": (-math.pi, math.pi),  # Full 360° rotation (radians)
            "ref_point": (0.5, 0.0, 0.0),
        },
    )


##
# Environment configuration
##


@configclass
class FR5SurgicalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the FR5 Surgical Peg Transfer environment."""

    # Scene settings
    scene: FR5SurgicalSceneCfg = FR5SurgicalSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Simulation settings
        self.decimation = 4  # 4 physics steps per env step
        self.episode_length_s = 10.0  # 10 seconds per episode
        self.sim.dt = 1 / 120.0  # 120Hz physics

        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Viewer settings
        self.viewer.eye = (1.5, 0.0, 1.0)
        self.viewer.lookat = (0.4, 0.0, 0.0)


##
# Play configuration (smaller scale for evaluation)
##


@configclass
class FR5SurgicalEnvCfg_PLAY(FR5SurgicalEnvCfg):
    """Configuration for playing/evaluating the FR5 Surgical environment."""

    def __post_init__(self):
        super().__post_init__()

        # Reduce number of environments for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        # Disable observation noise
        self.observations.policy.enable_corruption = False
