# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding the FR5 robot to an Isaac Lab environment with OSC."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

# Controller imports
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg

# Import robot configuration
# NOTE: FR5 USD currently has multiple articulation roots - using Franka as fallback
# To use FR5: 1) Fix the USD file (remove extra articulation roots)
#             2) Change the import below to use FR5_CFG from star_diff
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Uncomment below and comment out the Franka import when FR5 USD is fixed:
# from star_diff import FR5_CFG


@configclass
class FR5SceneCfg(InteractiveSceneCfg):
    """Designs the scene with the FR5 robot."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot: Using Franka as placeholder until FR5 USD is fixed
    # To use FR5, replace FRANKA_PANDA_HIGH_PD_CFG with FR5_CFG and update actuators
    fr5_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/FR5",
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Print joint names to verify ordering
    print(f"[INFO]: FR5 Joint Names: {scene['fr5_robot'].data.joint_names}")
    print(f"[INFO]: FR5 Joint Position Limits: {scene['fr5_robot'].data.joint_pos_limits}")
    
    # ---------------------------------------------------------
    # Initialize OSC Controller
    # ---------------------------------------------------------
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        motion_stiffness_task=[40.0] * 6,
        motion_damping_ratio_task=[1.5] * 6,
        nullspace_control="none", # FR5 is 6-DoF, so no nullspace available
        nullspace_stiffness=10.0,
        nullspace_damping_ratio=1.0,
    )
    controller = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)
    
    # Identify End-Effector
    # We assume the last body is the end-effector
    ee_idx = scene["fr5_robot"].data.body_names.index(scene["fr5_robot"].data.body_names[-1])
    print(f"[INFO]: Robot Body Names: {scene['fr5_robot'].data.body_names}")
    print(f"[INFO]: Is Fixed Base: {scene['fr5_robot'].is_fixed_base}")
    print(f"[INFO]: Using End-Effector Body: {scene['fr5_robot'].data.body_names[ee_idx]} (Index: {ee_idx})")

    # Markers for desired pose (Optional, could add visualization)
    from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    
    # Target frame marker (Green) - using the frame config but overriding scale if needed
    # Note: FRAME_MARKER_CFG uses a USD file props. We can use it directly.
    target_marker_cfg = FRAME_MARKER_CFG.copy()
    target_marker_cfg.prim_path = "/World/Visuals/TargetPose"
    target_marker = VisualizationMarkers(target_marker_cfg)
    
    # Current frame marker (Red) - distinct visual if possible, or another instance
    current_marker_cfg = FRAME_MARKER_CFG.copy()
    current_marker_cfg.prim_path = "/World/Visuals/CurrentPose"
    current_marker = VisualizationMarkers(current_marker_cfg)

    # Buffers
    initial_ee_pose = None
    
    while simulation_app.is_running():
        # reset logic
        if count % 1000 == 0:
            count = 0
            
            # Reset Robot State
            root_state = scene["fr5_robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["fr5_robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["fr5_robot"].write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = scene["fr5_robot"].data.default_joint_pos.clone()
            joint_vel = scene["fr5_robot"].data.default_joint_vel.clone()
            scene["fr5_robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            controller.reset()
            print("[INFO]: Resetting FR5 Robot state...")
            
            # Re-capture initial pose after reset
            # We need to step once to update data
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            
            # Capture initial pose in Base Frame
            root_pos_w = scene["fr5_robot"].data.root_pos_w
            root_quat_w = scene["fr5_robot"].data.root_quat_w
            ee_pose_w = scene["fr5_robot"].data.body_state_w[:, ee_idx, :7]
            
            initial_ee_pos_b, initial_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pose_w[:, :3], ee_pose_w[:, 3:]
            )
            initial_ee_pose = torch.cat([initial_ee_pos_b, initial_ee_quat_b], dim=-1)

        # ---------------------------------------------------------
        # OSC Control Loop
        # ---------------------------------------------------------
        
        # 1. Define Target Pose (Waving Motion) in Base Frame
        # ---------------------------------------------------
        if initial_ee_pose is not None:
            # Get current Base state (needed for transforms)
            root_pos_w = scene["fr5_robot"].data.root_pos_w
            root_quat_w = scene["fr5_robot"].data.root_quat_w
            
            desired_pose_b = initial_ee_pose.clone()
            
            # Create a waving motion
            # X: Move slightly forward/backward (fixed offset here)
            desired_pose_b[:, 0] += 0
            # Y: Side to side wave (Sine wave)
            desired_pose_b[:, 1] += 0
            # Z: Up and down (Sine wave)
            desired_pose_b[:, 2] -= 0.05
            
            # Set the command for the controller (in Base Frame)
            controller.set_command(desired_pose_b)

            # 2. Compute Torques
            # ------------------
            # Get Jacobian in World Frame
            jacobian_w = scene["fr5_robot"].root_physx_view.get_jacobians()
            print(f"[INFO]: Jacobian Shape: {jacobian_w.shape}, EE Index: {ee_idx}")
            
            # Adjust index for fixed base articulation
            # Check if dimensions allow direct indexing, otherwise fallback to shifted
            if ee_idx < jacobian_w.shape[1]:
                jacobi_ee_idx = ee_idx
            else:
                jacobi_ee_idx = ee_idx - 1
            
            print(f"[INFO]: Using Jacobi Index: {jacobi_ee_idx}")
            jacobian_w_ee = jacobian_w[:, jacobi_ee_idx, :, :]
            
            # Transform Jacobian J_w -> J_b
            # J_b = [R_inv 0; 0 R_inv] * J_w
            base_rot_inv = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
            jacobian_b = jacobian_w_ee.clone()
            jacobian_b[:, :3, :] = torch.bmm(base_rot_inv, jacobian_w_ee[:, :3, :])
            jacobian_b[:, 3:, :] = torch.bmm(base_rot_inv, jacobian_w_ee[:, 3:, :])
            
            # Get Current EE state in World Frame
            curr_ee_pose_w = scene["fr5_robot"].data.body_state_w[:, ee_idx, :7]
            curr_ee_vel_w = scene["fr5_robot"].data.body_state_w[:, ee_idx, 7:]
            
            # Transform Current EE state: World -> Base
            curr_ee_pos_b, curr_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, curr_ee_pose_w[:, :3], curr_ee_pose_w[:, 3:7]
            )
            curr_ee_pose_b = torch.cat([curr_ee_pos_b, curr_ee_quat_b], dim=-1)
            
            # Transform Velocity: World -> Base
            # v_b = R_inv * v_w, w_b = R_inv * w_w (Assuming static base)
            curr_ee_vel_b = curr_ee_vel_w.clone()
            curr_ee_vel_b[:, :3] = torch.bmm(base_rot_inv, curr_ee_vel_w[:, :3].unsqueeze(-1)).squeeze(-1)
            curr_ee_vel_b[:, 3:] = torch.bmm(base_rot_inv, curr_ee_vel_w[:, 3:].unsqueeze(-1)).squeeze(-1)

            # Mass matrix and Gravity
            mass_matrix = scene["fr5_robot"].root_physx_view.get_generalized_mass_matrices()
            gravity = scene["fr5_robot"].root_physx_view.get_gravity_compensation_forces()
            
            # Compute efforts
            efforts = controller.compute(
                 jacobian_b=jacobian_b,
                 current_ee_pose_b=curr_ee_pose_b,
                 current_ee_vel_b=curr_ee_vel_b,
                 mass_matrix=mass_matrix,
                 gravity=gravity,
                 current_joint_pos=scene["fr5_robot"].data.joint_pos,
                 current_joint_vel=scene["fr5_robot"].data.joint_vel,
            )
            
            # 3. Apply Action
            # ---------------
            scene["fr5_robot"].set_joint_effort_target(efforts)

            # 4. Visualize Markers (Needs World Frame)
            # --------------------
            # Transform Desired Pose Back to World Frame for visualization
            desired_pos_w, desired_quat_w = math_utils.combine_frame_transforms(
                root_pos_w, root_quat_w, desired_pose_b[:, :3], desired_pose_b[:, 3:7]
            )
            
            # Target Pose (World)
            target_marker.visualize(
                translations=desired_pos_w,
                orientations=desired_quat_w
            )
            # Current Pose (World)
            current_marker.visualize(
                translations=curr_ee_pose_w[:, :3],
                orientations=curr_ee_pose_w[:, 3:7]
            )
            if count<5:
                print("Target", desired_pos_w, desired_quat_w)
                print("Current", curr_ee_pose_w)
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set camera view for better visibility
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])
    
    # Design scene
    scene_cfg = FR5SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
