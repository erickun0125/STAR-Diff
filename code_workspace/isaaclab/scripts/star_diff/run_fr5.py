# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR5 robot with PSM gripper - Keyboard teleoperation demo.

Keyboard controls:
    W/S: Move along x-axis (forward/backward)
    A/D: Move along y-axis (left/right)
    Q/E: Move along z-axis (up/down)
    Z/X: Rotate around x-axis (roll)
    T/G: Rotate around y-axis (pitch)
    C/V: Rotate around z-axis (yaw)
    K: Toggle gripper (open/close)
    L: Reset pose
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="FR5 robot with PSM gripper teleoperation using keyboard."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sensitivity", type=float, default=0.005, help="Position sensitivity for keyboard control.")
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

# Controller imports
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg

# Keyboard device
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg

# Markers
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

# Import robot configuration
from star_diff import FR5_CFG


@configclass
class FR5SceneCfg(InteractiveSceneCfg):
    """Designs the scene with the FR5 robot."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # FR5 Robot with PSM gripper - effort control for OSC (zero stiffness/damping)
    fr5_robot = FR5_CFG.replace(
        prim_path="{ENV_REGEX_NS}/FR5",
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["j[1-6]"],
                stiffness=0.0,
                damping=0.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["psm_tool_gripper.*"],
                stiffness=1000.0,  # Position control for gripper
                damping=10.0,
            ),
        },
    )


def create_small_frame_marker(prim_path: str, scale: float = 0.1) -> VisualizationMarkers:
    """Create a small frame marker for visualization."""
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(scale, scale, scale),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0

    # Print robot info
    print(f"[INFO]: FR5 Joint Names: {scene['fr5_robot'].data.joint_names}")
    print(f"[INFO]: Robot Body Names: {scene['fr5_robot'].data.body_names}")
    print(f"[INFO]: Is Fixed Base: {scene['fr5_robot'].is_fixed_base}")

    # ---------------------------------------------------------
    # Initialize OSC Controller (for 6-DOF arm only)
    # ---------------------------------------------------------
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        motion_stiffness_task=[100.0] * 6,
        motion_damping_ratio_task=[1.0] * 6,
        nullspace_control="none",
    )
    controller = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Find end-effector body index (psm_tool_tip_link)
    ee_body_name = "psm_tool_tip_link"
    ee_idx = scene["fr5_robot"].data.body_names.index(ee_body_name)
    print(f"[INFO]: Using End-Effector Body: {ee_body_name} (Index: {ee_idx})")

    # Find gripper joint indices
    joint_names = scene["fr5_robot"].data.joint_names
    gripper1_idx = joint_names.index("psm_tool_gripper1_joint")
    gripper2_idx = joint_names.index("psm_tool_gripper2_joint")
    print(f"[INFO]: Gripper1 Index: {gripper1_idx}, Gripper2 Index: {gripper2_idx}")

    # Gripper limits (from USD: gripper1 [-0.524, 0], gripper2 [0, 0.524])
    gripper1_open = -0.524
    gripper1_close = 0.0
    gripper2_open = 0.524
    gripper2_close = 0.0

    # ---------------------------------------------------------
    # Initialize Keyboard Controller (only works in GUI mode)
    # ---------------------------------------------------------
    keyboard = None
    try:
        keyboard_cfg = Se3KeyboardCfg(
            pos_sensitivity=args_cli.sensitivity,
            rot_sensitivity=args_cli.sensitivity * 2,
            gripper_term=True,
            sim_device=sim.device,
        )
        keyboard = Se3Keyboard(keyboard_cfg)
        print(keyboard)
    except (AttributeError, RuntimeError) as e:
        print(f"[WARNING]: Keyboard not available (headless mode?): {e}")
        print("[INFO]: Running in autonomous mode without keyboard control.")

    # ---------------------------------------------------------
    # Visualization Markers (smaller size)
    # ---------------------------------------------------------
    target_marker = create_small_frame_marker("/World/Visuals/TargetPose", scale=0.08)
    current_marker = create_small_frame_marker("/World/Visuals/CurrentPose", scale=0.08)

    # ---------------------------------------------------------
    # Buffers
    # ---------------------------------------------------------
    desired_pose_b = None
    gripper_open = False

    # Reset and initialize
    def reset_robot():
        nonlocal desired_pose_b, gripper_open

        # Reset robot state
        root_state = scene["fr5_robot"].data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        scene["fr5_robot"].write_root_pose_to_sim(root_state[:, :7])
        scene["fr5_robot"].write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = scene["fr5_robot"].data.default_joint_pos.clone()
        joint_vel = scene["fr5_robot"].data.default_joint_vel.clone()
        scene["fr5_robot"].write_joint_state_to_sim(joint_pos, joint_vel)

        scene.reset()
        controller.reset()
        if keyboard is not None:
            keyboard.reset()
        gripper_open = False

        # Step once to update data
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Capture current EE pose as initial target
        root_pos_w = scene["fr5_robot"].data.root_pos_w
        root_quat_w = scene["fr5_robot"].data.root_quat_w
        ee_pose_w = scene["fr5_robot"].data.body_state_w[:, ee_idx, :7]

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pose_w[:, :3], ee_pose_w[:, 3:]
        )
        desired_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        print("[INFO]: Robot reset. Initial EE pose captured.")

    # Initial reset
    reset_robot()

    # Add reset callback to keyboard (if available)
    if keyboard is not None:
        keyboard.add_callback("L", reset_robot)
        print("\n" + "=" * 60)
        print("FR5 + PSM Gripper Teleoperation Ready!")
        print("=" * 60)
        print("Controls:")
        print("  W/S: Move X    A/D: Move Y    Q/E: Move Z")
        print("  Z/X: Roll      T/G: Pitch     C/V: Yaw")
        print("  K: Toggle Gripper    L: Reset")
        print("=" * 60 + "\n")
    else:
        print("\n[INFO]: Running without keyboard. Robot will maintain initial pose.")

    while simulation_app.is_running():
        # Get keyboard command (if available)
        if keyboard is not None:
            keyboard_cmd = keyboard.advance()
            delta_pos = keyboard_cmd[:3]  # Position delta
            delta_rot = keyboard_cmd[3:6]  # Rotation delta (rotation vector)
            gripper_cmd = keyboard_cmd[6].item() if len(keyboard_cmd) > 6 else 1.0
            # Update gripper state: K toggles, -1 = close, +1 = open
            gripper_open = gripper_cmd < 0
        else:
            # No keyboard: zero delta, gripper closed
            delta_pos = torch.zeros(3, device=sim.device)
            delta_rot = torch.zeros(3, device=sim.device)

        if desired_pose_b is not None:
            # Get current base frame
            root_pos_w = scene["fr5_robot"].data.root_pos_w
            root_quat_w = scene["fr5_robot"].data.root_quat_w

            # Apply position delta
            desired_pose_b[:, :3] += delta_pos.unsqueeze(0)

            # Apply rotation delta (convert rotation vector to quaternion delta)
            if torch.norm(delta_rot) > 1e-6:
                # Convert rotation vector to quaternion
                angle = torch.norm(delta_rot)
                axis = delta_rot / angle
                half_angle = angle / 2
                delta_quat = torch.cat([
                    torch.cos(half_angle).unsqueeze(0),
                    axis * torch.sin(half_angle)
                ])
                delta_quat = delta_quat.unsqueeze(0).to(desired_pose_b.device)

                # Apply rotation: q_new = q_delta * q_current
                current_quat = desired_pose_b[:, 3:7]
                desired_pose_b[:, 3:7] = math_utils.quat_mul(delta_quat, current_quat)

            # Set OSC command
            controller.set_command(desired_pose_b)

            # ---------------------------------------------------------
            # Compute OSC torques for arm (j1-j6)
            # ---------------------------------------------------------
            jacobian_w = scene["fr5_robot"].root_physx_view.get_jacobians()

            # Jacobian index adjustment for fixed base
            jacobi_ee_idx = ee_idx - 1 if ee_idx > 0 else ee_idx
            jacobian_w_ee = jacobian_w[:, jacobi_ee_idx, :, :6]  # Only first 6 joints (arm)

            # Transform Jacobian to base frame
            base_rot_inv = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
            jacobian_b = jacobian_w_ee.clone()
            jacobian_b[:, :3, :] = torch.bmm(base_rot_inv, jacobian_w_ee[:, :3, :])
            jacobian_b[:, 3:, :] = torch.bmm(base_rot_inv, jacobian_w_ee[:, 3:, :])

            # Current EE state
            curr_ee_pose_w = scene["fr5_robot"].data.body_state_w[:, ee_idx, :7]
            curr_ee_vel_w = scene["fr5_robot"].data.body_state_w[:, ee_idx, 7:]

            # Transform to base frame
            curr_ee_pos_b, curr_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, curr_ee_pose_w[:, :3], curr_ee_pose_w[:, 3:7]
            )
            curr_ee_pose_b = torch.cat([curr_ee_pos_b, curr_ee_quat_b], dim=-1)

            curr_ee_vel_b = curr_ee_vel_w.clone()
            curr_ee_vel_b[:, :3] = torch.bmm(base_rot_inv, curr_ee_vel_w[:, :3].unsqueeze(-1)).squeeze(-1)
            curr_ee_vel_b[:, 3:] = torch.bmm(base_rot_inv, curr_ee_vel_w[:, 3:].unsqueeze(-1)).squeeze(-1)

            # Mass matrix and gravity (only for arm joints)
            mass_matrix = scene["fr5_robot"].root_physx_view.get_generalized_mass_matrices()
            gravity = scene["fr5_robot"].root_physx_view.get_gravity_compensation_forces()

            # Compute arm torques
            arm_efforts = controller.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=curr_ee_pose_b,
                current_ee_vel_b=curr_ee_vel_b,
                mass_matrix=mass_matrix[:, :6, :6],
                gravity=gravity[:, :6],
                current_joint_pos=scene["fr5_robot"].data.joint_pos[:, :6],
                current_joint_vel=scene["fr5_robot"].data.joint_vel[:, :6],
            )

            # ---------------------------------------------------------
            # Gripper control (synchronized: both grippers open/close together)
            # ---------------------------------------------------------
            if gripper_open:
                # Open gripper: gripper1 → -0.524, gripper2 → +0.524
                gripper1_target = gripper1_open
                gripper2_target = gripper2_open
            else:
                # Close gripper: gripper1 → 0, gripper2 → 0
                gripper1_target = gripper1_close
                gripper2_target = gripper2_close

            # Create gripper position targets
            gripper_targets = torch.tensor(
                [[gripper1_target, gripper2_target]],
                dtype=torch.float32,
                device=sim.device
            ).expand(scene.num_envs, -1)

            # ---------------------------------------------------------
            # Apply actions
            # ---------------------------------------------------------
            # Apply arm torques
            scene["fr5_robot"].set_joint_effort_target(arm_efforts, joint_ids=[0, 1, 2, 3, 4, 5])

            # Apply gripper position targets
            scene["fr5_robot"].set_joint_position_target(gripper_targets, joint_ids=[gripper1_idx, gripper2_idx])

            # ---------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------
            # Transform desired pose to world frame
            desired_pos_w, desired_quat_w = math_utils.combine_frame_transforms(
                root_pos_w, root_quat_w, desired_pose_b[:, :3], desired_pose_b[:, 3:7]
            )

            target_marker.visualize(translations=desired_pos_w, orientations=desired_quat_w)
            current_marker.visualize(translations=curr_ee_pose_w[:, :3], orientations=curr_ee_pose_w[:, 3:7])

        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.3])

    # Design scene
    scene_cfg = FR5SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
