import torch
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

class OperationalSpaceControllerWrapper:
    """
    Wrapper for Operational Space Controller.
    Takes SE(3) pose command and outputs joint torques.
    """
    def __init__(self, cfg: OperationalSpaceControllerCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.osc_controller = OperationalSpaceController(cfg, num_envs, device)

    def reset(self, env_ids: torch.Tensor = None):
        self.osc_controller.reset() # OSC reset doesn't take env_ids in base implementation usually, but checking code it resets all.

    def compute(self, ee_pos_des: torch.Tensor, ee_quat_des: torch.Tensor,
                jacobian: torch.Tensor, current_ee_pose: torch.Tensor, current_ee_vel: torch.Tensor,
                mass_matrix: torch.Tensor, gravity: torch.Tensor = None) -> torch.Tensor:
        """
        Compute joint torques.
        Args:
            ee_pos_des: Desired EE position (N, 3)
            ee_quat_des: Desired EE orientation (N, 4)
            jacobian: EE Jacobian (N, 6, NJ)
            current_ee_pose: Current EE pose (N, 7)
            current_ee_vel: Current EE velocity (N, 6)
            mass_matrix: Mass matrix (N, NJ, NJ)
            gravity: Gravity vectors (N, NJ)
        Returns:
            joint_torques: (N, NJ)
        """
        # Construct command (N, 7) for pose
        # NOTE: OSC might have different command action dims depending on config (stiffness etc).
        # We assume fixed impedance mode for simplicity or that command includes gains if variable.
        # But the Requirement is "common SE(3) pose input". 
        # So we assume the gains are fixed in config or handled internally.
        
        command = torch.cat([ee_pos_des, ee_quat_des], dim=-1)
        
        self.osc_controller.set_command(command, current_ee_pose_b=current_ee_pose)
        
        joint_torques = self.osc_controller.compute(
            jacobian_b=jacobian,
            current_ee_pose_b=current_ee_pose,
            current_ee_vel_b=current_ee_vel,
            mass_matrix=mass_matrix,
            gravity=gravity
        )
        
        return joint_torques
