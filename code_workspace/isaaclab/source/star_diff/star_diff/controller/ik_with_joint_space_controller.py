import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

class IKWithJointSpaceController:
    """
    Wrapper for Differential IK Controller.
    Takes SE(3) pose command and outputs joint positions.
    """
    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.ik_controller = DifferentialIKController(cfg, num_envs, device)

    def reset(self, env_ids: torch.Tensor = None):
        self.ik_controller.reset(env_ids)

    def compute(self, ee_pos_des: torch.Tensor, ee_quat_des: torch.Tensor, 
                current_ee_pos: torch.Tensor, current_ee_quat: torch.Tensor, 
                jacobian: torch.Tensor, current_joint_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute joint position targets.
        Args:
            ee_pos_des: Desired EE position (N, 3)
            ee_quat_des: Desired EE orientation (N, 4)
            current_ee_pos: Current EE position (N, 3)
            current_ee_quat: Current EE orientation (N, 4)
            jacobian: EE Jacobian (N, 6, NJ)
            current_joint_pos: Current joint positions (N, NJ)
        Returns:
            joint_pos_target: (N, NJ)
        """
        # Set command directly as desired pose
        # We assume absolute pose command for simplicity in this wrapper,
        # or we adapt based on the internal DiffIK config.
        # But wait, DiffIK 'set_command' logic expects command in relative or absolute delta format 
        # depending on config.
        # Ideally, our Wrapper input IS the desired SE(3) pose. 
        # So we should configure DiffIK to be in "pose_abs" mode if possible, 
        # OR we compute the delta ourselves and send it.
        # Looking at DiffIK code: if command_type == "pose_rel", it expects delta.
        # If "pose_abs" (not explicitly supported in standard DiffIK options usually? 
        # Standard DiffIK supports "position", "pose_rel" (relative), "pose_abs" (maybe?)).
        # Actually checking DiffIK code again: 
        # if self.cfg.command_type == "position": ...
        # else: (implies pose)
        #   if self.cfg.use_relative_mode: ... apply_delta_pose
        #   else: ... self.ee_pos_des = command[:, 0:3]...
        
        # So if use_relative_mode=False, we can pass absolute pose.
        
        # Construct command tensor (N, 7)
        command = torch.cat([ee_pos_des, ee_quat_des], dim=-1)
        
        self.ik_controller.set_command(command, current_ee_pos, current_ee_quat)
        
        joint_pos_target = self.ik_controller.compute(current_ee_pos, current_ee_quat, jacobian, current_joint_pos)
        
        return joint_pos_target
