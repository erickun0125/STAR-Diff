import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers import JointImpedanceController, JointImpedanceControllerCfg

class ImpedanceController:
    """
    Impedance Controller (SE(3) -> IK -> Joint Impedance).
    Takes SE(3) pose command, converts to joint targets using IK, then tracks with Joint Impedance.
    """
    def __init__(self, ik_cfg: DifferentialIKControllerCfg, imp_cfg: JointImpedanceControllerCfg, 
                 num_envs: int, device: str, dof_pos_limits: torch.Tensor):
        self.num_envs = num_envs
        self.device = device
        
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs, device)
        self.imp_controller = JointImpedanceController(imp_cfg, num_envs, dof_pos_limits, device)

    def reset(self, env_ids: torch.Tensor = None):
        self.ik_controller.reset(env_ids)
        # JointImpedanceController reset_idx might not be implemented or needed depending on version,
        # but let's call it if available or just pass.
        # self.imp_controller.reset_idx(env_ids) 
        pass

    def compute(self, ee_pos_des: torch.Tensor, ee_quat_des: torch.Tensor,
                current_ee_pos: torch.Tensor, current_ee_quat: torch.Tensor,
                jacobian: torch.Tensor, current_joint_pos: torch.Tensor, current_joint_vel: torch.Tensor,
                mass_matrix: torch.Tensor = None, gravity: torch.Tensor = None) -> torch.Tensor:
        """
        Compute joint torques.
        """
        # 1. Run IK to get joint targets
        ik_command = torch.cat([ee_pos_des, ee_quat_des], dim=-1)
        self.ik_controller.set_command(ik_command, current_ee_pos, current_ee_quat)
        joint_pos_des = self.ik_controller.compute(current_ee_pos, current_ee_quat, jacobian, current_joint_pos)
        
        # 2. Run Joint Impedance
        self.imp_controller.set_command(joint_pos_des)
        joint_torques = self.imp_controller.compute(
            dof_pos=current_joint_pos,
            dof_vel=current_joint_vel,
            mass_matrix=mass_matrix,
            gravity=gravity
        )
        
        return joint_torques
