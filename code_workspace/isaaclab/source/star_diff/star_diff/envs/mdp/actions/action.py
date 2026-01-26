from __future__ import annotations

import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils.math import (
    quat_from_angle_axis, quat_mul, quat_conjugate, 
    quat_rotate_inverse, quat_rotate, combine_frame_transforms
)
from isaaclab.assets import Articulation

from .actions_cfg import RCMAwareActionCfg
from .....controller.ik_with_joint_space_controller import IKWithJointSpaceController
from .....controller.impedance_controller import ImpedanceController
from .....controller.operational_space_controller import OperationalSpaceControllerWrapper
from .....configs.controller_cfg import IKWithJointSpaceControllerCfg, ImpedanceControllerCfg, OperationalSpaceControllerWrapperCfg

def rotation_align(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """
    Computes quaternion that rotates v_from to v_to.
    Both vectors should be normalized.
    """
    # Using cross product for axis and dot product for angle
    # q = [cos(theta/2), sin(theta/2) * axis]
    # But a robust way is:
    # half_v = (v_from + v_to) / ||v_from + v_to||
    # q = [v_from . half_v, v_from x half_v]
    # This aligns v_from to v_to.
    
    # Standard quaternion from two vectors:
    # axis = cross(u, v)
    # angle = acos(dot(u, v))
    # This can be unstable near 180 degrees.
    
    # Efficient approach:
    # q.xyz = cross(u, v)
    # q.w = dot(u, v) + sqrt(norm(u)^2 * norm(v)^2)
    # Then normalize.
    
    # w = dot(v_from, v_to) + 1.0 (since they are unit vectors)
    dot = torch.sum(v_from * v_to, dim=-1, keepdim=True)
    
    # Check for antiparallel case (dot near -1)
    # If dot < -0.99999, we need an arbitrary axis perpendicular to v_from.
    # For now assume mostly forward facing for surgical tools.
    
    xyz = torch.linalg.cross(v_from, v_to)
    w = dot + 1.0
    
    q_unnorm = torch.cat([w, xyz], dim=-1)
    return torch.nn.functional.normalize(q_unnorm, dim=-1)

class RCMAwareAction(ActionTerm):
    """Action term for RCM-aware control."""
    
    cfg: RCMAwareActionCfg
    
    def __init__(self, cfg: RCMAwareActionCfg, env):
        super().__init__(cfg, env)
        
        # Get references to assets
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._trocar: Articulation = env.scene[cfg.trocar_asset_name]
        
        # Get body index
        self._body_idx = self._asset.find_bodies(cfg.body_name)[0][0]
        
        # Initialize Controller
        if isinstance(cfg.controller, IKWithJointSpaceControllerCfg):
            self._controller = IKWithJointSpaceController(cfg.controller, env.num_envs, env.device)
            self._output_type = "joint_pos"
        elif isinstance(cfg.controller, ImpedanceControllerCfg):
            # ImpedanceController wrapper requires limits for internal JointImpedance
            self._controller = ImpedanceController(
                cfg.controller.ik_cfg, 
                cfg.controller.imp_cfg, 
                env.num_envs, 
                env.device, 
                self._asset.data.soft_joint_pos_limits
            )
            self._output_type = "joint_effort"
        elif isinstance(cfg.controller, OperationalSpaceControllerWrapperCfg):
            self._controller = OperationalSpaceControllerWrapper(cfg.controller, env.num_envs, env.device)
            self._output_type = "joint_effort"
        else:
            raise ValueError(f"Unknown controller config type: {type(cfg.controller)}")

        # Buffers for actions
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Constants
        self._e_z = torch.zeros(self.num_envs, 3, device=self.device)
        self._e_z[:, 2] = 1.0

    def process_actions(self, actions: torch.Tensor):
        # Store raw actions
        self._raw_actions[:] = actions
        
        # 1. Get Trocar State
        trocar_pos = self._trocar.data.root_pos_w
        trocar_quat = self._trocar.data.root_quat_w # {T} orientation
        
        # 2. Parse Actions based on Variant
        # Assume actions are scaled properly prior to this or we apply simple scaling here if cfg.scale is set.
        # We'll skip explicit scale config usage here for brevity, relying on the input being correct scale.
        
        ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)
        
        if self.cfg.rcm_variant == 1:
            # (d_rel, R_rel)
            # R_rel assumed to be Axis-Angle (3 dims) + d_rel (1 dim) -> 4 dims
            # Or Euler? Summary table says "Relative rotation".
            # Let's assume input is [d_rel, rx, ry, rz] (axis-angle or euler).
            # We'll treat [rx, ry, rz] as axis-angle vector.
            d_rel = actions[:, 0:1]
            rot_rel_vec = actions[:, 1:4]
            
            # Convert rot_rel to quat
            angle = torch.norm(rot_rel_vec, dim=-1, keepdim=True)
            axis = torch.nn.functional.normalize(rot_rel_vec, dim=-1)
            # handle zero angle
            mask = angle.squeeze(-1) < 1e-6
            axis[mask] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            
            q_rel = quat_from_angle_axis(angle.squeeze(-1), axis)
            
            # R_E = R_T * R_rel
            ee_quat_des = quat_mul(trocar_quat, q_rel)
            
            # p_E = p_T + d * (R_E * e_z)
            dir_vec = quat_rotate(ee_quat_des, self._e_z)
            ee_pos_des = trocar_pos + d_rel * dir_vec
            
        elif self.cfg.rcm_variant == 2:
            # (gamma_rel, p_rel)
            # p_rel (3 dims), gamma_rel (1 dim)
            gamma_rel = actions[:, 0]
            p_rel = actions[:, 1:4]
            
            # p_E = T_T * p_rel
            # T_T: Trocar Frame to World. Pos: trocar_pos, Rot: trocar_quat
            ee_pos_des = quat_rotate(trocar_quat, p_rel) + trocar_pos
            
            # v = (p_E - p_T) normalized
            diff = ee_pos_des - trocar_pos
            v = torch.nn.functional.normalize(diff, dim=-1)
            
            # R_E = RotationAlign(e_z, v) * R_z(gamma)
            r_base = rotation_align(self._e_z, v)
            
            # R_z(gamma)
            axis_z = self._e_z
            q_gamma = quat_from_angle_axis(gamma_rel, axis_z)
            
            ee_quat_des = quat_mul(r_base, q_gamma)
            
        elif self.cfg.rcm_variant == 3:
            # (d_rel, R_abs)
            # R_abs (3 dims axis-angle), d_rel (1 dim)
            d_rel = actions[:, 0:1]
            rot_abs_vec = actions[:, 1:4]
            
            angle = torch.norm(rot_abs_vec, dim=-1, keepdim=True)
            axis = torch.nn.functional.normalize(rot_abs_vec, dim=-1)
            mask = angle.squeeze(-1) < 1e-6
            axis[mask] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            
            ee_quat_des = quat_from_angle_axis(angle.squeeze(-1), axis)
            
            # p_E = p_T + d * (R_E * e_z)
            dir_vec = quat_rotate(ee_quat_des, self._e_z)
            ee_pos_des = trocar_pos + d_rel * dir_vec
            
        elif self.cfg.rcm_variant == 4:
            # (gamma_rel, p_abs)
            # p_abs (3 dims), gamma_rel (1 dim)
            gamma_rel = actions[:, 0]
            p_abs = actions[:, 1:4]
            
            ee_pos_des = p_abs
            
            # v = (p_abs - p_T).normalized
            diff = ee_pos_des - trocar_pos
            v = torch.nn.functional.normalize(diff, dim=-1)
            
            r_base = rotation_align(self._e_z, v)
            
            axis_z = self._e_z
            q_gamma = quat_from_angle_axis(gamma_rel, axis_z)
            
            ee_quat_des = quat_mul(r_base, q_gamma)
            
        else:
            raise ValueError(f"Invalid RCM Variant: {self.cfg.rcm_variant}")

        # 3. Call Controller
        # Get robot state
        # Jacobian from `body_idx`
        jacobians = self._asset.data.body_jacobians[:, self._body_idx, :, :] # (N, 6, NJ)
        
        current_ee_pos = self._asset.data.body_pos_w[:, self._body_idx]
        current_ee_quat = self._asset.data.body_quat_w[:, self._body_idx]
        current_ee_pose = torch.cat([current_ee_pos, current_ee_quat], dim=-1) # (N, 7)
        current_ee_vel = self._asset.data.body_vel_w[:, self._body_idx] # (N, 6)
        
        current_joint_pos = self._asset.data.joint_pos
        current_joint_vel = self._asset.data.joint_vel
        
        mass_matrix = self._asset.data.mass_matrix # (N, NJ, NJ)
        
        if self._output_type == "joint_pos":
            # IK Controller
            targets = self._controller.compute(
                ee_pos_des, ee_quat_des,
                current_ee_pos, current_ee_quat,
                jacobians, current_joint_pos
            )
            self._joint_pos_target = targets
            
        elif self._output_type == "joint_effort":
            # Impedance or OSC
            targets = self._controller.compute(
                ee_pos_des=ee_pos_des, 
                ee_quat_des=ee_quat_des,
                jacobian=jacobians,
                current_ee_pose=current_ee_pose,
                current_ee_vel=current_ee_vel,
                mass_matrix=mass_matrix,
                current_joint_pos=current_joint_pos,
                current_joint_vel=current_joint_vel
            )
            self._joint_effort_target = targets

    def apply_actions(self):
        if self._output_type == "joint_pos":
            self._asset.set_joint_position_target(self._joint_pos_target)
        elif self._output_type == "joint_effort":
            self._asset.set_joint_effort_target(self._joint_effort_target)
