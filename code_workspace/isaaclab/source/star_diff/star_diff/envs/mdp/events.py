import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_matrix
import math

def randomize_trocar_pose(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, r_range: tuple[float, float], theta_range: tuple[float, float], phi_range: tuple[float, float], ref_point: tuple[float, float, float] = (0.5, 0.0, 0.0)):
    """
    Randomize Trocar Pose based on spherical coordinates.
    p_T = O_ref + [r sin(th) cos(phi), r sin(th) sin(phi), r cos(th)]
    """
    # Resolve asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    num_envs = len(env_ids)
    
    # Sample spherical coords
    r = torch.empty(num_envs, device=env.device).uniform_(*r_range)
    theta = torch.empty(num_envs, device=env.device).uniform_(*theta_range)
    phi = torch.empty(num_envs, device=env.device).uniform_(*phi_range)
    
    # Compute position
    ref = torch.tensor(ref_point, device=env.device).unsqueeze(0)
    
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    
    p_T = ref + torch.stack([x, y, z], dim=-1)
    
    # Orientation: Look at ref_point?
    # Vector from p_T to ref_point: v = ref - p_T
    # Align Z axis of Trocar to v? Or Identity?
    # Let's use Identity for now unless specified otherwise, 
    # as Trocar might just be a visual/physical ring.
    q_T = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)
    
    # Set state
    # Articulation root state: [pos, quat, lin_vel, ang_vel]
    # We set pos/quat.
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = p_T
    root_state[:, 3:7] = q_T
    
    asset.write_root_state_to_sim(root_state, env_ids)
