"""
RCM-aware observations.
"""
from isaaclab.managers import SceneEntityCfg
import torch

def trocar_position(env, actions):
    """Observation function for trocar position."""
    return torch.zeros(env.num_envs, 3)
