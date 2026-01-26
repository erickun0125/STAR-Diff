"""
Reward functions for STAR-Diff.
"""
from isaaclab.managers import SceneEntityCfg
import torch

def task_reward(env, actions):
    return torch.zeros(env.num_envs)
