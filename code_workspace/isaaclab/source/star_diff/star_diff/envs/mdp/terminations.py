"""
Termination conditions.
"""
import torch

def check_termination(env, actions):
    return torch.zeros(env.num_envs, dtype=torch.bool)
