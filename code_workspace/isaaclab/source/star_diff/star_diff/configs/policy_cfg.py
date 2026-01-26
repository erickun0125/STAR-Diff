"""
Policy Configurations.
"""
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class DiffusionPolicyCfg:
    """Configuration for Robomimic Diffusion Policy."""
    ckpt_path: str = ""
    """Path to the robomimic checkpoint file."""
    
    device: str = "cuda:0"
    """Device to load the model on."""
    
    obs_key_mapping: Dict[str, str] = field(default_factory=lambda: {
        # "isaaclab_key": "robomimic_key"
        "policy_state": "agentview_image", # Example default, should be updated by user
        "ee_pos": "robot0_eef_pos",
        "ee_quat": "robot0_eef_quat"
    })
    """Mapping from IsaacLab observation keys to Robomimic expected keys."""
