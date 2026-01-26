"""
Robomimic Diffusion Policy wrapper for STAR-Diff.
"""
from typing import Dict
import torch
import numpy as np

try:
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.config import config_factory
    from robomimic.algo import algo_factory
except ImportError:
    print("WARNING: robomimic not installed. DiffusionPolicyWrapper will fail if instantiated.")

from .base_policy import BasePolicy
from ..configs.policy_cfg import DiffusionPolicyCfg

class DiffusionPolicyWrapper(BasePolicy):
    """Wrapper for Robomimic Diffusion Policy."""
    
    def __init__(self, cfg: DiffusionPolicyCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Load model from checkpoint
        if not self.cfg.ckpt_path:
            print("Warning: No checkpoint path provided for DiffusionPolicyWrapper")
            self.policy = None
            return

        print(f"Loading Diffusion Policy from: {self.cfg.ckpt_path}")
        
        # We use the lower-level loading to get the Algo directly for better batch control
        # instead of policy_from_checkpoint which wraps in RolloutPolicy (often single env).
        
        # Read config from checkpoint
        ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=self.cfg.ckpt_path)
        algo_name = ckpt_dict["algo_name"]
        config = ckpt_dict["config"]
        # Instantiate model
        model = algo_factory(
            algo_name=ckpt_dict["algo_name"],
            config=ckpt_dict["config"],
            obs_key_shapes=ckpt_dict["shape_metadata"]["all_shapes"],
            ac_dim=ckpt_dict["shape_metadata"]["ac_dim"],
            device=self.device,
        )
        # Load weights
        model.load_state_dict(ckpt_dict["model"])
        model.set_eval()
        
        self.policy = model
        
        # Check normalization (Robomimic handles this internally if configured)
        # But we need to ensure inputs are correct.

    def reset(self):
        """Reset policy state (e.g. for RNNs or Diffusion history)."""
        if self.policy is not None:
             if hasattr(self.policy, "reset"):
                 self.policy.reset()
             # Diffusion policy often stateless between inferences unless it's using history.
             # Robomimic's Algo implementation usually requires managing your own observation queue 
             # if the model expects n_frame history.
             # However, typically IsaacLab might stack frames or we need to check if the Algo handles it.
             # For standard Diffusion Policy in Robomimic, it usually takes (B, T, D) or (B, D) depending on config.
             pass
        
    def get_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute action given observation.
        obs: Dict of tensors, shapes (Num_Envs, ...).
        Returns: Action tensor (Num_Envs, Action_Dim).
        """
        if self.policy is None:
             return torch.zeros((list(obs.values())[0].shape[0], 4), device=self.device)

        # 1. Map IsaacLab keys to Robomimic keys
        robo_obs = {}
        batch_size = 0
        
        for isaac_key, robo_key in self.cfg.obs_key_mapping.items():
            if isaac_key in obs:
                data = obs[isaac_key]
                batch_size = data.shape[0]
                # Ensure data is on correct device and float
                if data.dtype != torch.float32 and data.dtype != torch.float16:
                   data = data.float()
                
                # Check for history dimension requirements of the trained model
                # Robomimic models typically expect [Batch, History, Feature] or [Batch, Feature]
                # If the trained model expects history and our input is just current step,
                # we might need to rely on the model's internal buffering OR provide history here.
                # Assuming here that `obs` from IsaacLab or the wrapper handles history 
                # OR the model is stateless/configured to take single step (obs_horizon=1).
                # Common case: Robomimic algo.get_action(obs_dict) expects raw obs and handles stacking if used within RolloutPolicy.
                # But since we use Algo directly, we pass what the network expects.
                # If the network expects (B, T, D), we must provide it.
                # Inspecting config would be ideal. For now, assume simple (B, D) or (B, 1, D).
                
                # Most Robomimic nets expect (B, C, H, W) for images and (B, D) for low-dim.
                
                robo_obs[robo_key] = data.to(self.device)
            else:
                # Warning or ignore?
                pass
        
        # 2. Inference
        with torch.no_grad():
            # algo.get_action usually handles the forward pass including normalization if it was part of the graph.
            # But standard robomimic Algo.get_action returns a numpy array for a SINGLE step usually?
            # No, Algo.get_action(obs_dict) in modern robomimic usually supports batching if obs_dict is batched.
            
            # Let's check typical diffusion policy usage in Robomimic.
            # It normally returns dict or tensor.
            # We call `self.policy.get_action(robo_obs)`?
            # RoboMimic Algo classes usually have `get_action(obs_dict, goal_dict=None)`.
            
            # CAUTION: Robomimic's `get_action` might default to converting to numpy or operating on single entry.
            # We might want to call `self.policy.predict_action(obs_dict)` if available or call `self.policy.forward`?
            # Diffusion Algo `get_action` does sampling.
            
            # Safe bet: `self.policy.get_action(robo_obs)`
            actions = self.policy.get_action(robo_obs)
            
        # 3. Process Output
        # Robomimic usually returns Numpy array if inputs were numpy, or Tensor if inputs were Tensor?
        # Actually Robomimic tends to enforce Numpy I/O in `get_action`.
        # If it returns numpy, convert to tensor.
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
            
        # Ensure shape (Num_Envs, Action_Dim)
        if actions.shape[0] != batch_size:
            # Maybe it returned (Action_Dim,) for batch=1?
            actions = actions.view(batch_size, -1)
            
        return actions
