"""
Base policy interface for STAR-Diff.
"""
from abc import ABC, abstractmethod
import torch

class BasePolicy(ABC):
    """Abstract base class for policies."""
    
    @abstractmethod
    def reset(self):
        """Reset internal state."""
        pass
        
    @abstractmethod
    def get_action(self, obs: dict) -> torch.Tensor:
        """Compute action given observation."""
        pass
