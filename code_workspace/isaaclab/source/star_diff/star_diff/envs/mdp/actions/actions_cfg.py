from dataclasses import dataclass, MISSING
from isaaclab.managers import ActionTermCfg
from .action import RCMAwareAction
from .....configs.controller_cfg import IKWithJointSpaceControllerCfg, ImpedanceControllerCfg, OperationalSpaceControllerWrapperCfg

@dataclass
class RCMAwareActionCfg(ActionTermCfg):
    class_type: type = RCMAwareAction
    
    rcm_variant: int = MISSING # 1, 2, 3, or 4
    """The variant of the RCM parameterization (1-4)."""
    
    controller: IKWithJointSpaceControllerCfg | ImpedanceControllerCfg | OperationalSpaceControllerWrapperCfg = MISSING
    """The controller configuration to use."""
    
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    trocar_asset_name: str = "trocar"
    """Name of the trocar asset."""
    
    body_name: str = "ee_link" # Modify based on actual FR5 link name
    """Name of the end-effector body."""
    
    # Action dimension is 4 for all variants
    scale: float | dict[str, float] = 1.0
    """Scale factor for the actions."""
    
    debug_vis: bool = False
    """Whether to visualize the target pose."""
