from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from .action import RCMAwareAction
from ....configs.controller_cfg import (
    IKWithJointSpaceControllerCfg,
    ImpedanceControllerCfg,
    OperationalSpaceControllerWrapperCfg,
)


@configclass
class RCMAwareActionCfg(ActionTermCfg):
    """Configuration for RCM-aware action term."""

    class_type: type = RCMAwareAction

    rcm_variant: int = MISSING
    """The variant of the RCM parameterization (1-4)."""

    controller: IKWithJointSpaceControllerCfg | ImpedanceControllerCfg | OperationalSpaceControllerWrapperCfg = MISSING
    """The controller configuration to use."""

    asset_name: str = "robot"
    """Name of the robot asset."""

    trocar_asset_name: str = "trocar"
    """Name of the trocar asset."""

    body_name: str = "psm_tool_tip_link"
    """Name of the end-effector body."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the actions."""

    debug_vis: bool = False
    """Whether to visualize the target pose."""
