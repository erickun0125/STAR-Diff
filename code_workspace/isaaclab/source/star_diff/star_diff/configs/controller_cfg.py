from dataclasses import dataclass, MISSING
from isaaclab.controllers import DifferentialIKControllerCfg, JointImpedanceControllerCfg, OperationalSpaceControllerCfg
from isaaclab.utils import configclass

@configclass
class IKWithJointSpaceControllerCfg(DifferentialIKControllerCfg):
    """Configuration for IK with Joint Space Controller."""
    command_type: str = "pose_abs"
    use_relative_mode: bool = False
    # Default params can be overridden

@configclass
class ImpedanceControllerCfg:
    """Configuration for Impedance Controller (IK + Joint Imepdance)."""
    ik_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose_abs",
        use_relative_mode=False
    )
    imp_cfg: JointImpedanceControllerCfg = JointImpedanceControllerCfg(
        command_type="p_abs",
        impedance_mode="fixed",
        stiffness=100.0,
        damping_ratio=1.0, 
    )

@configclass
class OperationalSpaceControllerWrapperCfg(OperationalSpaceControllerCfg):
    """Configuration for Operational Space Controller Wrapper."""
    target_types: list[str] = ["pose_abs"]
    motion_stiffness_task: list[float] = [100.0] * 6
    motion_damping_ratio_task: list[float] = [2.0] * 6
    impedance_mode: str = "fixed"
    inertial_dynamics_decoupling: bool = True
