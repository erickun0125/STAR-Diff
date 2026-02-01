from dataclasses import dataclass, field
from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.utils import configclass


@configclass
class JointImpedanceControllerCfg:
    """Configuration for custom Joint Impedance Controller.

    This is a custom implementation since IsaacLab doesn't provide JointImpedanceController.
    Implements: τ = K_p * (q_des - q) - K_d * q_dot + gravity_compensation
    """

    stiffness: float = 100.0
    """Joint stiffness (Nm/rad)."""

    damping_ratio: float = 1.0
    """Damping ratio for critical damping. K_d = 2 * damping_ratio * sqrt(K_p * I)."""

    use_gravity_compensation: bool = True
    """Whether to add gravity compensation torques."""


@configclass
class IKWithJointSpaceControllerCfg(DifferentialIKControllerCfg):
    """Configuration for IK with Joint Space Controller."""

    command_type: str = "pose"  # "position" or "pose"
    use_relative_mode: bool = False
    ik_method: str = "pinv"  # "pinv", "svd", "trans", "dls"


@configclass
class ImpedanceControllerCfg:
    """Configuration for Impedance Controller (IK + Joint Impedance)."""

    ik_cfg: DifferentialIKControllerCfg = field(
        default_factory=lambda: DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="pinv",
        )
    )
    imp_cfg: JointImpedanceControllerCfg = field(
        default_factory=lambda: JointImpedanceControllerCfg(
            stiffness=100.0,
            damping_ratio=1.0,
        )
    )

@configclass
class OperationalSpaceControllerWrapperCfg(OperationalSpaceControllerCfg):
    """Configuration for Operational Space Controller Wrapper."""
    target_types: list[str] = ["pose_abs"]
    motion_stiffness_task: list[float] = [100.0] * 6
    motion_damping_ratio_task: list[float] = [2.0] * 6
    impedance_mode: str = "fixed"
    inertial_dynamics_decoupling: bool = True
