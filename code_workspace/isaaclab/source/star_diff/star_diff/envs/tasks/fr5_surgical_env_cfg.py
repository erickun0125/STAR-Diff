from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

from ..mdp.actions.actions_cfg import RCMAwareActionCfg
from ...configs.controller_cfg import IKWithJointSpaceControllerCfg
from ...assets.configs.fr5_cfg import FR5_CFG
from ...assets.configs.trocar_cfg import TROCAR_CFG
from ..mdp.events import randomize_trocar_pose
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp

@configclass
class FR5SurgicalSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with FR5, Trocar, and Peg."""
    
    # Robots
    robot = FR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Surgical Setup
    trocar = TROCAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Trocar")
    
    # Task Objects (Cube as Peg)
    peg = SceneEntityCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
    )
    # Target (Marker)
    target = SceneEntityCfg(
        prim_path="{ENV_REGEX_NS}/Target",
    )
    
    # Sensors
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.5), rot=(0.0, 0.0, 1.0, 0.0)), # Approx looking at workspace
        data_types=["rgb"],
        width=128,
        height=128,
    )

    # Peg Asset Spawn (Simple Cube)
    # Note: SceneEntityCfg doesn't spawn. We need to add it to the scene config as an asset if we want spawn.
    # We'll use RigidObjectCfg for Peg.
    pass

from isaaclab.assets import RigidObjectCfg

@configclass
class FR5SurgicalSceneCfg(InteractiveSceneCfg):
    robot = FR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    trocar = TROCAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Trocar")
    
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), color=(1.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )
    
    # Camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.8, 0.0, 0.6), lookat=(0.5, 0.0, 0.0)),
        data_types=["rgb"],
        width=256,
        height=256,
    )
    
    # Lights
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))

@configclass
class ActionsCfg:
    """Action dictionary."""
    # RCM Action with customizable variant and controller
    # Defaulting to Variant 4 and IK Controller
    arm_action = RCMAwareActionCfg(
        asset_name="robot",
        trocar_asset_name="trocar",
        rcm_variant=4,
        controller=IKWithJointSpaceControllerCfg(
            command_type="pose_abs",
        ),
    )

@configclass
class ObservationsCfg:
    """Observation dictionary."""
    @configclass
    class PolicyStateCfg(ObsGroup):
        """Observations for policy state."""
        # Robot Joint State
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # End Effector State (World Frame)
        ee_pos = ObsTerm(
            func=mdp.body_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link")} # Ensure body name is correct
        )
        ee_quat = ObsTerm(
            func=mdp.body_quat,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link")}
        )
        
        # Trocar State (World Frame)
        trocar_pos = ObsTerm(
            func=mdp.root_pos_w, 
            params={"asset_cfg": SceneEntityCfg("trocar")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenation = False 

    policy_state = PolicyStateCfg()

    @configclass
    class TiledCameraCfg(ObsGroup):
        """Observations for Tiled Camera."""
        rgb = ObsTerm(
            func=mdp.generated_image,
            params={"asset_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "convert_to_float": True}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenation = False
            
    tiled_camera = TiledCameraCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    randomize_trocar = EventTerm(
        func=randomize_trocar_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("trocar"),
            "r_range": (0.1, 0.2), # Example ranges
            "theta_range": (0.1, 0.5),
            "phi_range": (-0.5, 0.5),
            "ref_point": (0.5, 0.0, 0.0),
        },
    )

@configclass
class FR5SurgicalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for FR5 Surgical Environment."""
    scene = FR5SurgicalSceneCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()
    
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Set viewer params
        self.viewer.eye = (1.5, 0.0, 1.0)
        self.viewer.lookat = (0.5, 0.0, 0.0)
