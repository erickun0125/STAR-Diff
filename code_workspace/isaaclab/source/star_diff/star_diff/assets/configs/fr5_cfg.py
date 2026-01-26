"""
FR5 Articulation Configuration.
"""
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from star_diff import STAR_DIFF_ASSETS_DIR

FR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{STAR_DIFF_ASSETS_DIR}/fr5.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), # Updated in env
    ),
    actuators={}, # Will rely on USD definitions or default
)

