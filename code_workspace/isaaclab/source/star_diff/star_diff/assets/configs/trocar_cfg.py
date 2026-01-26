"""
Trocar Articulation Configuration.
"""
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from star_diff import STAR_DIFF_ASSETS_DIR

TROCAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{STAR_DIFF_ASSETS_DIR}/trocar.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.0),
    ),
    actuators={},
)

