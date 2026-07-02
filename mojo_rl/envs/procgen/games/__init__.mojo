"""Procgen games (maze). See `docs/PROCGEN_PORT.md`."""

from .maze import (
    MazeGame,
    MazeAssets,
    DIST_EASY,
    DIST_HARD,
    DIST_MEMORY,
    world_dim_for,
)
from .maze_env import MazeEnv, StepResult
from .maze_gym_env import MazeGymEnv, MazeState, MazeAction
from .chaser import (
    ChaserGame,
    ChaserAssets,
    DIST_EXTREME,
    LARGE_ORB,
    ENEMY_EGG,
    MAZE_WALL,
    ORB,
)
from .chaser_env import ChaserEnv
from .chaser_gym_env import ChaserGymEnv, ChaserState, ChaserAction
from .heist import (
    HeistGame,
    HeistAssets,
    heist_world_dim,
    LOCKED_DOOR,
    KEY,
    EXIT,
)
from .heist_env import HeistEnv
from .heist_gym_env import HeistGymEnv, HeistState, HeistAction
from .bigfish import BigfishGame, BigfishAssets, FISH
from .bigfish_env import BigfishEnv
from .bigfish_gym_env import BigfishGymEnv, BigfishState, BigfishAction
from .leaper import (
    LeaperGame,
    LeaperAssets,
    leaper_world_dim,
    LOG,
    ROAD,
    WATER,
    CAR,
    FINISH_LINE,
)
from .leaper_env import LeaperEnv
from .leaper_gym_env import LeaperGymEnv, LeaperState, LeaperAction
from .miner import MinerGame, MinerAssets, BOULDER, DIAMOND, EXIT, DIRT
from .miner_env import MinerEnv
from .miner_gym_env import MinerGymEnv, MinerState, MinerAction
from .starpilot import (
    StarpilotGame,
    StarpilotAssets,
    FLYER,
    METEOR,
    CLOUD,
    TURRET,
    FAST_FLYER,
)
from .starpilot_env import StarpilotEnv
from .starpilot_gym_env import StarpilotGymEnv, StarpilotState, StarpilotAction
from .plunder import PlunderGame, PlunderAssets, SHIP, PANEL, PLAYER_BULLET
from .plunder_env import PlunderEnv
from .plunder_gym_env import PlunderGymEnv, PlunderState, PlunderAction
from .fruitbot import FruitbotGame, FruitbotAssets
# NOTE: fruitbot's per-game object ids (GOOD_OBJ/BAD_OBJ/BARRIER/LOCKED_DOOR/LOCK)
# are NOT re-exported here — LOCKED_DOOR collides with heist's (different value).
# Import them from `mojo_rl.envs.procgen.games.fruitbot` directly.
from .fruitbot_env import FruitbotEnv
from .fruitbot_gym_env import FruitbotGymEnv, FruitbotState, FruitbotAction
from .bossfight import BossfightGame, BossfightAssets
# NOTE: bossfight's generic object ids (PLAYER_BULLET/BOSS/SHIELDS/ENEMY_BULLET/
# LASER_TRAIL/REFLECTED_BULLET/BARRIER) are NOT re-exported here — PLAYER_BULLET
# collides with plunder's and BARRIER with fruitbot's. Import them from
# `mojo_rl.envs.procgen.games.bossfight` directly.
from .bossfight_env import BossfightEnv
from .bossfight_gym_env import BossfightGymEnv, BossfightState, BossfightAction
from .coinrun import CoinrunGame
# NOTE: coinrun's generic object ids (GOAL/SAW/ENEMY/WALL_MID/CRATE/...) are NOT
# re-exported here — GOAL/CRATE/etc collide with other games'. Import them from
# `mojo_rl.envs.procgen.games.coinrun` directly.
