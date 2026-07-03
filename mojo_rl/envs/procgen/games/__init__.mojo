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
from .coinrun import CoinrunGame, CoinrunAssets
# NOTE: coinrun's generic object ids (GOAL/SAW/ENEMY/WALL_MID/CRATE/...) are NOT
# re-exported here — GOAL/CRATE/etc collide with other games'. Import them from
# `mojo_rl.envs.procgen.games.coinrun` directly.
from .coinrun_env import CoinrunEnv
from .coinrun_gym_env import CoinrunGymEnv, CoinrunState, CoinrunAction
from .caveflyer import CaveflyerGame, CaveflyerAssets
# NOTE: caveflyer's generic object ids (GOAL/TARGET/ENEMY/CAVEWALL/...) are NOT
# re-exported here — GOAL/ENEMY/etc collide with other games'. Import them from
# `mojo_rl.envs.procgen.games.caveflyer` directly.
from .caveflyer_env import CaveflyerEnv
from .caveflyer_gym_env import CaveflyerGymEnv, CaveflyerState, CaveflyerAction
from .climber import ClimberGame, ClimberAssets
# NOTE: climber's generic object ids (COIN/ENEMY/WALL_MID/...) stay module-local
# (collide with other games'). Import from `...games.climber` directly.
from .climber_env import ClimberEnv
from .climber_gym_env import ClimberGymEnv, ClimberState, ClimberAction
from .ninja import NinjaGame, NinjaAssets
# NOTE: ninja's generic object ids (GOAL/BOMB/WALL_MID/FIRE/...) stay module-local.
from .ninja_env import NinjaEnv
from .ninja_gym_env import NinjaGymEnv, NinjaState, NinjaAction
from .jumper import JumperGame, JumperAssets
# NOTE: jumper's generic object ids (GOAL/SPIKE/CAVEWALL/...) stay module-local.
from .jumper_env import JumperEnv
from .jumper_gym_env import JumperGymEnv, JumperState, JumperAction
from .dodgeball import DodgeballGame, DodgeballAssets
# NOTE: dodgeball's generic object ids (LAVA_WALL/ENEMY/DOOR/...) stay module-local.
from .dodgeball_env import DodgeballEnv
from .dodgeball_gym_env import DodgeballGymEnv, DodgeballState, DodgeballAction
