"""Procgen games (maze). See `docs/PROCGEN_PORT.md`."""

from mojo_rl.nn.constants import DT

from .procgen_env import (
    ProcgenGame,
    ProcgenEnv,
    ProcgenGymEnv,
    ProcgenState,
    ProcgenAction,
    StepResult,
)

from .maze import (
    MazeGame,
    MazeAssets,
    DIST_EASY,
    DIST_HARD,
    DIST_MEMORY,
    world_dim_for,
)
comptime MazeEnv = ProcgenEnv[MazeGame]
comptime MazeGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[MazeGame, DTYPE]
comptime MazeState = ProcgenState
comptime MazeAction = ProcgenAction
from .chaser import (
    ChaserGame,
    ChaserAssets,
    DIST_EXTREME,
    LARGE_ORB,
    ENEMY_EGG,
    MAZE_WALL,
    ORB,
)
comptime ChaserEnv = ProcgenEnv[ChaserGame]
comptime ChaserGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[ChaserGame, DTYPE]
comptime ChaserState = ProcgenState
comptime ChaserAction = ProcgenAction
from .heist import (
    HeistGame,
    HeistAssets,
    heist_world_dim,
    LOCKED_DOOR,
    KEY,
    EXIT,
)
comptime HeistEnv = ProcgenEnv[HeistGame]
comptime HeistGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[HeistGame, DTYPE]
comptime HeistState = ProcgenState
comptime HeistAction = ProcgenAction
from .bigfish import BigfishGame, BigfishAssets, FISH
comptime BigfishEnv = ProcgenEnv[BigfishGame]
comptime BigfishGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[BigfishGame, DTYPE]
comptime BigfishState = ProcgenState
comptime BigfishAction = ProcgenAction
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
comptime LeaperEnv = ProcgenEnv[LeaperGame]
comptime LeaperGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[LeaperGame, DTYPE]
comptime LeaperState = ProcgenState
comptime LeaperAction = ProcgenAction
# `EXIT` collides with heist's (heist=9, miner=6) — both are game-local tile ids.
# Heist keeps the unqualified name (pre-existing consumers depend on it); miner's is
# re-exported under a qualified name.
from .miner import (
    MinerGame,
    MinerAssets,
    BOULDER,
    DIAMOND,
    EXIT as MINER_EXIT,
    DIRT,
)
comptime MinerEnv = ProcgenEnv[MinerGame]
comptime MinerGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[MinerGame, DTYPE]
comptime MinerState = ProcgenState
comptime MinerAction = ProcgenAction
from .starpilot import (
    StarpilotGame,
    StarpilotAssets,
    FLYER,
    METEOR,
    CLOUD,
    TURRET,
    FAST_FLYER,
)
comptime StarpilotEnv = ProcgenEnv[StarpilotGame]
comptime StarpilotGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[StarpilotGame, DTYPE]
comptime StarpilotState = ProcgenState
comptime StarpilotAction = ProcgenAction
from .plunder import PlunderGame, PlunderAssets, SHIP, PANEL, PLAYER_BULLET
comptime PlunderEnv = ProcgenEnv[PlunderGame]
comptime PlunderGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[PlunderGame, DTYPE]
comptime PlunderState = ProcgenState
comptime PlunderAction = ProcgenAction
from .fruitbot import FruitbotGame, FruitbotAssets
# NOTE: fruitbot's per-game object ids (GOOD_OBJ/BAD_OBJ/BARRIER/LOCKED_DOOR/LOCK)
# are NOT re-exported here — LOCKED_DOOR collides with heist's (different value).
# Import them from `mojo_rl.envs.procgen.games.fruitbot` directly.
comptime FruitbotEnv = ProcgenEnv[FruitbotGame]
comptime FruitbotGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[FruitbotGame, DTYPE]
comptime FruitbotState = ProcgenState
comptime FruitbotAction = ProcgenAction
from .bossfight import BossfightGame, BossfightAssets
# NOTE: bossfight's generic object ids (PLAYER_BULLET/BOSS/SHIELDS/ENEMY_BULLET/
# LASER_TRAIL/REFLECTED_BULLET/BARRIER) are NOT re-exported here — PLAYER_BULLET
# collides with plunder's and BARRIER with fruitbot's. Import them from
# `mojo_rl.envs.procgen.games.bossfight` directly.
comptime BossfightEnv = ProcgenEnv[BossfightGame]
comptime BossfightGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[BossfightGame, DTYPE]
comptime BossfightState = ProcgenState
comptime BossfightAction = ProcgenAction
from .coinrun import CoinrunGame, CoinrunAssets
# NOTE: coinrun's generic object ids (GOAL/SAW/ENEMY/WALL_MID/CRATE/...) are NOT
# re-exported here — GOAL/CRATE/etc collide with other games'. Import them from
# `mojo_rl.envs.procgen.games.coinrun` directly.
comptime CoinrunEnv = ProcgenEnv[CoinrunGame]
comptime CoinrunGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[CoinrunGame, DTYPE]
comptime CoinrunState = ProcgenState
comptime CoinrunAction = ProcgenAction
from .caveflyer import CaveflyerGame, CaveflyerAssets
# NOTE: caveflyer's generic object ids (GOAL/TARGET/ENEMY/CAVEWALL/...) are NOT
# re-exported here — GOAL/ENEMY/etc collide with other games'. Import them from
# `mojo_rl.envs.procgen.games.caveflyer` directly.
comptime CaveflyerEnv = ProcgenEnv[CaveflyerGame]
comptime CaveflyerGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[CaveflyerGame, DTYPE]
comptime CaveflyerState = ProcgenState
comptime CaveflyerAction = ProcgenAction
from .climber import ClimberGame, ClimberAssets
# NOTE: climber's generic object ids (COIN/ENEMY/WALL_MID/...) stay module-local
# (collide with other games'). Import from `...games.climber` directly.
comptime ClimberEnv = ProcgenEnv[ClimberGame]
comptime ClimberGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[ClimberGame, DTYPE]
comptime ClimberState = ProcgenState
comptime ClimberAction = ProcgenAction
from .ninja import NinjaGame, NinjaAssets
# NOTE: ninja's generic object ids (GOAL/BOMB/WALL_MID/FIRE/...) stay module-local.
comptime NinjaEnv = ProcgenEnv[NinjaGame]
comptime NinjaGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[NinjaGame, DTYPE]
comptime NinjaState = ProcgenState
comptime NinjaAction = ProcgenAction
from .jumper import JumperGame, JumperAssets
# NOTE: jumper's generic object ids (GOAL/SPIKE/CAVEWALL/...) stay module-local.
comptime JumperEnv = ProcgenEnv[JumperGame]
comptime JumperGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[JumperGame, DTYPE]
comptime JumperState = ProcgenState
comptime JumperAction = ProcgenAction
from .dodgeball import DodgeballGame, DodgeballAssets
# NOTE: dodgeball's generic object ids (LAVA_WALL/ENEMY/DOOR/...) stay module-local.
comptime DodgeballEnv = ProcgenEnv[DodgeballGame]
comptime DodgeballGymEnv[DTYPE: DType = DT] = ProcgenGymEnv[DodgeballGame, DTYPE]
comptime DodgeballState = ProcgenState
comptime DodgeballAction = ProcgenAction
