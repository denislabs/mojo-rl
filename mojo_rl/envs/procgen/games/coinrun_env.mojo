"""`CoinrunEnv` — coinrun as a Procgen benchmark environment.

Wraps `CoinrunGame` with `LevelScheduler`. 64×64×3 RGB observation, 15-action
discrete space (0..8 move+jump). Assets shared read-only via
`ArcPointer[CoinrunAssets]`, passed into the renderer. Mirrors `BossfightEnv`.
See `docs/PROCGEN_COINRUN_SCOPE.md`.
"""

from std.memory import ArcPointer

from ..core.level_scheduler import LevelScheduler
from .coinrun import CoinrunGame, CoinrunAssets, DIST_EASY
from .maze_env import StepResult


struct CoinrunEnv(Copyable, Movable):
    comptime NUM_ACTIONS = 15
    comptime OBS_H = 64
    comptime OBS_W = 64
    comptime OBS_C = 3
    comptime OBS_DIM = 64 * 64 * 3

    var scheduler: LevelScheduler
    var game: CoinrunGame
    var assets: ArcPointer[CoinrunAssets]
    var current_level_seed: Int

    def __init__(
        out self,
        asset_root: String,
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ) raises:
        self = CoinrunEnv(
            ArcPointer(CoinrunAssets(asset_root)),
            rand_seed,
            num_levels,
            start_level,
            dist_mode,
        )

    def __init__(
        out self,
        assets: ArcPointer[CoinrunAssets],
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ):
        self.scheduler = LevelScheduler(rand_seed, num_levels, start_level)
        self.game = CoinrunGame(dist_mode)
        self.assets = assets
        self.current_level_seed = 0

    def reset(mut self) -> List[UInt8]:
        self.current_level_seed = self.scheduler.next_level_seed()
        self.game.reset(self.current_level_seed)
        return self.game.render_obs(self.assets[])

    def obs(self) -> List[UInt8]:
        return self.game.render_obs(self.assets[])

    def render(self, res: Int) -> List[UInt8]:
        return self.game.render(self.assets[], res)

    def step(mut self, action: Int) -> StepResult:
        var reward = self.game.step(action)
        return StepResult(
            self.game.render_obs(self.assets[]),
            reward,
            self.game.done,
            self.game.level_complete,
        )
