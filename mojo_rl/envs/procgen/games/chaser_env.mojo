"""`ChaserEnv` — chaser as a Procgen benchmark environment.

Wraps `ChaserGame` (one level) with `LevelScheduler` (train/test level selection):
each `reset()` samples a `current_level_seed` from the configured level set and
regenerates the level. 64×64×3 RGB observation, 15-action discrete space (0..8 are
the move directions incl diagonals; 9..14 are no-ops). Assets are shared read-only
via `ArcPointer[ChaserAssets]` and passed into the game's renderer.

Single-env CPU facade (mirrors `MazeEnv`). See `docs/PROCGEN_CHASER_SCOPE.md`.
"""

from std.memory import ArcPointer

from ..core.level_scheduler import LevelScheduler
from .chaser import ChaserGame, ChaserAssets, DIST_EASY
from .maze_env import StepResult


struct ChaserEnv(Copyable, Movable):
    comptime NUM_ACTIONS = 15
    comptime OBS_H = 64
    comptime OBS_W = 64
    comptime OBS_C = 3
    comptime OBS_DIM = 64 * 64 * 3

    var scheduler: LevelScheduler
    var game: ChaserGame
    var assets: ArcPointer[ChaserAssets]
    var current_level_seed: Int

    def __init__(
        out self,
        asset_root: String,
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ) raises:
        # Owns its own asset load.
        self = ChaserEnv(
            ArcPointer(ChaserAssets(asset_root)),
            rand_seed,
            num_levels,
            start_level,
            dist_mode,
        )

    def __init__(
        out self,
        assets: ArcPointer[ChaserAssets],
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ):
        # Shares an already-loaded asset bundle.
        self.scheduler = LevelScheduler(rand_seed, num_levels, start_level)
        self.game = ChaserGame(dist_mode)
        self.assets = assets
        self.current_level_seed = 0

    def reset(mut self) -> List[UInt8]:
        self.current_level_seed = self.scheduler.next_level_seed()
        self.game.reset(self.current_level_seed)
        return self.game.render_obs(self.assets[])

    def obs(self) -> List[UInt8]:
        return self.game.render_obs(self.assets[])

    def render(self, res: Int) -> List[UInt8]:
        """A square RGB frame at an arbitrary resolution (human play / debug)."""
        return self.game.render(self.assets[], res)

    def step(mut self, action: Int) -> StepResult:
        var reward = self.game.step(action)
        return StepResult(
            self.game.render_obs(self.assets[]),
            reward,
            self.game.done,
            self.game.level_complete,
        )
