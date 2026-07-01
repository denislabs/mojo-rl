"""`MazeEnv` — the maze as a Procgen benchmark environment.

Wraps `MazeGame` (one level) with `LevelScheduler` (train/test level selection),
matching Procgen's semantics: each `reset()` samples a `current_level_seed` from
the configured level set and regenerates the maze. 64×64×3 RGB observation,
15-action discrete space (only 0..8 move; 9..14 are no-ops, as in maze).

Single-env CPU facade. Full `BoxDiscreteActionEnv`/`GPUDiscreteEnv` trait
conformance + GPU batching is Phase 4 (a value-type-state refactor). HardMode
(world_dim 25) only for now; Easy/Memory (different world_dim + center_agent
camera) are a follow-up. See `docs/PROCGEN_PORT.md`.
"""

from ..core.level_scheduler import LevelScheduler
from .maze import MazeGame, DIST_HARD


struct StepResult(Copyable, Movable):
    var obs: List[UInt8]  # 64*64*3 RGB
    var reward: Float32
    var done: Bool
    var level_complete: Bool

    def __init__(
        out self,
        var obs: List[UInt8],
        reward: Float32,
        done: Bool,
        level_complete: Bool,
    ):
        self.obs = obs^
        self.reward = reward
        self.done = done
        self.level_complete = level_complete


struct MazeEnv(Copyable, Movable):
    comptime NUM_ACTIONS = 15
    comptime OBS_H = 64
    comptime OBS_W = 64
    comptime OBS_C = 3
    comptime OBS_DIM = 64 * 64 * 3

    var scheduler: LevelScheduler
    var game: MazeGame
    var current_level_seed: Int

    def __init__(
        out self,
        asset_root: String,
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = DIST_HARD,
    ) raises:
        self.scheduler = LevelScheduler(rand_seed, num_levels, start_level)
        self.game = MazeGame(asset_root, dist_mode)
        self.current_level_seed = 0

    def reset(mut self) -> List[UInt8]:
        self.current_level_seed = self.scheduler.next_level_seed()
        self.game.reset(self.current_level_seed)
        return self.game.render_obs()

    def obs(self) -> List[UInt8]:
        return self.game.render_obs()

    def render(self, res: Int) -> List[UInt8]:
        """A square RGB frame at an arbitrary resolution (for human play /
        debug). The 64×64 training observation comes from `obs()`/`step()`."""
        return self.game.render(res)

    def step(mut self, action: Int) -> StepResult:
        var reward = self.game.step(action)
        return StepResult(
            self.game.render_obs(),
            reward,
            self.game.done,
            self.game.level_complete,
        )
