"""Generic Procgen env wrappers — ONE `ProcgenEnv[G]` / `ProcgenGymEnv[G]`
pair over the `ProcgenGame` trait, replacing the 31 per-game
`<game>_env.mojo` / `<game>_gym_env.mojo` clone files (~3.2k lines that
were byte-identical modulo the game name; verified by normalized diff
before the collapse).

The per-game names (`BigfishEnv`, `BigfishGymEnv[DT]`, `BigfishState`, …)
remain available as comptime aliases in `games/__init__.mojo`, so no
consumer changes.

Game-side contract (`ProcgenGame`): each game struct adds a small
conformance-glue block (AssetsT / DEFAULT_DIST / GYM_MAX_STEPS +
load_assets/make statics + is_done/is_level_complete/gym_terminated
getters). Assets stay shared read-only via `ArcPointer[G.AssetsT]` and
are passed INTO the render calls (games do not own them — maze, the one
game that does, adapts with 3 ignore-the-arg overloads).

`gym_terminated` is the flag `ProcgenGymEnv.was_terminated()` reports
after a step: `done` for every game except maze, which reports
`level_complete` (maze's `done` fires on the timeout cap too, which the
gym layer already counts as truncation).
"""

from std.memory import ArcPointer

from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.state import State
from mojo_rl.core.action import Action
from mojo_rl.nn.constants import DT

from ..core.level_scheduler import LevelScheduler


trait ProcgenGame(Copyable, Movable, ImplicitlyDeletable):
    """Surface the generic Procgen env wrappers consume. See the module
    docstring for the per-game glue block this implies."""

    comptime AssetsT: ImplicitlyDeletable & Movable
    comptime DEFAULT_DIST: Int
    comptime GYM_MAX_STEPS: Int

    @staticmethod
    def load_assets(asset_root: String) raises -> Self.AssetsT:
        """Load the game's asset bundle from disk."""
        ...

    @staticmethod
    def make(assets: ArcPointer[Self.AssetsT], dist_mode: Int) -> Self:
        """Construct the game. Most games don't retain the assets (the
        env owns them and passes them into render calls) — maze does."""
        ...

    def reset(mut self, level_seed: Int):
        ...

    def step(mut self, action: Int) -> Float32:
        ...

    def is_done(self) -> Bool:
        ...

    def is_level_complete(self) -> Bool:
        ...

    def gym_terminated(self) -> Bool:
        """The real-terminal flag for `ProcgenGymEnv.was_terminated()`
        (vs the gym layer's own timeout truncation)."""
        ...

    # The pg_ prefix (adapters in each game's glue block) sidesteps
    # trait-conformance matching against the games' defaulted-argument
    # `render_obs(assets, res=…, ss=…)` methods — a defaulted method does
    # not satisfy a lower-arity trait requirement.

    def pg_render_obs(self, assets: Self.AssetsT) -> List[UInt8]:
        """The 64×64×3 RGB training observation."""
        ...

    def pg_render_obs_train(
        self, assets: Self.AssetsT, res: Int, ss: Int
    ) -> List[UInt8]:
        """`res×res×3` RGB observation with `ss`× supersampling."""
        ...

    def pg_render(self, assets: Self.AssetsT, res: Int) -> List[UInt8]:
        """A square RGB frame at an arbitrary resolution (human play /
        debug)."""
        ...


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


struct ProcgenEnv[G: ProcgenGame](Copyable, Movable):
    """Procgen benchmark environment: `G` + `LevelScheduler` (train/test
    level selection). Each `reset()` samples a `current_level_seed` from
    the configured level set and regenerates the level. 64×64×3 RGB
    observation, 15-action discrete space."""

    comptime NUM_ACTIONS = 15
    comptime OBS_H = 64
    comptime OBS_W = 64
    comptime OBS_C = 3
    comptime OBS_DIM = 64 * 64 * 3

    var scheduler: LevelScheduler
    var game: Self.G
    var assets: ArcPointer[Self.G.AssetsT]
    var current_level_seed: Int

    def __init__(
        out self,
        asset_root: String,
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = Self.G.DEFAULT_DIST,
    ) raises:
        # Owns its own asset load.
        self = Self(
            ArcPointer(Self.G.load_assets(asset_root)),
            rand_seed,
            num_levels,
            start_level,
            dist_mode,
        )

    def __init__(
        out self,
        assets: ArcPointer[Self.G.AssetsT],
        rand_seed: Int = 0,
        num_levels: Int = 0,
        start_level: Int = 0,
        dist_mode: Int = Self.G.DEFAULT_DIST,
    ):
        # Shares an already-loaded asset bundle.
        self.scheduler = LevelScheduler(rand_seed, num_levels, start_level)
        self.game = Self.G.make(assets, dist_mode)
        self.assets = assets
        self.current_level_seed = 0

    def reset(mut self) -> List[UInt8]:
        self.current_level_seed = self.scheduler.next_level_seed()
        self.game.reset(self.current_level_seed)
        return self.game.pg_render_obs(self.assets[])

    def obs(self) -> List[UInt8]:
        return self.game.pg_render_obs(self.assets[])

    def render(self, res: Int) -> List[UInt8]:
        """A square RGB frame at an arbitrary resolution (for human play /
        debug). The 64×64 training observation comes from `obs()`/`step()`."""
        return self.game.pg_render(self.assets[], res)

    def step(mut self, action: Int) -> StepResult:
        var reward = self.game.step(action)
        return StepResult(
            self.game.pg_render_obs(self.assets[]),
            reward,
            self.game.is_done(),
            self.game.is_level_complete(),
        )


@fieldwise_init
struct ProcgenState(State):
    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct ProcgenAction(Action):
    var value: Int


struct ProcgenGymEnv[G: ProcgenGame, DTYPE: DType = DT](
    BoxDiscreteActionEnv & Movable & ImplicitlyDeletable
):
    """`BoxDiscreteActionEnv` adapter for training: normalized `3×84×84`
    NCHW float observation + 15-action discrete space.
    `was_terminated()` reports the game's real terminal
    (`G.gym_terminated()`) vs the `MAX_STEPS` timeout truncation."""

    comptime dtype = Self.DTYPE
    comptime StateType = ProcgenState
    comptime ActionType = ProcgenAction

    comptime CNN_RES = 84
    comptime OBS_C = 3
    comptime OBS_SS_TRAIN = 2
    comptime STATE_SIZE: Int = 1
    comptime OBS_DIM: Int = Self.OBS_C * Self.CNN_RES * Self.CNN_RES  # 21168
    comptime NUM_ACTIONS: Int = 15
    comptime MAX_STEPS = Self.G.GYM_MAX_STEPS  # timeout → truncation

    var inner: ProcgenEnv[Self.G]
    var steps: Int
    var _terminated: Bool

    def __init__(
        out self,
        asset_root: String,
        rand_seed: Int = 0,
        num_levels: Int = 1,
        start_level: Int = 0,
        dist_mode: Int = Self.G.DEFAULT_DIST,
    ) raises:
        self = Self(
            ArcPointer(Self.G.load_assets(asset_root)),
            rand_seed,
            num_levels,
            start_level,
            dist_mode,
        )

    def __init__(
        out self,
        assets: ArcPointer[Self.G.AssetsT],
        rand_seed: Int = 0,
        num_levels: Int = 1,
        start_level: Int = 0,
        dist_mode: Int = Self.G.DEFAULT_DIST,
    ):
        self.inner = ProcgenEnv[Self.G](
            assets, rand_seed, num_levels, start_level, dist_mode
        )
        self.steps = 0
        self._terminated = False

    def _obs(self) -> List[Scalar[Self.DTYPE]]:
        var frame = self.inner.game.pg_render_obs_train(
            self.inner.assets[], Self.CNN_RES, Self.OBS_SS_TRAIN
        )
        comptime HW = Self.CNN_RES * Self.CNN_RES
        var out = List[Scalar[Self.DTYPE]]()
        out.resize(Self.OBS_DIM, 0)
        for y in range(Self.CNN_RES):
            for x in range(Self.CNN_RES):
                var poff = (y * Self.CNN_RES + x) * 3
                for c in range(3):
                    out[c * HW + y * Self.CNN_RES + x] = (
                        Scalar[Self.DTYPE](Int(frame[poff + c])) / 255.0
                    )
        return out^

    # --- Env ---
    def reset(mut self) -> ProcgenState:
        _ = self.reset_obs_list()
        return ProcgenState(0)

    def step(
        mut self, action: ProcgenAction, verbose: Bool = False
    ) -> Tuple[ProcgenState, Scalar[Self.DTYPE], Bool]:
        var res = self.step_obs(action.value)
        return (ProcgenState(0), res[1], res[2])

    def get_state(self) -> ProcgenState:
        return ProcgenState(0)

    def close(mut self):
        pass

    def was_terminated(self) -> Bool:
        return self._terminated

    # --- DiscreteActionEnv ---
    def action_from_index(self, action_idx: Int) -> ProcgenAction:
        return ProcgenAction(action_idx)

    def num_actions(self) -> Int:
        return Self.NUM_ACTIONS

    # --- ContinuousStateEnv ---
    def obs_dim(self) -> Int:
        return Self.OBS_DIM

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        return self._obs()

    def reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        _ = self.inner.reset()
        self.steps = 0
        self._terminated = False
        return self._obs()

    # --- BoxDiscreteActionEnv ---
    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        var reward = self.inner.game.step(action)
        self.steps += 1
        self._terminated = self.inner.game.gym_terminated()
        var truncated = self.steps >= Self.MAX_STEPS
        var done = self._terminated or truncated
        return (self._obs(), Scalar[Self.DTYPE](reward), done)
