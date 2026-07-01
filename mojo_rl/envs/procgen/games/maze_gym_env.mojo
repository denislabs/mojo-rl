"""`MazeGymEnv` — `BoxDiscreteActionEnv` adapter for training the maze.

Wraps `MazeEnv` for the deep_agents off-policy discrete driver
(`run_offpolicy_discrete_train`): emits a normalized `3×84×84` NCHW float
observation (the Nature-CNN input size — the canonical 64×64 obs stays on
`MazeEnv` for eval/human) and a 15-action discrete space. Adds a max-steps
truncation (`was_terminated()` reports goal-reached vs. timeout) so episodes that
never reach the cheese still terminate. Level selection (train/test split) is
handled by the wrapped `MazeEnv`'s scheduler. See `docs/PROCGEN_PORT.md`.
"""

from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.state import State
from mojo_rl.core.action import Action
from mojo_rl.nn.constants import DT

from std.memory import ArcPointer

from .maze_env import MazeEnv
from .maze import DIST_EASY, MazeAssets


@fieldwise_init
struct MazeState(State):
    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct MazeAction(Action):
    var value: Int


struct MazeGymEnv[DTYPE: DType = DT](
    BoxDiscreteActionEnv & Movable & ImplicitlyDeletable
):
    comptime dtype = Self.DTYPE
    comptime StateType = MazeState
    comptime ActionType = MazeAction

    comptime CNN_RES = 84  # Nature-CNN input size
    comptime OBS_C = 3
    comptime OBS_SS_TRAIN = 2  # obs supersample factor (168→84 box-avg)
    comptime STATE_SIZE: Int = 1
    comptime OBS_DIM: Int = Self.OBS_C * Self.CNN_RES * Self.CNN_RES  # 21168
    comptime NUM_ACTIONS: Int = 15
    comptime MAX_STEPS = 500  # maze timeout → truncation

    var inner: MazeEnv
    var steps: Int
    var _terminated: Bool

    def __init__(
        out self,
        asset_root: String,
        rand_seed: Int = 0,
        num_levels: Int = 1,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ) raises:
        # Owns its own asset load.
        self = MazeGymEnv[Self.DTYPE](
            ArcPointer(MazeAssets(asset_root)),
            rand_seed,
            num_levels,
            start_level,
            dist_mode,
        )

    def __init__(
        out self,
        assets: ArcPointer[MazeAssets],
        rand_seed: Int = 0,
        num_levels: Int = 1,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ):
        # Shares an already-loaded asset bundle (batched training).
        self.inner = MazeEnv(
            assets, rand_seed, num_levels, start_level, dist_mode
        )
        self.steps = 0
        self._terminated = False

    def _obs(self) -> List[Scalar[Self.DTYPE]]:
        # Render the maze at 84×84×3 (HWC uint8) → normalized NCHW float.
        var frame = self.inner.game.render_obs(Self.CNN_RES, Self.OBS_SS_TRAIN)
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
    def reset(mut self) -> MazeState:
        _ = self.reset_obs_list()
        return MazeState(0)

    def step(
        mut self, action: MazeAction, verbose: Bool = False
    ) -> Tuple[MazeState, Scalar[Self.DTYPE], Bool]:
        var res = self.step_obs(action.value)
        return (MazeState(0), res[1], res[2])

    def get_state(self) -> MazeState:
        return MazeState(0)

    def close(mut self):
        pass

    def was_terminated(self) -> Bool:
        return self._terminated

    # --- DiscreteActionEnv ---
    def action_from_index(self, action_idx: Int) -> MazeAction:
        return MazeAction(action_idx)

    def num_actions(self) -> Int:
        return Self.NUM_ACTIONS

    # --- ContinuousStateEnv ---
    def obs_dim(self) -> Int:
        return Self.OBS_DIM

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        return self._obs()

    def reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        _ = self.inner.reset()  # scheduler picks a level + regenerates
        self.steps = 0
        self._terminated = False
        return self._obs()

    # --- BoxDiscreteActionEnv ---
    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        var reward = self.inner.game.step(action)  # step the game (no 64px render)
        self.steps += 1
        self._terminated = self.inner.game.level_complete
        var truncated = self.steps >= Self.MAX_STEPS
        var done = self._terminated or truncated
        return (self._obs(), Scalar[Self.DTYPE](reward), done)
