"""`ClimberGymEnv` — `BoxDiscreteActionEnv` adapter for training climber.

Wraps `ClimberEnv` for the deep_agents off-policy discrete driver: normalized
3×84×84 NCHW obs, 15-action discrete space. `was_terminated()` reports a real
terminal (all coins collected OR death: enemy/off-screen) vs a timeout truncation
(`MAX_STEPS`). Mirrors `CoinrunGymEnv`. See `docs/PROCGEN_CLIMBER_SCOPE.md`.
"""

from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.state import State
from mojo_rl.core.action import Action
from mojo_rl.nn.constants import DT

from std.memory import ArcPointer

from .climber_env import ClimberEnv
from .climber import DIST_EASY, ClimberAssets


@fieldwise_init
struct ClimberState(State):
    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct ClimberAction(Action):
    var value: Int


struct ClimberGymEnv[DTYPE: DType = DT](
    BoxDiscreteActionEnv & Movable & ImplicitlyDeletable
):
    comptime dtype = Self.DTYPE
    comptime StateType = ClimberState
    comptime ActionType = ClimberAction

    comptime CNN_RES = 84
    comptime OBS_C = 3
    comptime OBS_SS_TRAIN = 2
    comptime STATE_SIZE: Int = 1
    comptime OBS_DIM: Int = Self.OBS_C * Self.CNN_RES * Self.CNN_RES
    comptime NUM_ACTIONS: Int = 15
    comptime MAX_STEPS = 1000

    var inner: ClimberEnv
    var steps: Int
    var _terminated: Bool

    def __init__(
        out self, asset_root: String, rand_seed: Int = 0, num_levels: Int = 1,
        start_level: Int = 0, dist_mode: Int = DIST_EASY,
    ) raises:
        self = ClimberGymEnv[Self.DTYPE](
            ArcPointer(ClimberAssets(asset_root)), rand_seed, num_levels, start_level, dist_mode
        )

    def __init__(
        out self, assets: ArcPointer[ClimberAssets], rand_seed: Int = 0, num_levels: Int = 1,
        start_level: Int = 0, dist_mode: Int = DIST_EASY,
    ):
        self.inner = ClimberEnv(assets, rand_seed, num_levels, start_level, dist_mode)
        self.steps = 0
        self._terminated = False

    def _obs(self) -> List[Scalar[Self.DTYPE]]:
        var frame = self.inner.game.render_obs(
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

    def reset(mut self) -> ClimberState:
        _ = self.reset_obs_list()
        return ClimberState(0)

    def step(
        mut self, action: ClimberAction, verbose: Bool = False
    ) -> Tuple[ClimberState, Scalar[Self.DTYPE], Bool]:
        var res = self.step_obs(action.value)
        return (ClimberState(0), res[1], res[2])

    def get_state(self) -> ClimberState:
        return ClimberState(0)

    def close(mut self):
        pass

    def was_terminated(self) -> Bool:
        return self._terminated

    def action_from_index(self, action_idx: Int) -> ClimberAction:
        return ClimberAction(action_idx)

    def num_actions(self) -> Int:
        return Self.NUM_ACTIONS

    def obs_dim(self) -> Int:
        return Self.OBS_DIM

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        return self._obs()

    def reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        _ = self.inner.reset()
        self.steps = 0
        self._terminated = False
        return self._obs()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        var reward = self.inner.game.step(action)
        self.steps += 1
        self._terminated = self.inner.game.done
        var truncated = self.steps >= Self.MAX_STEPS
        var done = self._terminated or truncated
        return (self._obs(), Scalar[Self.DTYPE](reward), done)
