"""`ChaserGymEnv` — `BoxDiscreteActionEnv` adapter for training chaser.

Wraps `ChaserEnv` for the deep_agents off-policy discrete driver: emits a
normalized `3×84×84` NCHW float observation (Nature-CNN input; the canonical
64×64 obs stays on `ChaserEnv` for eval/human) and a 15-action discrete space.
`was_terminated()` reports a real terminal (level complete OR agent caught) vs. a
timeout truncation (`MAX_STEPS`), so value bootstrapping is correct. Level
selection (train/test split) is handled by the wrapped `ChaserEnv` scheduler.

Mirrors `MazeGymEnv`. See `docs/PROCGEN_CHASER_SCOPE.md`.
"""

from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.state import State
from mojo_rl.core.action import Action
from mojo_rl.nn.constants import DT

from std.memory import ArcPointer

from .chaser_env import ChaserEnv
from .chaser import DIST_EASY, ChaserAssets


@fieldwise_init
struct ChaserState(State):
    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct ChaserAction(Action):
    var value: Int


struct ChaserGymEnv[DTYPE: DType = DT](
    BoxDiscreteActionEnv & Movable & ImplicitlyDeletable
):
    comptime dtype = Self.DTYPE
    comptime StateType = ChaserState
    comptime ActionType = ChaserAction

    comptime CNN_RES = 84  # Nature-CNN input size
    comptime OBS_C = 3
    comptime OBS_SS_TRAIN = 2  # obs supersample factor (168→84 box-avg)
    comptime STATE_SIZE: Int = 1
    comptime OBS_DIM: Int = Self.OBS_C * Self.CNN_RES * Self.CNN_RES  # 21168
    comptime NUM_ACTIONS: Int = 15
    comptime MAX_STEPS = 500  # timeout → truncation (chaser reference timeout=1000)

    var inner: ChaserEnv
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
        self = ChaserGymEnv[Self.DTYPE](
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
        num_levels: Int = 1,
        start_level: Int = 0,
        dist_mode: Int = DIST_EASY,
    ):
        # Shares an already-loaded asset bundle (batched training).
        self.inner = ChaserEnv(
            assets, rand_seed, num_levels, start_level, dist_mode
        )
        self.steps = 0
        self._terminated = False

    def _obs(self) -> List[Scalar[Self.DTYPE]]:
        # Render at 84×84×3 (HWC uint8) → normalized NCHW float.
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

    # --- Env ---
    def reset(mut self) -> ChaserState:
        _ = self.reset_obs_list()
        return ChaserState(0)

    def step(
        mut self, action: ChaserAction, verbose: Bool = False
    ) -> Tuple[ChaserState, Scalar[Self.DTYPE], Bool]:
        var res = self.step_obs(action.value)
        return (ChaserState(0), res[1], res[2])

    def get_state(self) -> ChaserState:
        return ChaserState(0)

    def close(mut self):
        pass

    def was_terminated(self) -> Bool:
        return self._terminated

    # --- DiscreteActionEnv ---
    def action_from_index(self, action_idx: Int) -> ChaserAction:
        return ChaserAction(action_idx)

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
        var reward = self.inner.game.step(action)  # step (no 64px render)
        self.steps += 1
        # Terminal (level complete or agent caught) vs. timeout truncation.
        self._terminated = self.inner.game.done
        var truncated = self.steps >= Self.MAX_STEPS
        var done = self._terminated or truncated
        return (self._obs(), Scalar[Self.DTYPE](reward), done)
