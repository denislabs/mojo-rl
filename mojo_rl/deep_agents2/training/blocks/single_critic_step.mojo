"""SingleCriticStep — DDPG critic update (single critic, sa concat).

Wraps `CriticUpdateBlock[CRITIC, BATCH, SA_DIM]`. Builds sa = concat(s, a)
internally (DDPG critic forward takes sa). Owns the sa scratch since the
SA shape is block-specific (TrainerState only carries OBS/ACT).
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from ...loss.critic_update_block import CriticUpdateBlock
from ..trainer_block import TrainerState


struct SingleCriticStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA = Self.OBS + Self.ACT
    comptime Inner = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA]

    var inner: Self.Inner
    var _mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.inner = Self.Inner()
        self._mb_sa = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "SingleCriticStep.make[target='gpu'] not yet supported"
        )
        var b = Self()
        b.inner = Self.Inner.make[target]()
        b._mb_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA)
        return b^

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut critic: Self.CRITIC,
        mut critic_opt: Adam,
    ) raises:
        # Build sa = concat(s, a) (CPU only).
        var mb_s_p = state.mb_s.cpu_ptr()
        var mb_a_p = state.mb_a.cpu_ptr()
        for b in range(Self.BATCH):
            for d in range(Self.OBS):
                self._mb_sa[b * Self.SA + d] = mb_s_p[b * Self.OBS + d]
            for j in range(Self.ACT):
                self._mb_sa[b * Self.SA + Self.OBS + j] = (
                    mb_a_p[b * Self.ACT + j]
                )
        var sa_t = TileTensor(
            self._mb_sa, row_major[Self.BATCH, Self.SA]()
        )
        var y_t = TileTensor(
            state.mb_y.cpu_ptr(), row_major[Self.BATCH, 1]()
        )
        var loss = self.inner.step[target](
            critic, critic_opt, sa_t, y_t,
        )
        state.critic_loss = loss
