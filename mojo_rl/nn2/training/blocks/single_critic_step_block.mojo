"""SingleCriticStepBlock — DDPG critic update (single critic, sa concat).

Wraps `CriticUpdateBlock[CRITIC, BATCH, SA_DIM]` for DDPG. Builds the
sa = concat(s, a) minibatch internally (DDPG critic forward takes sa,
unlike SAC which lets the actor block produce sa via the graph).
"""

from layout import TileTensor, row_major

from ...constants import DT
from ...core.module import Module
from ...optimizer.adam import Adam
from ...loss.critic_update_block import CriticUpdateBlock
from ..trainer_block import TrainerBlock, TrainerState


struct SingleCriticStepBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA = Self.OBS + Self.ACT
    alias Inner = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA]

    var critic_ptr:     UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var critic_opt_ptr: UnsafePointer[Adam, MutAnyOrigin]
    var inner_ptr:      UnsafePointer[Self.Inner, MutAnyOrigin]

    # Per-block sa scratch (concat of mb_s + mb_a). Owned here since
    # the SA shape is block-specific (SA = OBS + ACT for DDPG, but the
    # base TrainerState doesn't carry it).
    var _mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.critic_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic_opt_ptr = UnsafePointer[Adam, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.inner_ptr = UnsafePointer[Self.Inner, MutAnyOrigin](
            unsafe_from_address=0
        )
        self._mb_sa = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )

    def setup(mut self) raises:
        from std.memory import alloc
        self._mb_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA)

    def bind(
        mut self,
        critic_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        critic_opt_ptr: UnsafePointer[Adam, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.Inner, MutAnyOrigin],
    ):
        self.critic_ptr = critic_ptr
        self.critic_opt_ptr = critic_opt_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        # Build sa = concat(s, a) (CPU only — DDPG GPU not in scope J.1.e).
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
        var loss = self.inner_ptr[].step[target](
            self.critic_ptr[], self.critic_opt_ptr[], sa_t, y_t,
        )
        state.critic_loss = loss
