"""TwinCriticStepBlock — TrainerBlock wrapper around TwinCriticUpdateBlock.step.

Reads state.mb_s, state.mb_a, state.mb_y → writes state.critic_loss.
Holds pointers to (online critic1/critic2, both Adam optimizers, the
trainer's twin_critic_block instance)."""

from layout import TileTensor, row_major

from ...constants import DT
from ...core.module import Module
from ...optimizer.adam import Adam
from ...loss.critic_update_block import TwinCriticUpdateBlock
from ..trainer_block import TrainerBlock, TrainerState


struct TwinCriticStepBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias Inner = TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var critic1_ptr:     UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var critic1_opt_ptr: UnsafePointer[Adam, MutAnyOrigin]
    var critic2_ptr:     UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var critic2_opt_ptr: UnsafePointer[Adam, MutAnyOrigin]
    var inner_ptr:       UnsafePointer[Self.Inner, MutAnyOrigin]

    def __init__(out self):
        self.critic1_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic1_opt_ptr = UnsafePointer[Adam, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic2_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic2_opt_ptr = UnsafePointer[Adam, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.inner_ptr = UnsafePointer[Self.Inner, MutAnyOrigin](
            unsafe_from_address=0
        )

    def bind(
        mut self,
        critic1_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        critic1_opt_ptr: UnsafePointer[Adam, MutAnyOrigin],
        critic2_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        critic2_opt_ptr: UnsafePointer[Adam, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.Inner, MutAnyOrigin],
    ):
        self.critic1_ptr = critic1_ptr
        self.critic1_opt_ptr = critic1_opt_ptr
        self.critic2_ptr = critic2_ptr
        self.critic2_opt_ptr = critic2_opt_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        var mb_y_t = TileTensor(
            state.mb_y.cpu_ptr(), row_major[Self.BATCH, 1]()
        )
        var loss = self.inner_ptr[].step[target](
            self.critic1_ptr[], self.critic1_opt_ptr[],
            self.critic2_ptr[], self.critic2_opt_ptr[],
            state.mb_s.cpu_ptr(), state.mb_a.cpu_ptr(), mb_y_t,
        )
        state.critic_loss = loss
