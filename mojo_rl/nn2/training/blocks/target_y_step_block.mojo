"""TargetYStepBlock — TrainerBlock wrapper around TargetYBlock.step.

Holds pointers to (trainer.actor, trainer.pair1.target_net,
trainer.pair2.target_net, trainer.target_y_block). Reads state.alpha,
state.mb_sp, state.mb_r → writes state.mb_y."""

from ...constants import DT
from ...core.module import Module
from ..target_y_block import TargetYBlock
from ..trainer_block import TrainerBlock, TrainerState


struct TargetYStepBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias Inner = TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var actor_ptr:     UnsafePointer[Self.ACTOR, MutAnyOrigin]
    var critic1t_ptr:  UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var critic2t_ptr:  UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var inner_ptr:     UnsafePointer[Self.Inner, MutAnyOrigin]

    def __init__(out self):
        self.actor_ptr = UnsafePointer[Self.ACTOR, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic1t_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic2t_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.inner_ptr = UnsafePointer[Self.Inner, MutAnyOrigin](
            unsafe_from_address=0
        )

    def bind(
        mut self,
        actor_ptr: UnsafePointer[Self.ACTOR, MutAnyOrigin],
        critic1t_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        critic2t_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.Inner, MutAnyOrigin],
    ):
        self.actor_ptr = actor_ptr
        self.critic1t_ptr = critic1t_ptr
        self.critic2t_ptr = critic2t_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        comptime if target == "cpu":
            self.inner_ptr[].step[target](
                self.actor_ptr[],
                self.critic1t_ptr[],
                self.critic2t_ptr[],
                state.mb_sp.cpu_ptr(),
                state.mb_r.cpu_ptr(),
                state.alpha,
                state.mb_y.cpu_ptr(),
            )
        else:
            self.inner_ptr[].step[target](
                self.actor_ptr[],
                self.critic1t_ptr[],
                self.critic2t_ptr[],
                state.mb_sp.dev_ptr(),
                state.mb_r.dev_ptr(),
                state.alpha,
                state.mb_y.dev_ptr(),
            )
