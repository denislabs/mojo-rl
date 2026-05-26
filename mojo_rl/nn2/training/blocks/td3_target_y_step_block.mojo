"""TD3TargetYStepBlock — wraps TD3TargetYBlock.step.

Reads state.mb_sp, state.mb_r → writes state.mb_y. Holds pointers to
the target actor + twin target critics + the trainer's TD3TargetYBlock.
"""

from ...constants import DT
from ...core.module import Module
from ..td3_target_y_block import TD3TargetYBlock
from ..trainer_block import TrainerBlock, TrainerState


struct TD3TargetYStepBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias Inner = TD3TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var actor_t_ptr:    UnsafePointer[Self.ACTOR, MutAnyOrigin]
    var critic1_t_ptr:  UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var critic2_t_ptr:  UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var inner_ptr:      UnsafePointer[Self.Inner, MutAnyOrigin]

    def __init__(out self):
        self.actor_t_ptr = UnsafePointer[Self.ACTOR, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic1_t_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic2_t_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.inner_ptr = UnsafePointer[Self.Inner, MutAnyOrigin](
            unsafe_from_address=0
        )

    def bind(
        mut self,
        actor_t_ptr: UnsafePointer[Self.ACTOR, MutAnyOrigin],
        critic1_t_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        critic2_t_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.Inner, MutAnyOrigin],
    ):
        self.actor_t_ptr = actor_t_ptr
        self.critic1_t_ptr = critic1_t_ptr
        self.critic2_t_ptr = critic2_t_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        self.inner_ptr[].step[target](
            self.actor_t_ptr[],
            self.critic1_t_ptr[],
            self.critic2_t_ptr[],
            state.mb_sp.cpu_ptr(),
            state.mb_r.cpu_ptr(),
            state.mb_y.cpu_ptr(),
        )
