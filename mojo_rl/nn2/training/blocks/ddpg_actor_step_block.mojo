"""DDPGActorStepBlock — wraps DDPGActorLoss.forward_backward.

Deterministic policy gradient: writes state.actor_loss.
"""

from ...constants import DT
from ...core.module import Module
from ...optimizer.adam import Adam
from ...loss.ddpg_actor_loss import DDPGActorLoss
from ..trainer_block import TrainerBlock, TrainerState


struct DDPGActorStepBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias Inner = DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var actor_ptr:     UnsafePointer[Self.ACTOR, MutAnyOrigin]
    var actor_opt_ptr: UnsafePointer[Adam, MutAnyOrigin]
    var critic_ptr:    UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var inner_ptr:     UnsafePointer[Self.Inner, MutAnyOrigin]

    def __init__(out self):
        self.actor_ptr = UnsafePointer[Self.ACTOR, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.actor_opt_ptr = UnsafePointer[Adam, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.inner_ptr = UnsafePointer[Self.Inner, MutAnyOrigin](
            unsafe_from_address=0
        )

    def bind(
        mut self,
        actor_ptr: UnsafePointer[Self.ACTOR, MutAnyOrigin],
        actor_opt_ptr: UnsafePointer[Adam, MutAnyOrigin],
        critic_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.Inner, MutAnyOrigin],
    ):
        self.actor_ptr = actor_ptr
        self.actor_opt_ptr = actor_opt_ptr
        self.critic_ptr = critic_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        var loss = self.inner_ptr[].forward_backward[target, OPT=Adam](
            self.actor_ptr[], self.actor_opt_ptr[],
            self.critic_ptr[],
            state.mb_s.cpu_ptr(),
        )
        state.actor_loss = loss
