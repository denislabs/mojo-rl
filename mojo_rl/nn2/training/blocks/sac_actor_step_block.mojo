"""SACActorStepBlock — TrainerBlock wrapper around SACActorLoss.forward_backward.

Reads state.mb_s, state.alpha → writes state.actor_loss, state.log_prob_mean.
Holds pointers to (actor, actor_opt, online critic1/critic2, the trainer's
SACActorLoss instance — which also holds the shared `rsample` consumed by
select_action)."""

from ...constants import DT
from ...core.module import Module
from ...optimizer.adam import Adam
from ...loss.sac_actor_loss_cg import SACActorLossCG, SACActorLossOut
from ..trainer_block import TrainerBlock, TrainerState


struct SACActorStepBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias Inner = SACActorLossCG[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var actor_ptr:     UnsafePointer[Self.ACTOR, MutAnyOrigin]
    var actor_opt_ptr: UnsafePointer[Adam, MutAnyOrigin]
    var critic1_ptr:   UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var critic2_ptr:   UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var inner_ptr:     UnsafePointer[Self.Inner, MutAnyOrigin]

    def __init__(out self):
        self.actor_ptr = UnsafePointer[Self.ACTOR, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.actor_opt_ptr = UnsafePointer[Adam, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic1_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic2_ptr = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.inner_ptr = UnsafePointer[Self.Inner, MutAnyOrigin](
            unsafe_from_address=0
        )

    def bind(
        mut self,
        actor_ptr: UnsafePointer[Self.ACTOR, MutAnyOrigin],
        actor_opt_ptr: UnsafePointer[Adam, MutAnyOrigin],
        critic1_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        critic2_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.Inner, MutAnyOrigin],
    ):
        self.actor_ptr = actor_ptr
        self.actor_opt_ptr = actor_opt_ptr
        self.critic1_ptr = critic1_ptr
        self.critic2_ptr = critic2_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        comptime if target == "cpu":
            var res = self.inner_ptr[].forward_backward[target, OPT=Adam](
                self.actor_ptr[], self.actor_opt_ptr[],
                self.critic1_ptr[], self.critic2_ptr[],
                state.mb_s.cpu_ptr(),
                state.alpha,
            )
            state.actor_loss = res.loss
            state.log_prob_mean = res.log_prob_mean
        else:
            var res = self.inner_ptr[].forward_backward[target, OPT=Adam](
                self.actor_ptr[], self.actor_opt_ptr[],
                self.critic1_ptr[], self.critic2_ptr[],
                state.mb_s.dev_ptr(),
                state.alpha,
            )
            state.actor_loss = res.loss
            state.log_prob_mean = res.log_prob_mean
