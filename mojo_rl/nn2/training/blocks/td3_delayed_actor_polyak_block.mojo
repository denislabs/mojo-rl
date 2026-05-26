"""TD3DelayedActorPolyakBlock — TD3 actor update + 3-pair polyak, gated.

TD3's actor update and ALL THREE polyaks (actor + critic1 + critic2)
fire together every `policy_delay` critic steps. Block owns the
counter internally — no state pollution. When the counter hasn't
reached threshold, step_via is a no-op (does NOT set did_step=False
because the train_step itself still ran via Sample+TargetY+Critic).
"""

from ...constants import DT
from ...core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ...optimizer.adam import Adam
from ...loss.ddpg_actor_loss import DDPGActorLoss
from ..trainer_block import TrainerBlock, TrainerState


struct TD3DelayedActorPolyakBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias APair = OnlineTargetPair[Self.ACTOR]
    alias CPair = OnlineTargetPair[Self.CRITIC]
    alias ActorInner = DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var actor_ptr:       UnsafePointer[Self.ACTOR, MutAnyOrigin]
    var actor_opt_ptr:   UnsafePointer[Adam, MutAnyOrigin]
    var critic1_ptr:     UnsafePointer[Self.CRITIC, MutAnyOrigin]
    var actor_pair_ptr:  UnsafePointer[Self.APair, MutAnyOrigin]
    var pair1_ptr:       UnsafePointer[Self.CPair, MutAnyOrigin]
    var pair2_ptr:       UnsafePointer[Self.CPair, MutAnyOrigin]
    var inner_ptr:       UnsafePointer[Self.ActorInner, MutAnyOrigin]

    var tau:          Scalar[DT]
    var policy_delay: Int
    var _counter:     Int

    def __init__(out self):
        var null_a = UnsafePointer[Self.ACTOR, MutAnyOrigin](
            unsafe_from_address=0
        )
        var null_c = UnsafePointer[Self.CRITIC, MutAnyOrigin](
            unsafe_from_address=0
        )
        var null_o = UnsafePointer[Adam, MutAnyOrigin](unsafe_from_address=0)
        var null_ap = UnsafePointer[Self.APair, MutAnyOrigin](
            unsafe_from_address=0
        )
        var null_cp = UnsafePointer[Self.CPair, MutAnyOrigin](
            unsafe_from_address=0
        )
        var null_inner = UnsafePointer[Self.ActorInner, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.actor_ptr = null_a
        self.actor_opt_ptr = null_o
        self.critic1_ptr = null_c
        self.actor_pair_ptr = null_ap
        self.pair1_ptr = null_cp
        self.pair2_ptr = null_cp
        self.inner_ptr = null_inner
        self.tau = Scalar[DT](0.005)
        self.policy_delay = 2
        self._counter = 0

    def setup(mut self, policy_delay: Int, tau: Scalar[DT]):
        self.policy_delay = policy_delay
        self.tau = tau
        self._counter = 0

    def bind(
        mut self,
        actor_ptr: UnsafePointer[Self.ACTOR, MutAnyOrigin],
        actor_opt_ptr: UnsafePointer[Adam, MutAnyOrigin],
        critic1_ptr: UnsafePointer[Self.CRITIC, MutAnyOrigin],
        actor_pair_ptr: UnsafePointer[Self.APair, MutAnyOrigin],
        pair1_ptr: UnsafePointer[Self.CPair, MutAnyOrigin],
        pair2_ptr: UnsafePointer[Self.CPair, MutAnyOrigin],
        inner_ptr: UnsafePointer[Self.ActorInner, MutAnyOrigin],
    ):
        self.actor_ptr = actor_ptr
        self.actor_opt_ptr = actor_opt_ptr
        self.critic1_ptr = critic1_ptr
        self.actor_pair_ptr = actor_pair_ptr
        self.pair1_ptr = pair1_ptr
        self.pair2_ptr = pair2_ptr
        self.inner_ptr = inner_ptr

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        self._counter += 1
        if self._counter < self.policy_delay:
            return  # no-op this train_step; did_step stays True

        self._counter = 0
        # Actor update against critic1 (DDPG-style DPG on pair1.online).
        var loss = self.inner_ptr[].forward_backward[target, OPT=Adam](
            self.actor_ptr[], self.actor_opt_ptr[],
            self.critic1_ptr[],
            state.mb_s.cpu_ptr(),
        )
        state.actor_loss = loss

        # 3 polyaks: actor + critic1 + critic2.
        comptime if target == "cpu":
            self.actor_pair_ptr[].polyak_step["cpu"](self.tau)
            self.pair1_ptr[].polyak_step["cpu"](self.tau)
            self.pair2_ptr[].polyak_step["cpu"](self.tau)
        else:
            self.actor_pair_ptr[].polyak_step["gpu"](self.tau, state.ctx)
            self.pair1_ptr[].polyak_step["gpu"](self.tau, state.ctx)
            self.pair2_ptr[].polyak_step["gpu"](self.tau, state.ctx)
