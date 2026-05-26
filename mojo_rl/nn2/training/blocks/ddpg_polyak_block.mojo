"""DDPGPolyakBlock — polyak update for actor + critic pairs.

Used by DDPG (1 actor pair + 1 critic pair). For TD3, use
TD3DelayedActorPolyakBlock (handles policy_delay + 3 pairs)."""

from ...constants import DT
from ...core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ..trainer_block import TrainerBlock, TrainerState


struct DDPGPolyakBlock[
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

    var actor_pair_ptr:  UnsafePointer[Self.APair, MutAnyOrigin]
    var critic_pair_ptr: UnsafePointer[Self.CPair, MutAnyOrigin]
    var tau: Scalar[DT]

    def __init__(out self):
        self.actor_pair_ptr = UnsafePointer[Self.APair, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.critic_pair_ptr = UnsafePointer[Self.CPair, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.tau = Scalar[DT](0.005)

    def bind(
        mut self,
        actor_pair_ptr: UnsafePointer[Self.APair, MutAnyOrigin],
        critic_pair_ptr: UnsafePointer[Self.CPair, MutAnyOrigin],
        tau: Scalar[DT],
    ):
        self.actor_pair_ptr = actor_pair_ptr
        self.critic_pair_ptr = critic_pair_ptr
        self.tau = tau

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        comptime if target == "cpu":
            self.actor_pair_ptr[].polyak_step["cpu"](self.tau)
            self.critic_pair_ptr[].polyak_step["cpu"](self.tau)
        else:
            self.actor_pair_ptr[].polyak_step["gpu"](self.tau, state.ctx)
            self.critic_pair_ptr[].polyak_step["gpu"](self.tau, state.ctx)
