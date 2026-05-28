"""TD3DelayedActorPolyakStep — TD3 actor update + 3-pair polyak, gated.

TD3's actor update and ALL THREE polyaks (actor + critic1 + critic2) fire
together every `policy_delay` critic steps. The counter is internal — no
state pollution. When the counter hasn't reached threshold, `step` is a
no-op (does NOT touch state.did_step because Sample+TargetY+Critic still
ran).

Owns the inner DDPGActorLoss (TD3 uses DPG on critic1 for the actor).
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.optimizer.adam import Adam
from ...ddpg.actor_loss import DDPGActorLoss
from ...training.trainer_block import TrainerState


struct TD3DelayedActorPolyakStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, ACTOR: Module, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime APair = OnlineTargetPair[Self.ACTOR]
    comptime CPair = OnlineTargetPair[Self.CRITIC]
    comptime ActorInner = DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var inner: Self.ActorInner
    var tau:          Scalar[DT]
    var policy_delay: Int
    var _counter:     Int

    def __init__(out self):
        self.inner = Self.ActorInner()
        self.tau = Scalar[DT](0.005)
        self.policy_delay = 2
        self._counter = 0

    @staticmethod
    def make[target: StaticString](
        policy_delay: Int, tau: Scalar[DT],
    ) raises -> Self:
        comptime assert target == "cpu", (
            "TD3DelayedActorPolyakStep.make[target='gpu'] not yet supported"
        )
        var b = Self()
        b.inner = Self.ActorInner.make[target]()
        b.policy_delay = policy_delay
        b.tau = tau
        b._counter = 0
        return b^

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_opt: Adam,
        mut actor_pair: Self.APair,
        mut pair1: Self.CPair,
        mut pair2: Self.CPair,
    ) raises:
        """Accesses actor via `actor_pair.online` + critic1 via `pair1.online`
        internally — caller passes ONLY the pair refs (avoids Mojo aliasing
        rejection of passing `pair.online` + `pair` simultaneously)."""
        self._counter += 1
        if self._counter < self.policy_delay:
            return  # no-op this train_step
        self._counter = 0

        # Actor update against critic1 (DDPG-style DPG on pair1.online).
        var loss = self.inner.forward_backward[target, OPT=Adam](
            actor_pair.online, actor_opt, pair1.online,
            state.mb_s.cpu_ptr(),
        )
        state.actor_loss = loss

        # 3 polyaks: actor + critic1 + critic2.
        comptime if target == "cpu":
            actor_pair.polyak_step["cpu"](self.tau)
            pair1.polyak_step["cpu"](self.tau)
            pair2.polyak_step["cpu"](self.tau)
        else:
            actor_pair.polyak_step["gpu"](self.tau, state.ctx)
            pair1.polyak_step["gpu"](self.tau, state.ctx)
            pair2.polyak_step["gpu"](self.tau, state.ctx)
