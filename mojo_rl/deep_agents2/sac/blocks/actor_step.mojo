"""SACActorStep — SAC actor gradient step (owns SACActorLoss).

Reads state.mb_s, state.alpha → writes state.actor_loss + state.log_prob_mean.
The owned inner `SACActorLoss` also holds the shared `rsample` consumed
by the trainer's `select_action`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from ..actor_loss import SACActorLoss
from ...training.trainer_block import TrainerState


struct SACActorStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = SACActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        action_scale: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified make — single declaration for CPU + GPU. `ctx` is None
        on CPU; required on GPU (inner block raises if missing)."""
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx, action_scale=action_scale)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
    ) raises:
        var res = self.inner.forward_backward[
            target, OPT=Adam, POLICY=POLICY,
        ](
            actor,
            actor_opt,
            critic1,
            critic2,
            state.mb_s.target_ptr[target](),
            state.alpha,
        )
        state.actor_loss = res.loss
        state.log_prob_mean = res.log_prob_mean
