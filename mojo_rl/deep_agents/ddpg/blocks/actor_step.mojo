"""DDPGActorStep — DDPG actor (DPG) gradient step (owns storage DDPGActorLoss).

Writes state.actor_loss. Thin wrapper over the storage `DDPGActorLoss`
(forward_backward takes owned Tensors + Adam + ctx).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.optimizer.adam import Adam
from ..actor_loss import DDPGActorLoss
from ...training.trainer_block import TrainerState


struct DDPGActorStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic: Self.CRITIC,
    ) raises:
        var loss = self.inner.forward_backward[target, POLICY](
            actor,
            actor_opt,
            critic,
            state.mb_s,
            state.ctx,
        )
        # On GPU `loss` is a 0 sentinel — the real metric is drained from
        # the inner device accumulator at flush (read_loss_accum).
        state.actor_loss = loss

    # ── GPU loss-accumulator passthroughs (flush cadence; GPU only) ──
    def reset_loss_accum(mut self) raises:
        self.inner.reset_loss_accum()

    def read_loss_accum(mut self, ctx: DeviceContext) raises -> Scalar[DT]:
        return self.inner.read_loss_accum(ctx)
