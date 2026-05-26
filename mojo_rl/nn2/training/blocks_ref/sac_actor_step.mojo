"""SACActorStep — SAC actor gradient step (owns SACActorLossCG).

Reads state.mb_s, state.alpha → writes state.actor_loss + state.log_prob_mean.
The owned inner `SACActorLossCG` also holds the shared `rsample` consumed
by the trainer's `select_action`.
"""

from std.gpu.host import DeviceContext

from ...constants import DT
from ...core.module import Module
from ...optimizer.adam import Adam
from ...loss.sac_actor_loss_cg import SACActorLossCG
from ..trainer_block import TrainerState


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
    comptime Inner = SACActorLossCG[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        action_scale: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified make — POC for matmul-style `Optional[DeviceContext]` API.

        Single declaration for CPU + GPU. `ctx` is None on CPU; required on
        GPU (checked at runtime, mirrors `max.kernels.matmul`). The inner
        block (`SACActorLossCG`) still uses the two-overload pattern, so
        we translate `Optional → positional` at the boundary.
        """
        var b = Self()
        comptime if target == "cpu":
            b.inner = Self.Inner.make[target](action_scale=action_scale)
        else:
            if not ctx:
                raise Error(
                    "SACActorStep.make[target='gpu']: ctx required"
                )
            b.inner = Self.Inner.make[target](
                ctx.value(), action_scale=action_scale,
            )
        return b^

    def step[
        target: StaticString
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
    ) raises:
        var res = self.inner.forward_backward[target, OPT=Adam](
            actor,
            actor_opt,
            critic1,
            critic2,
            state.mb_s.target_ptr[target](),
            state.alpha,
        )
        state.actor_loss = res.loss
        state.log_prob_mean = res.log_prob_mean
