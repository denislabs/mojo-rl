"""EnsembleActorStep — TrainerState wrapper over EnsembleActorLoss.

Mirrors SAC's `SACActorStep` (`sac/blocks/actor_step.mojo`): reads
`state.mb_s` / `state.alpha`, runs `forward_backward`, writes
`state.actor_loss` + `state.log_prob_mean`. Trainer-facing surface
for R.3.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam

from ..ensemble import CriticEnsemble
from ..ensemble_actor_loss import EnsembleActorLoss
from ...training.trainer_block import TrainerState


struct EnsembleActorStep[
    ACTOR: Module,
    CRITIC: Module,
    N_: Int,
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime N = Self.N_
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = EnsembleActorLoss[
        Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](
            action_scale=action_scale, ctx=ctx,
        )
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
    ) raises:
        var res = self.inner.forward_backward[target, POLICY](
            actor,
            actor_opt,
            ensemble,
            state.mb_s.target_ptr[target](),
            state.alpha,
        )
        state.actor_loss = res.loss
        state.log_prob_mean = res.log_prob_mean
