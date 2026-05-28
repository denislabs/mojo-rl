"""PPOActorTrainStep — PPO actor gradient step.

Thin wrapper around `PPOActorLoss.forward_backward[target, OPT]`,
which is dual-target. Reads (mb_obs, mb_act, mb_olp,
mb_adv) from `OnPolicyState`. Returns the mean per-batch loss for
logging.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from ...loss.ppo_actor_loss import PPOActorLoss
from mojo_rl.nn2.optimizer.adam import Adam
from ..onpolicy_state import OnPolicyState


struct PPOActorTrainStep[
    OBS_: Int,
    ACT_: Int,
    MINIBATCH_: Int,
    ACTOR: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime MINIBATCH = Self.MINIBATCH_
    comptime Inner = PPOActorLoss[Self.ACTOR, Self.MINIBATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOActorTrainStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.inner = Self.Inner.make[target](
            ctx=ctx, clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        return b^

    def set_clip_eps(mut self, value: Scalar[DT]):
        self.inner.set_clip_eps(value)

    def set_entropy_coef(mut self, value: Scalar[DT]):
        self.inner.set_entropy_coef(value)

    def step[
        target: StaticString,
        ROLLOUT_LEN: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
    ) raises -> Scalar[DT]:
        var s_p   = state.mb_obs.target_ptr[target]()
        var a_p   = state.mb_act.target_ptr[target]()
        var olp_p = state.mb_olp.target_ptr[target]()
        var adv_p = state.mb_adv.target_ptr[target]()
        return self.inner.forward_backward[target, OPT=Adam, POLICY=POLICY](
            actor, actor_opt, s_p, a_p, olp_p, adv_p,
        )
