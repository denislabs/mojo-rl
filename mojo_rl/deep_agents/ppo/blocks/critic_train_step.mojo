"""PPOCriticTrainStep — PPO critic gradient step.

Wraps `critic.forward + MSELoss.forward + MSELoss.vjp + critic.vjp +
critic_opt.step` over the minibatch (s, ret) slot of `OnPolicyState`.

Reads state.mb_obs / state.mb_ret → writes state.mb_v / state.mb_gv /
state.mb_gi (the gradient flowing into the obs side is unused
downstream but the scratch is owned by the state so we just fill it
through).
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.loss.mse import MSELoss
from mojo_rl.nn.optimizer.adam import Adam
from ...training.onpolicy_state import OnPolicyState


struct PPOCriticTrainStep[
    OBS_: Int,
    MINIBATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime MINIBATCH = Self.MINIBATCH_
    comptime Inner = MSELoss[1]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOCriticTrainStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        return b^

    def step[
        target: StaticString,
        ACT: Int,
        ROLLOUT_LEN: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, ACT, ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        mut critic: Self.CRITIC,
        mut critic_opt: Adam,
    ) raises -> Scalar[DT]:
        var mb_obs_p = state.mb_obs.target_ptr[target]()
        var mb_v_p   = state.mb_v.target_ptr[target]()
        var mb_gv_p  = state.mb_gv.target_ptr[target]()
        var mb_gi_p  = state.mb_gi.target_ptr[target]()
        var mb_ret_p = state.mb_ret.target_ptr[target]()

        var mb_obs_t = TileTensor(
            mb_obs_p, row_major[Self.MINIBATCH, Self.OBS]()
        )
        var mb_v_t   = TileTensor(mb_v_p,   row_major[Self.MINIBATCH, 1]())
        var mb_gv_t  = TileTensor(mb_gv_p,  row_major[Self.MINIBATCH, 1]())
        var mb_gi_t  = TileTensor(mb_gi_p,  row_major[Self.MINIBATCH, Self.OBS]())
        var mb_ret_t = TileTensor(mb_ret_p, row_major[Self.MINIBATCH, 1]())

        critic.forward[target, Self.MINIBATCH](mb_obs_t, output=mb_v_t)
        var loss = self.inner.forward[target, Self.MINIBATCH](mb_v_t, mb_ret_t)
        self.inner.vjp[target, Self.MINIBATCH](mb_ret_t, mb_gv_t)
        critic_opt.zero_grad[target, M=Self.CRITIC](critic)
        critic.vjp[target, Self.MINIBATCH](mb_gv_t, mb_gi_t)
        critic_opt.step[target, M=Self.CRITIC](critic)
        return loss
