"""PPOGAEStep — Generalized Advantage Estimation over the rollout.

Bootstraps V(s_T) from `state.bootstrap_obs`, runs `compute_gae` over
the rollout buffers (rewards, values, terminated) → fills (adv_buf,
ret_buf). CPU path uses the existing free `compute_gae`. GPU path
(P.2) will use a one-thread-per-env sequential kernel.

Decoupled from minibatch normalisation: that lives in
`PPOMinibatchGatherStep` (CleanRL-style per-minibatch normalisation,
not whole-rollout normalisation).
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from ...constants import DT
from ...core.module import Module
from ..gae import compute_gae
from ..onpolicy_state import OnPolicyState


struct PPOGAEStep[
    OBS_: Int,
    ROLLOUT_LEN_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ROLLOUT_LEN = Self.ROLLOUT_LEN_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOGAEStep: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def step[
        target: StaticString,
        ACT: Int,
        MINIBATCH: Int,
    ](
        mut self,
        mut state: OnPolicyState[Self.OBS, ACT, Self.ROLLOUT_LEN, MINIBATCH],
        mut critic: Self.CRITIC,
        gamma: Scalar[DT],
        gae_lambda: Scalar[DT],
    ) raises:
        """Bootstrap V(s_T) via critic.forward[BATCH=1], then walk
        compute_gae backward over the host-side rollout buffers.

        GPU path: H2D bootstrap_obs → device, critic.forward on device,
        D2H v1[0] → host scalar. GAE itself runs on host (the
        ROLLOUT_LEN-long sequential recurrence is trivial CPU work and
        a per-step recurrence kernel adds no value at N_ENVS=1)."""
        var bo_cpu_p = state.bootstrap_obs.cpu_ptr()
        var v1_cpu_p = state.v1.cpu_ptr()

        comptime if target == "cpu":
            var bo_t = TileTensor(bo_cpu_p, row_major[1, Self.OBS]())
            var v1_t = TileTensor(v1_cpu_p, row_major[1, 1]())
            critic.forward[target, 1](bo_t, output=v1_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.bootstrap_obs.dev.value(), bo_cpu_p)
            var bo_dev_t = TileTensor(
                state.bootstrap_obs.dev_ptr(), row_major[1, Self.OBS](),
            )
            var v1_dev_t = TileTensor(
                state.v1.dev_ptr(), row_major[1, 1](),
            )
            critic.forward[target, 1](bo_dev_t, output=v1_dev_t)
            ctx.enqueue_copy(v1_cpu_p, state.v1.dev.value())
            ctx.synchronize()

        var next_value = v1_cpu_p[0]
        # GAE backward pass on host-side rollout buffers. Runtime
        # n_steps; comptime-templated form unrolls at ROLLOUT_LEN=2048
        # and explodes Mojo compile.
        compute_gae(
            Self.ROLLOUT_LEN,
            state.rew_buf.cpu_ptr(),
            state.val_buf.cpu_ptr(),
            state.term_buf.cpu_ptr(),
            next_value,
            gamma,
            gae_lambda,
            state.adv_buf.cpu_ptr(),
            state.ret_buf.cpu_ptr(),
        )
