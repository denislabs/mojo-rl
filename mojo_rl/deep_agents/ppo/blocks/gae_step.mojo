"""PPOGAEStep — per-env Generalized Advantage Estimation over the rollout.

Bootstraps V(s_T) for all N_ENVS via a BATCH=N_ENVS critic.forward on
`state.bootstrap_obs`, then walks `compute_gae` backward for each env
independently (T-major layout — strided reads at gap N_ENVS).

GPU path (hybrid N=1+): bootstrap critic forward on device, D2H v1,
GAE itself runs on host (sequential recurrence per env; a per-env
parallel scan kernel adds no value below very large N_ENVS).
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from ...training.onpolicy_state import OnPolicyState


struct PPOGAEStep[
    OBS_: Int,
    ROLLOUT_LEN_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
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
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut critic: Self.CRITIC,
        gamma: Scalar[DT],
        gae_lambda: Scalar[DT],
    ) raises:
        """Bootstrap V(s_T) per env via critic.forward[BATCH=N_ENVS],
        then walk GAE backward independently per env over the
        T-major host-side rollout buffers.

        GPU path: H2D bootstrap_obs → device, critic.forward on device,
        D2H v1 → host. GAE itself runs on host (sequential recurrence
        per env; ROLLOUT_LEN-long, trivial CPU work)."""
        var bo_cpu_p = state.bootstrap_obs.cpu_ptr()
        var v1_cpu_p = state.v1.cpu_ptr()

        comptime if target == "cpu":
            var bo_t = TileTensor(bo_cpu_p, row_major[N_ENVS, Self.OBS]())
            var v1_t = TileTensor(v1_cpu_p, row_major[N_ENVS, 1]())
            critic.forward[target, N_ENVS](bo_t, output=v1_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.bootstrap_obs.dev.value(), bo_cpu_p)
            var bo_dev_t = TileTensor(
                state.bootstrap_obs.dev_ptr(),
                row_major[N_ENVS, Self.OBS](),
            )
            var v1_dev_t = TileTensor(
                state.v1.dev_ptr(), row_major[N_ENVS, 1](),
            )
            critic.forward[target, N_ENVS](bo_dev_t, output=v1_dev_t)
            ctx.enqueue_copy(v1_cpu_p, state.v1.dev.value())
            ctx.synchronize()

        # Per-env GAE backward pass over T-major rollout buffers.
        # Layout: buf[t * N_ENVS + e] for time t, env e.
        var rew_p  = state.rew_buf.cpu_ptr()
        var val_p  = state.val_buf.cpu_ptr()
        var term_p = state.term_buf.cpu_ptr()
        var adv_p  = state.adv_buf.cpu_ptr()
        var ret_p  = state.ret_buf.cpu_ptr()
        for e in range(N_ENVS):
            var last_gae: Scalar[DT] = 0.0
            var next_value_e = v1_cpu_p[e]
            for t in range(Self.ROLLOUT_LEN - 1, -1, -1):
                var idx = t * N_ENVS + e
                var nonterm = Scalar[DT](1.0) - term_p[idx]
                var nv: Scalar[DT]
                if t == Self.ROLLOUT_LEN - 1:
                    nv = next_value_e
                else:
                    nv = val_p[(t + 1) * N_ENVS + e]
                var delta = rew_p[idx] + gamma * nv * nonterm - val_p[idx]
                last_gae = delta + gamma * gae_lambda * nonterm * last_gae
                adv_p[idx] = last_gae
                ret_p[idx] = last_gae + val_p[idx]
