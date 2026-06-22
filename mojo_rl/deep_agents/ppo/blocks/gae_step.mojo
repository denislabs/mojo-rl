"""PPOGAEStep — per-env Generalized Advantage Estimation over the rollout.

Bootstraps V(s_T) for all N_ENVS via a BATCH=N_ENVS critic.forward on
`state.bootstrap_obs`, then walks GAE backward for each env independently
(T-major layout — strided reads at gap N_ENVS).

GPU path (hybrid N=1+): bootstrap critic forward on device, D2H v1,
GAE itself runs on host (sequential recurrence per env; a per-env
parallel scan kernel adds no value below very large N_ENVS).

STORAGE migration: critic is a storage `Module` (`forward[target, B, POLICY](
TensorRefs[1](bootstrap_obs), v1, ctx)`). On GPU `bootstrap_obs.upload(ctx)`
stages H2D, the critic forward runs on device, then `v1.download(ctx)` reads
the bootstrap values back on host. The GAE recurrence reads/writes the rollout
buffers' host `.data` (sanctioned host loops).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
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
        POLICY: AMPPolicy = NoAMP,
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
        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.bootstrap_obs.upload(ctx)
            critic.forward[target, N_ENVS, POLICY=POLICY](
                TensorRefs[Self.CRITIC.ARITY](state.bootstrap_obs), state.v1, state.ctx
            )
            state.v1.download(ctx)
        else:
            critic.forward[target, N_ENVS, POLICY=POLICY](
                TensorRefs[Self.CRITIC.ARITY](state.bootstrap_obs), state.v1, state.ctx
            )

        # Per-env GAE backward pass over T-major rollout buffers (host-side
        # `.data` Lists, indexed directly — no raw pointers).
        # Layout: buf[t * N_ENVS + e] for time t, env e.
        ref v1 = state.v1.data
        ref rew = state.rew_buf.data
        ref val = state.val_buf.data
        ref term = state.term_buf.data
        ref adv = state.adv_buf.data
        ref ret = state.ret_buf.data
        for e in range(N_ENVS):
            var last_gae: Scalar[DT] = 0.0
            var next_value_e = v1[e]
            for t in range(Self.ROLLOUT_LEN - 1, -1, -1):
                var idx = t * N_ENVS + e
                var nonterm = Scalar[DT](1.0) - term[idx]
                var nv: Scalar[DT]
                if t == Self.ROLLOUT_LEN - 1:
                    nv = next_value_e
                else:
                    nv = val[(t + 1) * N_ENVS + e]
                var delta = rew[idx] + gamma * nv * nonterm - val[idx]
                last_gae = delta + gamma * gae_lambda * nonterm * last_gae
                adv[idx] = last_gae
                ret[idx] = last_gae + val[idx]
