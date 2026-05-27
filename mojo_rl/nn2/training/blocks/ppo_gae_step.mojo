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
        comptime assert target == "cpu", (
            "PPOGAEStep: P.1 is CPU-only (GPU lands in P.2)"
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
        # ── Bootstrap V(s_T).
        var bo_p = state.bootstrap_obs.target_ptr[target]()
        var v1_p = state.v1.target_ptr[target]()
        var bo_t = TileTensor(bo_p, row_major[1, Self.OBS]())
        var v1_t = TileTensor(v1_p, row_major[1, 1]())
        critic.forward[target, 1](bo_t, output=v1_t)
        var next_value = v1_p[0]

        # ── GAE backward pass. Runtime n_steps; comptime-templated
        # form unrolls at ROLLOUT_LEN=2048 and explodes Mojo compile.
        compute_gae(
            Self.ROLLOUT_LEN,
            state.rew_buf.target_ptr[target](),
            state.val_buf.target_ptr[target](),
            state.term_buf.target_ptr[target](),
            next_value,
            gamma,
            gae_lambda,
            state.adv_buf.target_ptr[target](),
            state.ret_buf.target_ptr[target](),
        )
