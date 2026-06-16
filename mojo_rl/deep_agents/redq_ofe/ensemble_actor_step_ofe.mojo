"""EnsembleActorStepOFE — REDQ-OFE actor loss + gradient step.

Phase O.2.b.2 (CPU). Port of `redq.EnsembleActorLoss` with the OFE
data path inserted between the concat step and the N critic forwards
on the forward side, and between the critic backwards and the
rsample backward on the backward side.

Forward chain (φ-aware delta tagged ▌):
  0.   actor_opt.zero_grad(actor)
  1.   actor.forward(φ(s)) → ao [B, 2·ACT]                  ▌ φ in
  2.   rsample.forward(ao) → alp [B, ACT+1] = (a | logπ)
  3.   sa_in = concat(φ(s), a), extract lp[b] = alp[b, ACT] ▌ wider
  4. ▌ action_branch.forward(sa_in) → φ(s, a) [B, PHI_SA]    ▌ NEW
  5.   for i in 0..N: q_i = critic_i(φ(s, a)); q_sum += q_i
  6.   loss = mean_b(α · lp[b] − q_sum[b] / N)
       lp_mean = mean_b(lp[b])

Backward chain:
  7.   for i in 0..N:
         grad_q_i[b] = −1 / (N · B)           (constant fill)
         critic_i.vjp[input_only](grad_q_i) → grad_φ(s, a)_i
         grad_φ(s, a)_sum += grad_φ(s, a)_i
  8. ▌ action_branch.vjp[input_only](grad_φ(s, a)_sum)       ▌ NEW
       → grad_sa_in [B, PHI_S+ACT]
  9.   grad_alp[b, :ACT] = grad_sa_in[b, PHI_S:PHI_S+ACT]
       grad_alp[b, ACT]  = α / B
       (grad_sa_in[b, :PHI_S] is DISCARDED — no backprop through SB)
  10.  rsample.vjp(grad_alp) → grad_ao
  11.  actor.vjp(grad_ao) → grad_φ(s) dummy  (DISCARDED)
  12.  actor_opt.step(actor)

Returns `EnsembleActorLossResult { loss, log_prob_mean }`.

Gradient policy
===============
  - actor (mode='all'): trained on this path
  - action_branch (mode='input_only'): stop-grad — trained only via
    the aux path (`OFEAuxLossStep`). The grad flowing back through
    AB feeds the rsample backward correctly even with input_only,
    because input_only only stops PARAM accumulation, the input
    gradient is still computed.
  - critics (mode='input_only'): stop-grad — they're trained on the
    REDQ critic-loss path. The actor loss must not perturb them.
  - state_branch: NEVER touched here — we discard grad_φ(s) from both
    the concat split AND from actor.vjp.

CPU-only for O.2.b.2; the GPU port lands alongside the REDQOFETrainer.
"""

from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.optimizer.adam import Adam

from ..primitives.rsample import RSample
from ..redq.ensemble import CriticEnsemble
from ..redq.ensemble_actor_loss import (
    EnsembleActorLossResult,
    _eal_zero_kernel,
    _eal_add_into_kernel,
    _eal_fill_const_kernel,
    _eal_concat_sa_extract_lp_kernel,
    _eal_build_grad_alp_kernel,
)


struct EnsembleActorStepOFE[
    ACTOR: Module,           # IN=PHI_S_DIM, OUT=2·ACT
    AB: Module,              # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    CRITIC: Module,          # IN=PHI_SA_DIM, OUT=1
    N_: Int,
    BATCH_: Int,
    PHI_S_DIM_: Int,
    ACT_: Int,
](Movable & ImplicitlyDestructible):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime PHI_S_DIM = Self.PHI_S_DIM_
    comptime ACT = Self.ACT_
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    # Forward scratches.
    var _mb_ao: Scratch["oas_mb_ao", Self.BATCH * (2 * Self.ACT)]
    var _mb_alp: Scratch["oas_mb_alp", Self.BATCH * (Self.ACT + 1)]
    var _mb_sa_in: Scratch["oas_mb_sa_in", Self.BATCH * Self.SA_IN_DIM]
    var _mb_phi_sa: Scratch["oas_mb_phi_sa", Self.BATCH * Self.PHI_SA_DIM]
    var _mb_q_i: Scratch["oas_mb_q_i", Self.BATCH]
    var _mb_q_sum: Scratch["oas_mb_q_sum", Self.BATCH]

    # Backward scratches.
    var _mb_grad_q_i: Scratch["oas_mb_grad_q_i", Self.BATCH]
    var _mb_grad_phi_sa_i: Scratch[
        "oas_mb_grad_phi_sa_i", Self.BATCH * Self.PHI_SA_DIM,
    ]
    var _mb_grad_phi_sa_sum: Scratch[
        "oas_mb_grad_phi_sa_sum", Self.BATCH * Self.PHI_SA_DIM,
    ]
    var _mb_grad_sa_in: Scratch[
        "oas_mb_grad_sa_in", Self.BATCH * Self.SA_IN_DIM,
    ]
    var _mb_grad_alp: Scratch[
        "oas_mb_grad_alp", Self.BATCH * (Self.ACT + 1),
    ]
    var _mb_grad_ao: Scratch[
        "oas_mb_grad_ao", Self.BATCH * (2 * Self.ACT),
    ]
    # actor.vjp writes grad_φ(s) which we discard (no SB backprop on
    # RL path) — the slab still has to exist for the vjp call.
    var _mb_grad_phi_s_dummy: Scratch[
        "oas_mb_grad_phi_s_dummy", Self.BATCH * Self.PHI_S_DIM,
    ]

    # GPU-only auxiliary buffers. `_mb_lp_dev` is a device-side lp
    # slab written by the concat+lp kernel; the two host buffers
    # mirror it + q_sum for the host-side loss reduction (REDQ-OFE
    # doesn't capture under CUDA graphs, so D2H is cheap).
    var _mb_lp_dev: Optional[DeviceBuffer[DT]]
    var _mb_q_sum_host: Optional[HostBuffer[DT]]
    var _mb_lp_host: Optional[HostBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Scratch[
            "oas_mb_ao", Self.BATCH * (2 * Self.ACT),
        ]()
        self._mb_alp = Scratch[
            "oas_mb_alp", Self.BATCH * (Self.ACT + 1),
        ]()
        self._mb_sa_in = Scratch[
            "oas_mb_sa_in", Self.BATCH * Self.SA_IN_DIM,
        ]()
        self._mb_phi_sa = Scratch[
            "oas_mb_phi_sa", Self.BATCH * Self.PHI_SA_DIM,
        ]()
        self._mb_q_i = Scratch["oas_mb_q_i", Self.BATCH]()
        self._mb_q_sum = Scratch["oas_mb_q_sum", Self.BATCH]()
        self._mb_grad_q_i = Scratch["oas_mb_grad_q_i", Self.BATCH]()
        self._mb_grad_phi_sa_i = Scratch[
            "oas_mb_grad_phi_sa_i", Self.BATCH * Self.PHI_SA_DIM,
        ]()
        self._mb_grad_phi_sa_sum = Scratch[
            "oas_mb_grad_phi_sa_sum", Self.BATCH * Self.PHI_SA_DIM,
        ]()
        self._mb_grad_sa_in = Scratch[
            "oas_mb_grad_sa_in", Self.BATCH * Self.SA_IN_DIM,
        ]()
        self._mb_grad_alp = Scratch[
            "oas_mb_grad_alp", Self.BATCH * (Self.ACT + 1),
        ]()
        self._mb_grad_ao = Scratch[
            "oas_mb_grad_ao", Self.BATCH * (2 * Self.ACT),
        ]()
        self._mb_grad_phi_s_dummy = Scratch[
            "oas_mb_grad_phi_s_dummy", Self.BATCH * Self.PHI_S_DIM,
        ]()
        self._mb_lp_dev = None
        self._mb_q_sum_host = None
        self._mb_lp_host = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleActorStepOFE: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleActorStepOFE.make[target='gpu']: ctx required"
                )
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.PHI_S_DIM, (
            "EnsembleActorStepOFE: ACTOR.IN must equal PHI_S_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleActorStepOFE: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "EnsembleActorStepOFE: AB.IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "EnsembleActorStepOFE: CRITIC.IN must equal PHI_SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleActorStepOFE: CRITIC.OUT_DIM must equal 1"
        )
        var b = Self()
        b.rsample = RSample[Self.ACT].make[target, Zero](ctx=ctx)
        b.rsample.action_scale = action_scale
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](b, ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            b._mb_lp_dev = c.enqueue_create_buffer[DT](Self.BATCH)
            b._mb_q_sum_host = c.enqueue_create_host_buffer[DT](
                Self.BATCH
            )
            b._mb_lp_host = c.enqueue_create_host_buffer[DT](Self.BATCH)
        return b^

    def forward_backward[
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut action_branch: Self.AB,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mb_phi_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> EnsembleActorLossResult:
        """Forward + backward + actor_opt.step. Returns (loss,
        log_prob_mean). The N online critics + the action branch are
        all touched in `mode='input_only'` — their param grads are
        NOT modified."""
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleActorStepOFE: target must be 'cpu' or 'gpu'"
        )
        assert_tag_for["EnsembleActorStepOFE", target](self.ts.target_tag)

        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        var grad_q_val: Scalar[DT] = -inv_n * inv_b
        var grad_lp_val: Scalar[DT] = alpha * inv_b

        # ── Step 0 — zero actor grad slab.
        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # ── Step 1 — actor.forward(φ(s)) → _mb_ao [B, 2·ACT].
        var ao_p = self._mb_ao.target_ptr[target]()
        var phi_s_t = TileTensor(
            mb_phi_s_ptr, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        var ao_t = TileTensor(
            ao_p, row_major[Self.BATCH, 2 * Self.ACT](),
        )
        actor.forward[target, Self.BATCH, POLICY](phi_s_t, output=ao_t)

        # ── Step 2 — rsample.forward(ao) → _mb_alp [B, ACT+1].
        var alp_p = self._mb_alp.target_ptr[target]()
        var alp_t = TileTensor(
            alp_p, row_major[Self.BATCH, Self.ALP_DIM](),
        )
        self.rsample.forward[target, Self.BATCH, POLICY](
            ao_t, output=alp_t,
        )

        # ── Step 3 — sa_in = concat(φ(s), a) (action portion of alp).
        # Also extract lp[b] = alp[b, ACT] into _mb_lp_dev on GPU (the
        # CPU host-side reduction reads it directly from alp_p).
        var sa_in_p = self._mb_sa_in.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for d in range(Self.PHI_S_DIM):
                    sa_in_p[b * Self.SA_IN_DIM + d] = mb_phi_s_ptr[
                        b * Self.PHI_S_DIM + d
                    ]
                for j in range(Self.ACT):
                    sa_in_p[b * Self.SA_IN_DIM + Self.PHI_S_DIM + j] = (
                        alp_p[b * Self.ALP_DIM + j]
                    )
        else:
            # Reuse REDQ's _eal_concat_sa_extract_lp_kernel with
            # PHI_S_DIM in the first-input width slot. The kernel
            # writes both sa_in AND _mb_lp_dev[b] = alp[b, ACT].
            var ctx = self.ts.ctx.value()
            var phi_s_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PHI_S_DIM),
                MutAnyOrigin,
            ](mb_phi_s_ptr)
            var alp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.ALP_DIM),
                MutAnyOrigin,
            ](alp_p)
            var sa_in_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.SA_IN_DIM),
                MutAnyOrigin,
            ](sa_in_p)
            var lp_dev = self._mb_lp_dev.value()
            var lp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](lp_dev.unsafe_ptr())
            comptime total_sa = Self.BATCH * Self.SA_IN_DIM
            comptime n_blocks_csa = (total_sa + TPB - 1) // TPB
            comptime concat_kernel = _eal_concat_sa_extract_lp_kernel[
                Self.PHI_S_DIM, Self.ACT, Self.BATCH,
                Self.SA_IN_DIM, Self.ALP_DIM,
            ]
            ctx.enqueue_function[concat_kernel](
                phi_s_lt, alp_lt, sa_in_lt, lp_lt,
                grid_dim=n_blocks_csa, block_dim=TPB,
            )

        # ── Step 4 — action_branch.forward(sa_in) → φ(s, a).
        var sa_in_t = TileTensor(
            sa_in_p, row_major[Self.BATCH, Self.SA_IN_DIM](),
        )
        var phi_sa_p = self._mb_phi_sa.target_ptr[target]()
        var phi_sa_t = TileTensor(
            phi_sa_p, row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        action_branch.forward[target, Self.BATCH, POLICY](
            sa_in_t, output=phi_sa_t,
        )

        # ── Step 5 — loop N online critic forwards; accumulate q_sum.
        var q_sum_p = self._mb_q_sum.target_ptr[target]()
        var q_i_p = self._mb_q_i.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                q_sum_p[b] = Scalar[DT](0.0)
        else:
            var ctx = self.ts.ctx.value()
            var q_sum_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](q_sum_p)
            comptime n_blocks_zb = (Self.BATCH + TPB - 1) // TPB
            comptime zero_b = _eal_zero_kernel[Self.BATCH]
            ctx.enqueue_function[zero_b](
                q_sum_lt,
                grid_dim=n_blocks_zb, block_dim=TPB,
            )

        for i in range(Self.N):
            var q_i_t = TileTensor(q_i_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].online.forward[
                target, Self.BATCH, POLICY,
            ](phi_sa_t, output=q_i_t)
            comptime if target == "cpu":
                for b in range(Self.BATCH):
                    q_sum_p[b] += q_i_p[b]
            else:
                var ctx = self.ts.ctx.value()
                var q_sum_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](q_sum_p)
                var q_i_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](q_i_p)
                comptime n_blocks_ab = (Self.BATCH + TPB - 1) // TPB
                comptime add_b = _eal_add_into_kernel[Self.BATCH]
                ctx.enqueue_function[add_b](
                    q_sum_lt, q_i_lt,
                    grid_dim=n_blocks_ab, block_dim=TPB,
                )

        # ── Step 6 — host-side scalar reduction: loss + log_prob_mean.
        var loss: Scalar[DT] = Scalar[DT](0.0)
        var lp_sum: Scalar[DT] = Scalar[DT](0.0)
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                var combined = q_sum_p[b] * inv_n
                var lp = alp_p[b * Self.ALP_DIM + Self.ACT]
                loss += alpha * lp - combined
                lp_sum += lp
        else:
            var ctx = self.ts.ctx.value()
            var q_sum_host = self._mb_q_sum_host.value()
            var lp_host = self._mb_lp_host.value()
            ctx.enqueue_copy(q_sum_host, self._mb_q_sum.dev.value())
            ctx.enqueue_copy(lp_host, self._mb_lp_dev.value())
            ctx.synchronize()
            var q_hp = q_sum_host.unsafe_ptr()
            var lp_hp = lp_host.unsafe_ptr()
            for b in range(Self.BATCH):
                var combined = q_hp[b] * inv_n
                var lp = lp_hp[b]
                loss += alpha * lp - combined
                lp_sum += lp
        loss = loss * inv_b
        var lp_mean = lp_sum * inv_b

        # ── Step 7 — backward through N critics into grad_φ(s, a)_sum.
        var grad_q_i_p = self._mb_grad_q_i.target_ptr[target]()
        var grad_phi_sa_i_p = self._mb_grad_phi_sa_i.target_ptr[target]()
        var grad_phi_sa_sum_p = (
            self._mb_grad_phi_sa_sum.target_ptr[target]()
        )
        comptime PSA_TOTAL = Self.BATCH * Self.PHI_SA_DIM
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                grad_q_i_p[b] = grad_q_val
            for k in range(PSA_TOTAL):
                grad_phi_sa_sum_p[k] = Scalar[DT](0.0)
        else:
            var ctx = self.ts.ctx.value()
            # grad_q_i_p fill = grad_q_val constant.
            var grad_q_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](grad_q_i_p)
            comptime n_blocks_qf = (Self.BATCH + TPB - 1) // TPB
            comptime fill_b = _eal_fill_const_kernel[Self.BATCH]
            ctx.enqueue_function[fill_b](
                grad_q_lt, grad_q_val,
                grid_dim=n_blocks_qf, block_dim=TPB,
            )
            # Zero grad_phi_sa_sum.
            var grad_psa_sum_lt = LayoutTensor[
                DT, Layout.row_major(PSA_TOTAL), MutAnyOrigin,
            ](grad_phi_sa_sum_p)
            comptime n_blocks_zps = (PSA_TOTAL + TPB - 1) // TPB
            comptime zero_psa = _eal_zero_kernel[PSA_TOTAL]
            ctx.enqueue_function[zero_psa](
                grad_psa_sum_lt,
                grid_dim=n_blocks_zps, block_dim=TPB,
            )
        var grad_q_i_t = TileTensor(
            grad_q_i_p, row_major[Self.BATCH, 1](),
        )
        var grad_phi_sa_i_t = TileTensor(
            grad_phi_sa_i_p, row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        for i in range(Self.N):
            ensemble.pairs[i].online.vjp[
                target, Self.BATCH, POLICY, mode="input_only",
            ](grad_q_i_t, grad_phi_sa_i_t)
            comptime if target == "cpu":
                for k in range(PSA_TOTAL):
                    grad_phi_sa_sum_p[k] += grad_phi_sa_i_p[k]
            else:
                var ctx = self.ts.ctx.value()
                var grad_psa_sum_lt = LayoutTensor[
                    DT, Layout.row_major(PSA_TOTAL), MutAnyOrigin,
                ](grad_phi_sa_sum_p)
                var grad_psa_i_lt = LayoutTensor[
                    DT, Layout.row_major(PSA_TOTAL), MutAnyOrigin,
                ](grad_phi_sa_i_p)
                comptime n_blocks_ips = (PSA_TOTAL + TPB - 1) // TPB
                comptime add_psa = _eal_add_into_kernel[PSA_TOTAL]
                ctx.enqueue_function[add_psa](
                    grad_psa_sum_lt, grad_psa_i_lt,
                    grid_dim=n_blocks_ips, block_dim=TPB,
                )

        # ── Step 8 — action_branch.vjp[input_only] → grad_sa_in.
        var grad_phi_sa_sum_t = TileTensor(
            grad_phi_sa_sum_p,
            row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        var grad_sa_in_p = self._mb_grad_sa_in.target_ptr[target]()
        var grad_sa_in_t = TileTensor(
            grad_sa_in_p, row_major[Self.BATCH, Self.SA_IN_DIM](),
        )
        action_branch.vjp[
            target, Self.BATCH, POLICY, mode="input_only",
        ](grad_phi_sa_sum_t, grad_sa_in_t)

        # ── Step 9 — build grad_alp from grad_sa_in[:, PHI_S:] + α/B.
        var grad_alp_p = self._mb_grad_alp.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for j in range(Self.ACT):
                    grad_alp_p[b * Self.ALP_DIM + j] = grad_sa_in_p[
                        b * Self.SA_IN_DIM + Self.PHI_S_DIM + j
                    ]
                grad_alp_p[b * Self.ALP_DIM + Self.ACT] = grad_lp_val
        else:
            # Reuse REDQ's `_eal_build_grad_alp_kernel`. Pass
            # PHI_S_DIM in the first-input width slot.
            var ctx = self.ts.ctx.value()
            var grad_sa_in_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.SA_IN_DIM),
                MutAnyOrigin,
            ](grad_sa_in_p)
            var grad_alp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.ALP_DIM),
                MutAnyOrigin,
            ](grad_alp_p)
            comptime total_galp = Self.BATCH * Self.ALP_DIM
            comptime n_blocks_g = (total_galp + TPB - 1) // TPB
            comptime build_galp = _eal_build_grad_alp_kernel[
                Self.BATCH, Self.PHI_S_DIM, Self.ACT,
                Self.SA_IN_DIM, Self.ALP_DIM,
            ]
            ctx.enqueue_function[build_galp](
                grad_sa_in_lt, grad_alp_lt, grad_lp_val,
                grid_dim=n_blocks_g, block_dim=TPB,
            )

        # ── Step 10 — rsample.vjp(grad_alp) → grad_ao.
        var grad_alp_t = TileTensor(
            grad_alp_p, row_major[Self.BATCH, Self.ALP_DIM](),
        )
        var grad_ao_p = self._mb_grad_ao.target_ptr[target]()
        var grad_ao_t = TileTensor(
            grad_ao_p, row_major[Self.BATCH, 2 * Self.ACT](),
        )
        self.rsample.vjp[target, Self.BATCH, POLICY](
            grad_alp_t, grad_ao_t,
        )

        # ── Step 11 — actor.vjp(grad_ao) → grad_φ(s) dummy.
        var grad_phi_s_dummy_p = (
            self._mb_grad_phi_s_dummy.target_ptr[target]()
        )
        var grad_phi_s_dummy_t = TileTensor(
            grad_phi_s_dummy_p,
            row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        actor.vjp[target, Self.BATCH, POLICY](
            grad_ao_t, grad_phi_s_dummy_t,
        )

        # ── Step 12 — actor_opt.step.
        actor_opt.step[target, M=Self.ACTOR](actor)

        return EnsembleActorLossResult(loss=loss, log_prob_mean=lp_mean)
