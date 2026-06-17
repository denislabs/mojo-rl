"""EnsembleActorLoss — REDQ SAC-style actor loss over N online critics.

Phase R.2 (CPU). Same shape as SAC's actor loss but uses the MEAN
of all N online critics instead of `min(Q1, Q2)`:

    loss_per_b[b] = α · log_prob[b] − combined_Q[b]
    combined_Q[b] = (1/N) · Σᵢ Qᵢ(s[b], rsample(π(s[b])))
    loss          = mean_b(loss_per_b)

Backward derivation:
    ∂loss/∂loss_per_b[b] = 1/B
    ∂loss/∂log_prob[b]   = α / B          (α treated as constant in actor opt)
    ∂loss/∂combined_Q[b] = −1 / B
    ∂loss/∂Qᵢ[b]         = (1/N) · ∂loss/∂combined_Q[b] = −1/(N·B)

Critic param-grad gating — `mode="input_only"` on every critic.vjp.
Per `Module.vjp` docs (`mojo_rl/nn/core/module.mojo`), this skips the
param-grad accumulation step entirely. We never call `opt.step` on
any critic in this block, so even if `mode` were ignored the critic
params would stay unchanged; the explicit `input_only` is the
semantic stop-grad (matches SAC's `ExternalNode[..., MODE="input_only"]`
pattern in `sac/actor_loss.mojo`).

Returns `EnsembleActorLossResult { loss, log_prob_mean }`. The
trainer reads `log_prob_mean` for `AlphaUpdateStep` (the entropy
temperature gradient is `log_prob_mean + target_entropy` — identical
to SAC, hence no `AlphaUpdateStep` changes needed in R.3).

R.2 is CPU-only. GPU comes alongside the full GPU REDQ trainer; the
forward + backward layout here is shaped so that each kernel call
swaps to its `_gpu` variant without restructuring the loop.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.optimizer.adam import Adam

from ..primitives.rsample import RSample
from .ensemble import CriticEnsemble


# ────────────────────────────────────────────────────────────────────
# GPU helper kernels.
# ────────────────────────────────────────────────────────────────────


def _eal_zero_kernel[
    N: Int
](dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],):
    """`dst[i] = 0`."""
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = Scalar[DT](0.0)


def _eal_add_into_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """`dst[i] += src[i]`."""
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = dst[idx] + src[idx]


def _eal_fill_const_kernel[
    N: Int
](dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin], value: Scalar[DT],):
    """`dst[i] = value` — seeds grad_q_i with the −1/(N·B) constant."""
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = value


def _eal_concat_sa_extract_lp_kernel[
    OBS: Int,
    ACT: Int,
    BATCH: Int,
    SA_DIM: Int,
    ALP_DIM: Int,
](
    s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    alp: LayoutTensor[DT, Layout.row_major(BATCH, ALP_DIM), MutAnyOrigin],
    sa: LayoutTensor[DT, Layout.row_major(BATCH, SA_DIM), MutAnyOrigin],
    lp: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """sa[b, :OBS] = s[b, :], sa[b, OBS:] = alp[b, :ACT], lp[b] =
    alp[b, ACT]. One thread per output cell of sa; the lp write is
    gated on d == 0 so each batch index writes lp[b] exactly once."""
    var idx = Int(global_idx.x)
    var total = BATCH * SA_DIM
    if idx >= total:
        return
    var b = idx // SA_DIM
    var d = idx % SA_DIM
    if d < OBS:
        sa[b, d] = rebind[Scalar[DT]](s[b, d])
    else:
        sa[b, d] = rebind[Scalar[DT]](alp[b, d - OBS])
    if d == 0:
        lp[b] = rebind[Scalar[DT]](alp[b, ACT])


def _eal_build_grad_alp_kernel[
    BATCH: Int,
    OBS: Int,
    ACT: Int,
    SA_DIM: Int,
    ALP_DIM: Int,
](
    grad_sa_sum: LayoutTensor[
        DT,
        Layout.row_major(BATCH, SA_DIM),
        MutAnyOrigin,
    ],
    grad_alp: LayoutTensor[
        DT,
        Layout.row_major(BATCH, ALP_DIM),
        MutAnyOrigin,
    ],
    grad_lp_const: Scalar[DT],
):
    """grad_alp[b, :ACT] = grad_sa_sum[b, OBS:]
       grad_alp[b, ACT]  = grad_lp_const   (= α / B)

    One thread per output cell of grad_alp."""
    var idx = Int(global_idx.x)
    var total = BATCH * ALP_DIM
    if idx >= total:
        return
    var b = idx // ALP_DIM
    var j = idx % ALP_DIM
    if j < ACT:
        grad_alp[b, j] = rebind[Scalar[DT]](grad_sa_sum[b, OBS + j])
    else:
        grad_alp[b, j] = grad_lp_const


@fieldwise_init
struct EnsembleActorLossResult(Movable & ImplicitlyDeletable):
    """Forward/backward result: scalar loss + log_prob_mean (the
    Σ_b log π(a|s) / B used by the AlphaUpdateStep)."""

    var loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]


struct EnsembleActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    N_: Int,
    BATCH_: Int,
    OBS_: Int,
    ACT_: Int,
](Movable & ImplicitlyDeletable):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime SA_DIM = Self.OBS + Self.ACT
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    # Forward scratches.
    var _mb_ao: Scratch["eal_mb_ao", Self.BATCH * (2 * Self.ACT)]
    var _mb_alp: Scratch["eal_mb_alp", Self.BATCH * (Self.ACT + 1)]
    var _mb_sa: Scratch["eal_mb_sa", Self.BATCH * Self.SA_DIM]
    var _mb_q_i: Scratch["eal_mb_q_i", Self.BATCH]
    var _mb_q_sum: Scratch["eal_mb_q_sum", Self.BATCH]

    # Backward scratches.
    var _mb_grad_q_i: Scratch["eal_mb_grad_q_i", Self.BATCH]
    var _mb_grad_sa_i: Scratch["eal_mb_grad_sa_i", Self.BATCH * Self.SA_DIM]
    var _mb_grad_sa_sum: Scratch[
        "eal_mb_grad_sa_sum",
        Self.BATCH * Self.SA_DIM,
    ]
    var _mb_grad_alp: Scratch[
        "eal_mb_grad_alp",
        Self.BATCH * (Self.ACT + 1),
    ]
    var _mb_grad_ao: Scratch[
        "eal_mb_grad_ao",
        Self.BATCH * (2 * Self.ACT),
    ]
    var _mb_grad_obs: Scratch["eal_mb_grad_obs", Self.BATCH * Self.OBS]

    # GPU-only auxiliary buffers (None on CPU).
    # The host mirrors hold the D2H'd q_sum + lp values used for the
    # scalar loss + log_prob_mean reduction in step 5. `_mb_lp_dev` is
    # the device-side scratch the concat+lp kernel writes into.
    var _mb_lp_dev: Optional[DeviceBuffer[DT]]
    var _mb_q_sum_host: Optional[HostBuffer[DT]]
    var _mb_lp_host: Optional[HostBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Scratch["eal_mb_ao", Self.BATCH * (2 * Self.ACT)]()
        self._mb_alp = Scratch["eal_mb_alp", Self.BATCH * (Self.ACT + 1)]()
        self._mb_sa = Scratch["eal_mb_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_q_i = Scratch["eal_mb_q_i", Self.BATCH]()
        self._mb_q_sum = Scratch["eal_mb_q_sum", Self.BATCH]()
        self._mb_grad_q_i = Scratch["eal_mb_grad_q_i", Self.BATCH]()
        self._mb_grad_sa_i = Scratch[
            "eal_mb_grad_sa_i",
            Self.BATCH * Self.SA_DIM,
        ]()
        self._mb_grad_sa_sum = Scratch[
            "eal_mb_grad_sa_sum",
            Self.BATCH * Self.SA_DIM,
        ]()
        self._mb_grad_alp = Scratch[
            "eal_mb_grad_alp",
            Self.BATCH * (Self.ACT + 1),
        ]()
        self._mb_grad_ao = Scratch[
            "eal_mb_grad_ao",
            Self.BATCH * (2 * Self.ACT),
        ]()
        self._mb_grad_obs = Scratch[
            "eal_mb_grad_obs",
            Self.BATCH * Self.OBS,
        ]()
        self._mb_lp_dev = None
        self._mb_q_sum_host = None
        self._mb_lp_host = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "EnsembleActorLoss: target must be 'cpu' or 'gpu'"
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleActorLoss.make[target='gpu']: ctx required"
                )
        comptime assert (
            Self.ACTOR.IN_DIMS[0] == Self.OBS
        ), "EnsembleActorLoss: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == 2 * Self.ACT
        ), "EnsembleActorLoss: ACTOR.OUT_DIM must equal 2·ACT"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "EnsembleActorLoss: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "EnsembleActorLoss: CRITIC.OUT_DIM must equal 1"
        var b = Self()
        b.rsample = RSample[Self.ACT].make[target, Zero](ctx=ctx)
        b.rsample.action_scale = action_scale
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](b, ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            b._mb_lp_dev = c.enqueue_create_buffer[DT](Self.BATCH)
            b._mb_q_sum_host = c.enqueue_create_host_buffer[DT](Self.BATCH)
            b._mb_lp_host = c.enqueue_create_host_buffer[DT](Self.BATCH)
        return b^

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> EnsembleActorLossResult:
        """One actor gradient step. Reads `mb_s_ptr` (BATCH × OBS),
        consumes `alpha`, writes through `actor` + `actor_opt`.
        Returns (loss, log_prob_mean).

        The N online critics get forward + vjp[input_only] — their
        param grads are NOT touched. Caller MUST NOT call
        `ensemble.opts[i].step` between this method and the next
        ensemble-critic update.
        """
        assert_tag_for["EnsembleActorLoss", target](self.ts.target_tag)

        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        var grad_q_val: Scalar[DT] = -inv_n * inv_b
        var grad_lp_val: Scalar[DT] = alpha * inv_b

        # ── Step 0 — zero actor grad slab.
        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # ── Step 1 — actor.forward(s) → _mb_ao [B, 2·ACT].
        var ao_p = self._mb_ao.target_ptr[target]()
        var s_t = TileTensor(mb_s_ptr, row_major[Self.BATCH, Self.OBS]())
        var ao_t = TileTensor(
            ao_p,
            row_major[Self.BATCH, 2 * Self.ACT](),
        )
        actor.forward[target, Self.BATCH, POLICY](s_t, output=ao_t)

        # ── Step 2 — rsample.forward(ao) → _mb_alp [B, ACT+1].
        var alp_p = self._mb_alp.target_ptr[target]()
        var alp_t = TileTensor(
            alp_p,
            row_major[Self.BATCH, Self.ALP_DIM](),
        )
        self.rsample.forward[target, Self.BATCH, POLICY](
            ao_t,
            output=alp_t,
        )

        # ── Step 3 — sa = concat(s, action) + extract lp[b] = alp[b, ACT].
        var sa_p = self._mb_sa.target_ptr[target]()
        # Borrow target_y's pattern: lp scratch lives in the actor-loss
        # block too — reused later for the host-side scalar reduction.
        # On CPU we walk both the concat and the lp extract inline; on
        # GPU one fused kernel writes both.
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for d in range(Self.OBS):
                    sa_p[b * Self.SA_DIM + d] = mb_s_ptr[b * Self.OBS + d]
                for j in range(Self.ACT):
                    sa_p[b * Self.SA_DIM + Self.OBS + j] = alp_p[
                        b * Self.ALP_DIM + j
                    ]
        else:
            # Note: GPU path uses a dedicated `_mb_lp` written by the
            # concat+lp kernel (declared below as a scratch slab). On
            # CPU the lp is read directly from `alp_p[b*ALP_DIM + ACT]`
            # in the scalar reduction below, so no dedicated scratch
            # is needed there.
            var ctx = self.ts.ctx.value()
            var s_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, Self.OBS),
                MutAnyOrigin,
            ](mb_s_ptr)
            var alp_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, Self.ALP_DIM),
                MutAnyOrigin,
            ](alp_p)
            var sa_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, Self.SA_DIM),
                MutAnyOrigin,
            ](sa_p)
            var lp_dev = self._mb_lp_dev.value()
            var lp_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
            ](lp_dev)
            comptime total_sa = Self.BATCH * Self.SA_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _eal_concat_sa_extract_lp_kernel[
                Self.OBS,
                Self.ACT,
                Self.BATCH,
                Self.SA_DIM,
                Self.ALP_DIM,
            ]
            ctx.enqueue_function[kernel](
                s_lt,
                alp_lt,
                sa_lt,
                lp_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
        var sa_t = TileTensor(
            sa_p,
            row_major[Self.BATCH, Self.SA_DIM](),
        )

        # ── Step 4 — loop N online critic forwards; accumulate Σᵢ Qᵢ(s,a).
        var q_sum_p = self._mb_q_sum.target_ptr[target]()
        var q_i_p = self._mb_q_i.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                q_sum_p[b] = Scalar[DT](0.0)
        else:
            var ctx = self.ts.ctx.value()
            var q_sum_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
                MutAnyOrigin,
            ](q_sum_p)
            comptime n_blocks_b = (Self.BATCH + TPB - 1) // TPB
            comptime zero_b = _eal_zero_kernel[Self.BATCH]
            ctx.enqueue_function[zero_b](
                q_sum_lt,
                grid_dim=n_blocks_b,
                block_dim=TPB,
            )

        for i in range(Self.N):
            var q_i_t = TileTensor(q_i_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].online.forward[
                target,
                Self.BATCH,
                POLICY,
            ](sa_t, output=q_i_t)
            comptime if target == "cpu":
                for b in range(Self.BATCH):
                    q_sum_p[b] += q_i_p[b]
            else:
                var ctx = self.ts.ctx.value()
                var q_sum_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH),
                    MutAnyOrigin,
                ](q_sum_p)
                var q_i_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH),
                    MutAnyOrigin,
                ](q_i_p)
                comptime n_blocks_a = (Self.BATCH + TPB - 1) // TPB
                comptime add_b = _eal_add_into_kernel[Self.BATCH]
                ctx.enqueue_function[add_b](
                    q_sum_lt,
                    q_i_lt,
                    grid_dim=n_blocks_a,
                    block_dim=TPB,
                )

        # ── Step 5 — host-side scalar reduction: loss + log_prob_mean.
        # GPU: D2H _mb_q_sum [BATCH] + _mb_lp_dev [BATCH] into host
        # mirrors, then the same scalar loop. Two tiny D2Hs per step —
        # acceptable since REDQ doesn't capture under CUDA graphs (host
        # control flow with subset sampling + policy delay).
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
        loss *= inv_b
        var log_prob_mean = lp_sum * inv_b

        # ── Step 6 — backward seed: grad_qᵢ[b] = −1/(N·B) for every (i, b).
        var grad_q_i_p = self._mb_grad_q_i.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                grad_q_i_p[b] = grad_q_val
        else:
            var ctx = self.ts.ctx.value()
            var grad_q_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
                MutAnyOrigin,
            ](grad_q_i_p)
            comptime n_blocks_q = (Self.BATCH + TPB - 1) // TPB
            comptime fill_b = _eal_fill_const_kernel[Self.BATCH]
            ctx.enqueue_function[fill_b](
                grad_q_lt,
                grad_q_val,
                grid_dim=n_blocks_q,
                block_dim=TPB,
            )
        var grad_q_i_t = TileTensor(
            grad_q_i_p,
            row_major[Self.BATCH, 1](),
        )

        # ── Step 7 — for each critic: vjp[input_only] → accumulate grad_sa.
        var grad_sa_sum_p = self._mb_grad_sa_sum.target_ptr[target]()
        comptime if target == "cpu":
            for k in range(Self.BATCH * Self.SA_DIM):
                grad_sa_sum_p[k] = Scalar[DT](0.0)
        else:
            var ctx = self.ts.ctx.value()
            var grad_sa_sum_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH * Self.SA_DIM),
                MutAnyOrigin,
            ](grad_sa_sum_p)
            comptime total_gss = Self.BATCH * Self.SA_DIM
            comptime n_blocks_z = (total_gss + TPB - 1) // TPB
            comptime zero_gss = _eal_zero_kernel[total_gss]
            ctx.enqueue_function[zero_gss](
                grad_sa_sum_lt,
                grid_dim=n_blocks_z,
                block_dim=TPB,
            )

        var grad_sa_i_p = self._mb_grad_sa_i.target_ptr[target]()
        for i in range(Self.N):
            var grad_sa_i_t = TileTensor(
                grad_sa_i_p,
                row_major[Self.BATCH, Self.SA_DIM](),
            )
            # Re-run critic.forward with the exact same sa — critic caches the
            # forward state for the immediately-following vjp call. We did
            # forward(sa) earlier per critic in step 4 but the cache may have
            # been clobbered by later critics' forwards (each critic owns its
            # OWN cache, so actually it survives — but we re-forward to be
            # robust to any future caching changes).
            var q_i_t = TileTensor(q_i_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].online.forward[
                target,
                Self.BATCH,
                POLICY,
            ](sa_t, output=q_i_t)
            ensemble.pairs[i].online.vjp[
                target,
                Self.BATCH,
                POLICY,
                mode="input_only",
            ](grad_q_i_t, grad_sa_i_t)
            comptime if target == "cpu":
                for k in range(Self.BATCH * Self.SA_DIM):
                    grad_sa_sum_p[k] += grad_sa_i_p[k]
            else:
                var ctx = self.ts.ctx.value()
                var grad_sa_sum_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH * Self.SA_DIM),
                    MutAnyOrigin,
                ](grad_sa_sum_p)
                var grad_sa_i_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH * Self.SA_DIM),
                    MutAnyOrigin,
                ](grad_sa_i_p)
                comptime total_gss = Self.BATCH * Self.SA_DIM
                comptime n_blocks_a = (total_gss + TPB - 1) // TPB
                comptime add_gss = _eal_add_into_kernel[total_gss]
                ctx.enqueue_function[add_gss](
                    grad_sa_sum_lt,
                    grad_sa_i_lt,
                    grid_dim=n_blocks_a,
                    block_dim=TPB,
                )

        # ── Step 8 — assemble grad_alp [B, ACT+1]:
        # grad_action[b, j] = grad_sa_sum[b, OBS + j]
        # grad_log_prob[b]  = α / B
        var grad_alp_p = self._mb_grad_alp.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for j in range(Self.ACT):
                    grad_alp_p[b * Self.ALP_DIM + j] = grad_sa_sum_p[
                        b * Self.SA_DIM + Self.OBS + j
                    ]
                grad_alp_p[b * Self.ALP_DIM + Self.ACT] = grad_lp_val
        else:
            var ctx = self.ts.ctx.value()
            var grad_sa_sum_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, Self.SA_DIM),
                MutAnyOrigin,
            ](grad_sa_sum_p)
            var grad_alp_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, Self.ALP_DIM),
                MutAnyOrigin,
            ](grad_alp_p)
            comptime total_galp = Self.BATCH * Self.ALP_DIM
            comptime n_blocks_g = (total_galp + TPB - 1) // TPB
            comptime build_galp = _eal_build_grad_alp_kernel[
                Self.BATCH,
                Self.OBS,
                Self.ACT,
                Self.SA_DIM,
                Self.ALP_DIM,
            ]
            ctx.enqueue_function[build_galp](
                grad_sa_sum_lt,
                grad_alp_lt,
                grad_lp_val,
                grid_dim=n_blocks_g,
                block_dim=TPB,
            )
        var grad_alp_t = TileTensor(
            grad_alp_p,
            row_major[Self.BATCH, Self.ALP_DIM](),
        )

        # ── Step 9 — rsample.vjp(grad_alp) → grad_ao [B, 2·ACT].
        var grad_ao_p = self._mb_grad_ao.target_ptr[target]()
        var grad_ao_t = TileTensor(
            grad_ao_p,
            row_major[Self.BATCH, 2 * Self.ACT](),
        )
        self.rsample.vjp[target, Self.BATCH, POLICY](
            grad_alp_t,
            grad_ao_t,
        )

        # ── Step 10 — actor.vjp(grad_ao) → grad_obs (discarded);
        # accumulates actor param grads.
        var grad_obs_p = self._mb_grad_obs.target_ptr[target]()
        var grad_obs_t = TileTensor(
            grad_obs_p,
            row_major[Self.BATCH, Self.OBS](),
        )
        actor.vjp[target, Self.BATCH, POLICY, mode="all"](
            grad_ao_t,
            grad_obs_t,
        )

        # ── Step 11 — actor_opt.step(actor).
        actor_opt.step[target, M=Self.ACTOR](actor)

        return EnsembleActorLossResult(
            loss=loss,
            log_prob_mean=log_prob_mean,
        )
