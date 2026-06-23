"""EnsembleActorLoss — REDQ SAC-style actor loss over N online critics (STORAGE).

Same shape as SAC's actor loss but uses the MEAN of all N online critics
instead of `min(Q1, Q2)`:

    loss_per_b[b] = α · log_prob[b] − combined_Q[b]
    combined_Q[b] = (1/N) · Σᵢ Qᵢ(s[b], rsample(π(s[b])))
    loss          = mean_b(loss_per_b)

Backward derivation:
    ∂loss/∂loss_per_b[b] = 1/B
    ∂loss/∂log_prob[b]   = α / B          (α treated as constant in actor opt)
    ∂loss/∂combined_Q[b] = −1 / B
    ∂loss/∂Qᵢ[b]         = (1/N) · ∂loss/∂combined_Q[b] = −1/(N·B)

STORAGE migration (Stage 5): legacy `Scratch`/`TargetStorage`/`mptr`/TileTensor
gone — scratch are owned `nn.storage.Tensor`s; the actor + RSample + critics use
the storage Module surface (`forward`/`vjp` over `TensorRefs`). The storage
`Module.vjp` has NO `mode` param (always computes both param + input grads); the
N online critics' PARAM grads computed here are HARMLESS / DISCARDED — the very
next `EnsembleCriticStep` calls `critic.zero_grad` before its own forward/vjp/
step, so nothing reads them (same contract SAC's storage `SACActorLoss` relies on
for its twin critics). Only the actor is stepped here.

Mean loss + mean log_prob are host reductions (per-step D2H on GPU; cheap at
REDQ scales — REDQ doesn't capture under CUDA graphs).

Returns `EnsembleActorLossResult { loss, log_prob_mean }`. The trainer reads
`log_prob_mean` for `AlphaUpdateStep`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Zero
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.rsample import RSample

from .ensemble import CriticEnsemble


# ────────────────────────────────────────────────────────────────────
# GPU helper kernels.
# ────────────────────────────────────────────────────────────────────


def _eal_zero_kernel[
    N: Int
](dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]):
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
](dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin], value: Scalar[DT]):
    """`dst[i] = value` — seeds grad_q_i with the −1/(N·B) constant."""
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = value


def _eal_concat_sa_extract_lp_kernel[
    OBS: Int, ACT: Int, BATCH: Int, SA_DIM: Int, ALP_DIM: Int,
](
    s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    alp: LayoutTensor[DT, Layout.row_major(BATCH, ALP_DIM), MutAnyOrigin],
    sa: LayoutTensor[DT, Layout.row_major(BATCH, SA_DIM), MutAnyOrigin],
    lp: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """sa[b, :OBS] = s[b, :], sa[b, OBS:] = alp[b, :ACT], lp[b] = alp[b, ACT]."""
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
    BATCH: Int, OBS: Int, ACT: Int, SA_DIM: Int, ALP_DIM: Int,
](
    grad_sa_sum: LayoutTensor[DT, Layout.row_major(BATCH, SA_DIM), MutAnyOrigin],
    grad_alp: LayoutTensor[DT, Layout.row_major(BATCH, ALP_DIM), MutAnyOrigin],
    grad_lp_const: Scalar[DT],
):
    """grad_alp[b, :ACT] = grad_sa_sum[b, OBS:]; grad_alp[b, ACT] = grad_lp_const
    (= α / B)."""
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
    """Forward/backward result: scalar loss + log_prob_mean."""

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

    # Forward scratch.
    var _mb_ao: Tensor          # [BATCH, 2*ACT]
    var _mb_alp: Tensor         # [BATCH, ACT+1]
    var _mb_sa: Tensor          # [BATCH, SA_DIM]
    var _mb_q_i: Tensor         # [BATCH]
    var _mb_q_sum: Tensor       # [BATCH]
    var _mb_lp: Tensor          # [BATCH] (GPU lp scratch; CPU reads alp directly)

    # Backward scratch.
    var _mb_grad_q_i: Tensor    # [BATCH]
    var _mb_grad_sa_i: Tensor   # [BATCH, SA_DIM]
    var _mb_grad_sa_sum: Tensor # [BATCH, SA_DIM]
    var _mb_grad_alp: Tensor    # [BATCH, ACT+1]
    var _mb_grad_ao: Tensor     # [BATCH, 2*ACT]
    var _mb_grad_obs: Tensor    # [BATCH, OBS]

    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Tensor()
        self._mb_alp = Tensor()
        self._mb_sa = Tensor()
        self._mb_q_i = Tensor()
        self._mb_q_sum = Tensor()
        self._mb_lp = Tensor()
        self._mb_grad_q_i = Tensor()
        self._mb_grad_sa_i = Tensor()
        self._mb_grad_sa_sum = Tensor()
        self._mb_grad_alp = Tensor()
        self._mb_grad_ao = Tensor()
        self._mb_grad_obs = Tensor()
        self.ctx = None

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
        b.ctx = ctx
        comptime if target == "cpu":
            b._mb_ao = Tensor.alloc(Self.BATCH * (2 * Self.ACT))
            b._mb_alp = Tensor.alloc(Self.BATCH * Self.ALP_DIM)
            b._mb_sa = Tensor.alloc(Self.BATCH * Self.SA_DIM)
            b._mb_q_i = Tensor.alloc(Self.BATCH)
            b._mb_q_sum = Tensor.alloc(Self.BATCH)
            b._mb_lp = Tensor.alloc(Self.BATCH)
            b._mb_grad_q_i = Tensor.alloc(Self.BATCH)
            b._mb_grad_sa_i = Tensor.alloc(Self.BATCH * Self.SA_DIM)
            b._mb_grad_sa_sum = Tensor.alloc(Self.BATCH * Self.SA_DIM)
            b._mb_grad_alp = Tensor.alloc(Self.BATCH * Self.ALP_DIM)
            b._mb_grad_ao = Tensor.alloc(Self.BATCH * (2 * Self.ACT))
            b._mb_grad_obs = Tensor.alloc(Self.BATCH * Self.OBS)
        else:
            var c = ctx.value()
            b._mb_ao = Tensor.alloc_gpu(c, Self.BATCH * (2 * Self.ACT))
            b._mb_alp = Tensor.alloc_gpu(c, Self.BATCH * Self.ALP_DIM)
            b._mb_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_DIM)
            b._mb_q_i = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_q_sum = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_lp = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_grad_q_i = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_grad_sa_i = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_DIM)
            b._mb_grad_sa_sum = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_DIM)
            b._mb_grad_alp = Tensor.alloc_gpu(c, Self.BATCH * Self.ALP_DIM)
            b._mb_grad_ao = Tensor.alloc_gpu(c, Self.BATCH * (2 * Self.ACT))
            b._mb_grad_obs = Tensor.alloc_gpu(c, Self.BATCH * Self.OBS)
        return b^

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mut mb_s: Tensor,
        alpha: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> EnsembleActorLossResult:
        """One actor gradient step. Reads `mb_s` (BATCH × OBS), consumes
        `alpha`, writes through `actor` + `actor_opt`. Returns (loss,
        log_prob_mean). The N online critics' param grads are discarded."""
        comptime BB = Self.BATCH
        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
        var grad_q_val: Scalar[DT] = -inv_n * inv_b
        var grad_lp_val: Scalar[DT] = alpha * inv_b

        # ── Step 0 — zero actor grad slab.
        actor.zero_grad[target](ctx)

        # ── Step 1 — actor.forward(s) → _mb_ao [B, 2·ACT].
        call_forward[target, BB, POLICY=POLICY](
            actor, TensorRefs[Self.ACTOR.ARITY](mb_s), self._mb_ao, ctx
        )

        # ── Step 2 — rsample.forward(ao) → _mb_alp [B, ACT+1].
        call_forward[target, BB, POLICY=POLICY](
            self.rsample, TensorRefs[1](self._mb_ao), self._mb_alp, ctx
        )

        # ── Step 3 — sa = concat(s, action) + extract lp[b] = alp[b, ACT].
        comptime if target == "cpu":
            for b in range(BB):
                for d in range(Self.OBS):
                    self._mb_sa.data[b * Self.SA_DIM + d] = (
                        mb_s.data[b * Self.OBS + d]
                    )
                for j in range(Self.ACT):
                    self._mb_sa.data[b * Self.SA_DIM + Self.OBS + j] = (
                        self._mb_alp.data[b * Self.ALP_DIM + j]
                    )
        else:
            var c = ctx.value()
            comptime total_sa = BB * Self.SA_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _eal_concat_sa_extract_lp_kernel[
                Self.OBS, Self.ACT, BB, Self.SA_DIM, Self.ALP_DIM,
            ]
            c.enqueue_function[kernel](
                mb_s.lt["gpu", Layout.row_major(BB, Self.OBS)](),
                self._mb_alp.lt["gpu", Layout.row_major(BB, Self.ALP_DIM)](),
                self._mb_sa.lt["gpu", Layout.row_major(BB, Self.SA_DIM)](),
                self._mb_lp.lt["gpu", Layout.row_major(BB)](),
                grid_dim=n_blocks, block_dim=TPB,
            )

        # ── Step 4 — loop N online critic forwards; accumulate Σᵢ Qᵢ(s,a).
        comptime if target == "cpu":
            for b in range(BB):
                self._mb_q_sum.data[b] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            comptime nbb = (BB + TPB - 1) // TPB
            c.enqueue_function[_eal_zero_kernel[BB]](
                self._mb_q_sum.lt["gpu", Layout.row_major(BB)](),
                grid_dim=nbb, block_dim=TPB,
            )
        for i in range(Self.N):
            call_forward[target, BB, POLICY=POLICY](
                ensemble.pairs[i].online,
                TensorRefs[Self.CRITIC.ARITY](self._mb_sa), self._mb_q_i, ctx
            )
            comptime if target == "cpu":
                for b in range(BB):
                    self._mb_q_sum.data[b] += self._mb_q_i.data[b]
            else:
                var c = ctx.value()
                comptime nbb = (BB + TPB - 1) // TPB
                c.enqueue_function[_eal_add_into_kernel[BB]](
                    self._mb_q_sum.lt["gpu", Layout.row_major(BB)](),
                    self._mb_q_i.lt["gpu", Layout.row_major(BB)](),
                    grid_dim=nbb, block_dim=TPB,
                )

        # ── Step 5 — host-side scalar reduction: loss + log_prob_mean.
        var loss: Scalar[DT] = Scalar[DT](0.0)
        var lp_sum: Scalar[DT] = Scalar[DT](0.0)
        comptime if target == "cpu":
            for b in range(BB):
                var combined = self._mb_q_sum.data[b] * inv_n
                var lp = self._mb_alp.data[b * Self.ALP_DIM + Self.ACT]
                loss += alpha * lp - combined
                lp_sum += lp
        else:
            var c = ctx.value()
            self._mb_q_sum.download(c)
            self._mb_lp.download(c)
            for b in range(BB):
                var combined = self._mb_q_sum.data[b] * inv_n
                var lp = self._mb_lp.data[b]
                loss += alpha * lp - combined
                lp_sum += lp
        loss *= inv_b
        var log_prob_mean = lp_sum * inv_b

        # ── Step 6 — backward seed: grad_qᵢ[b] = −1/(N·B) for every (i, b).
        comptime if target == "cpu":
            for b in range(BB):
                self._mb_grad_q_i.data[b] = grad_q_val
        else:
            var c = ctx.value()
            comptime nbb = (BB + TPB - 1) // TPB
            c.enqueue_function[_eal_fill_const_kernel[BB]](
                self._mb_grad_q_i.lt["gpu", Layout.row_major(BB)](),
                grad_q_val,
                grid_dim=nbb, block_dim=TPB,
            )

        # ── Step 7 — for each critic: vjp → accumulate grad_sa.
        comptime if target == "cpu":
            for k in range(BB * Self.SA_DIM):
                self._mb_grad_sa_sum.data[k] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            comptime total_gss = BB * Self.SA_DIM
            comptime n_blocks_z = (total_gss + TPB - 1) // TPB
            c.enqueue_function[_eal_zero_kernel[total_gss]](
                self._mb_grad_sa_sum.lt["gpu", Layout.row_major(total_gss)](),
                grid_dim=n_blocks_z, block_dim=TPB,
            )
        for i in range(Self.N):
            # Re-forward the SAME sa to refresh this critic's vjp cache, then
            # vjp. The critic's PARAM grads accumulate but are DISCARDED (the
            # next EnsembleCriticStep zero_grads before its own update).
            call_forward[target, BB, POLICY=POLICY](
                ensemble.pairs[i].online,
                TensorRefs[Self.CRITIC.ARITY](self._mb_sa), self._mb_q_i, ctx
            )
            call_vjp[target, BB, POLICY=POLICY](
                ensemble.pairs[i].online,
                TensorRefs[Self.CRITIC.ARITY](self._mb_sa),
                self._mb_grad_q_i,
                TensorRefs[Self.CRITIC.ARITY](self._mb_grad_sa_i),
                ctx,
            )
            comptime if target == "cpu":
                for k in range(BB * Self.SA_DIM):
                    self._mb_grad_sa_sum.data[k] += self._mb_grad_sa_i.data[k]
            else:
                var c = ctx.value()
                comptime total_gss = BB * Self.SA_DIM
                comptime n_blocks_a = (total_gss + TPB - 1) // TPB
                c.enqueue_function[_eal_add_into_kernel[total_gss]](
                    self._mb_grad_sa_sum.lt[
                        "gpu", Layout.row_major(total_gss)
                    ](),
                    self._mb_grad_sa_i.lt["gpu", Layout.row_major(total_gss)](),
                    grid_dim=n_blocks_a, block_dim=TPB,
                )

        # ── Step 8 — assemble grad_alp [B, ACT+1]:
        # grad_action[b, j] = grad_sa_sum[b, OBS + j]; grad_log_prob[b] = α / B.
        comptime if target == "cpu":
            for b in range(BB):
                for j in range(Self.ACT):
                    self._mb_grad_alp.data[b * Self.ALP_DIM + j] = (
                        self._mb_grad_sa_sum.data[b * Self.SA_DIM + Self.OBS + j]
                    )
                self._mb_grad_alp.data[b * Self.ALP_DIM + Self.ACT] = grad_lp_val
        else:
            var c = ctx.value()
            comptime total_galp = BB * Self.ALP_DIM
            comptime n_blocks_g = (total_galp + TPB - 1) // TPB
            comptime build_galp = _eal_build_grad_alp_kernel[
                BB, Self.OBS, Self.ACT, Self.SA_DIM, Self.ALP_DIM,
            ]
            c.enqueue_function[build_galp](
                self._mb_grad_sa_sum.lt[
                    "gpu", Layout.row_major(BB, Self.SA_DIM)
                ](),
                self._mb_grad_alp.lt["gpu", Layout.row_major(BB, Self.ALP_DIM)](),
                grad_lp_val,
                grid_dim=n_blocks_g, block_dim=TPB,
            )

        # ── Step 9 — rsample.vjp(grad_alp) → grad_ao [B, 2·ACT].
        call_vjp[target, BB, POLICY=POLICY](
            self.rsample,
            TensorRefs[1](self._mb_ao),
            self._mb_grad_alp,
            TensorRefs[1](self._mb_grad_ao),
            ctx,
        )

        # ── Step 10 — actor.vjp(grad_ao) → grad_obs (discarded); accumulates
        # actor param grads.
        call_vjp[target, BB, POLICY=POLICY](
            actor,
            TensorRefs[Self.ACTOR.ARITY](mb_s),
            self._mb_grad_ao,
            TensorRefs[Self.ACTOR.ARITY](self._mb_grad_obs),
            ctx,
        )

        # ── Step 11 — actor_opt.step(actor).
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)

        return EnsembleActorLossResult(
            loss=loss, log_prob_mean=log_prob_mean
        )
