"""BatchNorm2D[C, H, W, MOMENTUM, EPSILON] — per-channel BN for spatial inputs.

Phase 5 of `nn/PORTING_PLAN.md`. Mirrors `batch_norm_1d.mojo`'s
surface — γ/β as `Param`s with `decay=False`, running_mean/var as
decay-exempt zero-grad `Param`s too (M1 — they ride the `for_each_param`
walk into the v2 checkpoint envelope; the optimizer visits them but BN
backward never writes their grad, so they stay BIT-EXACT and evolve only
via the forward EMA), per-instance `training: Bool`, `cache_is_training`
flag.

The only structural difference vs BN1D is the reduction axis: stats
are reduced over batch *and* spatial position (H·W), giving
`N_eff = BATCH · H · W` samples per channel. Forward and backward are
otherwise the standard BN formulas, applied per channel.

Comptime shape: input `[BATCH, C, H, W]` flattened to `[BATCH, C·H·W]`;
output is the same shape. Used after every `Conv2D` in a CNN trunk
(NatureDQN doesn't use it, but ResNet-style trunks do).

GPU layout: one block per channel, threads stride over BATCH·SPATIAL
samples and reduce via `block.sum[block_size=BN2D_TPB]`.
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    State,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


comptime BN2D_DEFAULT_EPS: Float64 = 1e-5
comptime BN2D_DEFAULT_MOM: Float64 = 0.1
comptime BN2D_TPB: Int = 128
# Reduction blocks per channel (batch-shards). The training stats/grad
# reductions split each channel's BATCH·SPATIAL reduction across up to this
# many blocks (one per contiguous batch-row shard) → grid = C·G blocks instead
# of C, fixing the one-block-per-channel occupancy collapse on large C/spatial.
comptime BN2D_RBLOCKS: Int = 64


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one block per channel, threads stride over the joint
# (batch, spatial) sample axis. Flat storage layout `[BATCH, C, SPATIAL]`
# is consumed via explicit address arithmetic so the LayoutTensor stays
# at a single 1-D shape we can index directly.
# ──────────────────────────────────────────────────────────────────────


# ── Multi-block training forward: partial → finalize → normalize. ──
# Splits each channel's BATCH·SPATIAL reduction across G batch-shards
# (G = min(BATCH, BN2D_RBLOCKS)) so the grid is C·G blocks, not C. Stats use
# the Σx / Σx² one-pass form (var = E[x²] − E[x]², clamped ≥ 0) — one read of
# the input for stats, one for normalize (was 3 reads in the old single-block
# kernel). Not bit-identical to the CPU two-pass reference (reduction order +
# E[x²]−E[x]² differ), but matches within ~1e-3 (see test_batch_norm_2d_multiblock).
def _bn2d_partial_stats_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int, G: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    partial_sum:   LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
):
    """Pass 1: block (c, g) reduces channel c's batch rows [g·bpb, (g+1)·bpb)
    over all SPATIAL → Σx, Σx² into partial_{sum,sumsq}[c·G+g]. grid=(C·G,)."""
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var c_off = c * SPATIAL
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var x = rebind[Scalar[DT]](input[b, c_off + s])
            my_sum += x
            my_sumsq += x * x
            s += BN2D_TPB
    var bsum = block.sum[block_size=BN2D_TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=BN2D_TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        partial_sum[c * G + g] = bsum[0]
        partial_sumsq[c * G + g] = bsq[0]


def _bn2d_finalize_stats_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, G: Int,
    EPSILON: Float64, MOMENTUM: Float64,
](
    partial_sum:   LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_var:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_mean:    LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    """Pass 2: one block per channel sums its G partials → mean, var
    (E[x²]−E[x]², clamped ≥0), inv_std; folds the finite-guarded running-stat
    EMA. grid=(C,), block=(1,)."""
    var c = Int(block_idx.x)
    if c >= C:
        return
    if Int(thread_idx.x) != 0:
        return
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for g in range(G):
        s += rebind[Scalar[DT]](partial_sum[c * G + g])
        sq += rebind[Scalar[DT]](partial_sumsq[c * G + g])
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH * SPATIAL))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    var inv_std: Scalar[DT] = 1.0 / sqrt(var_ + Scalar[DT](EPSILON))
    cache_mean[c] = mean
    cache_inv_std[c] = inv_std
    # finite-guarded EMA (see the old single-block kernel's note): a float32
    # blow-up makes batch stats non-finite → folding them pins running_* at ±inf
    # → eval BN = inf·0 = NaN forever. Skip non-finite updates (`x-x==0` ⇔ finite).
    if (mean - mean == Scalar[DT](0.0)) and (var_ - var_ == Scalar[DT](0.0)):
        var mom = Scalar[DT](MOMENTUM)
        var one_m = Scalar[DT](1.0) - mom
        running_mean[c] = (
            one_m * rebind[Scalar[DT]](running_mean[c]) + mom * mean
        )
        running_var[c] = (
            one_m * rebind[Scalar[DT]](running_var[c]) + mom * var_
        )


def _bn2d_normalize_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int, G: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_mean:    LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
):
    """Pass 3: block (c, g) over batch rows [g·bpb,…) writes x̂ = (x−μ)·inv_std
    into cache + output = γ·x̂ + β. grid=(C·G,)."""
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var c_off = c * SPATIAL
    var mean = rebind[Scalar[DT]](cache_mean[c])
    var inv_std = rebind[Scalar[DT]](cache_inv_std[c])
    var gm = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var xh = (rebind[Scalar[DT]](input[b, off]) - mean) * inv_std
            cache_xhat[b, off] = xh
            output[b, off] = gm * xh + bt
            s += BN2D_TPB


def _bn2d_forward_eval_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int, EPSILON: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_var:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = rebind[Scalar[DT]](running_mean[c])
    var rv = rebind[Scalar[DT]](running_var[c])
    var inv_std: Scalar[DT] = 1.0 / sqrt(rv + eps)
    var g = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    var c_off = c * SPATIAL
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var x = rebind[Scalar[DT]](input[b, off])
            output[b, off] = g * (x - rm) * inv_std + bt
            s += BN2D_TPB


# ── Multi-block backward: partial → finalize → scatter (mirrors forward). ──
def _bn2d_bwd_partial_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int, G: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    p_dxhat:     LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dxhat_xhat: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dgamma:    LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dbeta:     LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
):
    """Pass 1: block (c, g) reduces its batch-shard → Σdx̂, Σdx̂·x̂, Σdγ, Σdβ
    into the four partial[c·G+g] buffers. grid=(C·G,)."""
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var gm = rebind[Scalar[DT]](gamma[c])
    var c_off = c * SPATIAL
    var s_dxhat: Scalar[DT] = 0.0
    var s_dxx: Scalar[DT] = 0.0
    var s_dg: Scalar[DT] = 0.0
    var s_db: Scalar[DT] = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var dy = rebind[Scalar[DT]](grad_output[b, off])
            var xh = rebind[Scalar[DT]](cache_xhat[b, off])
            var dxhat = dy * gm
            s_dxhat += dxhat
            s_dxx += dxhat * xh
            s_dg += dy * xh
            s_db += dy
            s += BN2D_TPB
    var a = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dxhat)
    var bb = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dxx)
    var cc = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dg)
    var dd = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_db)
    if t == 0:
        p_dxhat[c * G + g] = a[0]
        p_dxhat_xhat[c * G + g] = bb[0]
        p_dgamma[c * G + g] = cc[0]
        p_dbeta[c * G + g] = dd[0]


def _bn2d_bwd_finalize_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, G: Int, mode: StaticString,
](
    p_dxhat:     LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dxhat_xhat: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dgamma:    LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dbeta:     LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    m1_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    m2_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_beta:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    """Pass 2: one block per channel sums its G partials → m1 = Σdx̂/N,
    m2 = Σdx̂·x̂/N; accumulates grad_gamma/beta when mode=="all". grid=(C,)."""
    var c = Int(block_idx.x)
    if c >= C:
        return
    if Int(thread_idx.x) != 0:
        return
    var sa: Scalar[DT] = 0.0
    var sb: Scalar[DT] = 0.0
    var sg: Scalar[DT] = 0.0
    var sd: Scalar[DT] = 0.0
    for g in range(G):
        sa += rebind[Scalar[DT]](p_dxhat[c * G + g])
        sb += rebind[Scalar[DT]](p_dxhat_xhat[c * G + g])
        sg += rebind[Scalar[DT]](p_dgamma[c * G + g])
        sd += rebind[Scalar[DT]](p_dbeta[c * G + g])
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH * SPATIAL))
    m1_out[c] = sa * inv_n
    m2_out[c] = sb * inv_n
    comptime if mode == "all":
        grad_gamma[c] = rebind[Scalar[DT]](grad_gamma[c]) + sg
        grad_beta[c] = rebind[Scalar[DT]](grad_beta[c]) + sd


def _bn2d_bwd_scatter_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int, G: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    m1: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    m2: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
):
    """Pass 3: block (c, g) writes grad_input = inv_std·(dx̂ − m1 − x̂·m2)
    over its batch-shard. grid=(C·G,)."""
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var gm = rebind[Scalar[DT]](gamma[c])
    var inv_std = rebind[Scalar[DT]](cache_inv_std[c])
    var mm1 = rebind[Scalar[DT]](m1[c])
    var mm2 = rebind[Scalar[DT]](m2[c])
    var c_off = c * SPATIAL
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var dy = rebind[Scalar[DT]](grad_output[b, off])
            var xh = rebind[Scalar[DT]](cache_xhat[b, off])
            var dxhat = dy * gm
            grad_input[b, off] = inv_std * (dxhat - mm1 - xh * mm2)
            s += BN2D_TPB


struct BatchNorm2D[
    C: Int, H: Int, W: Int,
    MOMENTUM: Float64 = BN2D_DEFAULT_MOM,
    EPSILON: Float64 = BN2D_DEFAULT_EPS,
](Module):
    comptime ARITY: Int = 1
    comptime FLAT_DIM: Int = Self.C * Self.H * Self.W
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.FLAT_DIM)
    comptime OUT_DIM = Self.FLAT_DIM
    comptime SPATIAL: Int = Self.H * Self.W

    var gamma: Param["gamma", False, Self.C]
    var beta:  Param["beta",  False, Self.C]
    # Running stats — decay-exempt, zero-grad Params (M1); walked by
    # for_each_param into the v2 checkpoint, never moved by the optimizer.
    var running_mean: State["running_mean", Self.C]
    var running_var:  State["running_var", Self.C]
    var cache_xhat: List[Scalar[DT]]     # [BATCH, C, H, W] flat
    var cache_inv_std: List[Scalar[DT]]  # [C]
    var cache_xhat_dev: Optional[DeviceBuffer[DT]]
    var cache_inv_std_dev: Optional[DeviceBuffer[DT]]
    var cache_mean_dev: Optional[DeviceBuffer[DT]]   # [C] — multi-block forward
    # Multi-block reduction scratch (all [C·BN2D_RBLOCKS] or [C]; G≤RBLOCKS used).
    # Forward partials reuse psum/psumsq; backward adds dgamma/dbeta partials +
    # m1/m2 channel buffers.
    var bn_psum_dev:   Optional[DeviceBuffer[DT]]    # [C·RBLOCKS] Σx / Σdx̂
    var bn_psumsq_dev: Optional[DeviceBuffer[DT]]    # [C·RBLOCKS] Σx² / Σdx̂·x̂
    var bn_pdg_dev:    Optional[DeviceBuffer[DT]]    # [C·RBLOCKS] Σdγ
    var bn_pdb_dev:    Optional[DeviceBuffer[DT]]    # [C·RBLOCKS] Σdβ
    var bn_m1_dev:     Optional[DeviceBuffer[DT]]    # [C] backward m1
    var bn_m2_dev:     Optional[DeviceBuffer[DT]]    # [C] backward m2
    var cache_n_batch: Int
    var cache_is_training: Bool
    var training: Bool
    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.C]()
        self.beta  = Param["beta",  False, Self.C]()
        self.running_mean = State["running_mean", Self.C]()
        self.running_var  = State["running_var", Self.C]()
        self.cache_xhat = List[Scalar[DT]]()
        self.cache_inv_std = List[Scalar[DT]]()
        self.cache_xhat_dev = None
        self.cache_inv_std_dev = None
        self.cache_mean_dev = None
        self.bn_psum_dev = None
        self.bn_psumsq_dev = None
        self.bn_pdg_dev = None
        self.bn_pdb_dev = None
        self.bn_m1_dev = None
        self.bn_m2_dev = None
        self.cache_n_batch = 0
        self.cache_is_training = False
        self.training = True
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "BatchNorm2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.C > 0 and Self.H > 0 and Self.W > 0, (
            "BatchNorm2D: C, H, W must all be > 0"
        )
        comptime assert Self.MOMENTUM > 0.0 and Self.MOMENTUM <= 1.0, (
            "BatchNorm2D: MOMENTUM must be in (0, 1]"
        )
        var bn = Self()
        comptime if target == "cpu":
            bn.gamma = Param["gamma", False, Self.C].make_cpu()
            bn.beta  = Param["beta",  False, Self.C].make_cpu()
            var g_ptr = bn.gamma.value_unsafe_ptr_cpu()
            for k in range(Self.C):
                g_ptr[k] = Scalar[DT](1.0)
            bn.running_mean = State["running_mean", Self.C].make_cpu()
            bn.running_var  = State["running_var", Self.C].make_cpu()
            # make_cpu zero-fills value → running_mean already 0; set var←1.
            var rv_ptr = bn.running_var.value_unsafe_ptr_cpu()
            for k in range(Self.C):
                rv_ptr[k] = Scalar[DT](1.0)
            bn.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["BatchNorm2D.make[target='gpu']"](ctx)
            bn.gamma = Param["gamma", False, Self.C].make_gpu(ctx_v)
            bn.beta  = Param["beta",  False, Self.C].make_gpu(ctx_v)
            bn.gamma.val.dev.value().enqueue_fill(1.0)
            bn.beta.val.dev.value().enqueue_fill(0.0)
            bn.running_mean = State["running_mean", Self.C].make_gpu(
                ctx_v
            )
            bn.running_var = State["running_var", Self.C].make_gpu(
                ctx_v
            )
            bn.running_mean.t.dev.value().enqueue_fill(0.0)
            bn.running_var.t.dev.value().enqueue_fill(1.0)
            bn.cache_xhat_dev    = ctx_v.enqueue_create_buffer[DT](1)
            bn.cache_inv_std_dev = ctx_v.enqueue_create_buffer[DT](Self.C)
            bn.cache_mean_dev    = ctx_v.enqueue_create_buffer[DT](Self.C)
            comptime PR = Self.C * BN2D_RBLOCKS
            bn.bn_psum_dev   = ctx_v.enqueue_create_buffer[DT](PR)
            bn.bn_psumsq_dev = ctx_v.enqueue_create_buffer[DT](PR)
            bn.bn_pdg_dev    = ctx_v.enqueue_create_buffer[DT](PR)
            bn.bn_pdb_dev    = ctx_v.enqueue_create_buffer[DT](PR)
            bn.bn_m1_dev = ctx_v.enqueue_create_buffer[DT](Self.C)
            bn.bn_m2_dev = ctx_v.enqueue_create_buffer[DT](Self.C)
            bn.cache_n_batch = 0
            bn.ts = TargetStorage.make_gpu(ctx_v)
        return bn^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_xhat_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.FLAT_DIM
            )
            self.cache_n_batch = batch

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = input.ptr
            var out_p = output_v.ptr
            var g_p = self.gamma.value_unsafe_ptr_cpu()
            var b_p = self.bias_unsafe_ptr_cpu()
            var rm_v = TileTensor(self.running_mean.t.cpu, row_major[Self.C]())
            var rv_v = TileTensor(self.running_var.t.cpu,  row_major[Self.C]())
            var eps = Scalar[DT](Self.EPSILON)
            var n_eff = Scalar[DT](Float64(BATCH * Self.SPATIAL))
            var inv_n = Scalar[DT](1.0) / n_eff
            if self.training:
                ensure_cpu_buffer(
                    self.cache_xhat, BATCH * Self.FLAT_DIM,
                )
                ensure_cpu_buffer(self.cache_inv_std, Self.C)
                var xhat_p = self.cache_xhat.unsafe_ptr()
                var inv_v = TileTensor(
                    self.cache_inv_std, row_major[Self.C](),
                )
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for c in range(Self.C):
                    var g = g_p[c]
                    var bt = b_p[c]
                    var mean: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            mean += in_p[base + s]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            var d = in_p[base + s] - mean
                            var_ += d * d
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[c] = inv_std
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            var xh = (in_p[base + s] - mean) * inv_std
                            xhat_p[base + s] = xh
                            out_p[base + s] = g * xh + bt
                    rm_v[c] = one_m * rm_v[c] + mom * mean
                    rv_v[c] = one_m * rv_v[c] + mom * var_
                self.cache_is_training = True
            else:
                for c in range(Self.C):
                    var rm = rm_v[c]
                    var rv = rv_v[c]
                    var inv_std = Scalar[DT](1.0) / sqrt(rv + eps)
                    var g = g_p[c]
                    var bt = b_p[c]
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            out_p[base + s] = (
                                g * (in_p[base + s] - rm) * inv_std + bt
                            )
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.FLAT_DIM)
            comptime layout_c  = Layout.row_major(Self.C)
            var in_p_w  = input.ptr
            var out_p_w = output_v.ptr
            var in_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p_w)
            var g_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.gamma.val.dev.value()
            )
            var b_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.beta.val.dev.value()
            )
            var rm_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.running_mean.t.dev.value()
            )
            var rv_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.running_var.t.dev.value()
            )
            var ctx = self.ts.ctx.value()
            if self.training:
                self._ensure_cache_gpu(BATCH)
                var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                    self.cache_xhat_dev.value()
                )
                var is_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                    self.cache_inv_std_dev.value()
                )
                var mean_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                    self.cache_mean_dev.value()
                )
                # G batch-shards per channel (≤ BATCH and ≤ BN2D_RBLOCKS).
                comptime G = BATCH if BATCH < BN2D_RBLOCKS else BN2D_RBLOCKS
                comptime layout_pr = Layout.row_major(Self.C * G)
                var psum_lt = LayoutTensor[DT, layout_pr, MutAnyOrigin](
                    self.bn_psum_dev.value()
                )
                var psumsq_lt = LayoutTensor[DT, layout_pr, MutAnyOrigin](
                    self.bn_psumsq_dev.value()
                )
                # Pass 1: C·G blocks → partial Σx, Σx².
                comptime pk = _bn2d_partial_stats_kernel[
                    BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM, G,
                ]
                ctx.enqueue_function[pk](
                    in_lt, psum_lt, psumsq_lt,
                    grid_dim=Self.C * G, block_dim=BN2D_TPB,
                )
                # Pass 2: C blocks → mean/var/inv_std + running-stat EMA.
                comptime ck = _bn2d_finalize_stats_kernel[
                    BATCH, Self.C, Self.SPATIAL, G,
                    Self.EPSILON, Self.MOMENTUM,
                ]
                ctx.enqueue_function[ck](
                    psum_lt, psumsq_lt, rm_lt, rv_lt, mean_lt, is_lt,
                    grid_dim=Self.C, block_dim=1,
                )
                # Pass 3: C·G blocks → normalize + cache x̂.
                comptime nk = _bn2d_normalize_kernel[
                    BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM, G,
                ]
                ctx.enqueue_function[nk](
                    in_lt, out_lt, g_lt, b_lt, mean_lt, is_lt, xh_lt,
                    grid_dim=Self.C * G, block_dim=BN2D_TPB,
                )
                self.cache_is_training = True
            else:
                comptime ekernel = _bn2d_forward_eval_kernel[
                    BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM,
                    Self.EPSILON,
                ]
                ctx.enqueue_function[ekernel](
                    in_lt, out_lt, g_lt, b_lt, rm_lt, rv_lt,
                    grid_dim=Self.C, block_dim=BN2D_TPB,
                )

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        if not self.cache_is_training:
            raise Error(
                "BatchNorm2D.vjp: training-mode cache not populated."
                " Call forward(training=True) before vjp."
            )
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var g_p = self.gamma.value_unsafe_ptr_cpu()
            var dg_p = self.gamma.grad_unsafe_ptr_cpu()
            var db_p = self.beta.grad_unsafe_ptr_cpu()
            var xhat_p = self.cache_xhat.unsafe_ptr()
            var inv_v = TileTensor(
                self.cache_inv_std, row_major[Self.C](),
            )
            var n_eff = Scalar[DT](Float64(BATCH * Self.SPATIAL))
            var inv_n = Scalar[DT](1.0) / n_eff
            for c in range(Self.C):
                var g = g_p[c]
                var inv_std = inv_v[c]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta:  Scalar[DT] = 0.0
                for b in range(BATCH):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        sum_dxhat += dxhat
                        sum_dxhat_xhat += dxhat * xh
                        d_gamma += dy * xh
                        d_beta  += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(BATCH):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        gi_p[base + s] = inv_std * (
                            dxhat - m1 - xh * m2
                        )
                comptime if mode == "all":
                    dg_p[c] += d_gamma
                    db_p[c] += d_beta
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.FLAT_DIM)
            comptime layout_c  = Layout.row_major(Self.C)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            var g_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.gamma.val.dev.value()
            )
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat_dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.cache_inv_std_dev.value()
            )
            var dg_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.gamma.grd.dev.value()
            )
            var db_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.beta.grd.dev.value()
            )
            var ctx = self.ts.ctx.value()
            comptime G = BATCH if BATCH < BN2D_RBLOCKS else BN2D_RBLOCKS
            comptime layout_pr = Layout.row_major(Self.C * G)
            var pa_lt = LayoutTensor[DT, layout_pr, MutAnyOrigin](
                self.bn_psum_dev.value()        # reuse: Σdx̂
            )
            var pb_lt = LayoutTensor[DT, layout_pr, MutAnyOrigin](
                self.bn_psumsq_dev.value()      # reuse: Σdx̂·x̂
            )
            var pdg_lt = LayoutTensor[DT, layout_pr, MutAnyOrigin](
                self.bn_pdg_dev.value()
            )
            var pdb_lt = LayoutTensor[DT, layout_pr, MutAnyOrigin](
                self.bn_pdb_dev.value()
            )
            var m1_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.bn_m1_dev.value()
            )
            var m2_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.bn_m2_dev.value()
            )
            # Pass 1: C·G blocks → partial Σdx̂, Σdx̂·x̂, Σdγ, Σdβ.
            comptime bpk = _bn2d_bwd_partial_kernel[
                BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM, G,
            ]
            ctx.enqueue_function[bpk](
                go_lt, g_lt, xh_lt, pa_lt, pb_lt, pdg_lt, pdb_lt,
                grid_dim=Self.C * G, block_dim=BN2D_TPB,
            )
            # Pass 2: C blocks → m1, m2 + accumulate grad_gamma/beta (mode).
            comptime bfk = _bn2d_bwd_finalize_kernel[
                BATCH, Self.C, Self.SPATIAL, G, mode,
            ]
            ctx.enqueue_function[bfk](
                pa_lt, pb_lt, pdg_lt, pdb_lt, m1_lt, m2_lt, dg_lt, db_lt,
                grid_dim=Self.C, block_dim=1,
            )
            # Pass 3: C·G blocks → grad_input scatter.
            comptime bsk = _bn2d_bwd_scatter_kernel[
                BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM, G,
            ]
            ctx.enqueue_function[bsk](
                go_lt, g_lt, xh_lt, is_lt, m1_lt, m2_lt, gi_lt,
                grid_dim=Self.C * G, block_dim=BN2D_TPB,
            )

    # Inline helper so we don't read `beta.value` through `value_unsafe_ptr_cpu`
    # twice (one less symbol to update if Param's API shifts).
    def bias_unsafe_ptr_cpu(
        mut self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.beta.value_unsafe_ptr_cpu()

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "training":
            self.training = value > Scalar[DT](0.5)
