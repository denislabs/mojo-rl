"""Conv2D[IC, OC, K, S, P, H, W] — 2D convolution via im2col + `max_matmul`.

Phase 5 of `nn/PORTING_PLAN.md`. Reduces convolution to BATCH per-batch
matmuls (`out_b = weight @ im2col(input_b).T`), mirroring the legacy
`mojo_rl/nn/autodiff/primitives/conv2d.mojo:281` Apple/non-Apple path.
The matmul itself flows through `linalg.matmul`'s `max_matmul`, so on
Apple it lands on the Accelerate cblas kernel; on NVIDIA / generic CPU
it falls through to the platform-best implementation.

Layouts (per batch):
    weight:  [OC, IC·K·K]                         row-major (canonical)
    col:     [OH·OW, IC·K·K]                      row-major (im2col output)
    out:     [OC,   OH·OW]                        row-major
    out = weight @ col.T   (matmul `transpose_b=True`)

Why per-batch matmul instead of one big GEMM: keeps the convolution
free of any explicit reshape between BLAS-friendly `[OC, BATCH·OH·OW]`
and the trait-mandated `[BATCH, OC·OH·OW]` flat order. BATCH is
typically small (≤256), so the per-batch GEMM overhead is dominated by
the matmul itself. The legacy nn package has a batched-Apple variant
that re-packs the im2col across the batch to do one big sgemm; we
deliberately ship the per-batch path first and gate the Apple-batched
optimisation on a real CNN consumer that benchmarks the difference.

Backward (per batch):
    d_bias[oc] += Σ_{oh,ow} d_out[oc, oh, ow]
    d_weight   += d_out_b @ col_b     (`[OC, OH·OW] @ [OH·OW, IC·K·K]`,
                                       accumulated across batches)
    d_col_b     = d_out_b.T @ weight  (`max_matmul[transpose_a=True]`)
    d_input_b   = col2im(d_col_b)     (scatter-add into input shape)

Accumulation into `d_weight` uses Apple Accelerate's `cblas_sgemm` with
`beta=1` (single call, no temp alloc) when running on macOS fp32 — same
trick `linear.mojo` uses. On other targets we matmul into a temp slab
and add elementwise. Both paths produce identical numerics modulo
float32 rounding.

GPU path: **im2col + `max_matmul` (tensor-core GEMM)** — the same
reduction as the CPU path, but the im2col is a GPU kernel and the matmul
flows through `max_matmul[target="gpu"]` (cuBLAS-class GEMM on NVIDIA,
MPS on Apple), mirroring `Linear`'s GPU forward/backward. Replaces the
earlier naive direct-convolution kernels (one thread per output element)
which were compute-bound and 5-10× slower than the legacy fused conv on
conv nets (AlphaZero ResNet, Atari CNN, Dreamer image) — see
`feedback_nn2_gpu_conv_naive_slow`.

  * forward:  `col = im2col(x)` ([BS, COL]) → `out = col @ Wᵀ` ([BS, OC])
              → scatter-add bias into the [BATCH, OC·SO] trait layout.
  * d_weight: rebuild `col`, transpose grad_output → `goᵀ` ([OC, BS]),
              `dW += goᵀ @ col` ([OC, COL]).
  * d_bias:   block-per-OC reduction over BATCH·OH·OW (`block.sum`).
  * d_input:  one thread per input element gathering over (kh,kw,oc) —
              kept as the direct gather kernel (no atomics, already
              parallel; mirrors legacy `backward_dx_kernel_impl`).

`BS = BATCH·OH·OW`. The `col`/`out_packed`/`goᵀ` device scratch is
lazily sized to BATCH on first call and reused (CUDA-graph-capture-safe,
same pattern as `Linear.cacheT_dev`); `dW_tmp` is fixed [OC, COL] and
allocated at `make` time.
"""

from std.math import ceildiv
from std.memory import alloc
from std.sys import CompilationTarget
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from ..constants import DT, CPU_SIMD_W, TPB


comptime CONV_DW_TPB: Int = 128
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
    cast_fp32_to_bf16,
    cast_bf16_to_fp32,
    Conv2DAMPState,
)
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
    ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# im2col / col2im helpers — both produce `[OH·OW, IC·K·K]` row-major
# col matrices for one batch sample. Module-level so the Conv2D body
# stays terse and the compiler doesn't have to re-instantiate them per
# struct method.
# ──────────────────────────────────────────────────────────────────────


def _im2col_one_batch[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
](
    in_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    col_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Pack the IC·H·W input slab into an [OH·OW, IC·K·K] col matrix.

    Row index = `oh·OW + ow` (one row per output spatial position).
    Col index inside each row = `(ic·K + kh)·K + kw` (matches the
    weight flat layout `[OC, IC·K·K]` directly, so the matmul lines
    up with no further transpose). Padded receptive fields contribute
    zero — we write 0 for OOB lanes."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                var in_c_base = ic * H * W
                var col_ic_base = row_off + ic * K * K
                for kh in range(K):
                    var ih = oh * S + kh - P
                    var col_kh_base = col_ic_base + kh * K
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if ih < 0 or ih >= H or iw < 0 or iw >= W:
                            col_p[col_kh_base + kw] = Scalar[DT](0.0)
                        else:
                            col_p[col_kh_base + kw] = (
                                in_p[in_c_base + ih * W + iw]
                            )


def _col2im_one_batch[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
](
    d_col_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    d_in_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Scatter-add an [OH·OW, IC·K·K] col matrix back into a [IC·H·W]
    input gradient slab. Assumes `d_in_p` was zero-filled before the
    first call (typically by the Conv2D vjp body). Padded lanes are
    skipped — they never received a meaningful col entry."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                var in_c_base = ic * H * W
                var col_ic_base = row_off + ic * K * K
                for kh in range(K):
                    var ih = oh * S + kh - P
                    if ih < 0 or ih >= H:
                        continue
                    var col_kh_base = col_ic_base + kh * K
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if iw < 0 or iw >= W:
                            continue
                        d_in_p[in_c_base + ih * W + iw] += (
                            d_col_p[col_kh_base + kw]
                        )


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — im2col + GEMM (forward, d_weight) and direct gather/reduce
# (d_input, d_bias).
#
#   im2col:     1 thread per (b·SO + s, ck) col element.
#   scatter:    1 thread per output element — out_packed[BS,OC] → [B,OC·SO]
#               + bias.
#   go_transpose 1 thread per (oc, b·SO + s) — grad_output[B,OC·SO] → [OC,BS].
#   backward_dx 1 thread per (b, ic, ih, iw).
#   backward_db 1 thread per oc.               Sums over BATCH·OH·OW.
#
# No atomics — every kernel writes a unique destination slot, matching
# the deep_agents / nn convention (`c51/target_y_block.mojo:48`).
# ──────────────────────────────────────────────────────────────────────


def _conv2d_im2col_kernel[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int,
    H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, COL_SIZE: Int, SPATIAL_OUT: Int, BS: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    col: LayoutTensor[DT, Layout.row_major(BS, COL_SIZE), MutAnyOrigin],
):
    """im2col into a [BS, COL_SIZE] row-major matrix, row = `b·SO + s`,
    col index = `(ic·K + kh)·K + kw` (matches the weight flat layout so
    the matmul lines up). Padded receptive fields write 0. One thread per
    col element."""
    var idx = Int(global_idx.x)
    var total = BS * COL_SIZE
    if idx >= total:
        return
    var row = idx // COL_SIZE
    var ck = idx % COL_SIZE
    var b = row // SPATIAL_OUT
    var s = row % SPATIAL_OUT
    var oh = s // OW
    var ow = s % OW
    var ic = ck // (K * K)
    var rem = ck % (K * K)
    var kh = rem // K
    var kw = rem % K
    var ih = oh * S + kh - P
    var iw = ow * S + kw - P
    if ih < 0 or ih >= H or iw < 0 or iw >= W:
        col[row, ck] = Scalar[DT](0.0)
    else:
        col[row, ck] = rebind[Scalar[DT]](input[b, ic * H * W + ih * W + iw])


def _conv2d_scatter_bias_kernel[
    BATCH: Int, OC: Int, SPATIAL_OUT: Int, OUT_FLAT: Int, BS: Int,
](
    out_packed: LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
):
    """Scatter the GEMM result `out_packed[b·SO + s, oc]` into the
    trait-mandated `[BATCH, OC·SO]` flat layout and add the per-channel
    bias. One thread per output element."""
    var idx = Int(global_idx.x)
    var total = BATCH * OUT_FLAT
    if idx >= total:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var oc = out_pos // SPATIAL_OUT
    var s = out_pos % SPATIAL_OUT
    output[b, out_pos] = (
        rebind[Scalar[DT]](out_packed[b * SPATIAL_OUT + s, oc])
        + rebind[Scalar[DT]](bias[oc])
    )


def _conv2d_go_transpose_kernel[
    BATCH: Int, OC: Int, SPATIAL_OUT: Int, OUT_FLAT: Int, BS: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    go_T: LayoutTensor[DT, Layout.row_major(OC, BS), MutAnyOrigin],
):
    """Repack grad_output `[BATCH, OC·SO]` → `goᵀ[OC, BS]` (BS = BATCH·SO,
    column = `b·SO + s`) so `dW = goᵀ @ col` runs through `max_matmul`
    (which rejects `transpose_a`). One thread per goᵀ element."""
    var idx = Int(global_idx.x)
    var total = OC * BS
    if idx >= total:
        return
    var oc = idx // BS
    var col = idx % BS
    var b = col // SPATIAL_OUT
    var s = col % SPATIAL_OUT
    go_T[oc, col] = rebind[Scalar[DT]](grad_output[b, oc * SPATIAL_OUT + s])


def _conv2d_accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """dst[i] += src[i]. Accumulates the `max_matmul` dW result into
    grad_weight, preserving the vjp accumulate contract (mirrors
    Linear's `_accum_kernel`)."""
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


def _conv2d_backward_dx_kernel[
    BATCH: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int,
    H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, OUT_FLAT: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    weight: LayoutTensor[
        DT, Layout.row_major(OC * IC * K * K), MutAnyOrigin,
    ],
    grad_input: LayoutTensor[
        DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin,
    ],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * IN_FLAT
    if idx >= total:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var hw = H * W
    var ic = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W

    var acc: Scalar[DT] = 0.0
    for kh in range(K):
        var oh_num = ih + P - kh
        if oh_num < 0 or oh_num % S != 0:
            continue
        var oh = oh_num // S
        if oh >= OH:
            continue
        for kw in range(K):
            var ow_num = iw + P - kw
            if ow_num < 0 or ow_num % S != 0:
                continue
            var ow = ow_num // S
            if ow >= OW:
                continue
            var out_pos_base = oh * OW + ow
            for oc in range(OC):
                var w_off = (
                    ((oc * IC + ic) * K + kh) * K + kw
                )
                acc += (
                    rebind[Scalar[DT]](weight[w_off])
                    * rebind[Scalar[DT]](
                        grad_output[b, oc * OH * OW + out_pos_base]
                    )
                )
    grad_input[b, in_pos] = acc


# ── GEMM-based dx (replaces the K·K·OC direct gather). Three steps mirror
#    the CPU path + the forward: repack grad_output → go_packed[BS, OC], GEMM
#    d_col = go_packed @ weight[OC, COL] (tensor-core, contracts OC), then a
#    cheap K·K col2im-gather. The old direct gather did K·K·OC scattered reads
#    per input element with no tensor cores — the conv backward analog of the
#    naive-BatchNorm bug. ──
def _conv2d_go_pack_kernel[
    BATCH: Int, OC: Int, SPATIAL_OUT: Int, OUT_FLAT: Int, BS: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    go_packed: LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin],
):
    """Repack grad_output `[BATCH, OC·SO]` → `go_packed[BS, OC]` (row =
    `b·SO + s`) so `d_col = go_packed @ weight` runs as an untransposed
    `max_matmul`. One thread per go_packed element (coalesced write)."""
    var idx = Int(global_idx.x)
    var total = BS * OC
    if idx >= total:
        return
    var row = idx // OC
    var oc = idx % OC
    var b = row // SPATIAL_OUT
    var s = row % SPATIAL_OUT
    go_packed[row, oc] = rebind[Scalar[DT]](
        grad_output[b, oc * SPATIAL_OUT + s]
    )


def _conv2d_dx_col2im_kernel[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int,
    H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, COL_SIZE: Int, SPATIAL_OUT: Int, BS: Int,
](
    d_col: LayoutTensor[DT, Layout.row_major(BS, COL_SIZE), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
):
    """col2im gather: `grad_input[b,ic,ih,iw] = Σ_{kh,kw} d_col[b·SO+s,
    (ic·K+kh)·K+kw]` over the valid (oh,ow) that this input element feeds.
    OC is already contracted in `d_col` (the GEMM), so this is K·K reads per
    element (vs K·K·OC in the old direct gather). One thread per input
    element; unique destination → no atomics."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * IN_FLAT
    if idx >= total:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var hw = H * W
    var ic = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W

    var acc: Scalar[DT] = 0.0
    for kh in range(K):
        var oh_num = ih + P - kh
        if oh_num < 0 or oh_num % S != 0:
            continue
        var oh = oh_num // S
        if oh >= OH:
            continue
        for kw in range(K):
            var ow_num = iw + P - kw
            if ow_num < 0 or ow_num % S != 0:
                continue
            var ow = ow_num // S
            if ow >= OW:
                continue
            var row = b * SPATIAL_OUT + oh * OW + ow
            var col_idx = (ic * K + kh) * K + kw
            acc += rebind[Scalar[DT]](d_col[row, col_idx])
    grad_input[b, in_pos] = acc


def _conv2d_backward_db_kernel[
    BATCH: Int, OC: Int, OH: Int, OW: Int, OUT_FLAT: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    grad_bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
):
    """dB reduction — one block per OC, CONV_DW_TPB threads stride over
    BATCH·OH·OW and `block.sum` the partials. Same pattern as `dW`."""
    var oc = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if oc >= OC:
        return
    var spatial_out = OH * OW
    var n_eff = BATCH * spatial_out
    var out_c_off = oc * spatial_out

    var my_acc: Scalar[DT] = 0.0
    var idx = t
    while idx < n_eff:
        var b = idx // spatial_out
        var s_pos = idx % spatial_out
        my_acc += rebind[Scalar[DT]](grad_output[b, out_c_off + s_pos])
        idx += CONV_DW_TPB
    var total = block.sum[block_size=CONV_DW_TPB, broadcast=False](
        val=my_acc
    )
    if t == 0:
        grad_bias[oc] = (
            rebind[Scalar[DT]](grad_bias[oc]) + total[0]
        )


struct Conv2D[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
](Module):
    comptime ARITY: Int = 1
    comptime OH: Int = (Self.H + 2 * Self.P - Self.K) // Self.S + 1
    comptime OW: Int = (Self.W + 2 * Self.P - Self.K) // Self.S + 1
    comptime IN_DIM_FLAT: Int = Self.IC * Self.H * Self.W
    comptime OUT_DIM_FLAT: Int = Self.OC * Self.OH * Self.OW
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM_FLAT)
    comptime OUT_DIM = Self.OUT_DIM_FLAT
    comptime W_SIZE: Int = Self.OC * Self.IC * Self.K * Self.K
    comptime B_SIZE: Int = Self.OC
    comptime COL_SIZE: Int = Self.IC * Self.K * Self.K
    comptime SPATIAL_OUT: Int = Self.OH * Self.OW

    var weight: Param["weight", True,  Self.W_SIZE]
    var bias:   Param["bias",   False, Self.B_SIZE]
    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    # GPU im2col + GEMM scratch. `col`/`out_packed`/`goᵀ` are lazily sized
    # to BATCH·SPATIAL_OUT on first call and reused (capture-safe, mirrors
    # Linear.cacheT_dev); `dW_tmp` is fixed [OC, COL] and made at make-time.
    var col_dev: Optional[DeviceBuffer[DT]]
    var col_n: Int
    var outp_dev: Optional[DeviceBuffer[DT]]
    var outp_n: Int
    var goT_dev: Optional[DeviceBuffer[DT]]
    var goT_n: Int
    var dW_tmp_dev: Optional[DeviceBuffer[DT]]
    # bf16 scratch for the two GPU GEMMs (forward col@Wᵀ + backward goᵀ@col)
    # when POLICY.compute_dtype == bf16. Empty/None on the fp32 path.
    var amp: Conv2DAMPState[Self.OC, Self.COL_SIZE]
    var ts: TargetStorage

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = None
        self.col_dev = None
        self.col_n = 0
        self.outp_dev = None
        self.outp_n = 0
        self.goT_dev = None
        self.goT_n = 0
        self.dW_tmp_dev = None
        self.amp = Conv2DAMPState[Self.OC, Self.COL_SIZE].make()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Conv2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.K > 0 and Self.S > 0, (
            "Conv2D: K and S must be positive"
        )
        comptime assert Self.OH > 0 and Self.OW > 0, (
            "Conv2D: invalid spatial shape — check H/W/K/S/P"
        )
        var c = Self()
        comptime if target == "cpu":
            c.weight = Param["weight", True,  Self.W_SIZE].make_cpu()
            c.bias   = Param["bias",   False, Self.B_SIZE].make_cpu()
            # fan_in = IC·K·K, fan_out = OC·K·K — the canonical Kaiming
            # convention for conv weights.
            INIT.init_weight(
                c.weight.value_unsafe_ptr_cpu(),
                Self.W_SIZE,
                Self.IC * Self.K * Self.K,
                Self.OC * Self.K * Self.K,
            )
            INIT.init_bias(c.bias.value_unsafe_ptr_cpu(), Self.B_SIZE)
            c.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Conv2D.make[target='gpu']"](ctx)
            c.weight = Param["weight", True,  Self.W_SIZE].make_gpu(ctx_v)
            c.bias   = Param["bias",   False, Self.B_SIZE].make_gpu(ctx_v)
            # Initialise CPU storage with the chosen INIT, then enqueue
            # a copy to the device buffer — same pattern Linear uses for
            # Kaiming/Xavier weight init on GPU.
            var w_host = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
            var b_host = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
            INIT.init_weight(
                mptr(w_host.unsafe_ptr()),
                Self.W_SIZE,
                Self.IC * Self.K * Self.K,
                Self.OC * Self.K * Self.K,
            )
            INIT.init_bias(
                mptr(b_host.unsafe_ptr()),
                Self.B_SIZE,
            )
            var w_hb = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var b_hb = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            for k in range(Self.W_SIZE):
                w_hb.unsafe_ptr()[k] = w_host[k]
            for k in range(Self.B_SIZE):
                b_hb.unsafe_ptr()[k] = b_host[k]
            ctx_v.enqueue_copy(c.weight.val.dev.value(), w_hb)
            ctx_v.enqueue_copy(c.bias.val.dev.value(),   b_hb)
            ctx_v.synchronize()
            # Fixed [OC, COL] dW scratch for the max_matmul grad_w path; the
            # col / out_packed / goᵀ buffers stay None (lazily sized to
            # BATCH·SPATIAL_OUT on the first forward / backward).
            c.dW_tmp_dev = ctx_v.enqueue_create_buffer[DT](Self.W_SIZE)
            c.ts = TargetStorage.make_gpu(ctx_v)
        return c^

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
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        var in_p = input.ptr
        var out_p = output_v.ptr
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
            var w_p = self.weight.value_unsafe_ptr_cpu()
            var b_p = self.bias.value_unsafe_ptr_cpu()
            var col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var w_tt = TileTensor(
                self.weight.val.cpu, row_major[Self.OC, Self.COL_SIZE](),
            )
            for b in range(BATCH):
                _im2col_one_batch[
                    Self.IC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                ](
                    in_p + b * Self.IN_DIM_FLAT,
                    col_buf,
                )
                var col_tt = TileTensor(
                    col_buf,
                    row_major[Self.SPATIAL_OUT, Self.COL_SIZE](),
                )
                var out_b_p = out_p + b * Self.OUT_DIM_FLAT
                var out_tt = TileTensor(
                    out_b_p,
                    row_major[Self.OC, Self.SPATIAL_OUT](),
                )
                # out = W @ col.T  →  [OC, SPATIAL_OUT].
                max_matmul[transpose_b=True, target="cpu"](
                    out_tt, w_tt, col_tt, None,
                )
                # Bias broadcast across SPATIAL_OUT lanes.
                for oc in range(Self.OC):
                    var row = out_b_p + oc * Self.SPATIAL_OUT
                    var bv = b_p[oc]
                    var i = 0
                    while i + CPU_SIMD_W <= Self.SPATIAL_OUT:
                        var v = row.load[width=CPU_SIMD_W](i)
                        row.store(
                            i, v + SIMD[DT, CPU_SIMD_W](bv),
                        )
                        i += CPU_SIMD_W
                    while i < Self.SPATIAL_OUT:
                        row[i] = row[i] + bv
                        i += 1
            col_buf.free()
        else:
            # im2col + GEMM: col = im2col(x) [BS, COL]; out_packed = col @ Wᵀ
            # [BS, OC]; scatter → output [BATCH, OC·SO] + bias. The GEMM runs
            # through max_matmul (tensor cores), the same as Linear.
            comptime BS = BATCH * Self.SPATIAL_OUT
            var ctx = self.ts.ctx.value()
            ensure_gpu_buffer(self.col_dev, self.col_n, BS * Self.COL_SIZE, ctx)
            ensure_gpu_buffer(self.outp_dev, self.outp_n, BS * Self.OC, ctx)

            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(BATCH, Self.OUT_DIM_FLAT)
            comptime col_layout = Layout.row_major(BS, Self.COL_SIZE)
            comptime outp_layout = Layout.row_major(BS, Self.OC)
            comptime b_layout = Layout.row_major(Self.B_SIZE)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p)
            var col_lt = LayoutTensor[DT, col_layout, MutAnyOrigin](
                self.col_dev.value()
            )
            var b_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.bias.val.dev.value()
            )

            # ── (1) im2col → col[BS, COL] ──────────────────────────────
            comptime n_blocks_col = (BS * Self.COL_SIZE + TPB - 1) // TPB
            comptime im2col_kernel = _conv2d_im2col_kernel[
                BATCH, Self.IC, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.COL_SIZE, Self.SPATIAL_OUT, BS,
            ]
            ctx.enqueue_function[im2col_kernel](
                in_lt, col_lt, grid_dim=n_blocks_col, block_dim=TPB,
            )

            # ── (2) out_packed = col @ Wᵀ  ([BS, COL] @ [OC, COL]ᵀ) ────
            comptime if POLICY.compute_dtype == DT:
                var w_tt = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.OC, Self.COL_SIZE](),
                )
                var col_tt = TileTensor(
                    self.col_dev.value(), row_major[BS, Self.COL_SIZE](),
                )
                var outp_tt = TileTensor(
                    self.outp_dev.value(), row_major[BS, Self.OC](),
                )
                max_matmul[transpose_b=True, target="gpu"](
                    outp_tt, col_tt, w_tt, ctx,
                )
            else:
                # AMP: cast W + col → bf16, bf16 GEMM, upcast out → fp32.
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Conv2D GPU supports only fp32 and bf16 compute_dtype"
                )
                self.amp.ensure_gpu(BS, ctx)
                cast_fp32_to_bf16[target="gpu", N = Self.W_SIZE](
                    mptr(self.weight.val.dev.value().unsafe_ptr()),
                    mptr(self.amp.w_bf16_dev.value().unsafe_ptr()),
                    ctx,
                )
                cast_fp32_to_bf16[target="gpu", N = BS * Self.COL_SIZE](
                    mptr(self.col_dev.value().unsafe_ptr()),
                    mptr(self.amp.col_bf16_dev.value().unsafe_ptr()),
                    ctx,
                )
                var w_bf16_tt = TileTensor(
                    self.amp.w_bf16_dev.value(),
                    row_major[Self.OC, Self.COL_SIZE](),
                )
                var col_bf16_tt = TileTensor(
                    self.amp.col_bf16_dev.value(),
                    row_major[BS, Self.COL_SIZE](),
                )
                var outp_bf16_tt = TileTensor(
                    self.amp.outp_bf16_dev.value(), row_major[BS, Self.OC](),
                )
                max_matmul[transpose_b=True, target="gpu"](
                    outp_bf16_tt, col_bf16_tt, w_bf16_tt, ctx,
                )
                cast_bf16_to_fp32[target="gpu", N = BS * Self.OC](
                    mptr(self.amp.outp_bf16_dev.value().unsafe_ptr()),
                    mptr(self.outp_dev.value().unsafe_ptr()),
                    ctx,
                )

            # ── (3) scatter out_packed → output[B, OC·SO] + bias ───────
            var outp_lt = LayoutTensor[DT, outp_layout, MutAnyOrigin](
                self.outp_dev.value()
            )
            comptime n_blocks_sc = (BATCH * Self.OUT_DIM_FLAT + TPB - 1) // TPB
            comptime scatter_kernel = _conv2d_scatter_bias_kernel[
                BATCH, Self.OC, Self.SPATIAL_OUT, Self.OUT_DIM_FLAT, BS,
            ]
            ctx.enqueue_function[scatter_kernel](
                outp_lt, b_lt, out_lt,
                grid_dim=n_blocks_sc, block_dim=TPB,
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
        """Combined backward (S7) — the two phases in fixed order. Single
        source of truth for direct callers; Sequential calls the phases
        directly so the param-before-input order is the orchestrator's."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        self.vjp_param_grads[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output
        )
        self.vjp_grad_input[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, grad_inputs
        )

    def vjp_param_grads[
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
    ) raises:
        """Phase 1 (S7): d_bias + d_weight (mode=all). im2col reads the
        cached forward input (`_cached_input_ptr`); MUST run before
        `vjp_grad_input` writes the slab that input aliases under
        Sequential. d_weight needs the rebuilt `col`; the grad_input
        phase (d_col + col2im) does NOT, so im2col runs once, here."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        comptime if mode == "all":
            assert_tag_for["Conv2D", target](self.ts.target_tag)
            var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)

            comptime if target == "cpu":
                var go_p = grad_output_v.ptr
                var x_p = self._cached_input_ptr.value()
                var dw_p = self.weight.grad_unsafe_ptr_cpu()
                var db_p = self.bias.grad_unsafe_ptr_cpu()

                var col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                    Scalar[DT]
                ](Self.SPATIAL_OUT * Self.COL_SIZE)
                var dw_tmp: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
                comptime if (
                    CompilationTarget.is_macos() and DT == DType.float32
                ):
                    dw_tmp = None
                else:
                    dw_tmp = alloc[Scalar[DT]](Self.W_SIZE)

                for b in range(BATCH):
                    # ---- 1. Rebuild col_b for this batch (reads x) ----
                    _im2col_one_batch[
                        Self.IC, Self.K, Self.S, Self.P,
                        Self.H, Self.W, Self.OH, Self.OW,
                    ](
                        x_p + b * Self.IN_DIM_FLAT,
                        col_buf,
                    )
                    var col_tt = TileTensor(
                        col_buf,
                        row_major[Self.SPATIAL_OUT, Self.COL_SIZE](),
                    )

                    # ---- 2. d_out_b view + d_bias accumulate ----------
                    var go_b_p = go_p + b * Self.OUT_DIM_FLAT
                    var go_b_tt = TileTensor(
                        go_b_p, row_major[Self.OC, Self.SPATIAL_OUT](),
                    )
                    for oc in range(Self.OC):
                        var acc: Scalar[DT] = 0.0
                        var row_off = oc * Self.SPATIAL_OUT
                        for s in range(Self.SPATIAL_OUT):
                            acc += go_b_p[row_off + s]
                        db_p[oc] += acc

                    # ---- 3. d_weight += d_out_b @ col_b ---------------
                    #         d_out_b is [OC, SPATIAL_OUT],
                    #         col_b   is [SPATIAL_OUT, COL_SIZE],
                    #         result  is [OC, COL_SIZE] (= same flat as W).
                    # On Apple fp32 we use one cblas_sgemm with beta=1 (no
                    # temp). Elsewhere we matmul into dw_tmp and add.
                    comptime if (
                        CompilationTarget.is_macos()
                        and DT == DType.float32
                    ):
                        var cblas = get_cblas_f32_function()
                        cblas(
                            _CBLASOrder.ROW_MAJOR,
                            _CBLASTranspose.NO_TRANSPOSE,
                            _CBLASTranspose.NO_TRANSPOSE,
                            Int32(Self.OC),
                            Int32(Self.COL_SIZE),
                            Int32(Self.SPATIAL_OUT),
                            Float32(1.0),
                            rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                                go_b_p
                            ),
                            Int32(Self.SPATIAL_OUT),
                            rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                                col_buf
                            ),
                            Int32(Self.COL_SIZE),
                            Float32(1.0),
                            rebind[UnsafePointer[Float32, MutAnyOrigin]](
                                dw_p
                            ),
                            Int32(Self.COL_SIZE),
                        )
                    else:
                        var dw_tmp_p = dw_tmp.value()
                        var dw_tmp_tt = TileTensor(
                            dw_tmp_p,
                            row_major[Self.OC, Self.COL_SIZE](),
                        )
                        max_matmul[target="cpu"](
                            dw_tmp_tt, go_b_tt, col_tt, None,
                        )
                        var i = 0
                        while i + CPU_SIMD_W <= Self.W_SIZE:
                            var dwv = dw_p.load[width=CPU_SIMD_W](i)
                            var tv = dw_tmp_p.load[width=CPU_SIMD_W](i)
                            dw_p.store(i, dwv + tv)
                            i += CPU_SIMD_W
                        while i < Self.W_SIZE:
                            dw_p[i] = dw_p[i] + dw_tmp_p[i]
                            i += 1

                col_buf.free()
                comptime if not (
                    CompilationTarget.is_macos() and DT == DType.float32
                ):
                    dw_tmp.value().free()
            else:
                # im2col + GEMM dW: rebuild col[BS, COL] from the cached
                # input, transpose grad_output → goᵀ[OC, BS], then
                # dW_tmp = goᵀ @ col ([OC, BS] @ [BS, COL]) and accumulate
                # into grad_weight. d_bias keeps the block-per-OC reduction.
                # The im2col reads `in_lt` (the cached forward input) and is
                # enqueued in THIS phase so it precedes dx (the grad_input
                # phase), which clobbers the aliased slab — the original
                # Conv2D dx-first bug
                # (feedback_nn2_gpu_backward_order_aliased_slab) stays
                # structurally impossible.
                comptime BS = BATCH * Self.SPATIAL_OUT
                var go_p = grad_output_v.ptr
                var x_p = self._cached_input_ptr.value()
                var ctx = self.ts.ctx.value()
                ensure_gpu_buffer(
                    self.col_dev, self.col_n, BS * Self.COL_SIZE, ctx
                )
                ensure_gpu_buffer(self.goT_dev, self.goT_n, Self.OC * BS, ctx)

                comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
                comptime out_layout = Layout.row_major(
                    BATCH, Self.OUT_DIM_FLAT
                )
                comptime col_layout = Layout.row_major(BS, Self.COL_SIZE)
                comptime goT_layout = Layout.row_major(Self.OC, BS)
                comptime w_layout = Layout.row_major(Self.W_SIZE)
                comptime b_layout = Layout.row_major(Self.B_SIZE)
                var go_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](go_p)
                var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](x_p)
                var col_lt = LayoutTensor[DT, col_layout, MutAnyOrigin](
                    self.col_dev.value()
                )
                var goT_lt = LayoutTensor[DT, goT_layout, MutAnyOrigin](
                    self.goT_dev.value()
                )

                # ── (1) rebuild col[BS, COL] = im2col(x) ───────────────
                comptime n_blocks_col = (
                    BS * Self.COL_SIZE + TPB - 1
                ) // TPB
                comptime im2col_kernel = _conv2d_im2col_kernel[
                    BATCH, Self.IC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                    Self.IN_DIM_FLAT, Self.COL_SIZE, Self.SPATIAL_OUT, BS,
                ]
                ctx.enqueue_function[im2col_kernel](
                    in_lt, col_lt, grid_dim=n_blocks_col, block_dim=TPB,
                )

                # ── (2) goᵀ[OC, BS] = transpose(grad_output) ───────────
                comptime n_blocks_got = (Self.OC * BS + TPB - 1) // TPB
                comptime got_kernel = _conv2d_go_transpose_kernel[
                    BATCH, Self.OC, Self.SPATIAL_OUT, Self.OUT_DIM_FLAT, BS,
                ]
                ctx.enqueue_function[got_kernel](
                    go_lt, goT_lt, grid_dim=n_blocks_got, block_dim=TPB,
                )

                # ── (3) dW_tmp = goᵀ @ col  → accumulate into grad_w ───
                comptime if POLICY.compute_dtype == DT:
                    var goT_tt = TileTensor(
                        self.goT_dev.value(), row_major[Self.OC, BS](),
                    )
                    var col_tt = TileTensor(
                        self.col_dev.value(), row_major[BS, Self.COL_SIZE](),
                    )
                    var dW_tmp_tt = TileTensor(
                        self.dW_tmp_dev.value(),
                        row_major[Self.OC, Self.COL_SIZE](),
                    )
                    max_matmul[target="gpu"](dW_tmp_tt, goT_tt, col_tt, ctx)
                else:
                    # AMP: cast goᵀ + col → bf16, bf16 GEMM, upcast → fp32.
                    comptime assert POLICY.compute_dtype == DType.bfloat16, (
                        "Conv2D GPU supports only fp32 and bf16 compute_dtype"
                    )
                    self.amp.ensure_gpu(BS, ctx)
                    cast_fp32_to_bf16[target="gpu", N = Self.OC * BS](
                        mptr(self.goT_dev.value().unsafe_ptr()),
                        mptr(self.amp.goT_bf16_dev.value().unsafe_ptr()),
                        ctx,
                    )
                    cast_fp32_to_bf16[target="gpu", N = BS * Self.COL_SIZE](
                        mptr(self.col_dev.value().unsafe_ptr()),
                        mptr(self.amp.col_bf16_dev.value().unsafe_ptr()),
                        ctx,
                    )
                    var goT_bf16_tt = TileTensor(
                        self.amp.goT_bf16_dev.value(),
                        row_major[Self.OC, BS](),
                    )
                    var col_bf16_tt = TileTensor(
                        self.amp.col_bf16_dev.value(),
                        row_major[BS, Self.COL_SIZE](),
                    )
                    var dW_bf16_tt = TileTensor(
                        self.amp.dW_bf16_dev.value(),
                        row_major[Self.OC, Self.COL_SIZE](),
                    )
                    max_matmul[target="gpu"](
                        dW_bf16_tt, goT_bf16_tt, col_bf16_tt, ctx
                    )
                    cast_bf16_to_fp32[target="gpu", N = Self.W_SIZE](
                        mptr(self.amp.dW_bf16_dev.value().unsafe_ptr()),
                        mptr(self.dW_tmp_dev.value().unsafe_ptr()),
                        ctx,
                    )

                var dw_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                    self.weight.grd.dev.value()
                )
                var dW_tmp_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                    self.dW_tmp_dev.value()
                )
                comptime n_blocks_acc = (Self.W_SIZE + TPB - 1) // TPB
                comptime acc_kernel = _conv2d_accum_kernel[Self.W_SIZE]
                ctx.enqueue_function[acc_kernel](
                    dw_lt, dW_tmp_lt,
                    grid_dim=n_blocks_acc, block_dim=TPB,
                )

                # ── (4) d_bias — 1 block per OC, block.sum over BS ─────
                var db_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                    self.bias.grd.dev.value()
                )
                comptime db_kernel = _conv2d_backward_db_kernel[
                    BATCH, Self.OC, Self.OH, Self.OW, Self.OUT_DIM_FLAT,
                ]
                ctx.enqueue_function[db_kernel](
                    go_lt, db_lt,
                    grid_dim=Self.OC, block_dim=CONV_DW_TPB,
                )

    def vjp_grad_input[
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
        """Phase 2 (S7): d_input = col2im(d_out.T @ weight). Reads only
        grad_output + weight (NOT the cached input / col), so no im2col
        here; writes grad_inputs[0] (aliases the cached input — safe
        because phase 1 already consumed it). Runs in both modes."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var w_p = self.weight.value_unsafe_ptr_cpu()

            # Zero d_input — col2im is scatter-add.
            for k in range(BATCH * Self.IN_DIM_FLAT):
                gi_p[k] = Scalar[DT](0.0)

            var d_col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var go_b_T_buf: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
            comptime if (
                CompilationTarget.is_macos() and DT == DType.float32
            ):
                go_b_T_buf = None
            else:
                go_b_T_buf = alloc[Scalar[DT]](
                    Self.SPATIAL_OUT * Self.OC
                )

            var w_tt = TileTensor(
                self.weight.val.cpu, row_major[Self.OC, Self.COL_SIZE](),
            )

            for b in range(BATCH):
                var go_b_p = go_p + b * Self.OUT_DIM_FLAT

                # ---- 4. d_col_b = d_out_b.T @ weight ------------------
                #         d_out_b.T is [SPATIAL_OUT, OC],
                #         weight     is [OC, COL_SIZE],
                #         result     is [SPATIAL_OUT, COL_SIZE].
                # `max_matmul` does NOT support `transpose_a=True`, so on
                # Apple fp32 we drop through to cblas (which does); on
                # other targets we materialise the transpose explicitly
                # into `go_b_T_buf` and call max_matmul untransposed.
                comptime if (
                    CompilationTarget.is_macos() and DT == DType.float32
                ):
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.SPATIAL_OUT),
                        Int32(Self.COL_SIZE),
                        Int32(Self.OC),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            go_b_p
                        ),
                        Int32(Self.SPATIAL_OUT),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            w_p
                        ),
                        Int32(Self.COL_SIZE),
                        Float32(0.0),
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](
                            d_col_buf
                        ),
                        Int32(Self.COL_SIZE),
                    )
                else:
                    # Build d_out_b.T into go_b_T_buf, then untransposed
                    # matmul. The temp is SPATIAL_OUT × OC.
                    var go_b_T_buf_p = go_b_T_buf.value()
                    for s in range(Self.SPATIAL_OUT):
                        for oc in range(Self.OC):
                            go_b_T_buf_p[s * Self.OC + oc] = go_b_p[
                                oc * Self.SPATIAL_OUT + s
                            ]
                    var go_b_T_tt = TileTensor(
                        go_b_T_buf_p,
                        row_major[Self.SPATIAL_OUT, Self.OC](),
                    )
                    var d_col_tt = TileTensor(
                        d_col_buf,
                        row_major[Self.SPATIAL_OUT, Self.COL_SIZE](),
                    )
                    max_matmul[target="cpu"](
                        d_col_tt, go_b_T_tt, w_tt, None,
                    )

                # ---- 5. col2im → d_input_b ----------------------------
                _col2im_one_batch[
                    Self.IC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                ](
                    d_col_buf,
                    gi_p + b * Self.IN_DIM_FLAT,
                )

            d_col_buf.free()
            comptime if not (
                CompilationTarget.is_macos() and DT == DType.float32
            ):
                go_b_T_buf.value().free()
        else:
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(
                BATCH, Self.OUT_DIM_FLAT
            )
            comptime w_layout = Layout.row_major(Self.W_SIZE)
            var go_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](gi_p)
            var ctx = self.ts.ctx.value()
            comptime BS = BATCH * Self.SPATIAL_OUT

            # dx via GEMM + col2im (mirrors the CPU path + forward), instead of
            # the K·K·OC direct gather. Reuse col_dev as d_col[BS, COL] and
            # outp_dev as go_packed[BS, OC] (both free in backward); ensure here
            # since vjp_grad_input runs in input_only mode too (no phase-1 sizing).
            ensure_gpu_buffer(self.outp_dev, self.outp_n, BS * Self.OC, ctx)
            ensure_gpu_buffer(self.col_dev, self.col_n, BS * Self.COL_SIZE, ctx)

            # ── (1) go_packed[BS, OC] = repack(grad_output) ───────────────
            var gopack_lt = LayoutTensor[
                DT, Layout.row_major(BS, Self.OC), MutAnyOrigin,
            ](self.outp_dev.value())
            comptime n_blocks_gp = (BS * Self.OC + TPB - 1) // TPB
            comptime gopack_kernel = _conv2d_go_pack_kernel[
                BATCH, Self.OC, Self.SPATIAL_OUT, Self.OUT_DIM_FLAT, BS,
            ]
            ctx.enqueue_function[gopack_kernel](
                go_lt, gopack_lt, grid_dim=n_blocks_gp, block_dim=TPB,
            )

            # ── (2) d_col[BS, COL] = go_packed[BS, OC] @ weight[OC, COL] ──
            var gopack_tt = TileTensor(
                self.outp_dev.value(), row_major[BS, Self.OC](),
            )
            var w_tt = TileTensor(
                self.weight.val.dev.value(),
                row_major[Self.OC, Self.COL_SIZE](),
            )
            var dcol_tt = TileTensor(
                self.col_dev.value(), row_major[BS, Self.COL_SIZE](),
            )
            max_matmul[transpose_b=False, target="gpu"](
                dcol_tt, gopack_tt, w_tt, ctx
            )

            # ── (3) col2im gather → grad_input ───────────────────────────
            var dcol_lt = LayoutTensor[
                DT, Layout.row_major(BS, Self.COL_SIZE), MutAnyOrigin,
            ](self.col_dev.value())
            comptime total_in = BATCH * Self.IN_DIM_FLAT
            comptime n_blocks_in = (total_in + TPB - 1) // TPB
            comptime col2im_kernel = _conv2d_dx_col2im_kernel[
                BATCH, Self.IC, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.COL_SIZE, Self.SPATIAL_OUT, BS,
            ]
            ctx.enqueue_function[col2im_kernel](
                dcol_lt, gi_lt,
                grid_dim=n_blocks_in, block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
