"""Conv2D[IC, OC, K, S, P, H, W] — 2D convolution via im2col + `max_matmul`.

Phase 5 of `nn2/PORTING_PLAN.md`. Reduces convolution to BATCH per-batch
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

GPU path: **naive-but-correct direct convolution kernels** — one thread
per output position for the forward, one thread per input position
for `d_input`, one thread per weight scalar for `d_weight`, one thread
per output channel for `d_bias`. Mirrors the legacy nn
`backward_dx_kernel_impl` layout (one thread per input element with no
atomics — see `nn/autodiff/primitives/conv2d.mojo:790`). Tuned tiled +
im2col-on-GPU paths are a follow-up; the plan documents them as
consumer-gated under "Outstanding matmul-perf follow-ups".
"""

from std.math import ceildiv
from std.memory import alloc
from std.sys import CompilationTarget
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
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
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


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
# GPU kernels — direct convolution (no GPU im2col yet).
#
#   forward:    1 thread per (b, oc, oh, ow).
#   backward_dx 1 thread per (b, ic, ih, iw).
#   backward_dw 1 thread per (oc, ic, kh, kw). Sums over BATCH·OH·OW.
#   backward_db 1 thread per oc.               Sums over BATCH·OH·OW.
#
# No atomics — every kernel writes a unique destination slot, matching
# the deep_agents2 / nn2 convention (`c51/target_y_block.mojo:48`).
# ──────────────────────────────────────────────────────────────────────


def _conv2d_forward_kernel[
    BATCH: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int,
    H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, OUT_FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    weight: LayoutTensor[
        DT, Layout.row_major(OC * IC * K * K), MutAnyOrigin,
    ],
    bias:   LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
    output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * OUT_FLAT
    if idx >= total:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var spatial_out = OH * OW
    var oc = out_pos // spatial_out
    var rem = out_pos % spatial_out
    var oh = rem // OW
    var ow = rem % OW

    var acc = rebind[Scalar[DT]](bias[oc])
    var w_oc_off = oc * IC * K * K
    for ic in range(IC):
        var in_c_off = ic * H * W
        var w_ic_off = w_oc_off + ic * K * K
        for kh in range(K):
            var ih = oh * S + kh - P
            if ih < 0 or ih >= H:
                continue
            var w_kh_off = w_ic_off + kh * K
            for kw in range(K):
                var iw = ow * S + kw - P
                if iw < 0 or iw >= W:
                    continue
                acc += (
                    rebind[Scalar[DT]](input[b, in_c_off + ih * W + iw])
                    * rebind[Scalar[DT]](weight[w_kh_off + kw])
                )
    output[b, out_pos] = acc


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


def _conv2d_backward_dw_kernel[
    BATCH: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int,
    H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, OUT_FLAT: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    input: LayoutTensor[
        DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin,
    ],
    grad_weight: LayoutTensor[
        DT, Layout.row_major(OC * IC * K * K), MutAnyOrigin,
    ],
):
    """dW reduction — one block per weight scalar, CONV_DW_TPB threads
    stride over BATCH·OH·OW and `block.sum` the partials. Mirrors the
    LayerNorm dx pattern (one block per sample, threads reduce over DIM)
    but on the per-weight axis: every weight scalar accumulates over the
    full `(b, oh, ow)` set. Replaces the previous "1 thread per weight
    scalar with BATCH·OH·OW inner loop" layout, which had a 6k-iteration
    inner loop per thread on NatureDQN-sized inputs."""
    var weight_idx = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var total_weights = OC * IC * K * K
    if weight_idx >= total_weights:
        return

    var oc = weight_idx // (IC * K * K)
    var rem0 = weight_idx % (IC * K * K)
    var ic = rem0 // (K * K)
    var rem1 = rem0 % (K * K)
    var kh = rem1 // K
    var kw = rem1 % K

    var spatial_out = OH * OW
    var n_eff = BATCH * spatial_out
    var in_c_off = ic * H * W
    var out_c_off = oc * spatial_out

    var my_acc: Scalar[DT] = 0.0
    var idx = t
    while idx < n_eff:
        var b = idx // spatial_out
        var s_pos = idx % spatial_out
        var oh = s_pos // OW
        var ow = s_pos % OW
        var ih = oh * S + kh - P
        var iw = ow * S + kw - P
        if ih >= 0 and ih < H and iw >= 0 and iw < W:
            my_acc += (
                rebind[Scalar[DT]](
                    input[b, in_c_off + ih * W + iw]
                )
                * rebind[Scalar[DT]](
                    grad_output[b, out_c_off + s_pos]
                )
            )
        idx += CONV_DW_TPB
    var total = block.sum[block_size=CONV_DW_TPB, broadcast=False](
        val=my_acc
    )
    if t == 0:
        grad_weight[weight_idx] = (
            rebind[Scalar[DT]](grad_weight[weight_idx]) + total[0]
        )


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
    var ts: TargetStorage

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = None
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
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    w_host.unsafe_ptr()
                ),
                Self.W_SIZE,
                Self.IC * Self.K * Self.K,
                Self.OC * Self.K * Self.K,
            )
            INIT.init_bias(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    b_host.unsafe_ptr()
                ),
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
            c.ts = TargetStorage.make_gpu(ctx_v)
        return c^

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            input.ptr
        )
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            output_v.ptr
        )
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
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(
                BATCH, Self.OUT_DIM_FLAT
            )
            comptime w_layout = Layout.row_major(Self.W_SIZE)
            comptime b_layout = Layout.row_major(Self.B_SIZE)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](
                out_p
            )
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight.val.dev.value()
            )
            var b_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.bias.val.dev.value()
            )
            comptime total = BATCH * Self.OUT_DIM_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _conv2d_forward_kernel[
                BATCH, Self.IC, Self.OC, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, w_lt, b_lt, out_lt,
                grid_dim=n_blocks, block_dim=TPB,
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var x_p = self._cached_input_ptr.value()
            var w_p = self.weight.value_unsafe_ptr_cpu()
            var dw_p = self.weight.grad_unsafe_ptr_cpu()
            var db_p = self.bias.grad_unsafe_ptr_cpu()

            # Zero d_input — col2im is scatter-add.
            for k in range(BATCH * Self.IN_DIM_FLAT):
                gi_p[k] = Scalar[DT](0.0)

            var col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var d_col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var dw_tmp: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
            var go_b_T_buf: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
            comptime if (
                CompilationTarget.is_macos() and DT == DType.float32
            ):
                dw_tmp = None
                go_b_T_buf = None
            else:
                dw_tmp = alloc[Scalar[DT]](Self.W_SIZE)
                go_b_T_buf = alloc[Scalar[DT]](
                    Self.SPATIAL_OUT * Self.OC
                )

            var w_tt = TileTensor(
                self.weight.val.cpu, row_major[Self.OC, Self.COL_SIZE](),
            )

            for b in range(BATCH):
                # ---- 1. Rebuild col_b for this batch ------------------
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

                # ---- 2. d_out_b view + d_bias accumulate --------------
                var go_b_p = go_p + b * Self.OUT_DIM_FLAT
                var go_b_tt = TileTensor(
                    go_b_p, row_major[Self.OC, Self.SPATIAL_OUT](),
                )
                comptime if mode == "all":
                    for oc in range(Self.OC):
                        var acc: Scalar[DT] = 0.0
                        var row_off = oc * Self.SPATIAL_OUT
                        for s in range(Self.SPATIAL_OUT):
                            acc += go_b_p[row_off + s]
                        db_p[oc] += acc

                # ---- 3. d_weight += d_out_b @ col_b -------------------
                #         d_out_b is [OC, SPATIAL_OUT],
                #         col_b   is [SPATIAL_OUT, COL_SIZE],
                #         result  is [OC, COL_SIZE] (= same flat as W).
                # On Apple fp32 we use one cblas_sgemm with beta=1 (no
                # temp). Elsewhere we matmul into dw_tmp and add.
                comptime if mode == "all":
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

                # ---- 4. d_col_b = d_out_b.T @ weight ------------------
                #         d_out_b.T is [SPATIAL_OUT, OC],
                #         weight     is [OC, COL_SIZE],
                #         result     is [SPATIAL_OUT, COL_SIZE].
                # `max_matmul` does NOT support `transpose_a=True`, so on
                # Apple fp32 we drop through to cblas (which does); on
                # other targets we materialise the transpose explicitly
                # into `go_b_T_buf` and call max_matmul untransposed.
                # Mirrors `linear.mojo` grad_w's Apple-vs-other split.
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

            col_buf.free()
            d_col_buf.free()
            comptime if not (
                CompilationTarget.is_macos() and DT == DType.float32
            ):
                dw_tmp.value().free()
                go_b_T_buf.value().free()
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var x_p = self._cached_input_ptr.value()
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(
                BATCH, Self.OUT_DIM_FLAT
            )
            comptime w_layout = Layout.row_major(Self.W_SIZE)
            comptime b_layout = Layout.row_major(Self.B_SIZE)
            var go_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](go_p)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](x_p)
            var gi_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](gi_p)
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight.val.dev.value()
            )
            var ctx = self.ts.ctx.value()

            # BACKWARD-ORDER INVARIANT (mirrors CPU + Linear): the param
            # grads (dW, dB) read `in_lt` (the cached forward input), so
            # they MUST run BEFORE dx, which writes `gi_lt`. When this
            # conv is not the first layer, Sequential aliases its
            # grad_input slab onto the same buffer that holds its cached
            # forward input (memory reuse). Since the GPU queue is
            # in-order, running dx first would clobber the input with
            # gradients before dw/db read it → corrupt dW (silent;
            # diverges only across multi-batch training, where the stale
            # input differs between steps). Enqueue dw/db first.
            comptime if mode == "all":
                var dw_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                    self.weight.grd.dev.value()
                )
                var db_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                    self.bias.grd.dev.value()
                )

                # d_weight — 1 block per weight scalar, CONV_DW_TPB
                # threads reduce over BATCH·OH·OW via `block.sum`.
                comptime dw_kernel = _conv2d_backward_dw_kernel[
                    BATCH, Self.IC, Self.OC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                    Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
                ]
                ctx.enqueue_function[dw_kernel](
                    go_lt, in_lt, dw_lt,
                    grid_dim=Self.W_SIZE, block_dim=CONV_DW_TPB,
                )

                # d_bias — 1 block per OC, CONV_DW_TPB threads reduce
                # over BATCH·OH·OW via `block.sum`.
                comptime db_kernel = _conv2d_backward_db_kernel[
                    BATCH, Self.OC, Self.OH, Self.OW, Self.OUT_DIM_FLAT,
                ]
                ctx.enqueue_function[db_kernel](
                    go_lt, db_lt,
                    grid_dim=Self.OC, block_dim=CONV_DW_TPB,
                )

            # d_input — 1 thread per (b, ic, ih, iw). Runs LAST so it may
            # safely overwrite an aliased input/grad_input slab.
            comptime total_in = BATCH * Self.IN_DIM_FLAT
            comptime n_blocks_in = (total_in + TPB - 1) // TPB
            comptime dx_kernel = _conv2d_backward_dx_kernel[
                BATCH, Self.IC, Self.OC, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
            ]
            ctx.enqueue_function[dx_kernel](
                go_lt, w_lt, gi_lt,
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
