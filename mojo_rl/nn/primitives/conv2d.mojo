"""Conv2D[IC, OC, K, S, P, H, W] — 2D convolution on the storage surface.

The Conv2D de-risk for the storage migration (plan §2/§5: the ONE unproven
kernel). Reduction is identical to legacy `nn.primitives.Conv2D` — im2col +
`max_matmul` GEMM — but on the storage surface (`ref/mut Tensor`, `TensorRefs`,
`lt_gpu`), and the legacy's two-phase vjp split collapses to ONE `vjp` because
the storage surface passes `forward_input` (x) explicitly (invariant §3.1) — no
`_cached_input_ptr`, no param-before-input ordering hazard.

Layouts (flat trait order):
    input    [BATCH, IC·H·W]
    weight   [OC, IC·K·K]            (col index = (ic·K + kh)·K + kw)
    col      [BS, IC·K·K]            BS = BATCH·OH·OW   (im2col output)
    out      [BATCH, OC·OH·OW]

  forward:  col = im2col(x); out_packed[BS,OC] = col @ Wᵀ; scatter + bias.
  vjp:      d_bias = colsum(go); col = im2col(x); dW += goᵀ @ col;
            d_col = go_packed @ W; d_input = col2im(d_col).

CPU uses portable `max_matmul`-into-temp + accumulate (no Apple-cblas beta=1
special-case — correctness first; gated to tolerance vs a direct-conv ref). GPU
re-derives the legacy kernels here so `nn/storage` stays independent of the
legacy package (which gets deleted at the end of the migration).
"""

from std.sys import CompilationTarget
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from mojo_rl.nn.constants import DT, TPB, LAYOUT_NCHW, LAYOUT_NHWC
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor
from .linear import _cast_f2b_kernel, BF16


comptime CONV_DW_TPB: Int = 128

# A/B knob: block size for the flat thread-per-element im2col/scatter launches.
# These are bandwidth-bound elementwise kernels over BS·COL / B·OUT_FLAT, so 128
# vs 256 only changes occupancy granularity (both are warp multiples) — prediction
# is ~0% on NVIDIA (cf. O1). Bumped to 256 to A/B against Modular's im2col block
# size; revert to TPB (128) if no NVIDIA gain. Apple is parity-only here.
comptime CONV_TPB: Int = 256


# ── Layout-2D index helpers (NCHW default / NHWC) ────────────────────────────
# ONE source of truth for the channels-first vs channels-last index math, shared
# by every CPU + GPU conv kernel. Pure integer arithmetic, @always_inline →
# device-safe. NCHW reproduces the pre-LAYOUT formulas exactly (bit-identical).
#   input  (ic, ih, iw): NCHW ic*H*W + ih*W + iw   | NHWC (ih*W+iw)*IC + ic
#   COL    (ic, kh, kw): NCHW (ic*K+kh)*K + kw     | NHWC (kh*K+kw)*IC + ic
#   output (oc, s)     : NCHW oc*SO + s            | NHWC s*OC + oc
@always_inline
def _in_off[
    LAYOUT: Int, IC: Int, H: Int, W: Int
](ic: Int, ih: Int, iw: Int) -> Int:
    comptime if LAYOUT == LAYOUT_NHWC:
        return (ih * W + iw) * IC + ic
    else:
        return ic * H * W + ih * W + iw


@always_inline
def _in_decode[
    LAYOUT: Int, IC: Int, H: Int, W: Int
](in_pos: Int) -> Tuple[Int, Int, Int]:
    """Inverse of `_in_off`: flat within-sample offset → (ic, ih, iw)."""
    comptime if LAYOUT == LAYOUT_NHWC:
        var ic = in_pos % IC
        var sp = in_pos // IC
        return (ic, sp // W, sp % W)
    else:
        var hw = H * W
        var rem = in_pos % hw
        return (in_pos // hw, rem // W, rem % W)


@always_inline
def _col_off[LAYOUT: Int, IC: Int, K: Int](ic: Int, kh: Int, kw: Int) -> Int:
    comptime if LAYOUT == LAYOUT_NHWC:
        return (kh * K + kw) * IC + ic
    else:
        return (ic * K + kh) * K + kw


@always_inline
def _col_decode[
    LAYOUT: Int, IC: Int, K: Int
](ck: Int) -> Tuple[Int, Int, Int]:
    """Inverse of `_col_off`: COL index → (ic, kh, kw)."""
    comptime if LAYOUT == LAYOUT_NHWC:
        var ic = ck % IC
        var khkw = ck // IC
        return (ic, khkw // K, khkw % K)
    else:
        var ic = ck // (K * K)
        var rem = ck % (K * K)
        return (ic, rem // K, rem % K)


@always_inline
def _out_off[LAYOUT: Int, OC: Int, SO: Int](oc: Int, s: Int) -> Int:
    comptime if LAYOUT == LAYOUT_NHWC:
        return s * OC + oc
    else:
        return oc * SO + s


@always_inline
def _out_decode[
    LAYOUT: Int, OC: Int, SO: Int
](out_pos: Int) -> Tuple[Int, Int]:
    """Inverse of `_out_off`: flat within-sample offset → (oc, s)."""
    comptime if LAYOUT == LAYOUT_NHWC:
        return (out_pos % OC, out_pos // OC)
    else:
        return (out_pos // SO, out_pos % SO)


# ── CPU im2col / col2im over List storage (no pointers, no origins) ──────
def _im2col_cpu[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](ref in_list: List[Scalar[DT]], in_off: Int, mut col_list: List[Scalar[DT]]):
    """x[IC·H·W] slab at `in_off` → col_list[OH·OW, COL] row-major. COL axis and
    input slab follow LAYOUT (NCHW default)."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                for kh in range(K):
                    var ih = oh * S + kh - P
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        var c_idx = row_off + _col_off[LAYOUT, IC, K](ic, kh, kw)
                        if ih < 0 or ih >= H or iw < 0 or iw >= W:
                            col_list[c_idx] = Scalar[DT](0)
                        else:
                            col_list[c_idx] = in_list[
                                in_off + _in_off[LAYOUT, IC, H, W](ic, ih, iw)
                            ]


def _col2im_cpu[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    ref d_col_list: List[Scalar[DT]],
    mut d_in_list: List[Scalar[DT]],
    in_off: Int,
):
    """Scatter-add d_col[OH·OW, COL] back into d_in_list[IC·H·W] at `in_off`
    (must be pre-zeroed). COL axis and input slab follow LAYOUT (NCHW default)."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                for kh in range(K):
                    var ih = oh * S + kh - P
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if iw < 0 or iw >= W:
                            continue
                        d_in_list[
                            in_off + _in_off[LAYOUT, IC, H, W](ic, ih, iw)
                        ] += d_col_list[
                            row_off + _col_off[LAYOUT, IC, K](ic, kh, kw)
                        ]


# ── GPU kernels (re-derived; args MutAnyOrigin = the GPU ABI boundary) ──
# The GEMM-flanking kernels are dtype-parametric on the ACTIVATION dtype (`ADT`):
# the fp32 path calls them with DT, the bf16-flow path with bfloat16 (im2col,
# scatter, col2im, transpose, pack are pure gather/copy — dtype-transparent).
def _im2col_kernel[
    BATCH: Int,
    IC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    OH: Int,
    OW: Int,
    IN_FLAT: Int,
    COL: Int,
    SO: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    col: LayoutTensor[ADT, Layout.row_major(BS, COL), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BS * COL:
        return
    var row = idx // COL
    var ck = idx % COL
    var b = row // SO
    var s = row % SO
    var oh = s // OW
    var ow = s % OW
    var ic, kh, kw = _col_decode[LAYOUT, IC, K](ck)
    var ih = oh * S + kh - P
    var iw = ow * S + kw - P
    if ih < 0 or ih >= H or iw < 0 or iw >= W:
        col[row, ck] = Scalar[ADT](0)
    else:
        col[row, ck] = rebind[Scalar[ADT]](
            input[b, _in_off[LAYOUT, IC, H, W](ic, ih, iw)]
        )


def _scatter_bias_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    out_packed: LayoutTensor[ADT, Layout.row_major(BS, OC), MutAnyOrigin],
    bias: LayoutTensor[ADT, Layout.row_major(OC), MutAnyOrigin],
    output: LayoutTensor[ADT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var oc, s = _out_decode[LAYOUT, OC, SO](out_pos)
    output[b, out_pos] = rebind[Scalar[ADT]](
        out_packed[b * SO + s, oc]
    ) + rebind[Scalar[ADT]](bias[oc])


def _fwd_oc1_matvec_kernel[
    BATCH: Int,
    SO: Int,
    COL: Int,
    BS: Int,
    ADT: DType = DT,
](
    col: LayoutTensor[ADT, Layout.row_major(BS, COL), MutAnyOrigin],
    weight: LayoutTensor[ADT, Layout.row_major(COL), MutAnyOrigin],
    bias: LayoutTensor[ADT, Layout.row_major(1), MutAnyOrigin],
    output: LayoutTensor[ADT, Layout.row_major(BATCH, SO), MutAnyOrigin],
):
    """OC==1 forward: `max_matmul[transpose_b=True]` on GPU SILENTLY
    MISCOMPUTES N=1 GEMMs (out[BS,1] = col @ Wᵀ; verified wrong on Metal,
    N>=2 exact — sibling of the documented N=1 abort in the CPU vjp path).
    Compute the matvec + bias directly instead; with OC=1, OUT_FLAT == SO
    and the LAYOUT scatter is the identity, so this fuses GEMM + scatter.
    Accumulates in fp32 (matches the GEMM's accumulation for bf16)."""
    var idx = Int(global_idx.x)
    if idx >= BS:
        return
    var acc: Scalar[DT] = 0.0
    for k in range(COL):
        acc += (
            rebind[Scalar[ADT]](col[idx, k]).cast[DT]()
            * rebind[Scalar[ADT]](weight[k]).cast[DT]()
        )
    acc += rebind[Scalar[ADT]](bias[0]).cast[DT]()
    output[idx // SO, idx % SO] = acc.cast[ADT]()


def _go_transpose_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: LayoutTensor[
        ADT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    go_T: LayoutTensor[ADT, Layout.row_major(OC, BS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= OC * BS:
        return
    var oc = idx // BS
    var col = idx % BS
    var b = col // SO
    var s = col % SO
    go_T[oc, col] = rebind[Scalar[ADT]](
        grad_output[b, _out_off[LAYOUT, OC, SO](oc, s)]
    )


def _accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


def _wT_transpose_kernel[
    OC: Int, COL: Int, ADT: DType = DT
](
    w: LayoutTensor[ADT, Layout.row_major(OC, COL), MutAnyOrigin],
    wT: LayoutTensor[ADT, Layout.row_major(COL, OC), MutAnyOrigin],
):
    """O2: transpose weight `W[OC, COL]` → `Wᵀ[COL, OC]` so the input-grad GEMM
    can compute `d_colᵀ[COL, BS] = Wᵀ[COL, OC] @ goᵀ[OC, BS]` (reusing the `goᵀ`
    already built for dW — no `_go_pack` kernel, no `[BS,COL]` d_col). `W` is
    small (`OC·COL`), so the transpose is cheap relative to the col2im it
    coalesces. `max_matmul` rejects `transpose_a`, hence the explicit transpose."""
    var idx = Int(global_idx.x)
    if idx >= OC * COL:
        return
    var oc = idx // COL
    var c = idx % COL
    wT[c, oc] = rebind[Scalar[ADT]](w[oc, c])


def _dx_col2im_kernel[
    BATCH: Int,
    IC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    OH: Int,
    OW: Int,
    IN_FLAT: Int,
    COL: Int,
    SO: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    d_col: LayoutTensor[ADT, Layout.row_major(COL, BS), MutAnyOrigin],
    grad_input: LayoutTensor[
        ADT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin
    ],
):
    # O2: `d_col` is the TRANSPOSED layout `[COL, BS]` (vs the natural `[BS,COL]`
    # GEMM output). Adjacent threads differ in `iw` → `row` (col2im is a gather:
    # one input element ← up to K² d_col entries, each read exactly once, so no
    # cross-thread reuse and shared-mem tiling buys nothing — coalescing is the
    # only lever). In `[COL, BS]` the per-(kh,kw) read `d_col[col_idx, row]` has
    # adjacent threads at adjacent `row` → CONTIGUOUS, coalesced. Measured 1.59×
    # on the S=1 hot shape (NVIDIA); strided/`[BS,COL]` was stride-COL scattered.
    # O3: the `S==1` branch (the whole MuZero/EZv2 residual tower) drops the
    # `% S` / `// S` integer ops and a divergence source.
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * IN_FLAT:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var ic, ih, iw = _in_decode[LAYOUT, IC, H, W](in_pos)
    # fp32 accumulator (the col2im sum) even on the bf16-flow path; only the
    # written grad_input is bf16 (activation dtype).
    var acc: Scalar[DT] = 0
    comptime if S == 1:
        for kh in range(K):
            var oh = ih + P - kh
            if oh < 0 or oh >= OH:
                continue
            for kw in range(K):
                var ow = iw + P - kw
                if ow < 0 or ow >= OW:
                    continue
                var row = b * SO + oh * OW + ow
                var col_idx = _col_off[LAYOUT, IC, K](ic, kh, kw)
                acc += rebind[Scalar[ADT]](d_col[col_idx, row]).cast[DT]()
    else:
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
                var row = b * SO + oh * OW + ow
                var col_idx = _col_off[LAYOUT, IC, K](ic, kh, kw)
                acc += rebind[Scalar[ADT]](d_col[col_idx, row]).cast[DT]()
    grad_input[b, in_pos] = acc.cast[ADT]()


def _backward_db_kernel[
    BATCH: Int,
    OC: Int,
    OH: Int,
    OW: Int,
    OUT_FLAT: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: LayoutTensor[
        ADT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    grad_bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
):
    # grad_output is the ACTIVATION dtype (`ADT`); grad_bias is the FP32 master
    # grad. Each element casts to DT before the block-sum (fp32 accumulator).
    var oc = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if oc >= OC:
        return
    var so = OH * OW
    var n_eff = BATCH * so
    var my_acc: Scalar[DT] = 0
    var idx = t
    while idx < n_eff:
        var b = idx // so
        var s_pos = idx % so
        my_acc += rebind[Scalar[ADT]](
            grad_output[b, _out_off[LAYOUT, OC, OH * OW](oc, s_pos)]
        ).cast[DT]()
        idx += CONV_DW_TPB
    var total = block.sum[block_size=CONV_DW_TPB, broadcast=False](val=my_acc)
    if t == 0:
        grad_bias[oc] = rebind[Scalar[DT]](grad_bias[oc]) + total[0]


# ── Conv2D ────────────────────────────────────────────────────────────────
struct Conv2D[
    IC_: Int,
    OC_: Int,
    K_: Int,
    S_: Int,
    P_: Int,
    H_: Int,
    W_: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](Module):
    comptime ARITY = 1
    comptime OH = (Self.H_ + 2 * Self.P_ - Self.K_) // Self.S_ + 1
    comptime OW = (Self.W_ + 2 * Self.P_ - Self.K_) // Self.S_ + 1
    comptime IN_FLAT = Self.IC_ * Self.H_ * Self.W_
    comptime OUT_FLAT = Self.OC_ * Self.OH * Self.OW
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_FLAT)
    comptime OUT_DIM = Self.OUT_FLAT
    comptime W_SIZE = Self.OC_ * Self.IC_ * Self.K_ * Self.K_
    comptime B_SIZE = Self.OC_
    comptime COL = Self.IC_ * Self.K_ * Self.K_
    comptime SO = Self.OH * Self.OW
    # Activation-flow dtype (satisfies the Module trait). `Conv2D[...]` = fp32
    # (ACT_DT == DT, the legacy NoAMP path); `Conv2D[..., DType.bfloat16]` flows
    # activations at bf16 (the AMP "Step B" memory win). Master weights/grads/
    # bias STAY fp32 (`Param` is always `DT`); only the CACHED bf16 weight copy
    # (`w_bf`, version-gated) and bf16 bias (`b_a`) are low-precision.
    comptime ACT_DT = Self.ADT

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # GPU im2col + GEMM scratch (lazy, reused — capture-safe). fp32-path scratch
    # (`dW_tmp` is ALSO used by the bf16 path: the dW GEMM writes a FP32 output).
    var col_t: Tensor  # [BS, COL] (im2col col) reused as [COL, BS] d_colᵀ (O2)
    var outp_t: Tensor  # [BS, OC]   (out_packed; fwd only)
    var goT_t: Tensor  # [OC, BS]   (goᵀ — for dW AND d_colᵀ, O2)
    var dW_tmp: Tensor  # [OC, COL]  (fp32 dW temp; bf16-in → fp32-out GEMM)
    var wT_t: Tensor  # [COL, OC]  (O2: fp32 Wᵀ for the d_colᵀ GEMM)
    # bf16-flow compute scratch (lazy; used only when ACT_DT == bf16 and
    # target == "gpu"). `col_t_bf`/`outp_t_bf`/`goT_t_bf` are the bf16 activation
    # scratch (im2col col, out_packed/go_packed, goᵀ). `w_bf` is the CACHED bf16
    # weight: recast from `weight.val` only when the optimizer bumped its version
    # since the last cast (`_w_cast_version`), so the W cast happens ONCE per
    # optimizer step, not per forward. `b_a` is the cached bf16 bias.
    var col_t_bf: TensorImpl[Self.ADT]
    var outp_t_bf: TensorImpl[Self.ADT]
    var goT_t_bf: TensorImpl[Self.ADT]
    var w_bf: TensorImpl[Self.ADT]
    var b_a: TensorImpl[Self.ADT]
    var wT_bf: TensorImpl[Self.ADT]  # [COL, OC] (O2: bf16 Wᵀ for d_colᵀ GEMM)
    var _w_cast_version: Int  # `weight.val.version` at last bf16 weight cast
    # Capture mode (set via `set_attr["capture_recast"]`): when True, the bf16
    # weight recast is UNCONDITIONAL so the cast kernel is always recorded into a
    # CUDA graph and reads the live fp32 master on every replay — the version
    # gate would skip it on replay and serve STALE weights. Off → version-gated.
    var _force_recast: Bool
    # A2: device address of the input the forward last im2col'd into
    # `col_t`/`col_t_bf`. The vjp REUSES that col (skips its redundant im2col)
    # iff its `forward_input` is the SAME buffer; else it recomputes (safe
    # fallback = today's behavior, never wrong). Valid under the framework's
    # re-forward-before-vjp cache contract — the same one `BatchNorm2D.vjp`
    # already relies on for `cache_xhat` (conv sits next to BN in every block,
    # so wherever BN's single-field cache is valid at vjp, `col_t` is too).
    # 0 = no forward has populated the col yet.
    var _col_src_ptr: Int

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.col_t = Tensor()
        self.outp_t = Tensor()
        self.goT_t = Tensor()
        self.dW_tmp = Tensor()
        self.wT_t = Tensor()
        self.col_t_bf = TensorImpl[Self.ADT]()
        self.outp_t_bf = TensorImpl[Self.ADT]()
        self.goT_t_bf = TensorImpl[Self.ADT]()
        self.w_bf = TensorImpl[Self.ADT]()
        self.b_a = TensorImpl[Self.ADT]()
        self.wT_bf = TensorImpl[Self.ADT]()
        self._w_cast_version = -1  # < any real version → first forward casts
        self._force_recast = False
        self._col_src_ptr = 0

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "capture_recast":
            self._force_recast = value != Scalar[DT](0.0)

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Ensure the cached bf16 weight `w_bf` reflects the current fp32
        `weight.val`. Recasts ONLY when the optimizer bumped `val.version` since
        the last cast (so the weight cast is ONCE per step, not per fwd/bwd).
        Shared by forward (the cast) and vjp (which REUSES it — no optimizer step
        intervenes between a fwd and its bwd)."""
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        if self._force_recast or self.weight.val.version != self._w_cast_version:
            c.enqueue_function[_cast_f2b_kernel[Self.W_SIZE]](
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.w_bf.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=(Self.W_SIZE + 255) // 256,
                block_dim=256,
            )
            self._w_cast_version = self.weight.val.version

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var c = Self()
        c.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        c.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        # Receptive-field-scaled fan_in/fan_out for a conv weight [OC, IC, K, K]
        # so Kaiming/Xavier get the RIGHT bound. This previously called a fixed
        # `(k%7-3)*0.1` placeholder that IGNORED `INIT` — a degenerate, non-random
        # init that BatchNorm masked (CIFAR/ResNet) but BN-FREE conv nets (MuZero
        # spatial h/g/f) could not recover from. `INIT=Deterministic` reproduces
        # that exact pattern, so bit-parity gates are unchanged. `init_weight`/
        # `init_bias` upload to device on GPU.
        comptime fan_in = Self.IC_ * Self.K_ * Self.K_
        comptime fan_out = Self.OC_ * Self.K_ * Self.K_
        INIT.init_weight[target](c.weight.val, Self.W_SIZE, fan_in, fan_out, ctx)
        INIT.init_bias[target](c.bias.val, Self.B_SIZE, ctx)
        return c^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT in this branch, but the compiler doesn't collapse the
            # opaque `Self.ACT_DT` param to `DT` for type-unification against the
            # fp32 weight/bias views — so rebind the activation refs (sound: the
            # dtypes are equal here). `TensorImpl[Self.ACT_DT]` ≡ `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.OUT_FLAT)
                var col = List[Scalar[DT]](
                    length=Self.SO * Self.COL, fill=Scalar[DT](0)
                )
                var out_b = List[Scalar[DT]](
                    length=Self.OC_ * Self.SO, fill=Scalar[DT](0)
                )
                var w_tt = TileTensor(
                    self.weight.val.data, row_major[Self.OC_, Self.COL]()
                )
                for b in range(B):
                    _im2col_cpu[
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.LAYOUT,
                    ](in0d.data, b * Self.IN_FLAT, col)
                    var col_tt = TileTensor(col, row_major[Self.SO, Self.COL]())
                    var out_b_tt = TileTensor(
                        out_b, row_major[Self.OC_, Self.SO]()
                    )
                    # out_b[OC,SO] = W[OC,COL] @ col[SO,COL]ᵀ
                    max_matmul[transpose_b=True, target="cpu"](
                        out_b_tt, w_tt, col_tt, None
                    )
                    # scatter + bias broadcast into out.data[b*OUT_FLAT:] (out_b
                    # is the [OC,SO] GEMM result; the output offset follows LAYOUT)
                    var base = b * Self.OUT_FLAT
                    for oc in range(Self.OC_):
                        var bv = self.bias.val.data[oc]
                        for s in range(Self.SO):
                            outd.data[
                                base + _out_off[Self.LAYOUT, Self.OC_, Self.SO](
                                    oc, s
                                )
                            ] = (out_b[oc * Self.SO + s] + bv)
            else:
                var c = ctx.value()
                comptime BS = B * Self.SO
                outd.ensure_gpu(c, B * Self.OUT_FLAT)
                self.col_t.ensure_gpu(c, BS * Self.COL)
                self.outp_t.ensure_gpu(c, BS * Self.OC_)
                # (1) im2col → col[BS, COL]
                comptime nb_col = (BS * Self.COL + CONV_TPB - 1) // CONV_TPB
                c.enqueue_function[
                    _im2col_kernel[
                        B,
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.IN_FLAT,
                        Self.COL,
                        Self.SO,
                        BS,
                        DT,
                        Self.LAYOUT,
                    ]
                ](
                    in0d.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                    self.col_t.lt["gpu", Layout.row_major(BS, Self.COL)](),
                    grid_dim=nb_col,
                    block_dim=CONV_TPB,
                )
                # A2: record the input buffer so this forward's col_t can be
                # reused by the matching vjp (skipping a redundant im2col).
                self._col_src_ptr = Int(in0d.dev.value().unsafe_ptr())
                comptime if Self.OC_ == 1:
                    # (2+3 fused) OC==1: max_matmul GPU miscomputes N=1 —
                    # direct matvec + bias (see _fwd_oc1_matvec_kernel).
                    comptime nb_mv = (BS + CONV_TPB - 1) // CONV_TPB
                    c.enqueue_function[
                        _fwd_oc1_matvec_kernel[B, Self.SO, Self.COL, BS, DT]
                    ](
                        self.col_t.lt["gpu", Layout.row_major(BS, Self.COL)](),
                        self.weight.val.lt["gpu", Layout.row_major(Self.COL)](),
                        self.bias.val.lt["gpu", Layout.row_major(1)](),
                        outd.lt["gpu", Layout.row_major(B, Self.SO)](),
                        grid_dim=nb_mv,
                        block_dim=CONV_TPB,
                    )
                else:
                    # (2) out_packed[BS,OC] = col[BS,COL] @ W[OC,COL]ᵀ
                    var col_tt = TileTensor(
                        self.col_t.dev.value(), row_major[BS, Self.COL]()
                    )
                    var w_tt = TileTensor(
                        self.weight.val.dev.value(),
                        row_major[Self.OC_, Self.COL](),
                    )
                    var outp_tt = TileTensor(
                        self.outp_t.dev.value(), row_major[BS, Self.OC_]()
                    )
                    max_matmul[transpose_b=True, target="gpu"](
                        outp_tt, col_tt, w_tt, c
                    )
                    # (3) scatter → output[B, OC·SO] + bias
                    comptime nb_sc = (
                        B * Self.OUT_FLAT + CONV_TPB - 1
                    ) // CONV_TPB
                    c.enqueue_function[
                        _scatter_bias_kernel[
                            B,
                            Self.OC_,
                            Self.SO,
                            Self.OUT_FLAT,
                            BS,
                            DT,
                            Self.LAYOUT,
                        ]
                    ](
                        self.outp_t.lt["gpu", Layout.row_major(BS, Self.OC_)](),
                        self.bias.val.lt["gpu", Layout.row_major(Self.OC_)](),
                        outd.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                        grid_dim=nb_sc,
                        block_dim=CONV_TPB,
                    )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert target == "gpu", "bf16-flow Conv2D is GPU-only"
            var c = ctx.value()
            comptime BS = B * Self.SO
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            self.col_t_bf.ensure_gpu(c, BS * Self.COL)
            self.outp_t_bf.ensure_gpu(c, BS * Self.OC_)
            # x (in0) is ALREADY bf16 — no input cast. W: cached bf16 (recast
            # only on a version bump). bias: cheap per-forward DT→bf16 cast.
            self._ensure_w_bf(c)
            self.b_a.ensure_gpu(c, Self.B_SIZE)
            c.enqueue_function[_cast_f2b_kernel[Self.B_SIZE]](
                self.bias.val.lt["gpu", Layout.row_major(Self.B_SIZE)](),
                self.b_a.lt["gpu", Layout.row_major(Self.B_SIZE)](),
                grid_dim=(Self.B_SIZE + 255) // 256,
                block_dim=256,
            )
            # (1) im2col → col[BS, COL] (bf16-in → bf16-out)
            comptime nb_col = (BS * Self.COL + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _im2col_kernel[
                    B,
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.IN_FLAT,
                    Self.COL,
                    Self.SO,
                    BS,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                in0.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                self.col_t_bf.lt["gpu", Layout.row_major(BS, Self.COL)](),
                grid_dim=nb_col,
                block_dim=CONV_TPB,
            )
            # A2: record the input buffer for col_t_bf reuse in the matching vjp.
            self._col_src_ptr = Int(in0.dev.value().unsafe_ptr())
            comptime if Self.OC_ == 1:
                # (2+3 fused) OC==1: max_matmul GPU miscomputes N=1 —
                # direct matvec + bias (see _fwd_oc1_matvec_kernel).
                comptime nb_mv = (BS + CONV_TPB - 1) // CONV_TPB
                c.enqueue_function[
                    _fwd_oc1_matvec_kernel[B, Self.SO, Self.COL, BS, Self.ADT]
                ](
                    self.col_t_bf.lt["gpu", Layout.row_major(BS, Self.COL)](),
                    self.w_bf.lt["gpu", Layout.row_major(Self.COL)](),
                    self.b_a.lt["gpu", Layout.row_major(1)](),
                    out.lt["gpu", Layout.row_major(B, Self.SO)](),
                    grid_dim=nb_mv,
                    block_dim=CONV_TPB,
                )
            else:
                # (2) out_packed[BS,OC] = col[BS,COL] @ W[OC,COL]ᵀ — bf16-in →
                # bf16-out GEMM (fp32 accumulation is automatic).
                var col_tt = TileTensor(
                    self.col_t_bf.dev.value(), row_major[BS, Self.COL]()
                )
                var w_tt = TileTensor(
                    self.w_bf.dev.value(), row_major[Self.OC_, Self.COL]()
                )
                var outp_tt = TileTensor(
                    self.outp_t_bf.dev.value(), row_major[BS, Self.OC_]()
                )
                max_matmul[transpose_b=True, target="gpu"](
                    outp_tt, col_tt, w_tt, c
                )
                # (3) scatter → output[B, OC·SO] + bf16 bias
                comptime nb_sc = (B * Self.OUT_FLAT + CONV_TPB - 1) // CONV_TPB
                c.enqueue_function[
                    _scatter_bias_kernel[
                        B,
                        Self.OC_,
                        Self.SO,
                        Self.OUT_FLAT,
                        BS,
                        Self.ADT,
                        Self.LAYOUT,
                    ]
                ](
                    self.outp_t_bf.lt["gpu", Layout.row_major(BS, Self.OC_)](),
                    self.b_a.lt["gpu", Layout.row_major(Self.OC_)](),
                    out.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                    grid_dim=nb_sc,
                    block_dim=CONV_TPB,
                )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
          # ── fp32 path (legacy NoAMP, byte-identical) ──
          # ACT_DT IS DT here — rebind the activation refs (sound; see forward).
          ref find = rebind[Tensor](fin)
          ref gind = rebind[Tensor](gin)
          ref god = rebind[Tensor](grad_output)
          comptime if target == "cpu":
            gind.ensure(B * Self.IN_FLAT)
            for k in range(B * Self.IN_FLAT):
                gind.data[k] = Scalar[DT](0)
            var col = List[Scalar[DT]](
                length=Self.SO * Self.COL, fill=Scalar[DT](0)
            )
            var d_col = List[Scalar[DT]](
                length=Self.SO * Self.COL, fill=Scalar[DT](0)
            )
            var w_tt = TileTensor(
                self.weight.val.data, row_major[Self.OC_, Self.COL]()
            )
            # Apple-fp32: fused cblas paths (beta=1 dW-accumulate + TRANSPOSE
            # d_col), matching legacy Conv2D. Elsewhere: portable max_matmul
            # into a temp + add (one extra W_SIZE pass) and an explicit
            # transpose (max_matmul rejects transpose_a).
            comptime IS_APPLE_F32 = (
                CompilationTarget.is_macos() and DT == DType.float32
            )
            for b in range(B):
                var go_base = b * Self.OUT_FLAT
                # d_bias[oc] += Σ_s go[oc, s]
                for oc in range(Self.OC_):
                    var acc: Scalar[DT] = 0
                    for s in range(Self.SO):
                        acc += god.data[
                            go_base
                            + _out_off[Self.LAYOUT, Self.OC_, Self.SO](oc, s)
                        ]
                    self.bias.grd.data[oc] += acc
                # rebuild col_b = im2col(x_b)
                _im2col_cpu[
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.LAYOUT,
                ](find.data, b * Self.IN_FLAT, col)
                # The Apple-cblas fused path reinterprets grad_output memory as a
                # contiguous [OC, SO] matrix — only valid for NCHW. NHWC falls back
                # to the portable gather-into-[OC,SO] path below.
                comptime if IS_APPLE_F32 and Self.LAYOUT == LAYOUT_NCHW:
                    var cblas = get_cblas_f32_function()
                    var go_b_p = god.data.unsafe_ptr().unsafe_offset(go_base)
                    # dW += go_b[OC,SO] @ col_b[SO,COL]  (beta=1, no temp)
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.NO_TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.OC_),
                        Int32(Self.COL),
                        Int32(Self.SO),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](go_b_p),
                        Int32(Self.SO),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            col.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](
                            self.weight.grd.data.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                    )
                    # d_col[SO,COL] = go_bᵀ[SO,OC] @ W[OC,COL]  (beta=0, A^T)
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.SO),
                        Int32(Self.COL),
                        Int32(Self.OC_),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](go_b_p),
                        Int32(Self.SO),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            self.weight.val.data.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                        Float32(0.0),
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](
                            d_col.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                    )
                else:
                    # gather grad_output (LAYOUT-ordered) into packed [OC, SO]
                    var go_b = List[Scalar[DT]](
                        length=Self.OC_ * Self.SO, fill=Scalar[DT](0)
                    )
                    for oc in range(Self.OC_):
                        for s in range(Self.SO):
                            go_b[oc * Self.SO + s] = god.data[
                                go_base
                                + _out_off[Self.LAYOUT, Self.OC_, Self.SO](oc, s)
                            ]
                    comptime if Self.COL == 1:
                        # Degenerate N=COL=1 GEMM (a 1x1 conv with IC=1 — the
                        # MuZero action-embedding). `max_matmul` aborts on N=1 on
                        # the non-Apple path, so compute the two matrix-vector
                        # products directly. col / d_col are [SO, 1] == [SO];
                        # W is [OC, 1] == weight[oc].
                        #   dW[oc]  += Σ_s go[oc,s]·col[s]
                        #   d_col[s] = Σ_oc go[oc,s]·W[oc]
                        for oc in range(Self.OC_):
                            var acc = Scalar[DT](0)
                            for s in range(Self.SO):
                                acc += go_b[oc * Self.SO + s] * col[s]
                            self.weight.grd.data[oc] += acc
                        for s in range(Self.SO):
                            var acc = Scalar[DT](0)
                            for oc in range(Self.OC_):
                                acc += go_b[oc * Self.SO + s] * self.weight.val.data[oc]
                            d_col[s] = acc
                    else:
                        var col_tt = TileTensor(
                            col, row_major[Self.SO, Self.COL]()
                        )
                        var go_b_tt = TileTensor(
                            go_b, row_major[Self.OC_, Self.SO]()
                        )
                        # dW += go_b[OC,SO] @ col[SO,COL]
                        var dw_tmp = List[Scalar[DT]](
                            length=Self.W_SIZE, fill=Scalar[DT](0)
                        )
                        var dw_tmp_tt = TileTensor(
                            dw_tmp, row_major[Self.OC_, Self.COL]()
                        )
                        max_matmul[target="cpu"](dw_tmp_tt, go_b_tt, col_tt, None)
                        for k in range(Self.W_SIZE):
                            self.weight.grd.data[k] += dw_tmp[k]
                        # d_col[SO,COL] = go_bᵀ[SO,OC] @ W[OC,COL]
                        var go_b_T = List[Scalar[DT]](
                            length=Self.SO * Self.OC_, fill=Scalar[DT](0)
                        )
                        for s in range(Self.SO):
                            for oc in range(Self.OC_):
                                go_b_T[s * Self.OC_ + oc] = go_b[
                                    oc * Self.SO + s
                                ]
                        var go_b_T_tt = TileTensor(
                            go_b_T, row_major[Self.SO, Self.OC_]()
                        )
                        var d_col_tt = TileTensor(
                            d_col, row_major[Self.SO, Self.COL]()
                        )
                        max_matmul[target="cpu"](
                            d_col_tt, go_b_T_tt, w_tt, None
                        )
                # col2im → grad_input_b (scatter-add)
                _col2im_cpu[
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.LAYOUT,
                ](d_col, gind.data, b * Self.IN_FLAT)
          else:
            var c = ctx.value()
            comptime BS = B * Self.SO
            gind.ensure_gpu(c, B * Self.IN_FLAT)
            self.col_t.ensure_gpu(c, BS * Self.COL)
            self.goT_t.ensure_gpu(c, Self.OC_ * BS)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # (1) col = im2col(x) — A2: REUSE the forward's col_t when this vjp's
            # forward_input is the buffer the forward im2col'd; else recompute
            # (safe fallback). See `_col_src_ptr`.
            if (
                self._col_src_ptr == 0
                or Int(find.dev.value().unsafe_ptr()) != self._col_src_ptr
            ):
                comptime nb_col = (BS * Self.COL + TPB - 1) // TPB
                c.enqueue_function[
                    _im2col_kernel[
                        B,
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.IN_FLAT,
                        Self.COL,
                        Self.SO,
                        BS,
                        DT,
                        Self.LAYOUT,
                    ]
                ](
                    find.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                    self.col_t.lt["gpu", Layout.row_major(BS, Self.COL)](),
                    grid_dim=nb_col,
                    block_dim=TPB,
                )
            # (2) goᵀ[OC,BS] = transpose(grad_output)
            comptime nb_got = (Self.OC_ * BS + TPB - 1) // TPB
            c.enqueue_function[
                _go_transpose_kernel[
                    B,
                    Self.OC_,
                    Self.SO,
                    Self.OUT_FLAT,
                    BS,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                god.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.goT_t.lt["gpu", Layout.row_major(Self.OC_, BS)](),
                grid_dim=nb_got,
                block_dim=TPB,
            )
            # (3) dW_tmp = goᵀ @ col → accumulate into grad_w
            var goT_tt = TileTensor(
                self.goT_t.dev.value(), row_major[Self.OC_, BS]()
            )
            var col_tt = TileTensor(
                self.col_t.dev.value(), row_major[BS, Self.COL]()
            )
            var dW_tmp_tt = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.OC_, Self.COL]()
            )
            max_matmul[target="gpu"](dW_tmp_tt, goT_tt, col_tt, c)
            comptime nb_acc = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=nb_acc,
                block_dim=TPB,
            )
            # (4) d_bias — 1 block per OC
            c.enqueue_function[
                _backward_db_kernel[
                    B,
                    Self.OC_,
                    Self.OH,
                    Self.OW,
                    Self.OUT_FLAT,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                god.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OC_)](),
                grid_dim=Self.OC_,
                block_dim=CONV_DW_TPB,
            )
            # (5) d_input (O2): d_colᵀ[COL,BS] = Wᵀ[COL,OC] @ goᵀ[OC,BS] reusing
            # the goᵀ from (2) — NO `_go_pack` kernel — then the coalesced col2im
            # reading the transposed d_colᵀ. col_t (free after the dW GEMM) is
            # reused as the [COL, BS] d_colᵀ buffer.
            self.wT_t.ensure_gpu(c, Self.COL * Self.OC_)
            comptime nb_wt = (Self.OC_ * Self.COL + TPB - 1) // TPB
            c.enqueue_function[_wT_transpose_kernel[Self.OC_, Self.COL]](
                self.weight.val.lt["gpu", Layout.row_major(Self.OC_, Self.COL)](),
                self.wT_t.lt["gpu", Layout.row_major(Self.COL, Self.OC_)](),
                grid_dim=nb_wt,
                block_dim=TPB,
            )
            var wT_tt = TileTensor(
                self.wT_t.dev.value(), row_major[Self.COL, Self.OC_]()
            )
            var goT2_tt = TileTensor(
                self.goT_t.dev.value(), row_major[Self.OC_, BS]()
            )
            var dcolT_tt = TileTensor(
                self.col_t.dev.value(), row_major[Self.COL, BS]()
            )
            max_matmul[target="gpu"](dcolT_tt, wT_tt, goT2_tt, c)
            comptime nb_dx = (B * Self.IN_FLAT + CONV_DW_TPB - 1) // CONV_DW_TPB
            c.enqueue_function[
                _dx_col2im_kernel[
                    B,
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.IN_FLAT,
                    Self.COL,
                    Self.SO,
                    BS,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                self.col_t.lt["gpu", Layout.row_major(Self.COL, BS)](),
                gind.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                grid_dim=nb_dx,
                block_dim=CONV_DW_TPB,
            )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert target == "gpu", "bf16-flow Conv2D is GPU-only"
            var c = ctx.value()
            comptime BS = B * Self.SO
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            self.col_t_bf.ensure_gpu(c, BS * Self.COL)
            self.goT_t_bf.ensure_gpu(c, Self.OC_ * BS)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)  # fp32 dW temp
            self._ensure_w_bf(c)  # cached bf16 weight (reused from forward)
            # (1) col = im2col(x): x (fin) ALREADY bf16 → bf16 col. A2: REUSE the
            # forward's col_t_bf when fin is the buffer the forward im2col'd; else
            # recompute (safe fallback). See `_col_src_ptr`.
            if (
                self._col_src_ptr == 0
                or Int(fin.dev.value().unsafe_ptr()) != self._col_src_ptr
            ):
                comptime nb_col = (BS * Self.COL + TPB - 1) // TPB
                c.enqueue_function[
                    _im2col_kernel[
                        B,
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.IN_FLAT,
                        Self.COL,
                        Self.SO,
                        BS,
                        Self.ADT,
                        Self.LAYOUT,
                    ]
                ](
                    fin.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                    self.col_t_bf.lt["gpu", Layout.row_major(BS, Self.COL)](),
                    grid_dim=nb_col,
                    block_dim=TPB,
                )
            # (2) goᵀ[OC,BS] = transpose(grad_output): bf16 go → bf16 goᵀ.
            comptime nb_got = (Self.OC_ * BS + TPB - 1) // TPB
            c.enqueue_function[
                _go_transpose_kernel[
                    B,
                    Self.OC_,
                    Self.SO,
                    Self.OUT_FLAT,
                    BS,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.goT_t_bf.lt["gpu", Layout.row_major(Self.OC_, BS)](),
                grid_dim=nb_got,
                block_dim=TPB,
            )
            # (3) dW_tmp = goᵀ @ col → bf16-in, FP32-out GEMM → accumulate into
            # the fp32 master grad.
            var goT_tt = TileTensor(
                self.goT_t_bf.dev.value(), row_major[Self.OC_, BS]()
            )
            var col_tt = TileTensor(
                self.col_t_bf.dev.value(), row_major[BS, Self.COL]()
            )
            var dW_tmp_tt = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.OC_, Self.COL]()
            )
            max_matmul[target="gpu"](dW_tmp_tt, goT_tt, col_tt, c)
            comptime nb_acc = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=nb_acc,
                block_dim=TPB,
            )
            # (4) d_bias — bf16 go → FP32 master grad (fp32 accumulator).
            c.enqueue_function[
                _backward_db_kernel[
                    B,
                    Self.OC_,
                    Self.OH,
                    Self.OW,
                    Self.OUT_FLAT,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OC_)](),
                grid_dim=Self.OC_,
                block_dim=CONV_DW_TPB,
            )
            # (5) d_input (O2): d_colᵀ[COL,BS] = Wᵀ_bf[COL,OC] @ goᵀ[OC,BS]
            # reusing goᵀ from (2) — NO `_go_pack` kernel. All bf16 (grad_input
            # flows bf16); Wᵀ_bf transposes the forward's cached bf16 weight.
            # col_t_bf (free after the dW GEMM) is reused as the [COL,BS] d_colᵀ.
            self.wT_bf.ensure_gpu(c, Self.COL * Self.OC_)
            comptime nb_wt = (Self.OC_ * Self.COL + TPB - 1) // TPB
            c.enqueue_function[
                _wT_transpose_kernel[Self.OC_, Self.COL, Self.ADT]
            ](
                self.w_bf.lt["gpu", Layout.row_major(Self.OC_, Self.COL)](),
                self.wT_bf.lt["gpu", Layout.row_major(Self.COL, Self.OC_)](),
                grid_dim=nb_wt,
                block_dim=TPB,
            )
            var wbT_tt = TileTensor(
                self.wT_bf.dev.value(), row_major[Self.COL, Self.OC_]()
            )
            var goT2_tt = TileTensor(
                self.goT_t_bf.dev.value(), row_major[Self.OC_, BS]()
            )
            var dcolT_tt = TileTensor(
                self.col_t_bf.dev.value(), row_major[Self.COL, BS]()
            )
            max_matmul[target="gpu"](dcolT_tt, wbT_tt, goT2_tt, c)
            comptime nb_dx = (B * Self.IN_FLAT + CONV_DW_TPB - 1) // CONV_DW_TPB
            c.enqueue_function[
                _dx_col2im_kernel[
                    B,
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.IN_FLAT,
                    Self.COL,
                    Self.SO,
                    BS,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                self.col_t_bf.lt["gpu", Layout.row_major(Self.COL, BS)](),
                gin.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                grid_dim=nb_dx,
                block_dim=CONV_DW_TPB,
            )

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update weight + bias toward `src` (target ← online). Required
        for use as a target net (pixel-obs critics use Conv2D); the Module
        default is a no-op which would silently freeze the target (see
        LinearReLU polyak bug, Stage-5)."""
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
