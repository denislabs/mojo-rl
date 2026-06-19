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

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer


comptime CONV_DW_TPB: Int = 128


# ── CPU im2col / col2im over List storage (no pointers, no origins) ──────
def _im2col_cpu[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int
](ref in_list: List[Scalar[DT]], in_off: Int, mut col_list: List[Scalar[DT]]):
    """x[IC·H·W] slab at `in_off` → col_list[OH·OW, IC·K·K] row-major."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                var in_c_base = in_off + ic * H * W
                var col_ic_base = row_off + ic * K * K
                for kh in range(K):
                    var ih = oh * S + kh - P
                    var col_kh_base = col_ic_base + kh * K
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if ih < 0 or ih >= H or iw < 0 or iw >= W:
                            col_list[col_kh_base + kw] = Scalar[DT](0)
                        else:
                            col_list[col_kh_base + kw] = in_list[
                                in_c_base + ih * W + iw
                            ]


def _col2im_cpu[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int
](
    ref d_col_list: List[Scalar[DT]],
    mut d_in_list: List[Scalar[DT]],
    in_off: Int,
):
    """Scatter-add d_col[OH·OW, IC·K·K] back into d_in_list[IC·H·W] at
    `in_off` (must be pre-zeroed)."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                var in_c_base = in_off + ic * H * W
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
                        d_in_list[in_c_base + ih * W + iw] += d_col_list[
                            col_kh_base + kw
                        ]


# ── GPU kernels (re-derived; args MutAnyOrigin = the GPU ABI boundary) ──
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
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    col: LayoutTensor[DT, Layout.row_major(BS, COL), MutAnyOrigin],
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
    var ic = ck // (K * K)
    var rem = ck % (K * K)
    var kh = rem // K
    var kw = rem % K
    var ih = oh * S + kh - P
    var iw = ow * S + kw - P
    if ih < 0 or ih >= H or iw < 0 or iw >= W:
        col[row, ck] = Scalar[DT](0)
    else:
        col[row, ck] = rebind[Scalar[DT]](input[b, ic * H * W + ih * W + iw])


def _scatter_bias_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
](
    out_packed: LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var oc = out_pos // SO
    var s = out_pos % SO
    output[b, out_pos] = rebind[Scalar[DT]](
        out_packed[b * SO + s, oc]
    ) + rebind[Scalar[DT]](bias[oc])


def _go_transpose_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    go_T: LayoutTensor[DT, Layout.row_major(OC, BS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= OC * BS:
        return
    var oc = idx // BS
    var col = idx % BS
    var b = col // SO
    var s = col % SO
    go_T[oc, col] = rebind[Scalar[DT]](grad_output[b, oc * SO + s])


def _accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


def _go_pack_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    go_packed: LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BS * OC:
        return
    var row = idx // OC
    var oc = idx % OC
    var b = row // SO
    var s = row % SO
    go_packed[row, oc] = rebind[Scalar[DT]](grad_output[b, oc * SO + s])


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
](
    d_col: LayoutTensor[DT, Layout.row_major(BS, COL), MutAnyOrigin],
    grad_input: LayoutTensor[
        DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin
    ],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * IN_FLAT:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var hw = H * W
    var ic = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W
    var acc: Scalar[DT] = 0
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
            var col_idx = (ic * K + kh) * K + kw
            acc += rebind[Scalar[DT]](d_col[row, col_idx])
    grad_input[b, in_pos] = acc


def _backward_db_kernel[
    BATCH: Int,
    OC: Int,
    OH: Int,
    OW: Int,
    OUT_FLAT: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    grad_bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
):
    var oc = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if oc >= OC:
        return
    var so = OH * OW
    var n_eff = BATCH * so
    var out_c_off = oc * so
    var my_acc: Scalar[DT] = 0
    var idx = t
    while idx < n_eff:
        var b = idx // so
        var s_pos = idx % so
        my_acc += rebind[Scalar[DT]](grad_output[b, out_c_off + s_pos])
        idx += CONV_DW_TPB
    var total = block.sum[block_size=CONV_DW_TPB, broadcast=False](val=my_acc)
    if t == 0:
        grad_bias[oc] = rebind[Scalar[DT]](grad_bias[oc]) + total[0]


# ── Conv2D ────────────────────────────────────────────────────────────────
struct Conv2D[IC_: Int, OC_: Int, K_: Int, S_: Int, P_: Int, H_: Int, W_: Int](
    Module
):
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

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # GPU im2col + GEMM scratch (lazy, reused — capture-safe).
    var col_t: Tensor  # [BS, COL]  (im2col / d_col)
    var outp_t: Tensor  # [BS, OC]   (out_packed / go_packed)
    var goT_t: Tensor  # [OC, BS]   (goᵀ for dW)
    var dW_tmp: Tensor  # [OC, COL]

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.col_t = Tensor()
        self.outp_t = Tensor()
        self.goT_t = Tensor()
        self.dW_tmp = Tensor()

    @staticmethod
    def _init_w(mut w: Tensor):
        # Deterministic small init (parity harness overwrites this anyway).
        for k in range(Self.W_SIZE):
            w.data[k] = Scalar[DT]((k % 7) - 3) * 0.1

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var c = Self()
        c.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        c.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        Self._init_w(c.weight.val)
        comptime if target != "cpu":
            var dctx = ctx.value()
            c.weight.val.upload(dctx)
            c.bias.val.upload(dctx)
        return c^

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_FLAT)
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
                ](in0.data, b * Self.IN_FLAT, col)
                var col_tt = TileTensor(col, row_major[Self.SO, Self.COL]())
                var out_b_tt = TileTensor(out_b, row_major[Self.OC_, Self.SO]())
                # out_b[OC,SO] = W[OC,COL] @ col[SO,COL]ᵀ
                max_matmul[transpose_b=True, target="cpu"](
                    out_b_tt, w_tt, col_tt, None
                )
                # scatter + bias broadcast into out.data[b*OUT_FLAT:]
                var base = b * Self.OUT_FLAT
                for oc in range(Self.OC_):
                    var bv = self.bias.val.data[oc]
                    for s in range(Self.SO):
                        out.data[base + oc * Self.SO + s] = (
                            out_b[oc * Self.SO + s] + bv
                        )
        else:
            var c = ctx.value()
            comptime BS = B * Self.SO
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            self.col_t.ensure_gpu(c, BS * Self.COL)
            self.outp_t.ensure_gpu(c, BS * Self.OC_)
            # (1) im2col → col[BS, COL]
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
                ]
            ](
                in0.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                self.col_t.lt["gpu", Layout.row_major(BS, Self.COL)](),
                grid_dim=nb_col,
                block_dim=TPB,
            )
            # (2) out_packed[BS,OC] = col[BS,COL] @ W[OC,COL]ᵀ
            var col_tt = TileTensor(
                self.col_t.dev.value(), row_major[BS, Self.COL]()
            )
            var w_tt = TileTensor(
                self.weight.val.dev.value(), row_major[Self.OC_, Self.COL]()
            )
            var outp_tt = TileTensor(
                self.outp_t.dev.value(), row_major[BS, Self.OC_]()
            )
            max_matmul[transpose_b=True, target="gpu"](outp_tt, col_tt, w_tt, c)
            # (3) scatter → output[B, OC·SO] + bias
            comptime nb_sc = (B * Self.OUT_FLAT + TPB - 1) // TPB
            c.enqueue_function[
                _scatter_bias_kernel[
                    B,
                    Self.OC_,
                    Self.SO,
                    Self.OUT_FLAT,
                    BS,
                ]
            ](
                self.outp_t.lt["gpu", Layout.row_major(BS, Self.OC_)](),
                self.bias.val.lt["gpu", Layout.row_major(Self.OC_)](),
                out.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                grid_dim=nb_sc,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_FLAT)
            for k in range(B * Self.IN_FLAT):
                gin.data[k] = Scalar[DT](0)
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
                        acc += grad_output.data[go_base + oc * Self.SO + s]
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
                ](fin.data, b * Self.IN_FLAT, col)
                comptime if IS_APPLE_F32:
                    var cblas = get_cblas_f32_function()
                    var go_b_p = grad_output.data.unsafe_ptr() + go_base
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
                    var col_tt = TileTensor(col, row_major[Self.SO, Self.COL]())
                    var go_b = List[Scalar[DT]](
                        length=Self.OC_ * Self.SO, fill=Scalar[DT](0)
                    )
                    for i in range(Self.OC_ * Self.SO):
                        go_b[i] = grad_output.data[go_base + i]
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
                            go_b_T[s * Self.OC_ + oc] = go_b[oc * Self.SO + s]
                    var go_b_T_tt = TileTensor(
                        go_b_T, row_major[Self.SO, Self.OC_]()
                    )
                    var d_col_tt = TileTensor(
                        d_col, row_major[Self.SO, Self.COL]()
                    )
                    max_matmul[target="cpu"](d_col_tt, go_b_T_tt, w_tt, None)
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
                ](d_col, gin.data, b * Self.IN_FLAT)
        else:
            var c = ctx.value()
            comptime BS = B * Self.SO
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            self.col_t.ensure_gpu(c, BS * Self.COL)
            self.outp_t.ensure_gpu(c, BS * Self.OC_)
            self.goT_t.ensure_gpu(c, Self.OC_ * BS)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # (1) col = im2col(x)
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
                ]
            ](
                fin.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
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
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
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
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OC_)](),
                grid_dim=Self.OC_,
                block_dim=CONV_DW_TPB,
            )
            # (5) d_input: go_packed → d_col = go_packed @ W → col2im
            comptime nb_gp = (BS * Self.OC_ + TPB - 1) // TPB
            c.enqueue_function[
                _go_pack_kernel[
                    B,
                    Self.OC_,
                    Self.SO,
                    Self.OUT_FLAT,
                    BS,
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.outp_t.lt["gpu", Layout.row_major(BS, Self.OC_)](),
                grid_dim=nb_gp,
                block_dim=TPB,
            )
            var gopack_tt = TileTensor(
                self.outp_t.dev.value(), row_major[BS, Self.OC_]()
            )
            var w_tt = TileTensor(
                self.weight.val.dev.value(), row_major[Self.OC_, Self.COL]()
            )
            var dcol_tt = TileTensor(
                self.col_t.dev.value(), row_major[BS, Self.COL]()
            )
            max_matmul[target="gpu"](dcol_tt, gopack_tt, w_tt, c)
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
                ]
            ](
                self.col_t.lt["gpu", Layout.row_major(BS, Self.COL)](),
                gin.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                grid_dim=nb_dx,
                block_dim=CONV_DW_TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        self.weight.visit_with[target](visitor, ctx)
        self.bias.visit_with[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.weight.zero_grad[target](ctx)
        self.bias.zero_grad[target](ctx)
