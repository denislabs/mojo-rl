"""Linear — Module conformer (CPU + GPU). y = x @ W + b via max_matmul.

Each forward/vjp branches `comptime if target == "cpu"` (tracked `TileTensor`
over `.data`) `else` (device `LayoutTensor` via `lt_gpu` + a naive kernel).
The storage surface (`ref/mut Tensor`, `TensorRefs`) is identical on both
targets; the only GPU erasure is the kernel-arg `MutAnyOrigin`. Params are
`Param` (two `Tensor`s, cpu+dev).

LIFETIME NOTE: a pack subscript (`inputs[k]`) returns a TEMPORARY ref. Building
a view from `inputs[k].data` directly and using it LATER dangles (the temporary
dies at the end of the statement; a later op clobbers the stack). So each body
first binds the element to a named `ref` (`ref in0 = inputs[0]`) that lives for
the whole function, then builds views from that.
"""

from std.sys import CompilationTarget
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..loss.sac import polyak_tensor


# ── kernels (non-GEMM ops; the three matmuls go through max_matmul) ──────
def _bias_add_kernel[
    B: Int, OUT: Int
](
    o: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        o[idx // OUT, idx % OUT] += bias[idx % OUT]


# Naive grad_w transpose (one thread/elem, strided read). Still used by
# linear_relu / noisy_linear; linear + linear_act use the tiled B1' kernel below.
def _transpose_kernel[
    ROWS: Int, COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < ROWS * COLS:
        dst[idx % COLS, idx // COLS] = src[idx // COLS, idx % COLS]


comptime _T_TILE = 32
comptime _T_BR = 8        # 32x8 BLOCK_ROWS, 4 elems/thread (B1')


# Tiled grad_w transpose: 32x8 BLOCK_ROWS shared-mem tile (B1'). Coalesces the
# strided src read the naive kernel suffers; bench (RTX 5090) showed up to 1.65x
# on skinny grad_w (large ROWS·COLS), neutral on square. Launch with
# grid_dim=(ceildiv(COLS,32), ceildiv(ROWS,32)), block_dim=(32, 8).
def _transpose_tiled_kernel[
    ROWS: Int, COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var tile = LayoutTensor[
        DT,
        Layout.row_major(_T_TILE, _T_TILE + 1),   # +1 pad → no bank conflicts
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var cy = Int(block_idx.y) * _T_TILE       # tile origin in ROWS
    var cx = Int(block_idx.x) * _T_TILE       # tile origin in COLS
    var tx = Int(thread_idx.x)                # [0, _T_TILE)
    var ty = Int(thread_idx.y)                # [0, _T_BR)

    var c = cx + tx
    comptime for r in range(0, _T_TILE, _T_BR):
        var rr = cy + ty + r
        if rr < ROWS and c < COLS:
            tile[ty + r, tx] = rebind[Scalar[DT]](src[rr, c])
    barrier()

    var r2 = cy + tx                          # dst col (coalesced, stride 1)
    comptime for r in range(0, _T_TILE, _T_BR):
        var c2 = cx + ty + r                  # dst row
        if r2 < ROWS and c2 < COLS:
            dst[c2, r2] = rebind[Scalar[DT]](tile[tx, ty + r])


def _accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


def _lin_gb_kernel[
    B: Int, OUT: Int
](
    go: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
    gb: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < OUT:
        var s: go.element_type = 0
        for b in range(B):
            s += go[b, j]
        gb[j] += s


comptime BF16 = DType.bfloat16


def _cast_f2b_kernel[
    N: Int
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[BF16, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[BF16]()


def _cast_b2f_kernel[
    N: Int
](
    src: LayoutTensor[BF16, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[DT]()


# ── Linear ─────────────────────────────────────────────────────────────
struct Linear[IN_: Int, OUT_: Int, AMP: Bool = False](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_

    @staticmethod
    def display_label() -> String:
        return String("Linear")
    comptime B_SIZE = Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # grad_w scratch (lazy): cacheᵀ [IN, B] + dW_tmp [IN, OUT] for the
    # transpose + max_matmul + accumulate path (max_matmul rejects transpose_a).
    var cacheT: Tensor
    var dW_tmp: Tensor
    # AMP bf16 compute scratch (lazy; used only when AMP and target == "gpu").
    var x_bf: TensorImpl[BF16]
    var w_bf: TensorImpl[BF16]
    var o_bf: TensorImpl[BF16]

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.cacheT = Tensor()
        self.dW_tmp = Tensor()
        self.x_bf = TensorImpl[BF16]()
        self.w_bf = TensorImpl[BF16]()
        self.o_bf = TensorImpl[BF16]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var l = Self()
        l.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        l.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        INIT.init_weight[target](
            l.weight.val, Self.W_SIZE, Self.IN_, Self.OUT_, ctx
        )
        INIT.init_bias[target](l.bias.val, Self.B_SIZE, ctx)
        return l^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        # y = x @ W (max_matmul), then += bias.
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_)
            var x_v = TileTensor(in0.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.data, row_major[B, Self.OUT_]())
            max_matmul[target="cpu"](out_v, x_v, w_v, None)
            var bt = TileTensor(self.bias.val.data, row_major[Self.OUT_]())
            for b in range(B):
                for j in range(Self.OUT_):
                    out_v[b, j] += bt[j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            comptime if Self.AMP:
                # AMP: cast x,W → bf16, bf16 max_matmul → o_bf, cast → fp32.
                # Master weights/grads stay fp32; only the matmul is bf16.
                self.x_bf.ensure_gpu(c, B * Self.IN_)
                self.w_bf.ensure_gpu(c, Self.W_SIZE)
                self.o_bf.ensure_gpu(c, B * Self.OUT_)
                c.enqueue_function[_cast_f2b_kernel[B * Self.IN_]](
                    in0.lt["gpu", Layout.row_major(B * Self.IN_)](),
                    self.x_bf.lt["gpu", Layout.row_major(B * Self.IN_)](),
                    grid_dim=(B * Self.IN_ + 255) // 256,
                    block_dim=256,
                )
                c.enqueue_function[_cast_f2b_kernel[Self.W_SIZE]](
                    self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                    self.w_bf.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                    grid_dim=(Self.W_SIZE + 255) // 256,
                    block_dim=256,
                )
                var x_bf_v = TileTensor(
                    self.x_bf.dev.value(), row_major[B, Self.IN_]()
                )
                var w_bf_v = TileTensor(
                    self.w_bf.dev.value(), row_major[Self.IN_, Self.OUT_]()
                )
                var o_bf_v = TileTensor(
                    self.o_bf.dev.value(), row_major[B, Self.OUT_]()
                )
                max_matmul[target="gpu"](o_bf_v, x_bf_v, w_bf_v, c)
                c.enqueue_function[_cast_b2f_kernel[B * Self.OUT_]](
                    self.o_bf.lt["gpu", Layout.row_major(B * Self.OUT_)](),
                    out.lt["gpu", Layout.row_major(B * Self.OUT_)](),
                    grid_dim=(B * Self.OUT_ + 255) // 256,
                    block_dim=256,
                )
            else:
                var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.IN_, Self.OUT_](),
                )
                var out_v = TileTensor(
                    out.dev.value(), row_major[B, Self.OUT_]()
                )
                max_matmul[target="gpu"](out_v, x_v, w_v, c)
            var ol = out.lt["gpu", Layout.row_major(B, Self.OUT_)]()
            var bl = self.bias.val.lt["gpu", Layout.row_major(Self.OUT_)]()
            c.enqueue_function[_bias_add_kernel[B, Self.OUT_]](
                ol, bl, grid_dim=(B * Self.OUT_ + 255) // 256, block_dim=256
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
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
            gin.ensure(B * Self.IN_)
            self.cacheT.ensure(Self.IN_ * B)
            self.dW_tmp.ensure(Self.W_SIZE)
            var x_v = TileTensor(fin.data, row_major[B, Self.IN_]())
            var go_v = TileTensor(grad_output.data, row_major[B, Self.OUT_]())
            var gi_v = TileTensor(gin.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
            )
            var gw_v = TileTensor(
                self.weight.grd.data, row_major[Self.IN_, Self.OUT_]()
            )
            var gb_v = TileTensor(self.bias.grd.data, row_major[Self.OUT_]())
            # grad_b += colsum(go)
            for b in range(B):
                for j in range(Self.OUT_):
                    gb_v[j] += go_v[b, j]
            # grad_w += xᵀ @ go. Apple-fp32: ONE fused cblas_sgemm (TRANSPOSE A=x,
            # beta=1 → no transpose buffer, no temp, no accumulate loop — matches
            # legacy Linear). Else: transpose into cacheT + max_matmul + add.
            comptime IS_APPLE_F32 = (
                CompilationTarget.is_macos() and DT == DType.float32
            )
            comptime if IS_APPLE_F32:
                var cblas = get_cblas_f32_function()
                cblas(
                    _CBLASOrder.ROW_MAJOR,
                    _CBLASTranspose.TRANSPOSE,
                    _CBLASTranspose.NO_TRANSPOSE,
                    Int32(Self.IN_),
                    Int32(Self.OUT_),
                    Int32(B),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        fin.data.unsafe_ptr()
                    ),
                    Int32(Self.IN_),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        grad_output.data.unsafe_ptr()
                    ),
                    Int32(Self.OUT_),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](
                        self.weight.grd.data.unsafe_ptr()
                    ),
                    Int32(Self.OUT_),
                )
            else:
                var cT_v = TileTensor(
                    self.cacheT.data, row_major[Self.IN_, B]()
                )
                for b in range(B):
                    for k in range(Self.IN_):
                        cT_v[k, b] = x_v[b, k]
                var dW_v = TileTensor(
                    self.dW_tmp.data, row_major[Self.IN_, Self.OUT_]()
                )
                max_matmul[target="cpu"](dW_v, cT_v, go_v, None)
                for k in range(Self.IN_):
                    for j in range(Self.OUT_):
                        gw_v[k, j] += dW_v[k, j]
            # grad_input = go @ Wᵀ
            max_matmul[transpose_b=True, target="cpu"](gi_v, go_v, w_v, None)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            self.cacheT.ensure_gpu(c, Self.IN_ * B)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # grad_b += colsum(go)
            var gol = grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)]()
            var gbl = self.bias.grd.lt["gpu", Layout.row_major(Self.OUT_)]()
            c.enqueue_function[_lin_gb_kernel[B, Self.OUT_]](
                gol, gbl, grid_dim=(Self.OUT_ + 255) // 256, block_dim=256
            )
            # grad_w += cacheᵀ @ go: transpose x → cacheT (B1' tiled), matmul,
            # accumulate.
            var xl = fin.lt["gpu", Layout.row_major(B, Self.IN_)]()
            var cTl = self.cacheT.lt["gpu", Layout.row_major(Self.IN_, B)]()
            c.enqueue_function[_transpose_tiled_kernel[B, Self.IN_]](
                xl,
                cTl,
                grid_dim=(
                    (Self.IN_ + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )
            var cT_v = TileTensor(
                self.cacheT.dev.value(), row_major[Self.IN_, B]()
            )
            var go_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var dW_v = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            max_matmul[target="gpu"](dW_v, cT_v, go_v, c)
            var gwl = self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)]()
            var dWl = self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)]()
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                gwl, dWl, grid_dim=(Self.W_SIZE + 255) // 256, block_dim=256
            )
            # grad_input = go @ Wᵀ
            var go_v2 = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var w_v = TileTensor(
                self.weight.val.dev.value(),
                row_major[Self.IN_, Self.OUT_](),
            )
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.IN_]())
            max_matmul[transpose_b=True, target="gpu"](gi_v, go_v2, w_v, c)

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `weight` + `bias` Params).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )
