"""Embedding[VOCAB, EMBED_DIM] — one-hot embedding lookup (storage surface).

Transformed from legacy `nn.primitives.Embedding`. The input is a `[BATCH,
VOCAB]` one-hot (DT); output `[BATCH, EMBED_DIM] = input @ weight`. One Param
(`weight` [VOCAB, EMBED_DIM], decay=True) + a leaf-owned cache of the one-hot
input. CPU uses the naive triple loops, GPU the 3 kernels (fwd / grad_in /
grad_w), carried over verbatim.

Backward: grad_in = grad_out @ Wᵀ ; grad_W += cache_inᵀ @ grad_out.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernels: the three matmuls go through max_matmul (mirrors Linear);
#    only the input-transpose (for grad_w = inputᵀ@grad_out, since max_matmul
#    rejects transpose_a) and the grad_weight accumulate remain hand-rolled. ──
def _emb_transpose_kernel[
    ROWS: Int, COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < ROWS * COLS:
        dst[idx % COLS, idx // COLS] = src[idx // COLS, idx % COLS]


def _emb_accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


struct Embedding[VOCAB_: Int, EMBED_DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.VOCAB_)
    comptime OUT_DIM = Self.EMBED_DIM_
    comptime W_SIZE = Self.VOCAB_ * Self.EMBED_DIM_

    var weight: Param["weight", True, Self.W_SIZE]
    var cache_in: Tensor  # [BATCH, VOCAB] one-hot (CPU grad_w)
    # GPU grad_w scratch (lazy): cache_inᵀ [VOCAB, B] (transposed in forward) +
    # gw_tmp [VOCAB, EMBED_DIM] (max_matmul output before accumulate).
    var cache_inT: Tensor
    var gw_tmp: Tensor

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.cache_in = Tensor()
        self.cache_inT = Tensor()
        self.gw_tmp = Tensor()

    @staticmethod
    def _init_w(mut w: Tensor):
        for k in range(Self.W_SIZE):
            w.data[k] = Scalar[DT](((k % 11) - 5)) * 0.07

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var e = Self()
        e.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        Self._init_w(e.weight.val)
        comptime if target != "cpu":
            e.weight.val.upload(ctx.value())
        return e^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.EMBED_DIM_)
            self.cache_in.ensure(B * Self.VOCAB_)
            var input = TileTensor(in0.data, row_major[B, Self.VOCAB_]())
            var output_v = TileTensor(out.data, row_major[B, Self.EMBED_DIM_]())
            var w_v = TileTensor(
                self.weight.val.data, row_major[Self.VOCAB_, Self.EMBED_DIM_]()
            )
            var cin = TileTensor(
                self.cache_in.data, row_major[B, Self.VOCAB_]()
            )
            for b in range(B):
                for v in range(Self.VOCAB_):
                    cin[b, v] = input[b, v]
                for j in range(Self.EMBED_DIM_):
                    var acc: Scalar[DT] = 0.0
                    for v in range(Self.VOCAB_):
                        acc += input[b, v] * w_v[v, j]
                    output_v[b, j] = acc
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.EMBED_DIM_)
            self.cache_inT.ensure_gpu(c, Self.VOCAB_ * B)
            comptime lbv = Layout.row_major(B, Self.VOCAB_)
            comptime lvb = Layout.row_major(Self.VOCAB_, B)
            # out[B, ED] = input[B, VOCAB] @ weight[VOCAB, ED]
            var in_v = TileTensor(in0.dev.value(), row_major[B, Self.VOCAB_]())
            var w_v = TileTensor(
                self.weight.val.dev.value(),
                row_major[Self.VOCAB_, Self.EMBED_DIM_](),
            )
            var out_v = TileTensor(
                out.dev.value(), row_major[B, Self.EMBED_DIM_]()
            )
            max_matmul[target="gpu"](out_v, in_v, w_v, c)
            # cache_inᵀ[VOCAB, B] = input[B, VOCAB]ᵀ  (for grad_w in backward)
            comptime tot = B * Self.VOCAB_
            comptime nblk = (tot + TPB - 1) // TPB
            c.enqueue_function[_emb_transpose_kernel[B, Self.VOCAB_]](
                in0.lt["gpu", lbv](),
                self.cache_inT.lt["gpu", lvb](),
                grid_dim=nblk,
                block_dim=TPB,
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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.VOCAB_)
            var go_v = TileTensor(
                grad_output.data, row_major[B, Self.EMBED_DIM_]()
            )
            var gi_v = TileTensor(gin.data, row_major[B, Self.VOCAB_]())
            var w_v = TileTensor(
                self.weight.val.data, row_major[Self.VOCAB_, Self.EMBED_DIM_]()
            )
            for b in range(B):
                for v in range(Self.VOCAB_):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.EMBED_DIM_):
                        acc += go_v[b, j] * w_v[v, j]
                    gi_v[b, v] = acc
            var gw_v = TileTensor(
                self.weight.grd.data, row_major[Self.VOCAB_, Self.EMBED_DIM_]()
            )
            var cin = TileTensor(
                self.cache_in.data, row_major[B, Self.VOCAB_]()
            )
            for v in range(Self.VOCAB_):
                for j in range(Self.EMBED_DIM_):
                    var acc: Scalar[DT] = 0.0
                    for b in range(B):
                        acc += cin[b, v] * go_v[b, j]
                    gw_v[v, j] += acc
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.VOCAB_)
            self.gw_tmp.ensure_gpu(c, Self.W_SIZE)
            comptime lw = Layout.row_major(Self.VOCAB_, Self.EMBED_DIM_)
            comptime lwf = Layout.row_major(Self.W_SIZE)
            # grad_in[B, VOCAB] = grad_out[B, ED] @ weight[VOCAB, ED]ᵀ
            var go_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.EMBED_DIM_]()
            )
            var w_v = TileTensor(
                self.weight.val.dev.value(),
                row_major[Self.VOCAB_, Self.EMBED_DIM_](),
            )
            var gi_v = TileTensor(
                gin.dev.value(), row_major[B, Self.VOCAB_]()
            )
            max_matmul[transpose_b=True, target="gpu"](gi_v, go_v, w_v, c)
            # gw_tmp[VOCAB, ED] = cache_inᵀ[VOCAB, B] @ grad_out[B, ED]
            var cinT_v = TileTensor(
                self.cache_inT.dev.value(), row_major[Self.VOCAB_, B]()
            )
            var gwtmp_v = TileTensor(
                self.gw_tmp.dev.value(),
                row_major[Self.VOCAB_, Self.EMBED_DIM_](),
            )
            max_matmul[target="gpu"](gwtmp_v, cinT_v, go_v, c)
            # weight.grad += gw_tmp  (accumulate, matches legacy semantics)
            comptime gw_blk = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_emb_accum_kernel[Self.W_SIZE]](
                self.weight.grd.lt["gpu", lwf](),
                self.gw_tmp.lt["gpu", lwf](),
                grid_dim=gw_blk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
