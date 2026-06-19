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

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _embedding_fwd_kernel[
    BATCH: Int, VOCAB: Int, EMBED_DIM: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(VOCAB, EMBED_DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, EMBED_DIM), MutAnyOrigin],
    cache_in: LayoutTensor[DT, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * EMBED_DIM
    if gid >= total:
        return
    var b = gid // EMBED_DIM
    var j = gid % EMBED_DIM
    var acc: Scalar[DT] = 0.0
    for v in range(VOCAB):
        var x = rebind[Scalar[DT]](input[b, v])
        acc += x * rebind[Scalar[DT]](weight[v, j])
        if j == 0:
            cache_in[b, v] = x
    output[b, j] = acc


def _embedding_grad_in_kernel[
    BATCH: Int, VOCAB: Int, EMBED_DIM: Int
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, EMBED_DIM), MutAnyOrigin
    ],
    weight: LayoutTensor[DT, Layout.row_major(VOCAB, EMBED_DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * VOCAB
    if gid >= total:
        return
    var b = gid // VOCAB
    var v = gid % VOCAB
    var acc: Scalar[DT] = 0.0
    for j in range(EMBED_DIM):
        acc += rebind[Scalar[DT]](grad_output[b, j]) * rebind[Scalar[DT]](
            weight[v, j]
        )
    grad_input[b, v] = acc


def _embedding_grad_w_kernel[
    BATCH: Int, VOCAB: Int, EMBED_DIM: Int
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, EMBED_DIM), MutAnyOrigin
    ],
    cache_in: LayoutTensor[DT, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
    grad_weight: LayoutTensor[
        DT, Layout.row_major(VOCAB, EMBED_DIM), MutAnyOrigin
    ],
):
    var gid = Int(global_idx.x)
    comptime total = VOCAB * EMBED_DIM
    if gid >= total:
        return
    var v = gid // EMBED_DIM
    var j = gid % EMBED_DIM
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[DT]](cache_in[b, v]) * rebind[Scalar[DT]](
            grad_output[b, j]
        )
    grad_weight[v, j] = rebind[Scalar[DT]](grad_weight[v, j]) + acc


struct Embedding[VOCAB_: Int, EMBED_DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.VOCAB_)
    comptime OUT_DIM = Self.EMBED_DIM_
    comptime W_SIZE = Self.VOCAB_ * Self.EMBED_DIM_

    var weight: Param["weight", True, Self.W_SIZE]
    var cache_in: Tensor  # [BATCH, VOCAB] one-hot

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.cache_in = Tensor()

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
            self.cache_in.ensure_gpu(c, B * Self.VOCAB_)
            comptime lbv = Layout.row_major(B, Self.VOCAB_)
            comptime lbe = Layout.row_major(B, Self.EMBED_DIM_)
            comptime lw = Layout.row_major(Self.VOCAB_, Self.EMBED_DIM_)
            comptime total = B * Self.EMBED_DIM_
            comptime nblk = (total + TPB - 1) // TPB
            c.enqueue_function[
                _embedding_fwd_kernel[B, Self.VOCAB_, Self.EMBED_DIM_]
            ](
                in0.lt["gpu", lbv](),
                self.weight.val.lt["gpu", lw](),
                out.lt["gpu", lbe](),
                self.cache_in.lt["gpu", lbv](),
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
            comptime lbv = Layout.row_major(B, Self.VOCAB_)
            comptime lbe = Layout.row_major(B, Self.EMBED_DIM_)
            comptime lw = Layout.row_major(Self.VOCAB_, Self.EMBED_DIM_)
            comptime gi_total = B * Self.VOCAB_
            comptime gi_blk = (gi_total + TPB - 1) // TPB
            c.enqueue_function[
                _embedding_grad_in_kernel[B, Self.VOCAB_, Self.EMBED_DIM_]
            ](
                grad_output.lt["gpu", lbe](),
                self.weight.val.lt["gpu", lw](),
                gin.lt["gpu", lbv](),
                grid_dim=gi_blk,
                block_dim=TPB,
            )
            comptime gw_total = Self.W_SIZE
            comptime gw_blk = (gw_total + TPB - 1) // TPB
            c.enqueue_function[
                _embedding_grad_w_kernel[B, Self.VOCAB_, Self.EMBED_DIM_]
            ](
                grad_output.lt["gpu", lbe](),
                self.cache_in.lt["gpu", lbv](),
                self.weight.grd.lt["gpu", lw](),
                grid_dim=gw_blk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
