"""Embedding[VOCAB, EMBED_DIM] — one-hot lookup table.

Input is a one-hot row of width `VOCAB`; output is the corresponding
`EMBED_DIM` embedding row. Implemented as a dense one-hot @ table matmul
so it composes with the rest of nn unchanged (the caller feeds one-hot
rows; `Tokenwise[seq_len, Embedding]` looks up every token position):

    out[b, j] = sum_v in[b, v] * W[v, j]      W laid out (VOCAB, EMBED_DIM)

Params: `weight` (VOCAB*EMBED_DIM, decay-enabled). Cache: the one-hot
input `[BATCH, VOCAB]`, leaf-owned (its own buffer, NOT the Sequential
input slab) so backward order is unconstrained.

Backward:
  * grad_in[b, v] = sum_j grad_out[b, j] * W[v, j]
  * grad_W[v, j] += sum_b cache_in[b, v] * grad_out[b, j]

Init: `INIT.init_weight` over the table (same path as Linear's weight);
the composite caller picks the initializer (GPT typically Normal(0, 0.02)).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Cache,
    Param,
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
)


def _embedding_fwd_kernel[
    BATCH: Int, VOCAB: Int, EMBED_DIM: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
    weight: LayoutTensor[
        DT, Layout.row_major(VOCAB, EMBED_DIM), MutAnyOrigin
    ],
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
        # Cache the one-hot input (column j==0 thread writes each row once).
        if j == 0:
            cache_in[b, v] = x
    output[b, j] = acc


def _embedding_grad_in_kernel[
    BATCH: Int, VOCAB: Int, EMBED_DIM: Int
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, EMBED_DIM), MutAnyOrigin
    ],
    weight: LayoutTensor[
        DT, Layout.row_major(VOCAB, EMBED_DIM), MutAnyOrigin
    ],
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


struct Embedding[VOCAB: Int, EMBED_DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.VOCAB)
    comptime OUT_DIM = Self.EMBED_DIM

    var weight: Param["weight", True, Self.VOCAB * Self.EMBED_DIM]

    # Cache (leaf-owned): the one-hot input.
    var cache_in: Cache["cache_in"]

    var ts: TargetStorage

    def __init__(out self):
        self.weight = Param["weight", True, Self.VOCAB * Self.EMBED_DIM]()
        self.cache_in = Cache["cache_in"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Embedding: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.VOCAB > 0 and Self.EMBED_DIM > 0, (
            "Embedding: VOCAB, EMBED_DIM must be > 0"
        )
        comptime W_SIZE = Self.VOCAB * Self.EMBED_DIM
        var e = Self()
        comptime if target == "cpu":
            e.weight = Param["weight", True, W_SIZE].make_cpu()
            INIT.init_weight(
                e.weight.value_unsafe_ptr_cpu(),
                W_SIZE, Self.VOCAB, Self.EMBED_DIM,
            )
            e.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Embedding.make[target='gpu']"](ctx)
            e.weight = Param["weight", True, W_SIZE].make_gpu(ctx_v)
            var w_host = ctx_v.enqueue_create_host_buffer[DT](W_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(
                w_host.unsafe_ptr(), W_SIZE, Self.VOCAB, Self.EMBED_DIM
            )
            ctx_v.enqueue_copy(e.weight.val.dev.value(), w_host)
            ctx_v.synchronize()
            e.ts = TargetStorage.make_gpu(ctx_v)
        return e^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_in.ensure_gpu(ctx, batch * Self.VOCAB)
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
        assert_tag_for["Embedding", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_in.ensure_cpu(BATCH * Self.VOCAB)
            var w_v = TileTensor(
                self.weight.val.cpu, row_major[Self.VOCAB, Self.EMBED_DIM]()
            )
            var cin = TileTensor(
                self.cache_in.cpu, row_major[BATCH, Self.VOCAB]()
            )
            for b in range(BATCH):
                for v in range(Self.VOCAB):
                    cin[b, v] = input[b, v]
                for j in range(Self.EMBED_DIM):
                    var acc: Scalar[DT] = 0.0
                    for v in range(Self.VOCAB):
                        acc += input[b, v] * w_v[v, j]
                    output_v[b, j] = acc
        else:
            self._ensure_cache_gpu(BATCH)
            comptime lay_bv = Layout.row_major(BATCH, Self.VOCAB)
            comptime lay_be = Layout.row_major(BATCH, Self.EMBED_DIM)
            comptime lay_w = Layout.row_major(Self.VOCAB, Self.EMBED_DIM)
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[DT, lay_bv, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, lay_be, MutAnyOrigin](out_p)
            var w_lt = LayoutTensor[DT, lay_w, MutAnyOrigin](
                self.weight.val.dev.value()
            )
            var cin_lt = LayoutTensor[DT, lay_bv, MutAnyOrigin](
                self.cache_in.dev.value()
            )
            comptime total = BATCH * Self.EMBED_DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _embedding_fwd_kernel[
                BATCH, Self.VOCAB, Self.EMBED_DIM
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, w_lt, out_lt, cin_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

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
        assert_tag_for["Embedding", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var w_v = TileTensor(
                self.weight.val.cpu, row_major[Self.VOCAB, Self.EMBED_DIM]()
            )
            for b in range(BATCH):
                for v in range(Self.VOCAB):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.EMBED_DIM):
                        acc += grad_output_v[b, j] * w_v[v, j]
                    grad_input_v[b, v] = acc
            comptime if mode == "all":
                var gw_v = TileTensor(
                    self.weight.grd.cpu, row_major[Self.VOCAB, Self.EMBED_DIM]()
                )
                var cin = TileTensor(
                    self.cache_in.cpu, row_major[BATCH, Self.VOCAB]()
                )
                for v in range(Self.VOCAB):
                    for j in range(Self.EMBED_DIM):
                        var acc: Scalar[DT] = 0.0
                        for b in range(BATCH):
                            acc += cin[b, v] * grad_output_v[b, j]
                        gw_v[v, j] += acc
        else:
            var ctx = self.ts.ctx.value()
            comptime lay_bv = Layout.row_major(BATCH, Self.VOCAB)
            comptime lay_be = Layout.row_major(BATCH, Self.EMBED_DIM)
            comptime lay_w = Layout.row_major(Self.VOCAB, Self.EMBED_DIM)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, lay_be, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, lay_bv, MutAnyOrigin](gi_p)
            var w_lt = LayoutTensor[DT, lay_w, MutAnyOrigin](
                self.weight.val.dev.value()
            )
            comptime gi_total = BATCH * Self.VOCAB
            comptime gi_blocks = (gi_total + TPB - 1) // TPB
            comptime gi_kernel = _embedding_grad_in_kernel[
                BATCH, Self.VOCAB, Self.EMBED_DIM
            ]
            ctx.enqueue_function[gi_kernel](
                go_lt, w_lt, gi_lt, grid_dim=gi_blocks, block_dim=TPB,
            )
            comptime if mode == "all":
                var cin_lt = LayoutTensor[DT, lay_bv, MutAnyOrigin](
                    self.cache_in.dev.value()
                )
                var gw_lt = LayoutTensor[DT, lay_w, MutAnyOrigin](
                    self.weight.grd.dev.value()
                )
                comptime gw_total = Self.VOCAB * Self.EMBED_DIM
                comptime gw_blocks = (gw_total + TPB - 1) // TPB
                comptime gw_kernel = _embedding_grad_w_kernel[
                    BATCH, Self.VOCAB, Self.EMBED_DIM
                ]
                ctx.enqueue_function[gw_kernel](
                    go_lt, cin_lt, gw_lt,
                    grid_dim=gw_blocks, block_dim=TPB,
                )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Embedding", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Embedding", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
