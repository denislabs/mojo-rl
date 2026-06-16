"""LearnedQueries[IGNORE_DIM, N, D] — a learned constant token sequence.

A `N·D` parameter, the same for every row of the batch — `N` learned query
tokens of width `D`. The (only) input is read for its BATCH count *and
nothing else*; its values are ignored and its gradient is zero:

    out[b, i] = queries[i]                         (∀ b)
    grad_input[b, ·] = 0
    grad_queries[i]  = sum_b grad_out[b, i]         (tokens shared → reduce)

This is the DETR/LeWM decoder's "fixed set of learnable query tokens, one
per target patch". It differs from `LearnedTokens` (which *concatenates*
learned tokens onto a real input stream): here the queries ARE the whole
stream, and the carrier input exists only so the op can live as a graph
node hanging off an upstream tensor (e.g. the encoder `emb`) to inherit
the batch dimension. `IGNORE_DIM` is that carrier's width.

Param is weight-decay-exempt (like `LearnedTokens`) and INIT-filled (NOT
zero — zero queries would make every patch identical and unrecoverable by
symmetry). CPU + GPU.
"""

from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import (
    Initializer, AMPPolicy, NoAMP, Param, ParamVisitor,
    for_each_param_auto, zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


comptime LQ_RTPB = 64  # reduction block size for the param-grad batch sum


def _lq_forward_kernel[
    BATCH: Int, OUT_DIM: Int
](
    param: LayoutTensor[DT, Layout.row_major(OUT_DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_DIM:
        return
    var i = idx % OUT_DIM
    output.ptr[idx] = rebind[Scalar[DT]](param.ptr[i])


def _lq_grad_input_zero_kernel[
    N: Int
](
    grad_input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        grad_input.ptr[idx] = Scalar[DT](0.0)


def _lq_grad_param_kernel[
    BATCH: Int, OUT_DIM: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    grad_param: LayoutTensor[DT, Layout.row_major(OUT_DIM), MutAnyOrigin],
):
    # One block per param element; threads reduce over the batch.
    var col = Int(block_idx.x)
    if col >= OUT_DIM:
        return
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        acc += rebind[Scalar[DT]](grad_output.ptr[bi * OUT_DIM + col])
        bi += LQ_RTPB
    var total = block.sum[block_size=LQ_RTPB, broadcast=False](val=acc)
    if t == 0:
        grad_param.ptr[col] = rebind[Scalar[DT]](grad_param.ptr[col]) + total[0]


struct LearnedQueries[IGNORE_DIM: Int, N: Int, D: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IGNORE_DIM)
    comptime OUT_DIM = Self.N * Self.D

    var queries: Param["queries", False, Self.N * Self.D]
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N > 0 and Self.D > 0, (
            "LearnedQueries: N, D must be > 0"
        )
        self.queries = Param["queries", False, Self.N * Self.D]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "LearnedQueries: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime SZ = Self.N * Self.D
        comptime if target == "cpu":
            m.queries = Param["queries", False, SZ].make_cpu()
            INIT.init_weight(
                m.queries.value_unsafe_ptr_cpu(), SZ, Self.D, Self.D
            )
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["LearnedQueries.make[gpu]"](ctx)
            m.queries = Param["queries", False, SZ].make_gpu(ctx_v)
            var host = ctx_v.enqueue_create_host_buffer[DT](SZ)
            ctx_v.synchronize()
            INIT.init_weight(host.unsafe_ptr(), SZ, Self.D, Self.D)
            ctx_v.enqueue_copy(m.queries.val.dev.value(), host)
            ctx_v.synchronize()
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("LearnedQueries")

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
        assert_tag_for["LearnedQueries", target](self.ts.target_tag)
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime if target == "cpu":
            var q = TileTensor(self.queries.val.cpu, row_major[Self.OUT_DIM]())
            for b in range(BATCH):
                for i in range(Self.OUT_DIM):
                    out[b, i] = q[i]
        else:
            var p_lt = LayoutTensor[
                DT, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ](self.queries.val.dev.value())
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ](out.ptr)
            comptime n_blocks = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _lq_forward_kernel[BATCH, Self.OUT_DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                p_lt, o_lt, grid_dim=n_blocks, block_dim=TPB
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
        assert_tag_for["LearnedQueries", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = grad_inputs.tile[0, BATCH, Self.IGNORE_DIM]()
        comptime gi_total = BATCH * Self.IGNORE_DIM

        comptime if target == "cpu":
            # grad_input = 0 (queries don't depend on the carrier input).
            for k in range(gi_total):
                gi.ptr[k] = Scalar[DT](0.0)
            comptime if mode == "all":
                var gq = TileTensor(
                    self.queries.grd.cpu, row_major[Self.OUT_DIM]()
                )
                for b in range(BATCH):
                    for i in range(Self.OUT_DIM):
                        gq[i] += go[b, i]
        else:
            var ctx = self.ts.ctx.value()
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(gi_total), MutAnyOrigin
            ](gi.ptr)
            comptime zblocks = (gi_total + TPB - 1) // TPB
            ctx.enqueue_function[_lq_grad_input_zero_kernel[gi_total]](
                gi_lt, grid_dim=zblocks, block_dim=TPB
            )
            comptime if mode == "all":
                var go_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
                ](go.ptr)
                var gp_lt = LayoutTensor[
                    DT, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
                ](self.queries.grd.dev.value())
                comptime gpk = _lq_grad_param_kernel[BATCH, Self.OUT_DIM]
                ctx.enqueue_function[gpk](
                    go_lt, gp_lt, grid_dim=Self.OUT_DIM, block_dim=LQ_RTPB
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["LearnedQueries", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["LearnedQueries", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
