"""MSEPerSample[DIM] — per-row mean squared error, a graph-node loss.

    out[b, 0] = (1/DIM) * sum_i (a[b,i] - b[b,i])^2

ARITY=2, OUT_DIM=1, PARAM-free. Produces a (B,1) per-sample loss so it can
be the single output of a ComputeGraph (the SAC actor-loss pattern): the
Step block then reduces to the batch mean and seeds grad=1/B.

Backward (c = grad_out·2/DIM):
    grad_a[b,i] =  c · (a-b);   grad_b[b,i] = -c · (a-b)
Cache: diff = a-b (DIM per row).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage, assert_tag_for, ensure_cpu_buffer,
)


def _mps_forward_kernel[BATCH: Int, DIM: Int](
    a: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    diff: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var r = Int(global_idx.x)
    if r >= BATCH:
        return
    var s = Scalar[DT](0)
    for i in range(DIM):
        var d = rebind[Scalar[DT]](a[r, i]) - rebind[Scalar[DT]](b[r, i])
        diff[r, i] = d
        s += d * d
    o[r, 0] = s / Scalar[DT](DIM)


def _mps_backward_kernel[BATCH: Int, DIM: Int](
    go: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    diff: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ga: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gb: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var r = idx // DIM
    var i = idx % DIM
    var c = rebind[Scalar[DT]](go[r, 0]) * Scalar[DT](2.0 / Float64(DIM))
    var v = c * rebind[Scalar[DT]](diff[r, i])
    ga[r, i] = v
    gb[r, i] = -v


struct MSEPerSample[DIM: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.DIM)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("MSEPerSample")

    var cache_diff: Cache["cache_diff"]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_diff = Cache["cache_diff"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["MSEPerSample.make[gpu]"](ctx)
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    def _ensure_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["MSEPerSample", target](self.ts.target_tag)
        var a = inputs.tile[0, BATCH, Self.DIM]()
        var b = inputs.tile[1, BATCH, Self.DIM]()
        var out = typed_view_mut[BATCH, 1](output)

        comptime if target == "cpu":
            self.cache_diff.ensure_cpu(BATCH * Self.DIM)
            var diff = TileTensor(self.cache_diff.cpu, row_major[BATCH, Self.DIM]())
            for r in range(BATCH):
                var s = Scalar[DT](0)
                for i in range(Self.DIM):
                    var d = a[r, i] - b[r, i]
                    diff[r, i] = d
                    s += d * d
                out[r, 0] = s / Scalar[DT](Self.DIM)
        else:
            self._ensure_gpu(BATCH)
            comptime lay2 = Layout.row_major(BATCH, Self.DIM)
            comptime lay1 = Layout.row_major(BATCH, 1)
            var a_lt = LayoutTensor[DT, lay2, MutAnyOrigin](
                a.ptr
            )
            var b_lt = LayoutTensor[DT, lay2, MutAnyOrigin](
                b.ptr
            )
            var o_lt = LayoutTensor[DT, lay1, MutAnyOrigin](
                out.ptr
            )
            var d_lt = LayoutTensor[DT, lay2, MutAnyOrigin](
                self.cache_diff.dev.value()
            )
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            self.ts.ctx.value().enqueue_function[
                _mps_forward_kernel[BATCH, Self.DIM]
            ](a_lt, b_lt, o_lt, d_lt, grid_dim=n_blocks, block_dim=TPB)

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        assert_tag_for["MSEPerSample", target](self.ts.target_tag)
        var go = typed_view[BATCH, 1](grad_output)
        var ga = grad_inputs.tile[0, BATCH, Self.DIM]()
        var gb = grad_inputs.tile[1, BATCH, Self.DIM]()

        comptime if target == "cpu":
            var diff = TileTensor(self.cache_diff.cpu, row_major[BATCH, Self.DIM]())
            for r in range(BATCH):
                var c = go[r, 0] * Scalar[DT](2.0 / Float64(Self.DIM))
                for i in range(Self.DIM):
                    var v = c * diff[r, i]
                    ga[r, i] = v
                    gb[r, i] = -v
        else:
            comptime lay2 = Layout.row_major(BATCH, Self.DIM)
            comptime lay1 = Layout.row_major(BATCH, 1)
            var go_lt = LayoutTensor[DT, lay1, MutAnyOrigin](
                go.ptr
            )
            var ga_lt = LayoutTensor[DT, lay2, MutAnyOrigin](
                ga.ptr
            )
            var gb_lt = LayoutTensor[DT, lay2, MutAnyOrigin](
                gb.ptr
            )
            var d_lt = LayoutTensor[DT, lay2, MutAnyOrigin](
                self.cache_diff.dev.value()
            )
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            self.ts.ctx.value().enqueue_function[
                _mps_backward_kernel[BATCH, Self.DIM]
            ](go_lt, d_lt, ga_lt, gb_lt, grid_dim=n_blocks, block_dim=TPB)
