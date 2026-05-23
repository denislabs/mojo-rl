"""Slice[IN, START, END] — extracts column range `[START, END)` from input.

Zero-fills the rest of grad_input on backward so that ComputeGraph's
scatter-add into a shared predecessor `_grad_out_buf` interleaves
correctly with parallel slicers (e.g. the q1/q2/log_prob unpack in
`SACActorLossCG`).

No params. Conforms to `Module`. Orchestrator owns slabs;
`backward[mode]` accepted, has no effect (no params to skip).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


def _slice_forward_kernel[
    BATCH: Int, IN: Int, START: Int, OUT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var j = idx % OUT
        output[b, j] = rebind[Scalar[DT]](input[b, START + j])


def _slice_backward_kernel[
    BATCH: Int, IN: Int, START: Int, OUT: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    # Zero the whole grad_input and scatter the slice in. One thread
    # per [b, k] over the FULL input shape — k in [START, START+OUT)
    # gets grad_output, the rest gets 0.
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var k = idx % IN
        var zero: Scalar[DT] = 0.0
        if k >= START and k < START + OUT:
            grad_input[b, k] = rebind[Scalar[DT]](grad_output[b, k - START])
        else:
            grad_input[b, k] = zero


struct Slice[IN: Int, START: Int, END: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIM = Self.IN
    comptime IN1_DIM: Int = 0
    comptime IN2_DIM: Int = 0
    comptime OUT_DIM = Self.END - Self.START

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Slice.make[target='gpu', INIT] requires a DeviceContext"
        )
        comptime assert Self.START >= 0, "Slice.START must be >= 0"
        comptime assert Self.END > Self.START, "Slice.END must be > START"
        comptime assert Self.END <= Self.IN, "Slice.END must be <= IN_DIM"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Slice.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        comptime assert Self.START >= 0, "Slice.START must be >= 0"
        comptime assert Self.END > Self.START, "Slice.END must be > START"
        comptime assert Self.END <= Self.IN, "Slice.END must be <= IN_DIM"
        var s = Self()
        s.ts = TargetStorage.make_gpu(ctx)
        return s^

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
        assert_tag_for["Slice", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIM](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    output_v[b, j] = input[b, Self.START + j]
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](out_p)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _slice_forward_kernel[
                BATCH, Self.IN, Self.START, Self.OUT_DIM,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Slice", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIM](grad_inputs[0])

        comptime if target == "cpu":
            # Zero whole grad_input first; scatter the slice in afterward.
            # Zeros required for ComputeGraph scatter-add: when multiple
            # slicers share a predecessor, each writes its slice range and
            # leaves the rest at 0 so the scatter-add sums correctly.
            for b in range(BATCH):
                for k in range(Self.IN_DIM):
                    grad_input_v[b, k] = Scalar[DT](0.0)
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    grad_input_v[b, Self.START + j] = grad_output_v[b, j]
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN), MutAnyOrigin,
            ](gi_p)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.IN + TPB - 1) // TPB
            comptime kernel = _slice_backward_kernel[
                BATCH, Self.IN, Self.START, Self.OUT_DIM,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
