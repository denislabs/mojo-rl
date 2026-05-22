"""BinaryAdd[DIM] — pure element-wise addition.

  output[b, d]    = in0[b, d] + in1[b, d]
  grad_in0[b, d]  = grad_output[b, d]
  grad_in1[b, d]  = grad_output[b, d]

No params. Conforms to `BinaryModule`. The orchestrator (Sequential /
ComputeGraph) owns every slab; `backward[mode]` collapses backward +
backward_input.

Used by `TargetYBlock` (Phase 3.2) to form `y = r + γ·soft_v`, and
expected to be useful for any future loss block that adds two
intermediate tensors (e.g. DreamerV3's value-prediction + entropy
regularizer).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.binary_module import BinaryModule
from ..core.target_storage import TargetStorage, assert_tag_for


def _badd_forward_kernel[
    N: Int,
](
    in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](in0[idx]) + rebind[Scalar[DT]](in1[idx])


def _badd_backward_kernel[
    N: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var go = rebind[Scalar[DT]](grad_output[idx])
        grad_in0[idx] = go
        grad_in1[idx] = go


struct BinaryAdd[DIM: Int](BinaryModule):
    comptime IN0_DIM = Self.DIM
    comptime IN1_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryAdd.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryAdd.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.ts = TargetStorage.make_gpu(ctx)
        return s^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        in1: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert in0.flat_rank == 2, "in0 rank-2 [BATCH, DIM]"
        comptime assert in1.flat_rank == 2, "in1 rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        assert_tag_for["BinaryAdd", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    output[b, d] = in0[b, d] + in1[b, d]
        else:
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in0.ptr)
            var i1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in1.ptr)
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            comptime N = BATCH * Self.DIM
            var i0_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](i0_p)
            var i1_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](i1_p)
            var o_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](o_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _badd_forward_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                i0_lt, i1_lt, o_lt, grid_dim=n_blocks, block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_in0.flat_rank == 2, "grad_in0 rank-2"
        comptime assert grad_in1.flat_rank == 2, "grad_in1 rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["BinaryAdd", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var go = grad_output[b, d]
                    grad_in0[b, d] = go
                    grad_in1[b, d] = go
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in0.ptr)
            var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in1.ptr)
            comptime N = BATCH * Self.DIM
            var go_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](go_p)
            var gi0_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](gi0_p)
            var gi1_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](gi1_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _badd_backward_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi0_lt, gi1_lt, grid_dim=n_blocks, block_dim=TPB,
            )
