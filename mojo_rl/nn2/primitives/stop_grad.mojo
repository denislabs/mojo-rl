"""StopGrad[DIM] — identity forward, zero-fill backward.

Severs gradient flow. No cache —
the forward just copies input → output, and backward writes zeros
unconditionally without needing to know what the input was. Even
simpler than ReLU.

Conforms to `Module`. `mode` is accepted but has no behavioral
effect — StopGrad zeros grad_input either way.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


def _stop_grad_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        output[b, d] = rebind[Scalar[DT]](input[b, d])


def _stop_grad_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        grad_input[b, d] = Scalar[DT](0.0)


struct StopGrad[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "StopGrad.make[target='gpu', INIT] requires a DeviceContext"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "StopGrad.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var s = Self()
        s.ts = TargetStorage.make_gpu(ctx)
        return s^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank  == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        assert_tag_for["StopGrad", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    output[b, d] = input[b, d]
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var in_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var input_lt  = LayoutTensor[DT, layout, MutAnyOrigin](in_ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _stop_grad_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                input_lt, output_lt, grid_dim=n_blocks, block_dim=TPB,
            )

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["StopGrad", target](self.ts.target_tag)
        # mode has no effect: grad_input always zero.

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    grad_input[b, d] = Scalar[DT](0.0)
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var gi_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _stop_grad_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
