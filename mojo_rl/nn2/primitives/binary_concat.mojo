"""BinaryConcat[IN0_DIM, IN1_DIM].

Horizontal stack of two `[BATCH, *]` tiles into `[BATCH, IN0_DIM+IN1_DIM]`.

  output[b, d]              = in0[b, d]                     d in [0, IN0_DIM)
  output[b, IN0_DIM + d]    = in1[b, d]                     d in [0, IN1_DIM)
  grad_in0[b, d]            = grad_output[b, d]             d in [0, IN0_DIM)
  grad_in1[b, d]            = grad_output[b, IN0_DIM + d]   d in [0, IN1_DIM)

No params. Conforms to `BinaryModule`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


def _bconcat_forward_kernel[
    BATCH: Int, IN0_DIM: Int, IN1_DIM: Int,
](
    in0: LayoutTensor[DT, Layout.row_major(BATCH, IN0_DIM), MutAnyOrigin],
    in1: LayoutTensor[DT, Layout.row_major(BATCH, IN1_DIM), MutAnyOrigin],
    output: LayoutTensor[
        DT, Layout.row_major(BATCH, IN0_DIM + IN1_DIM), MutAnyOrigin,
    ],
):
    var idx = Int(global_idx.x)
    comptime OUT = IN0_DIM + IN1_DIM
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var d = idx % OUT
        if d < IN0_DIM:
            output[b, d] = rebind[Scalar[DT]](in0[b, d])
        else:
            output[b, d] = rebind[Scalar[DT]](in1[b, d - IN0_DIM])


def _bconcat_backward_kernel[
    BATCH: Int, IN0_DIM: Int, IN1_DIM: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, IN0_DIM + IN1_DIM), MutAnyOrigin,
    ],
    grad_in0: LayoutTensor[DT, Layout.row_major(BATCH, IN0_DIM), MutAnyOrigin],
    grad_in1: LayoutTensor[DT, Layout.row_major(BATCH, IN1_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    comptime OUT = IN0_DIM + IN1_DIM
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var d = idx % OUT
        var v = rebind[Scalar[DT]](grad_output[b, d])
        if d < IN0_DIM:
            grad_in0[b, d] = v
        else:
            grad_in1[b, d - IN0_DIM] = v


struct BinaryConcat[IN0_DIM_: Int, IN1_DIM_: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIM = Self.IN0_DIM_
    comptime IN0_DIM = Self.IN0_DIM_
    comptime IN1_DIM = Self.IN1_DIM_
    comptime IN2_DIM: Int = 0
    comptime OUT_DIM = Self.IN0_DIM_ + Self.IN1_DIM_

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryConcat.make[target='gpu', INIT] requires a DeviceContext"
        )
        var c = Self()
        c.ts = TargetStorage.make_cpu()
        return c^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryConcat.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var c = Self()
        c.ts = TargetStorage.make_gpu(ctx)
        return c^

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
        assert_tag_for["BinaryConcat", target](self.ts.target_tag)
        var in0 = typed_view[BATCH, Self.IN0_DIM](inputs[0])
        var in1 = typed_view[BATCH, Self.IN1_DIM](inputs[1])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.IN0_DIM):
                    output_v[b, d] = in0[b, d]
                for d in range(Self.IN1_DIM):
                    output_v[b, Self.IN0_DIM + d] = in1[b, d]
        else:
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in0.ptr)
            var i1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in1.ptr)
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            var i0_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN0_DIM), MutAnyOrigin,
            ](i0_p)
            var i1_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN1_DIM), MutAnyOrigin,
            ](i1_p)
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](o_p)
            comptime TPB = 128
            comptime total = BATCH * Self.OUT_DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _bconcat_forward_kernel[
                BATCH, Self.IN0_DIM, Self.IN1_DIM,
            ]
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
        assert_tag_for["BinaryConcat", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_in0 = typed_view_mut[BATCH, Self.IN0_DIM](grad_inputs[0])
        var grad_in1 = typed_view_mut[BATCH, Self.IN1_DIM](grad_inputs[1])

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.IN0_DIM):
                    grad_in0[b, d] = grad_output_v[b, d]
                for d in range(Self.IN1_DIM):
                    grad_in1[b, d] = grad_output_v[b, Self.IN0_DIM + d]
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in0.ptr)
            var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in1.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](go_p)
            var gi0_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN0_DIM), MutAnyOrigin,
            ](gi0_p)
            var gi1_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN1_DIM), MutAnyOrigin,
            ](gi1_p)
            comptime TPB = 128
            comptime total = BATCH * Self.OUT_DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _bconcat_backward_kernel[
                BATCH, Self.IN0_DIM, Self.IN1_DIM,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi0_lt, gi1_lt, grid_dim=n_blocks, block_dim=TPB,
            )
