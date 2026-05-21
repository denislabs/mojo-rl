"""BinaryElemMin[DIM] — retrofit (Phase B).

  output[b, d]   = min(in0[b, d], in1[b, d])
  grad_in0[b, d] = grad_output[b, d] if in0 wins, else 0
  grad_in1[b, d] = grad_output[b, d] if in1 wins, else 0   (ties → in0)

Cache: mask byte per output element (1.0 = in0 won, 0.0 = in1 won),
stored as `Scalar[DT]`. Leaf-owned, no aliasing concern.

No params. Conforms to `BinaryModule`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.binary_module import BinaryModule
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer, ensure_gpu_buffer,
)


def _bemin_forward_kernel[
    N: Int,
](
    in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var a = rebind[Scalar[DT]](in0[idx])
        var b = rebind[Scalar[DT]](in1[idx])
        if a < b:
            output[idx] = a
            mask[idx] = Scalar[DT](1.0)
        else:
            output[idx] = b
            mask[idx] = Scalar[DT](0.0)


def _bemin_backward_kernel[
    N: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var m = rebind[Scalar[DT]](mask[idx])
        var go = rebind[Scalar[DT]](grad_output[idx])
        var zero: Scalar[DT] = 0.0
        if m > Scalar[DT](0.5):
            grad_in0[idx] = go
            grad_in1[idx] = zero
        else:
            grad_in0[idx] = zero
            grad_in1[idx] = go


struct BinaryElemMin[DIM: Int](BinaryModule):
    comptime IN0_DIM = Self.DIM
    comptime IN1_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var mask: List[Scalar[DT]]                   # [BATCH, DIM] CPU mask cache
    var mask_dev: Optional[DeviceBuffer[DT]]     # [BATCH, DIM] GPU mask cache
    var _mask_dev_n: Int
    var ts: TargetStorage

    def __init__(out self):
        self.mask = List[Scalar[DT]]()
        self.mask_dev = None
        self._mask_dev_n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryElemMin.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryElemMin.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var m = Self()
        m.ts = TargetStorage.make_gpu(ctx)
        return m^

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
        assert_tag_for["BinaryElemMin", target](self.ts.target_tag)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.mask, BATCH * Self.DIM)
            var m_p = self.mask.unsafe_ptr()
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var a = in0[b, d]
                    var bv = in1[b, d]
                    if a < bv:
                        output[b, d] = a
                        m_p[b * Self.DIM + d] = Scalar[DT](1.0)
                    else:
                        output[b, d] = bv
                        m_p[b * Self.DIM + d] = Scalar[DT](0.0)
        else:
            var ctx = self.ts.ctx.value()
            comptime N = BATCH * Self.DIM
            ensure_gpu_buffer(self.mask_dev, self._mask_dev_n, N, ctx)
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in0.ptr)
            var i1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in1.ptr)
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var m_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.mask_dev.value().unsafe_ptr()
            )
            var i0_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](i0_p)
            var i1_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](i1_p)
            var o_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](o_p)
            var m_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](m_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _bemin_forward_kernel[N]
            ctx.enqueue_function[kernel](
                i0_lt, i1_lt, o_lt, m_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["BinaryElemMin", target](self.ts.target_tag)

        comptime if target == "cpu":
            var m_p = self.mask.unsafe_ptr()
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var mask_v = m_p[b * Self.DIM + d]
                    var go = grad_output[b, d]
                    if mask_v > Scalar[DT](0.5):
                        grad_in0[b, d] = go
                        grad_in1[b, d] = Scalar[DT](0.0)
                    else:
                        grad_in0[b, d] = Scalar[DT](0.0)
                        grad_in1[b, d] = go
        else:
            var ctx = self.ts.ctx.value()
            comptime N = BATCH * Self.DIM
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in0.ptr)
            var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in1.ptr)
            var m_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.mask_dev.value().unsafe_ptr()
            )
            var go_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](go_p)
            var gi0_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](gi0_p)
            var gi1_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](gi1_p)
            var m_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](m_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _bemin_backward_kernel[N]
            ctx.enqueue_function[kernel](
                go_lt, m_lt, gi0_lt, gi1_lt, grid_dim=n_blocks, block_dim=TPB,
            )
