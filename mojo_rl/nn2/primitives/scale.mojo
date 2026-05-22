"""Scale[DIM] — multiplies by a runtime scalar `multiplier`.

Forward: `out = m·in`,
Backward: `grad_in = m·grad_out`. The multiplier is a public mut field
the caller updates per-step (SAC tracks moving α this way).

No cache: multiplier lives on the struct; no need to remember anything
from forward. Conforms to `Module`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


def _scale_kernel[
    N: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    multiplier: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](input[idx]) * multiplier


struct Scale[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var multiplier: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self.multiplier = Scalar[DT](1.0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Scale.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Scale.make[target='cpu', INIT](ctx) — drop ctx for CPU"
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
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        assert_tag_for["Scale", target](self.ts.target_tag)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                out_p.store(k, in_p.load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                out_p[k] = in_p[k] * self.multiplier
                k += 1
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            comptime N = BATCH * Self.DIM
            var in_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](out_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _scale_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, self.multiplier,
                grid_dim=n_blocks, block_dim=TPB,
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
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Scale", target](self.ts.target_tag)

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                gi_p.store(k, go_p.load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                gi_p[k] = go_p[k] * self.multiplier
                k += 1
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            comptime N = BATCH * Self.DIM
            var go_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](gi_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _scale_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, self.multiplier,
                grid_dim=n_blocks, block_dim=TPB,
            )
