"""Reduction Modules — retrofit (Phase B).

Sum[DIM] and Mean[DIM] across the feature axis. Same algorithm as v1,
just the scaffold collapses: `ts: TargetStorage` replaces the per-leaf
tag/inference/ctx triplet, `backward[mode]` collapses `backward` +
`backward_input`, and Phase 10A buffer surface is dropped.

No params on either struct → no Param wrappers; `for_each_param` is
a no-op trait conformance.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per batch row.
# ──────────────────────────────────────────────────────────────────────


def _sum_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var acc: Scalar[DT] = 0.0
        for d in range(DIM):
            acc += rebind[Scalar[DT]](input[b, d])
        output[b, 0] = acc


def _mean_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var acc: Scalar[DT] = 0.0
        for d in range(DIM):
            acc += rebind[Scalar[DT]](input[b, d])
        output[b, 0] = acc * (Scalar[DT](1.0) / Scalar[DT](DIM))


def _broadcast_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    multiplier: Scalar[DT],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        grad_input[b, d] = rebind[Scalar[DT]](grad_output[b, 0]) * multiplier


# ──────────────────────────────────────────────────────────────────────
# Sum
# ──────────────────────────────────────────────────────────────────────


struct Sum[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Sum.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Sum.make[target='cpu', INIT](ctx) — drop ctx for CPU"
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
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, 1]"
        assert_tag_for["Sum", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](out_p)
            comptime TPB = 128
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _sum_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["Sum", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var go = grad_output[b, 0]
                for d in range(Self.DIM):
                    grad_input[b, d] = go
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin,
            ](gi_p)
            comptime TPB = 128
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _broadcast_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, Scalar[DT](1.0),
                grid_dim=n_blocks, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# Mean
# ──────────────────────────────────────────────────────────────────────


struct Mean[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1
    comptime _INV_DIM: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.DIM)

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Mean.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Mean.make[target='cpu', INIT](ctx) — drop ctx for CPU"
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
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, 1]"
        assert_tag_for["Mean", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc * Self._INV_DIM
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](out_p)
            comptime TPB = 128
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _mean_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["Mean", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var go_inv = grad_output[b, 0] * Self._INV_DIM
                for d in range(Self.DIM):
                    grad_input[b, d] = go_inv
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin,
            ](gi_p)
            comptime TPB = 128
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _broadcast_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, Self._INV_DIM,
                grid_dim=n_blocks, block_dim=TPB,
            )
