"""Reduction Modules — `Reduce[DIM, OP: ReduceOp]` + `Sum` / `Mean` aliases.

Phase 2.5.C migration: pre-Phase-2.5 this file held two near-identical
135-LOC structs (`Sum[DIM]` and `Mean[DIM]`) differing only in a `1/DIM`
scale factor. Post-migration both are one-line aliases over
`Reduce[DIM, OP]`, with the per-op math factored into `SumOp` /
`MeanOp` (`ReduceOp` trait).

Forward:  `out[b, 0] = OP.scale_factor[DIM]() · Σ_d input[b, d]`
Backward: `grad_in[b, d] = OP.scale_factor[DIM]() · grad_out[b, 0]`
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.reduce_op import ReduceOp
from ..core.target_storage import TargetStorage, assert_tag_for
from .ops.sum_op import SumOp
from .ops.mean_op import MeanOp


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — OP supplies the scale factor via `OP.scale_factor[DIM]()`.
# Inside top-level kernel functions, must use bare `OP` (not `Self.OP`):
# see `feedback_mojo_kernel_op_param_scope`.
# ──────────────────────────────────────────────────────────────────────


def _reduce_forward_kernel[
    BATCH: Int, DIM: Int, OP: ReduceOp,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var acc: Scalar[DT] = 0.0
        for d in range(DIM):
            acc += rebind[Scalar[DT]](input[b, d])
        output[b, 0] = acc * OP.scale_factor[DIM]()


def _reduce_broadcast_kernel[
    BATCH: Int, DIM: Int, OP: ReduceOp,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        grad_input[b, d] = (
            rebind[Scalar[DT]](grad_output[b, 0]) * OP.scale_factor[DIM]()
        )


# ──────────────────────────────────────────────────────────────────────
# Reduce[DIM, OP] — shared body for every linear reduction. Two leaf
# aliases (`Sum` / `Mean`) cover all current consumers; new ones (e.g.
# weighted means with a comptime per-element weight) plug in by adding
# a new `ReduceOp` impl.
# ──────────────────────────────────────────────────────────────────────


struct Reduce[DIM: Int, OP: ReduceOp](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. INIT is ignored (no parameters) but accepted for
        Sequential.make[target, INIT] uniformity."""
        comptime assert target == "cpu", (
            "Reduce.make[target='gpu', INIT] requires a DeviceContext"
        )
        var r = Self()
        r.ts = TargetStorage.make_cpu()
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Reduce.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var r = Self()
        r.ts = TargetStorage.make_gpu(ctx)
        return r^

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
        assert_tag_for["Reduce", target](self.ts.target_tag)

        comptime if target == "cpu":
            var scale = Self.OP.scale_factor[Self.DIM]()
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc * scale
        else:
            comptime layout_in = Layout.row_major(BATCH, Self.DIM)
            comptime layout_out = Layout.row_major(BATCH, 1)
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var in_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, layout_out, MutAnyOrigin](out_p)
            comptime TPB = 128
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _reduce_forward_kernel[BATCH, Self.DIM, Self.OP]
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
        assert_tag_for["Reduce", target](self.ts.target_tag)

        comptime if target == "cpu":
            var scale = Self.OP.scale_factor[Self.DIM]()
            for b in range(BATCH):
                var go_scaled = grad_output[b, 0] * scale
                for d in range(Self.DIM):
                    grad_input[b, d] = go_scaled
        else:
            comptime layout_go = Layout.row_major(BATCH, 1)
            comptime layout_gi = Layout.row_major(BATCH, Self.DIM)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[DT, layout_go, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_gi, MutAnyOrigin](gi_p)
            comptime TPB = 128
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _reduce_broadcast_kernel[BATCH, Self.DIM, Self.OP]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# Leaf aliases — pre-Phase-2.5 these were 135-LOC structs each.
# ──────────────────────────────────────────────────────────────────────


comptime Sum[DIM: Int] = Reduce[DIM, SumOp]
comptime Mean[DIM: Int] = Reduce[DIM, MeanOp]
