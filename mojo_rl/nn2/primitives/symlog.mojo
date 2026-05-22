"""Symlog[DIM] — `y = sign(x) * log(1 + |x|)`.

Symmetric logarithmic transform that compresses large magnitudes while
preserving sign. Used by DreamerV3 (reward/return rescaling) and TD-MPC2
(distributional value head encoding).

  forward:  y = sign(x) * log(1 + |x|)
  vjp:      dx = dy / (1 + |x|)

The backward needs the original input `x`. Sequential keeps the input
slab live across forward → backward, so we use the same input-pointer-
alias pattern as `ReLU` — no owned cache buffer.
"""

from std.math import log
from std.math import abs as math_abs
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _symlog_forward_kernel[
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
        var x = rebind[Scalar[DT]](input[b, d])
        var zero: Scalar[DT] = 0.0
        var one: Scalar[DT] = 1.0
        var abs_x = x if x >= zero else -x
        var sign: Scalar[DT] = one if x >= zero else -one
        output[b, d] = sign * log(one + abs_x)


def _symlog_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        var x = rebind[Scalar[DT]](cache[b, d])
        var zero: Scalar[DT] = 0.0
        var one: Scalar[DT] = 1.0
        var abs_x = x if x >= zero else -x
        grad_input[b, d] = grad_output[b, d] / (one + abs_x)


# ──────────────────────────────────────────────────────────────────────
# Symlog — parameterless leaf with input-pointer-alias cache.
# ──────────────────────────────────────────────────────────────────────


struct Symlog[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage
    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "Symlog.make[target='gpu', INIT] requires a DeviceContext"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "Symlog.make[target='cpu', INIT](ctx) — drop ctx for CPU"
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
        comptime assert input.flat_rank  == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        assert_tag_for["Symlog", target](self.ts.target_tag)

        self._cached_input_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](input.ptr)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var one_v = SIMD[DT, CPU_SIMD_W](1)
            var pos_v = SIMD[DT, CPU_SIMD_W](1)
            var neg_v = SIMD[DT, CPU_SIMD_W](-1)
            var zero_v = SIMD[DT, CPU_SIMD_W](0)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var x = in_p.load[width=CPU_SIMD_W](k)
                var abs_x = math_abs(x)
                var sign = x.ge(zero_v).select(pos_v, neg_v)
                out_p.store(k, sign * log(one_v + abs_x))
                k += CPU_SIMD_W
            while k < N:
                var x = in_p[k]
                var ax = x if x >= 0 else -x
                var sgn: Scalar[DT] = 1 if x >= 0 else -1
                out_p[k] = sgn * log(Scalar[DT](1) + ax)
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var in_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var input_lt = LayoutTensor[DT, layout, MutAnyOrigin](in_ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _symlog_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                input_lt, output_lt,
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
        assert_tag_for["Symlog", target](self.ts.target_tag)

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var c_p = self._cached_input_ptr
            var one_v = SIMD[DT, CPU_SIMD_W](1)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var x = c_p.load[width=CPU_SIMD_W](k)
                var dy = go_p.load[width=CPU_SIMD_W](k)
                gi_p.store(k, dy / (one_v + math_abs(x)))
                k += CPU_SIMD_W
            while k < N:
                var x = c_p[k]
                var ax = x if x >= 0 else -x
                gi_p[k] = go_p[k] / (Scalar[DT](1) + ax)
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var go_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_ptr)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self._cached_input_ptr
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _symlog_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cache_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
