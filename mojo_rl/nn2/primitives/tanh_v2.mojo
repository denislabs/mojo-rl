"""Tanh[DIM] — retrofit (Phase A, output-caching).

Diffs vs `tanh.mojo`:
  - `TargetStorage` field replaces (_target_tag, _inference, ctx).
  - Free `assert_tag_for["TanhV2", target]` replaces `_assert_tag`.
  - Free `ensure_cpu_buffer` / `ensure_gpu_buffer` replace per-leaf helpers.
  - `backward[mode]` replaces separate `backward_input`.

UNLIKE ReLU/StopGrad, Tanh OWNS ITS CACHE BUFFER. Cache stores
`y = tanh(x)` because `dy/dx = 1 - y²` is cheaper from `y` than
recomputing `tanh(x)`. The slab-aliasing trick from audit Spike #1
doesn't apply: in `Sequential[..., Tanh, OtherLayer, ...]`, slab[i] (=
Tanh's output) gets clobbered by `OtherLayer.backward` before Tanh's
backward reads it. So Tanh holds its own buffer. This is option 2 from
the audit decision.

Conforms to `ModuleV2` (slim trait — no Phase 10A buffer surface).
"""

from std.math import tanh
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module_v2 import ModuleV2
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
    ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (identical to v1).
# ──────────────────────────────────────────────────────────────────────


def _tanh_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        var x = rebind[Scalar[DT]](input[b, d])
        var y = tanh(x)
        output[b, d] = y
        cache[b, d] = y


def _tanh_backward_kernel[
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
        var y = rebind[Scalar[DT]](cache[b, d])
        var go = rebind[Scalar[DT]](grad_output[b, d])
        grad_input[b, d] = go * (Scalar[DT](1.0) - y * y)


# ──────────────────────────────────────────────────────────────────────
# TanhV2 — owns its cache buffer.
# ──────────────────────────────────────────────────────────────────────


struct TanhV2[DIM: Int](ModuleV2):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage
    var cache: List[Scalar[DT]]
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.cache = List[Scalar[DT]]()
        self.cache_dev = None
        self.cache_dev_n = 0

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "TanhV2.make[target='gpu', INIT] requires a DeviceContext"
        var t = Self()
        t.ts = TargetStorage.make_cpu()
        return t^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "TanhV2.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var t = Self()
        t.ts = TargetStorage.make_gpu(ctx)
        t.cache_dev = ctx.enqueue_create_buffer[DT](1)
        t.cache_dev_n = 0
        return t^

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
        assert_tag_for["TanhV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.cache, BATCH * Self.DIM)
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var cache_p = self.cache.unsafe_ptr()
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.load[width=CPU_SIMD_W](k)
                var t = tanh(v)
                cache_p.store(k, t)
                out_p.store(k, t)
                k += CPU_SIMD_W
            while k < N:
                var y = tanh(in_p[k])
                cache_p[k] = y
                out_p[k] = y
                k += 1
        else:
            ensure_gpu_buffer(
                self.cache_dev, self.cache_dev_n,
                BATCH * Self.DIM, self.ts.ctx.value(),
            )
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var in_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var input_lt = LayoutTensor[DT, layout, MutAnyOrigin](in_ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _tanh_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                input_lt, output_lt, cache_lt,
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
        assert_tag_for["TanhV2", target](self.ts.target_tag)
        # mode has no behavioral effect — Tanh has no params.

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var c_p = self.cache.unsafe_ptr()
            var one_v = SIMD[DT, CPU_SIMD_W](1)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var y = c_p.load[width=CPU_SIMD_W](k)
                var g = go_p.load[width=CPU_SIMD_W](k)
                gi_p.store(k, g * (one_v - y * y))
                k += CPU_SIMD_W
            while k < N:
                var y = c_p[k]
                gi_p[k] = go_p[k] * (1.0 - y * y)
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var go_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_ptr)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _tanh_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cache_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
