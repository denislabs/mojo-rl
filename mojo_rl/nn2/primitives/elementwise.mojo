"""Elementwise[DIM, OP: ElementOp] — single template for all elementwise activations.

Phase 1.3. Replaces the per-leaf duplication (`relu.mojo`, `tanh.mojo`,
`sigmoid.mojo`, `mish.mojo`, `symlog.mojo`, `scale.mojo`, …) with one
Module body that takes the per-lane math via a comptime `OP: ElementOp`
parameter. Each leaf collapses to a one-line type alias:

    alias Tanh[DIM: Int] = Elementwise[DIM, TanhOp]
    alias ReLU[DIM: Int] = Elementwise[DIM, ReLUOp]
    alias Sigmoid[DIM: Int] = Elementwise[DIM, SigmoidOp]

Cache strategy is chosen by `Self.OP.owns_cache`:

  - `Self.OP.owns_cache = True`  (Tanh / Sigmoid / …):
        forward writes `y = OP.forward(x)` to an OWNED cache buffer
        (`cache` CPU `List` + `cache_dev` GPU `DeviceBuffer`). Backward
        reads `y` from the cache. Mirrors `Tanh[DIM]`'s pre-1.3 layout.
  - `Self.OP.owns_cache = False` (ReLU / Mish / Symlog / Scale / …):
        forward aliases the input pointer into `_cached_input_ptr`
        (no copy). Backward reads `x` back through that alias. The
        orchestrator (Sequential / ComputeGraph) owns the input slab
        and guarantees it survives until backward completes.

Both CPU paths use SIMD via `Self.OP.forward_simd[W]` / `Self.OP.backward_simd[W]`
with `CPU_SIMD_W = 8` (project-wide constant). GPU paths use one
shared kernel per direction, parameterised on `OP` via comptime.

BACKWARD-ORDER INVARIANT: this leaf has no params, so `mode="all"` and
`mode="input_only"` are identical. The invariant doesn't apply.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.element_op import ElementOp
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — OP supplies the scalar math via `Self.OP.forward_scalar` /
# `Self.OP.backward_scalar`. The cache write is comptime-branched on
# `Self.OP.owns_cache`: owned-cache stores `y`, input-alias stores `x`.
# ──────────────────────────────────────────────────────────────────────


def _elementwise_forward_kernel[
    BATCH: Int, DIM: Int, OP: ElementOp,
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
        var y = OP.forward_scalar(x)
        output[b, d] = y
        comptime if OP.owns_cache:
            cache[b, d] = y
        else:
            cache[b, d] = x


def _elementwise_backward_kernel[
    BATCH: Int, DIM: Int, OP: ElementOp,
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
        var c = rebind[Scalar[DT]](cache[b, d])
        var go = rebind[Scalar[DT]](grad_output[b, d])
        grad_input[b, d] = OP.backward_scalar(c, go)


# ──────────────────────────────────────────────────────────────────────
# Elementwise[DIM, OP] — owns: TargetStorage + cache (for owns_cache=True)
# OR _cached_input_ptr (for owns_cache=False). Both field clusters are
# always present; only the matching one is populated (same pattern as
# CPU/GPU dual storage).
# ──────────────────────────────────────────────────────────────────────


struct Elementwise[DIM: Int, OP: ElementOp](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    @staticmethod
    def display_label() -> String:
        return Self.OP.display_label()

    var ts: TargetStorage

    # owns_cache=True path: own cache (S5 dynamic Cache role, lazy-grown).
    var cache: Cache["elem_cache"]

    # owns_cache=False path: alias the orchestrator's input slab (borrow,
    # not owned storage — stays a raw pointer, excluded from the Tensor core).
    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.cache = Cache["elem_cache"]()
        self._cached_input_ptr = None

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (Elementwise is
        parameterless) but accepted for uniformity so
        Sequential.make[target, INIT] can recurse."""
        comptime assert target == "cpu" or target == "gpu", (
            "Elementwise: target must be 'cpu' or 'gpu'"
        )
        var e = Self()
        comptime if target == "cpu":
            e.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Elementwise.make[target='gpu']"](ctx)
            e.ts = TargetStorage.make_gpu(ctx_v)
            # cache is lazy (S5 Cache) — grown at forward via ensure_gpu.
        return e^

    # ------------------------------------------------------------------
    # Forward — SIMD CPU loop or one GPU kernel launch. Cache write is
    # comptime-branched on Self.OP.owns_cache.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        # POLICY accepted for trait conformance; Elementwise stays in DT.
        assert_tag_for["Elementwise", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            comptime N = BATCH * Self.DIM
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)

            comptime if Self.OP.owns_cache:
                # Own cache: write y to both output and cache.
                self.cache.ensure_cpu(N)
                var cache_p = self.cache.cpu_ptr()
                var k = 0
                while k + CPU_SIMD_W <= N:
                    var v = in_p.load[width=CPU_SIMD_W](k)
                    var y = Self.OP.forward_simd[CPU_SIMD_W](v)
                    cache_p.store(k, y)
                    out_p.store(k, y)
                    k += CPU_SIMD_W
                while k < N:
                    var y = Self.OP.forward_scalar(in_p[k])
                    cache_p[k] = y
                    out_p[k] = y
                    k += 1
            else:
                # Input-alias: record the input pointer; write only y to
                # output. Orchestrator keeps input slab live.
                self._cached_input_ptr = in_p
                var k = 0
                while k + CPU_SIMD_W <= N:
                    var v = in_p.load[width=CPU_SIMD_W](k)
                    out_p.store(k, Self.OP.forward_simd[CPU_SIMD_W](v))
                    k += CPU_SIMD_W
                while k < N:
                    out_p[k] = Self.OP.forward_scalar(in_p[k])
                    k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var in_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            var input_lt = LayoutTensor[DT, layout, MutAnyOrigin](in_ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_ptr)

            comptime if Self.OP.owns_cache:
                self.cache.ensure_gpu(self.ts.ctx.value(), BATCH * Self.DIM)
                var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                    self.cache.dev.value()
                )
                comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
                comptime kernel = _elementwise_forward_kernel[
                    BATCH, Self.DIM, Self.OP,
                ]
                self.ts.ctx.value().enqueue_function[kernel](
                    input_lt, output_lt, cache_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                # Input-alias: cache the input device pointer; the GPU
                # backward kernel reads it back. No kernel-side cache
                # write needed — but `_elementwise_forward_kernel` always
                # takes a cache LT; we pass `input_lt` itself, which is
                # harmless because `Self.OP.owns_cache == False` makes the
                # kernel write `cache[b, d] = x` to the same buffer it
                # just read from (idempotent).
                self._cached_input_ptr = in_ptr
                comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
                comptime kernel = _elementwise_forward_kernel[
                    BATCH, Self.DIM, Self.OP,
                ]
                self.ts.ctx.value().enqueue_function[kernel](
                    input_lt, output_lt, input_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

    # ------------------------------------------------------------------
    # Backward — read cached `c` (= y or x), compute gi = OP.backward(c, go).
    # `mode` is a no-op (no params).
    # ------------------------------------------------------------------

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Elementwise", target](self.ts.target_tag)
        var go_view = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi_view = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_view.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi_view.ptr)
            var c_p: UnsafePointer[Scalar[DT], MutAnyOrigin]
            comptime if Self.OP.owns_cache:
                c_p = self.cache.cpu_ptr()
            else:
                c_p = self._cached_input_ptr.value()
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var c = c_p.load[width=CPU_SIMD_W](k)
                var g = go_p.load[width=CPU_SIMD_W](k)
                gi_p.store(k, Self.OP.backward_simd[CPU_SIMD_W](c, g))
                k += CPU_SIMD_W
            while k < N:
                gi_p[k] = Self.OP.backward_scalar(c_p[k], go_p[k])
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var go_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_view.ptr)
            var gi_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi_view.ptr)
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_ptr)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_ptr)
            var cache_lt: LayoutTensor[DT, layout, MutAnyOrigin]
            comptime if Self.OP.owns_cache:
                cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                    self.cache.dev.value()
                )
            else:
                cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                    self._cached_input_ptr.value()
                )
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _elementwise_backward_kernel[
                BATCH, Self.DIM, Self.OP,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cache_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
