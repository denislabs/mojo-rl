"""ReLU[DIM] — retrofit lighthouse (NN2_AUDIT Follow-up #7).

First leaf migrated to the audit-Spike patterns. Sets the template
that every other leaf will follow. Diffs vs `relu.mojo`:

  1. **TargetStorage composition.** The 3 fields `_target_tag`,
     `_inference`, `ctx` collapse to `var ts: TargetStorage`.

  2. **Free `assert_tag_for`.** The per-leaf `_assert_tag[target]`
     method is gone. Method bodies call
     `assert_tag_for["ReLU", target](self.ts.target_tag)` directly.

  3. **Unified-buffer cache (no owned cache field).** ReLU is
     input-caching. Forward records the input pointer; backward
     reconstructs a TileTensor view from it. NO `cache: List[Scalar[DT]]`
     field, NO `cache_dev: Optional[DeviceBuffer]` field, NO
     `_ensure_cache_*` helpers. Memory saved per ReLU instance:
     `BATCH * DIM * 4 bytes` (the cache buffer the v1 leaf would own).

  4. **`backward[mode]` instead of `backward` + `backward_input`.**
     ReLU is element-wise so `mode` doesn't change behavior, but the
     uniform signature lets `StopGradParams` and twin-critic actor
     updates use a single dispatch.

  5. **Slim `Module` conformance.** No Phase 10A `out_ptr/grad_in_ptr/
     grad_out_ptr/ensure_buffers` to override (those don't exist on
     `Module`).

LOC: 289 (relu.mojo) → 220 (this file). The remaining headroom comes
from the SIMD/GPU bodies, which stay mechanically identical to v1.

Validates against the v1 ReLU element-for-element — see
`tests/nn2/test_relu.mojo`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
)
from ..core.module import Module
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# ReLU is input-caching: forward writes nothing extra to a cache; the
# cache pointer aliases `input.ptr`. Backward reads the SAME pointer
# back (which is still valid because the orchestrator owns the slab
# across forward → backward).
# ──────────────────────────────────────────────────────────────────────


def _relu_forward_kernel[
    BATCH: Int,
    DIM: Int,
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
        output[b, d] = x if x > zero else zero


def _relu_backward_kernel[
    BATCH: Int,
    DIM: Int,
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
        var zero: Scalar[DT] = 0.0
        var cached = rebind[Scalar[DT]](cache[b, d])
        grad_input[b, d] = grad_output[b, d] if cached > zero else zero


# ──────────────────────────────────────────────────────────────────────
# ReLU — retrofitted ReLU. Two fields: `ts` and `_cached_input_ptr`.
# ──────────────────────────────────────────────────────────────────────


struct ReLU[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage
    # Pointer aliasing the forward input. NOT an owned buffer — the
    # caller (Sequential, user, etc.) keeps the slab live across
    # forward → backward. See audit Spike #1 for the contract.
    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. INIT ignored (ReLU is parameterless) but accepted
        for uniformity so Sequential.make[target, INIT] can recurse."""
        comptime assert (
            target == "cpu"
        ), "ReLU.make[target='gpu', INIT] requires a DeviceContext"
        var r = Self()
        r.ts = TargetStorage.make_cpu()
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert (
            target == "gpu"
        ), "ReLU.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var r = Self()
        r.ts = TargetStorage.make_gpu(ctx)
        return r^

    # ------------------------------------------------------------------
    # Forward — element-wise ReLU, alias the input pointer for backward.
    # ------------------------------------------------------------------

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
        # POLICY is accepted for trait conformance but ReLU stays in DT.
        comptime assert input.flat_rank  == 2, "input must be rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, DIM]"
        assert_tag_for["ReLU", target](self.ts.target_tag)

        # Alias input pointer — NO COPY. The orchestrator owning the
        # input slab keeps it live until backward. See audit Spike #1.
        self._cached_input_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](input.ptr)

        comptime if target == "cpu":
            # SIMD path. Mojo nightly does not autovectorize the scalar
            # form — manual `load[width=W]` is 3-5x faster (memory:
            # feedback_mojo_cpu_manual_simd_required).
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var zero_v = SIMD[DT, CPU_SIMD_W](0)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.load[width=CPU_SIMD_W](k)
                out_p.store(k, v.gt(zero_v).select(v, zero_v))
                k += CPU_SIMD_W
            while k < N:
                var v = in_p[k]
                out_p[k] = v if v > 0 else 0
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var in_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var input_lt = LayoutTensor[DT, layout, MutAnyOrigin](in_ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _relu_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                input_lt,
                output_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # Backward — element-wise mask. `mode` ignored (no params, identical
    # work for both "all" and "input_only").
    # ------------------------------------------------------------------

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
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["ReLU", target](self.ts.target_tag)

        # Element-wise: writing grad_input element-by-element while
        # reading the same element from cache is safe even when they
        # alias (the slab-aliasing case from audit Spike #1).
        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var c_p = self._cached_input_ptr
            var zero_v = SIMD[DT, CPU_SIMD_W](0)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var c = c_p.load[width=CPU_SIMD_W](k)
                var g = go_p.load[width=CPU_SIMD_W](k)
                gi_p.store(k, c.gt(zero_v).select(g, zero_v))
                k += CPU_SIMD_W
            while k < N:
                gi_p[k] = go_p[k] if c_p[k] > 0 else 0
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var go_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_ptr)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](self._cached_input_ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _relu_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt,
                cache_lt,
                gi_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
