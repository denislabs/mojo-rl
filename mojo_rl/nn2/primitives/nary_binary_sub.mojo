"""NaryBinarySub[DIM] — Phase 4.6a foundation demo leaf.

A standalone element-wise subtraction module conforming to `NaryModule`
(the new unified trait). Demonstrates the full pattern that Phase 4.6b
will apply to every leaf:

  - `comptime ARITY: Int = 2` field declares binary input arity.
  - `forward[BATCH](*inputs, mut output)` — variadic origin-erased pack.
  - `vjp[BATCH, mode](grad_output, mut *grad_inputs)` — mutable variadic.
  - Body uses `typed_view[BATCH, DIM]` to rebuild typed rank-2 views
    from the opaque-layout variadic elements.
  - `comptime if Self.ARITY == 2:` dispatch chooses the typed-view
    branch (no per-iteration overhead, just one rebuild per input).

CPU-only for the foundation. Phase 4.6b adds the GPU kernel + SIMD path
when migrating the full BinaryElementwise template.

Semantics (mirror BinarySub):
    output[b, d]   = in0[b, d] - in1[b, d]
    grad_in0[b, d] =  grad_output[b, d]
    grad_in1[b, d] = -grad_output[b, d]
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


struct NaryBinarySub[DIM_: Int](Module):
    """Element-wise subtraction (binary, parameter-less). Conforms to
    the unified Module trait (Phase 4.6b — NaryModule was folded back
    into Module)."""

    comptime DIM: Int = Self.DIM_
    comptime ARITY: Int = 2
    comptime IN_DIM: Int = Self.DIM_
    comptime IN0_DIM: Int = Self.DIM_
    comptime IN1_DIM: Int = Self.DIM_
    comptime IN2_DIM: Int = 0
    comptime OUT_DIM: Int = Self.DIM_

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "NaryBinarySub.make[target='gpu', INIT] requires DeviceContext"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "NaryBinarySub.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var s = Self()
        s.ts = TargetStorage.make_gpu(ctx)
        return s^

    # ------------------------------------------------------------------
    # Forward — output[b, d] = inputs[0][b, d] - inputs[1][b, d].
    # ------------------------------------------------------------------

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
        assert_tag_for["NaryBinarySub", target](self.ts.target_tag)
        comptime assert target == "cpu", "NaryBinarySub: CPU-only foundation"

        comptime if Self.ARITY == 2:
            # Rebuild typed rank-2 views from variadic elements.
            var in0 = typed_view[BATCH, Self.IN0_DIM](inputs[0])
            var in1 = typed_view[BATCH, Self.IN1_DIM](inputs[1])
            var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

            # SIMD inner loop on flat pointers (mirrors BinaryElementwise).
            comptime N = BATCH * Self.DIM
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in0.ptr)
            var i1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in1.ptr)
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)

            var k = 0
            while k + CPU_SIMD_W <= N:
                var xv = i0_p.load[width=CPU_SIMD_W](k)
                var yv = i1_p.load[width=CPU_SIMD_W](k)
                o_p.store(k, xv - yv)
                k += CPU_SIMD_W
            while k < N:
                o_p[k] = i0_p[k] - i1_p[k]
                k += 1
        else:
            raise Error("NaryBinarySub: ARITY must be 2 — got non-binary")

    # ------------------------------------------------------------------
    # vjp — grad_inputs[0] = grad_output; grad_inputs[1] = -grad_output.
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["NaryBinarySub", target](self.ts.target_tag)
        comptime assert target == "cpu", "NaryBinarySub: CPU-only foundation"

        comptime if Self.ARITY == 2:
            var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
            var gi0 = typed_view_mut[BATCH, Self.IN0_DIM](grad_inputs[0])
            var gi1 = typed_view_mut[BATCH, Self.IN1_DIM](grad_inputs[1])

            comptime N = BATCH * Self.DIM
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi0.ptr)
            var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi1.ptr)

            var k = 0
            while k + CPU_SIMD_W <= N:
                var gv = go_p.load[width=CPU_SIMD_W](k)
                gi0_p.store(k, gv)
                gi1_p.store(k, -gv)
                k += CPU_SIMD_W
            while k < N:
                gi0_p[k] = go_p[k]
                gi1_p[k] = -go_p[k]
                k += 1
        else:
            raise Error("NaryBinarySub: ARITY must be 2 — got non-binary")
