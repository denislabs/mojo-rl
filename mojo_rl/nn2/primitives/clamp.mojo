"""Clamp[DIM] — element-wise hard clamp to [min_val, max_val].

Forward:  out[b, d] = max(min_val, min(max_val, x[b, d]))
Backward: grad_in[b, d] = grad_out[b, d] if min_val < x < max_val else 0

`min_val` and `max_val` are runtime per-instance scalars (mirrors `Scale`).
Caller sets them via `set_attr["min_val"](v)` / `set_attr["max_val"](v)`
on either the bare struct or through `ComputeGraph.set_node_attr[NAME, ATTR]`.

No cache field — the backward kernel re-reads `x` through the orchestrator-
owned input slab (input-alias pattern, mirrors ReLU/Mish in `Elementwise`).
The orchestrator (Sequential / ComputeGraph) keeps the input slab live
until backward completes.

Phase 4.5 — used by `DDPGTargetYBlock` (1 instance, action clamp) and
`TD3TargetYBlock` (2 instances: noise clip + smoothed-action clamp).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


def _clamp_forward_kernel[
    N: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    min_val: Scalar[DT],
    max_val: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](input[idx])
        var y = x
        if y > max_val:
            y = max_val
        if y < min_val:
            y = min_val
        output[idx] = y


def _clamp_backward_kernel[
    N: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    min_val: Scalar[DT],
    max_val: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](input[idx])
        var zero: Scalar[DT] = 0.0
        if x < min_val or x > max_val:
            grad_input[idx] = zero
        else:
            grad_input[idx] = rebind[Scalar[DT]](grad_output[idx])


struct Clamp[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    var min_val: Scalar[DT]
    var max_val: Scalar[DT]
    var ts: TargetStorage

    # Input-alias cache: backward reads x back through this pointer (the
    # orchestrator's input slab). Mirrors ReLU/Mish in `Elementwise`.
    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.min_val = Scalar[DT](-1.0)
        self.max_val = Scalar[DT](1.0)
        self.ts = TargetStorage.make_uninit()
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Clamp: target must be 'cpu' or 'gpu'"
        )
        var c = Self()
        comptime if target == "cpu":
            c.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Clamp.make[target='gpu']: ctx required")
            c.ts = TargetStorage.make_gpu(ctx.value())
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
        assert_tag_for["Clamp", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            self._cached_input_ptr = in_p
            var min_v = SIMD[DT, CPU_SIMD_W](self.min_val)
            var max_v = SIMD[DT, CPU_SIMD_W](self.max_val)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.load[width=CPU_SIMD_W](k)
                # min(max(v, min_v), max_v)
                v = v.gt(min_v).select(v, min_v)
                v = v.lt(max_v).select(v, max_v)
                out_p.store(k, v)
                k += CPU_SIMD_W
            while k < N:
                var v = in_p[k]
                if v > self.max_val:
                    v = self.max_val
                if v < self.min_val:
                    v = self.min_val
                out_p[k] = v
                k += 1
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            self._cached_input_ptr = in_p
            comptime N = BATCH * Self.DIM
            var in_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](out_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _clamp_forward_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, self.min_val, self.max_val,
                grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["Clamp", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
            var x_p = self._cached_input_ptr
            var min_v = SIMD[DT, CPU_SIMD_W](self.min_val)
            var max_v = SIMD[DT, CPU_SIMD_W](self.max_val)
            var zero = SIMD[DT, CPU_SIMD_W](0.0)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var x = x_p.load[width=CPU_SIMD_W](k)
                var go = go_p.load[width=CPU_SIMD_W](k)
                # in_range = (x > min_v) AND (x < max_v); else zero.
                var in_range = x.gt(min_v) & x.lt(max_v)
                gi_p.store(k, in_range.select(go, zero))
                k += CPU_SIMD_W
            while k < N:
                var x = x_p[k]
                if x < self.min_val or x > self.max_val:
                    gi_p[k] = Scalar[DT](0.0)
                else:
                    gi_p[k] = go_p[k]
                k += 1
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
            var x_p = self._cached_input_ptr
            comptime N = BATCH * Self.DIM
            var go_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](go_p)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](x_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](gi_p)
            comptime TPB = 128
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _clamp_backward_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, in_lt, gi_lt, self.min_val, self.max_val,
                grid_dim=n_blocks, block_dim=TPB,
            )

    # Override of Module.set_attr — supports ATTR="min_val" / "max_val".
    # Other ATTR strings are no-ops (Mojo nightly can't error on unknown
    # ATTR from a comptime if without a constexpr-assert).
    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "min_val":
            self.min_val = value
        comptime if ATTR == "max_val":
            self.max_val = value
