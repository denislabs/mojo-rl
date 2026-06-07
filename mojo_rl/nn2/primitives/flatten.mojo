"""Flatten[DIM] — identity passthrough Module.

Phase 2 of `nn2/PORTING_PLAN.md`. From the module's standpoint a
flatten is the identity: a `[BATCH, DIM]` slab in == a `[BATCH, DIM]`
slab out. The shape change is purely in the caller's view layout
(e.g. a Conv2D producing `[BATCH, C·H·W]` flat output reinterpreted
as `[BATCH, DIM]` with DIM = C·H·W). The Module trait carries
`OUT_DIM = IN_DIM` and the kernel is a memcpy in both directions.

Lives in nn2 ahead of Conv2D so Phase 5's `NatureDQN = Sequential[
Conv2D, ReLU, Conv2D, ReLU, Conv2D, ReLU, Flatten, Linear, ReLU, Linear]`
composition lands the moment Conv2D arrives — no glue change to
`Sequential`.

No params, no cache. backward is the same identity as forward.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


def _flatten_copy_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


struct Flatten[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params) but accepted
        for `Sequential.make[target, INIT]` uniformity."""
        comptime assert target == "cpu" or target == "gpu", (
            "Flatten: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.DIM > 0, "Flatten: DIM must be > 0"
        var f = Self()
        comptime if target == "cpu":
            f.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Flatten.make[target='gpu']: ctx required")
            f.ts = TargetStorage.make_gpu(ctx.value())
        return f^

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
        assert_tag_for["Flatten", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            output_v.ptr
        )
        comptime N = BATCH * Self.DIM

        comptime if target == "cpu":
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.load[width=CPU_SIMD_W](k)
                out_p.store(k, v)
                k += CPU_SIMD_W
            while k < N:
                out_p[k] = in_p[k]
                k += 1
        else:
            var in_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](out_p)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _flatten_copy_kernel[N]
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
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Flatten", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_output_v.ptr
        )
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_input_v.ptr
        )
        comptime N = BATCH * Self.DIM

        comptime if target == "cpu":
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = go_p.load[width=CPU_SIMD_W](k)
                gi_p.store(k, v)
                k += CPU_SIMD_W
            while k < N:
                gi_p[k] = go_p[k]
                k += 1
        else:
            var go_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](gi_p)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime kernel = _flatten_copy_kernel[N]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
