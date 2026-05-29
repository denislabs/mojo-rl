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

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
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


def _scale_dev_kernel[
    N: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    mptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    # Device-resident multiplier variant — reads the scale factor from
    # `mptr[0]` instead of a baked scalar arg, so the value can be updated
    # by another GPU kernel (SAC's on-device α) without breaking CUDA-graph
    # capture. Every thread reads the same `mptr[0]`.
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](input[idx]) * mptr[0]


struct Scale[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    var multiplier: Scalar[DT]
    # Slice 4 — optional device-resident multiplier source. When non-null
    # (set via `set_multiplier_ptr`), the GPU forward/vjp read the scale
    # factor from `multiplier_ptr[0]` instead of baking `multiplier` into the
    # kernel args — required for CUDA-graph capture when another GPU kernel
    # (SAC's on-device α) updates the value each step. Null → baked-scalar
    # path (bit-identical to pre-Slice-4). CPU always uses `multiplier`.
    var multiplier_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self.multiplier = Scalar[DT](1.0)
        self.multiplier_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Scale: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Scale.make[target='gpu']: ctx required")
            s.ts = TargetStorage.make_gpu(ctx.value())
        return s^

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
        assert_tag_for["Scale", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
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
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            comptime N = BATCH * Self.DIM
            var in_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](out_p)
            comptime n_blocks = (N + TPB - 1) // TPB
            if Int(self.multiplier_ptr) != 0:
                comptime dev_kernel = _scale_dev_kernel[N]
                self.ts.ctx.value().enqueue_function[dev_kernel](
                    in_lt, out_lt, self.multiplier_ptr,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                comptime kernel = _scale_kernel[N]
                self.ts.ctx.value().enqueue_function[kernel](
                    in_lt, out_lt, self.multiplier,
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
        assert_tag_for["Scale", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
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
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
            comptime N = BATCH * Self.DIM
            var go_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](gi_p)
            comptime n_blocks = (N + TPB - 1) // TPB
            if Int(self.multiplier_ptr) != 0:
                comptime dev_kernel = _scale_dev_kernel[N]
                self.ts.ctx.value().enqueue_function[dev_kernel](
                    go_lt, gi_lt, self.multiplier_ptr,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                comptime kernel = _scale_kernel[N]
                self.ts.ctx.value().enqueue_function[kernel](
                    go_lt, gi_lt, self.multiplier,
                    grid_dim=n_blocks, block_dim=TPB,
                )

    # Override of Module.set_attr — supports ATTR="multiplier". Other
    # ATTR strings are no-ops (Mojo nightly can't error on unknown
    # ATTR from a comptime if without a constexpr-assert).
    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "multiplier":
            self.multiplier = value

    # Slice 4 — point the multiplier at a device buffer holding the live
    # scale factor (e.g. SAC's on-device α). Pass a null pointer to revert
    # to the baked-scalar `multiplier` path. GPU-only effect; CPU ignores it.
    def set_multiplier_ptr(
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ):
        self.multiplier_ptr = p
