"""Add[DIM, N] — variadic elementwise sum primitive.

Replaces the legacy `BinaryAdd[DIM]` (alias of `BinaryElementwise[DIM,
BinaryAddOp]`) + `TernaryFusedAdd[DIM]` with one variadic primitive.

  output[b, d]         = Σ_i inputs[i][b, d]
  grad_inputs[i][b, d] = grad_output[b, d]    (∀ i ∈ [0, N))

`ARITY = N`, all inputs share `DIM` (homogeneous; broadcasting is out of
scope). CPU SIMD + GPU. GPU forward uses `init + (N-1) × accum` kernel
launches; vjp launches N copy kernels.

Variadic inputs follow the same-Layout hetero-shape workaround (Phase
4.6c) — though for Add the shapes are already homogeneous, so the
workaround is a no-op in practice.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — init (output = src) and accum (output += src). Forward
# emits 1 init + (N-1) accum launches; vjp emits N init (copy) launches
# (one per grad-input).
# ──────────────────────────────────────────────────────────────────────


def _add_init_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](src[idx])


def _add_accum_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = output[idx] + rebind[Scalar[DT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# Add[DIM, N]
# ──────────────────────────────────────────────────────────────────────


struct Add[DIM_: Int, N_: Int](Module):
    comptime ARITY: Int = Self.N_
    comptime IN_DIMS = InlineArray[Int, Self.N_](fill=Self.DIM_)
    comptime IN0_DIM: Int = Self.DIM_
    comptime OUT_DIM: Int = Self.DIM_

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N_ >= 2, "Add: needs at least 2 inputs"
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Add: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        comptime if target == "cpu":
            a.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Add.make[target='gpu']: ctx required")
            a.ts = TargetStorage.make_gpu(ctx.value())
        return a^

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
        assert_tag_for["Add", target](self.ts.target_tag)
        comptime TOTAL = BATCH * Self.DIM_

        comptime if target == "cpu":
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            # Init: output = inputs[0]
            var k = 0
            while k + CPU_SIMD_W <= TOTAL:
                o_p.store(k, i0_p.load[width=CPU_SIMD_W](k))
                k += CPU_SIMD_W
            while k < TOTAL:
                o_p[k] = i0_p[k]
                k += 1
            # Accumulate inputs[1..N)
            comptime for i in range(1, Self.N_):
                var ii_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    inputs[i].ptr
                )
                var kk = 0
                while kk + CPU_SIMD_W <= TOTAL:
                    o_p.store(
                        kk,
                        o_p.load[width=CPU_SIMD_W](kk)
                        + ii_p.load[width=CPU_SIMD_W](kk),
                    )
                    kk += CPU_SIMD_W
                while kk < TOTAL:
                    o_p[kk] = o_p[kk] + ii_p[kk]
                    kk += 1
        else:
            comptime layout = Layout.row_major(TOTAL)
            comptime n_blocks = (TOTAL + TPB - 1) // TPB
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var o_lt = LayoutTensor[DT, layout, MutAnyOrigin](o_p)

            # Init from inputs[0].
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            var i0_lt = LayoutTensor[DT, layout, MutAnyOrigin](i0_p)
            comptime init_kernel = _add_init_kernel[TOTAL]
            self.ts.ctx.value().enqueue_function[init_kernel](
                i0_lt, o_lt, grid_dim=n_blocks, block_dim=TPB,
            )

            # Accumulate inputs[1..N).
            comptime for i in range(1, Self.N_):
                var ii_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    inputs[i].ptr
                )
                var ii_lt = LayoutTensor[DT, layout, MutAnyOrigin](ii_p)
                comptime accum_kernel = _add_accum_kernel[TOTAL]
                self.ts.ctx.value().enqueue_function[accum_kernel](
                    ii_lt, o_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["Add", target](self.ts.target_tag)
        comptime TOTAL = BATCH * Self.DIM_

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output.ptr
            )
            comptime for i in range(Self.N_):
                var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    grad_inputs[i].ptr
                )
                var k = 0
                while k + CPU_SIMD_W <= TOTAL:
                    gi_p.store(k, go_p.load[width=CPU_SIMD_W](k))
                    k += CPU_SIMD_W
                while k < TOTAL:
                    gi_p[k] = go_p[k]
                    k += 1
        else:
            comptime layout = Layout.row_major(TOTAL)
            comptime n_blocks = (TOTAL + TPB - 1) // TPB
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output.ptr
            )
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_p)
            comptime for i in range(Self.N_):
                var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    grad_inputs[i].ptr
                )
                var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_p)
                comptime copy_kernel = _add_init_kernel[TOTAL]
                self.ts.ctx.value().enqueue_function[copy_kernel](
                    go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
                )
