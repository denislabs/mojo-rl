"""Parallel[A, B] — 2-branch column-concat.

Scaffold:
`ts: TargetStorage`, `backward[mode]` collapses `backward` +
`backward_input` (mode flows into each branch), walkers recurse into
each branch.

Forward:  `output = [A(input) | B(input)]`
Backward: `grad_input = A.backward(go[:,:OUT_A]) + B.backward(go[:,OUT_A:])`

A and B share the same input dim; their grad_inputs are SUMMED into
the caller's grad_input via `_elementwise_add_kernel` (shared with
Residual).

Scratch layout (4 slabs per side, lazy-grown to BATCH):
  - `out_a`/`out_b` (BATCH×OUT_*): forward outputs; reused as
    grad-output halves on backward
  - `gi_a`/`gi_b` (BATCH×IN_DIM): each branch's grad_input
"""

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for
from .residual import _elementwise_add_kernel


# ──────────────────────────────────────────────────────────────────────
# Concat / split kernels (same as v1).
# ──────────────────────────────────────────────────────────────────────


def _parallel_concat_kernel[
    BATCH: Int, OUT_A: Int, OUT_B: Int,
](
    a: LayoutTensor[DT, Layout.row_major(BATCH, OUT_A), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, OUT_B), MutAnyOrigin],
    packed: LayoutTensor[DT, Layout.row_major(BATCH, OUT_A + OUT_B), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * (OUT_A + OUT_B)
    if idx < total:
        var bi = idx // (OUT_A + OUT_B)
        var ji = idx % (OUT_A + OUT_B)
        if ji < OUT_A:
            packed[bi, ji] = rebind[Scalar[DT]](a[bi, ji])
        else:
            packed[bi, ji] = rebind[Scalar[DT]](b[bi, ji - OUT_A])


def _parallel_split_kernel[
    BATCH: Int, OUT_A: Int, OUT_B: Int,
](
    packed: LayoutTensor[DT, Layout.row_major(BATCH, OUT_A + OUT_B), MutAnyOrigin],
    a: LayoutTensor[DT, Layout.row_major(BATCH, OUT_A), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, OUT_B), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * (OUT_A + OUT_B)
    if idx < total:
        var bi = idx // (OUT_A + OUT_B)
        var ji = idx % (OUT_A + OUT_B)
        if ji < OUT_A:
            a[bi, ji] = rebind[Scalar[DT]](packed[bi, ji])
        else:
            b[bi, ji - OUT_A] = rebind[Scalar[DT]](packed[bi, ji])


# ──────────────────────────────────────────────────────────────────────
# Parallel
# ──────────────────────────────────────────────────────────────────────


struct Parallel[A: Module, B: Module](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.A.IN_DIMS[0])
    comptime OUT_DIM = Self.A.OUT_DIM + Self.B.OUT_DIM
    comptime OUT_A = Self.A.OUT_DIM
    comptime OUT_B = Self.B.OUT_DIM

    var branch_a: Self.A
    var branch_b: Self.B

    var out_a_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var out_b_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var gi_a_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin]
    var gi_b_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin]

    var out_a_dev: Optional[DeviceBuffer[DT]]
    var out_b_dev: Optional[DeviceBuffer[DT]]
    var gi_a_dev:  Optional[DeviceBuffer[DT]]
    var gi_b_dev:  Optional[DeviceBuffer[DT]]
    var scratch_n_batch: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.A.IN_DIMS[0] == Self.B.IN_DIMS[0]
        ), "Parallel requires A.IN_DIMS[0] == B.IN_DIMS[0]"
        self.branch_a = Self.A()
        self.branch_b = Self.B()
        self.out_a_cpu = alloc[Scalar[DT]](1)
        self.out_b_cpu = alloc[Scalar[DT]](1)
        self.gi_a_cpu  = alloc[Scalar[DT]](1)
        self.gi_b_cpu  = alloc[Scalar[DT]](1)
        self.out_a_dev = None
        self.out_b_dev = None
        self.gi_a_dev  = None
        self.gi_b_dev  = None
        self.scratch_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Parallel: target must be 'cpu' or 'gpu'"
        )
        var p = Self()
        p.branch_a = Self.A.make[target, INIT](ctx=ctx)
        p.branch_b = Self.B.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            p.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Parallel.make[target='gpu']"](ctx)
            p.out_a_dev = ctx_v.enqueue_create_buffer[DT](1)
            p.out_b_dev = ctx_v.enqueue_create_buffer[DT](1)
            p.gi_a_dev  = ctx_v.enqueue_create_buffer[DT](1)
            p.gi_b_dev  = ctx_v.enqueue_create_buffer[DT](1)
            p.ts = TargetStorage.make_gpu(ctx_v)
        return p^

    def __del__(deinit self):
        self.out_a_cpu.free()
        self.out_b_cpu.free()
        self.gi_a_cpu.free()
        self.gi_b_cpu.free()

    def _ensure_scratch_cpu(mut self, batch: Int):
        if self.scratch_n_batch < batch:
            self.out_a_cpu.free()
            self.out_b_cpu.free()
            self.gi_a_cpu.free()
            self.gi_b_cpu.free()
            self.out_a_cpu = alloc[Scalar[DT]](batch * Self.OUT_A)
            self.out_b_cpu = alloc[Scalar[DT]](batch * Self.OUT_B)
            self.gi_a_cpu  = alloc[Scalar[DT]](batch * Self.IN_DIMS[0])
            self.gi_b_cpu  = alloc[Scalar[DT]](batch * Self.IN_DIMS[0])
            self.scratch_n_batch = batch

    def _ensure_scratch_gpu(mut self, batch: Int) raises:
        if self.scratch_n_batch < batch:
            var c = self.ts.ctx.value()
            self.out_a_dev = c.enqueue_create_buffer[DT](batch * Self.OUT_A)
            self.out_b_dev = c.enqueue_create_buffer[DT](batch * Self.OUT_B)
            self.gi_a_dev  = c.enqueue_create_buffer[DT](batch * Self.IN_DIMS[0])
            self.gi_b_dev  = c.enqueue_create_buffer[DT](batch * Self.IN_DIMS[0])
            self.scratch_n_batch = batch

    # ----- Forward ---------------------------------------------------------

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
        assert_tag_for["Parallel", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self._ensure_scratch_cpu(BATCH)
            var out_a = TileTensor(self.out_a_cpu, row_major[BATCH, Self.OUT_A]())
            var out_b = TileTensor(self.out_b_cpu, row_major[BATCH, Self.OUT_B]())
            self.branch_a.forward[target, BATCH, POLICY=POLICY](input, output=out_a)
            self.branch_b.forward[target, BATCH, POLICY=POLICY](input, output=out_b)
            for b in range(BATCH):
                for j in range(Self.OUT_A):
                    output_v[b, j] = out_a[b, j]
                for j in range(Self.OUT_B):
                    output_v[b, Self.OUT_A + j] = out_b[b, j]
        else:
            self._ensure_scratch_gpu(BATCH)
            var out_p_w = mptr(output_v.ptr)
            var pa: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_a_dev.value().unsafe_ptr()
            var pb: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_b_dev.value().unsafe_ptr()
            var out_a_tt = TileTensor(pa, row_major[BATCH, Self.OUT_A]())
            var out_b_tt = TileTensor(pb, row_major[BATCH, Self.OUT_B]())
            self.branch_a.forward[target, BATCH, POLICY=POLICY](input, output=out_a_tt)
            self.branch_b.forward[target, BATCH, POLICY=POLICY](input, output=out_b_tt)
            comptime layout_a = Layout.row_major(BATCH, Self.OUT_A)
            comptime layout_b = Layout.row_major(BATCH, Self.OUT_B)
            comptime layout_p = Layout.row_major(BATCH, Self.OUT_DIM)
            var a_lt = LayoutTensor[DT, layout_a, MutAnyOrigin](self.out_a_dev.value())
            var b_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](self.out_b_dev.value())
            var p_lt = LayoutTensor[DT, layout_p, MutAnyOrigin](out_p_w)
            comptime n_blocks = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _parallel_concat_kernel[BATCH, Self.OUT_A, Self.OUT_B]
            self.ts.ctx.value().enqueue_function[kernel](
                a_lt, b_lt, p_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

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
        assert_tag_for["Parallel", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            self._ensure_scratch_cpu(BATCH)
            var go_a = TileTensor(self.out_a_cpu, row_major[BATCH, Self.OUT_A]())
            var go_b = TileTensor(self.out_b_cpu, row_major[BATCH, Self.OUT_B]())
            for b in range(BATCH):
                for j in range(Self.OUT_A):
                    go_a[b, j] = grad_output_v[b, j]
                for j in range(Self.OUT_B):
                    go_b[b, j] = grad_output_v[b, Self.OUT_A + j]
            var gi_a = TileTensor(self.gi_a_cpu, row_major[BATCH, Self.IN_DIMS[0]]())
            var gi_b = TileTensor(self.gi_b_cpu, row_major[BATCH, Self.IN_DIMS[0]]())
            self.branch_a.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](go_a, gi_a)
            self.branch_b.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](go_b, gi_b)
            var ap = self.gi_a_cpu
            var bp = self.gi_b_cpu
            var gp = mptr(grad_input_v.ptr)
            comptime N = BATCH * Self.IN_DIMS[0]
            var k = 0
            while k + CPU_SIMD_W <= N:
                gp.store(
                    k,
                    ap.load[width=CPU_SIMD_W](k) + bp.load[width=CPU_SIMD_W](k),
                )
                k += CPU_SIMD_W
            while k < N:
                gp[k] = ap[k] + bp[k]
                k += 1
        else:
            self._ensure_scratch_gpu(BATCH)
            var go_p_w = mptr(grad_output_v.ptr)
            var gi_p_w = mptr(grad_input_v.ptr)
            var pa: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_a_dev.value().unsafe_ptr()
            var pb: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_b_dev.value().unsafe_ptr()
            var pia: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.gi_a_dev.value().unsafe_ptr()
            var pib: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.gi_b_dev.value().unsafe_ptr()

            comptime layout_a = Layout.row_major(BATCH, Self.OUT_A)
            comptime layout_b = Layout.row_major(BATCH, Self.OUT_B)
            comptime layout_p = Layout.row_major(BATCH, Self.OUT_DIM)
            var p_lt = LayoutTensor[DT, layout_p, MutAnyOrigin](go_p_w)
            var a_lt = LayoutTensor[DT, layout_a, MutAnyOrigin](self.out_a_dev.value())
            var b_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](self.out_b_dev.value())
            comptime n_blocks_split = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime split_kernel = _parallel_split_kernel[
                BATCH, Self.OUT_A, Self.OUT_B
            ]
            self.ts.ctx.value().enqueue_function[split_kernel](
                p_lt, a_lt, b_lt,
                grid_dim=n_blocks_split, block_dim=TPB,
            )

            var go_a_tt = TileTensor(pa, row_major[BATCH, Self.OUT_A]())
            var go_b_tt = TileTensor(pb, row_major[BATCH, Self.OUT_B]())
            var gi_a_tt = TileTensor(pia, row_major[BATCH, Self.IN_DIMS[0]]())
            var gi_b_tt = TileTensor(pib, row_major[BATCH, Self.IN_DIMS[0]]())
            self.branch_a.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](go_a_tt, gi_a_tt)
            self.branch_b.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](go_b_tt, gi_b_tt)

            comptime layout_in = Layout.row_major(BATCH, Self.IN_DIMS[0])
            var gi_a_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](self.gi_a_dev.value())
            var gi_b_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](self.gi_b_dev.value())
            var gi_out_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](gi_p_w)
            comptime n_blocks_sum = (BATCH * Self.IN_DIMS[0] + TPB - 1) // TPB
            comptime sum_kernel = _elementwise_add_kernel[BATCH, Self.IN_DIMS[0]]
            self.ts.ctx.value().enqueue_function[sum_kernel](
                gi_a_lt, gi_b_lt, gi_out_lt,
                grid_dim=n_blocks_sum, block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Parallel", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.branch_a.for_each_param[target, V](prefix + sep + "a", visitor)
        self.branch_b.for_each_param[target, V](prefix + sep + "b", visitor)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Parallel", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.branch_a.for_each_state[target, V](prefix + sep + "a", visitor)
        self.branch_b.for_each_state[target, V](prefix + sep + "b", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Parallel", target](self.ts.target_tag)
        self.branch_a.zero_grad[target]()
        self.branch_b.zero_grad[target]()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.branch_a.set_attr[ATTR](value)
        self.branch_b.set_attr[ATTR](value)
