"""Parallel[A, B] — column-concat 2-way branching. Phase 5.6.

Two branches sharing the same input dimension. The output packs both
branches side-by-side:

    output[b, j]              = A(input)[b, j]                 for j in [0, A.OUT)
    output[b, A.OUT + j]      = B(input)[b, j]                 for j in [0, B.OUT)

Constraint: `A.IN_DIM == B.IN_DIM` (enforced at comptime).
OUT_DIM     = A.OUT_DIM + B.OUT_DIM.

Backward:
    grad_input = A.backward(grad_output[:, :A.OUT])
               + B.backward(grad_output[:, A.OUT:])

Internal scratch (4 buffers, lazy-grown):
    out_a / out_b — A and B forward outputs (also reused as grad-output
                    halves on backward).
    gi_a  / gi_b  — A's and B's grad_input. Summed into the caller's
                    grad_input by `_elementwise_add_kernel` (shared with
                    `Residual`).

Why scratch instead of strided TileTensor views: the slice
`packed[:, :A.OUT]` is non-contiguous (stride = A.OUT + B.OUT, not
A.OUT) — Mojo's `row_major[BATCH, A.OUT]` layout bakes in stride =
A.OUT, so passing the packed pointer would mis-index. Scratch is the
simplest correct path for Phase 5; PPO action heads are tiny anyway
(A.OUT, B.OUT ≤ ~30) so the copy cost is negligible.

Variadic Parallel[*BRANCHES] is deferred — Phase 5 ships only the 2-way
form (the one PPO's StochasticActor actually needs).
"""

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Initializer,
    AMPPolicy, NoAMP,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)
from .residual import _elementwise_add_kernel


# ──────────────────────────────────────────────────────────────────────────
# GPU helpers: concat (forward) and split (backward grad_output).
# ──────────────────────────────────────────────────────────────────────────


def _parallel_concat_kernel[
    BATCH: Int, OUT_A: Int, OUT_B: Int,
](
    a: LayoutTensor[DT, Layout.row_major(BATCH, OUT_A), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, OUT_B), MutAnyOrigin],
    packed: LayoutTensor[DT, Layout.row_major(BATCH, OUT_A + OUT_B), MutAnyOrigin],
):
    """packed[b, j<OUT_A] = a[b, j]; packed[b, OUT_A+j] = b[b, j]."""
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
    """a[b, j] = packed[b, j]; b[b, j] = packed[b, OUT_A + j]."""
    var idx = Int(global_idx.x)
    var total = BATCH * (OUT_A + OUT_B)
    if idx < total:
        var bi = idx // (OUT_A + OUT_B)
        var ji = idx % (OUT_A + OUT_B)
        if ji < OUT_A:
            a[bi, ji] = rebind[Scalar[DT]](packed[bi, ji])
        else:
            b[bi, ji - OUT_A] = rebind[Scalar[DT]](packed[bi, ji])


# ──────────────────────────────────────────────────────────────────────────
# Parallel — owns A and B + four scratch slabs.
# ──────────────────────────────────────────────────────────────────────────


struct Parallel[A: Module, B: Module](Module):
    comptime IN_DIM = Self.A.IN_DIM
    comptime OUT_DIM = Self.A.OUT_DIM + Self.B.OUT_DIM
    comptime OUT_A = Self.A.OUT_DIM
    comptime OUT_B = Self.B.OUT_DIM

    var branch_a: Self.A
    var branch_b: Self.B
    var ctx: Optional[DeviceContext]

    # Scratch slabs (4 each side, lazy-grown to BATCH × shape).
    var out_a_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]   # BATCH × OUT_A
    var out_b_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]   # BATCH × OUT_B
    var gi_a_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin]   # BATCH × IN_DIM
    var gi_b_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin]   # BATCH × IN_DIM

    var out_a_dev: Optional[DeviceBuffer[DT]]
    var out_b_dev: Optional[DeviceBuffer[DT]]
    var gi_a_dev:  Optional[DeviceBuffer[DT]]
    var gi_b_dev:  Optional[DeviceBuffer[DT]]
    var scratch_n_batch: Int   # current capacity (BATCH)

    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        comptime assert (
            Self.A.IN_DIM == Self.B.IN_DIM
        ), "Parallel requires A.IN_DIM == B.IN_DIM"
        self.branch_a = Self.A()
        self.branch_b = Self.B()
        self.ctx = None
        self.out_a_cpu = alloc[Scalar[DT]](1)
        self.out_b_cpu = alloc[Scalar[DT]](1)
        self.gi_a_cpu  = alloc[Scalar[DT]](1)
        self.gi_b_cpu  = alloc[Scalar[DT]](1)
        self.out_a_dev = None
        self.out_b_dev = None
        self.gi_a_dev  = None
        self.gi_b_dev  = None
        self.scratch_n_batch = 0
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "Parallel.make[target='gpu', INIT] requires a DeviceContext"
        var p = Self()
        p.branch_a = Self.A.make[target, INIT]()
        p.branch_b = Self.B.make[target, INIT]()
        p._target_tag = TARGET_CPU
        return p^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "Parallel.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var p = Self()
        p.branch_a = Self.A.make[target, INIT](ctx)
        p.branch_b = Self.B.make[target, INIT](ctx)
        p.ctx = ctx
        p.out_a_dev = ctx.enqueue_create_buffer[DT](1)
        p.out_b_dev = ctx.enqueue_create_buffer[DT](1)
        p.gi_a_dev  = ctx.enqueue_create_buffer[DT](1)
        p.gi_b_dev  = ctx.enqueue_create_buffer[DT](1)
        p._target_tag = TARGET_GPU
        return p^

    def __del__(deinit self):
        self.out_a_cpu.free()
        self.out_b_cpu.free()
        self.gi_a_cpu.free()
        self.gi_b_cpu.free()

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Parallel: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_scratch_cpu(mut self, batch: Int):
        if self.scratch_n_batch < batch:
            self.out_a_cpu.free()
            self.out_b_cpu.free()
            self.gi_a_cpu.free()
            self.gi_b_cpu.free()
            self.out_a_cpu = alloc[Scalar[DT]](batch * Self.OUT_A)
            self.out_b_cpu = alloc[Scalar[DT]](batch * Self.OUT_B)
            self.gi_a_cpu  = alloc[Scalar[DT]](batch * Self.IN_DIM)
            self.gi_b_cpu  = alloc[Scalar[DT]](batch * Self.IN_DIM)
            self.scratch_n_batch = batch

    def _ensure_scratch_gpu(mut self, batch: Int) raises:
        if self.scratch_n_batch < batch:
            var c = self.ctx.value()
            self.out_a_dev = c.enqueue_create_buffer[DT](batch * Self.OUT_A)
            self.out_b_dev = c.enqueue_create_buffer[DT](batch * Self.OUT_B)
            self.gi_a_dev  = c.enqueue_create_buffer[DT](batch * Self.IN_DIM)
            self.gi_b_dev  = c.enqueue_create_buffer[DT](batch * Self.IN_DIM)
            self.scratch_n_batch = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout, LOUT: TensorLayout,
        OIN: MutOrigin,    OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_scratch_cpu(BATCH)
            var out_a = TileTensor(self.out_a_cpu, row_major[BATCH, Self.OUT_A]())
            var out_b = TileTensor(self.out_b_cpu, row_major[BATCH, Self.OUT_B]())
            self.branch_a.forward[target, BATCH, POLICY=POLICY](input, out_a)
            self.branch_b.forward[target, BATCH, POLICY=POLICY](input, out_b)
            # Concat into output: [out_a | out_b].
            for b in range(BATCH):
                for j in range(Self.OUT_A):
                    output[b, j] = out_a[b, j]
                for j in range(Self.OUT_B):
                    output[b, Self.OUT_A + j] = out_b[b, j]
        else:
            self._ensure_scratch_gpu(BATCH)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var pa: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_a_dev.value().unsafe_ptr()
            var pb: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_b_dev.value().unsafe_ptr()
            var out_a_tt = TileTensor(pa, row_major[BATCH, Self.OUT_A]())
            var out_b_tt = TileTensor(pb, row_major[BATCH, Self.OUT_B]())
            self.branch_a.forward[target, BATCH, POLICY=POLICY](input, out_a_tt)
            self.branch_b.forward[target, BATCH, POLICY=POLICY](input, out_b_tt)
            # Concat into packed output.
            comptime layout_a = Layout.row_major(BATCH, Self.OUT_A)
            comptime layout_b = Layout.row_major(BATCH, Self.OUT_B)
            comptime layout_p = Layout.row_major(BATCH, Self.OUT_DIM)
            var a_lt = LayoutTensor[DT, layout_a, MutAnyOrigin](self.out_a_dev.value())
            var b_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](self.out_b_dev.value())
            var p_lt = LayoutTensor[DT, layout_p, MutAnyOrigin](output_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _parallel_concat_kernel[BATCH, Self.OUT_A, Self.OUT_B]
            self.ctx.value().enqueue_function[kernel](
                a_lt, b_lt, p_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout, LGI: TensorLayout,
        OGO: MutOrigin,    OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_scratch_cpu(BATCH)
            # Split grad_output into out_a_cpu / out_b_cpu (reusing those
            # buffers as grad-output scratch on backward).
            var go_a = TileTensor(self.out_a_cpu, row_major[BATCH, Self.OUT_A]())
            var go_b = TileTensor(self.out_b_cpu, row_major[BATCH, Self.OUT_B]())
            for b in range(BATCH):
                for j in range(Self.OUT_A):
                    go_a[b, j] = grad_output[b, j]
                for j in range(Self.OUT_B):
                    go_b[b, j] = grad_output[b, Self.OUT_A + j]
            # Each branch writes grad_input into its own scratch.
            var gi_a = TileTensor(self.gi_a_cpu, row_major[BATCH, Self.IN_DIM]())
            var gi_b = TileTensor(self.gi_b_cpu, row_major[BATCH, Self.IN_DIM]())
            self.branch_a.backward[target, BATCH, POLICY=POLICY](go_a, gi_a)
            self.branch_b.backward[target, BATCH, POLICY=POLICY](go_b, gi_b)
            # Sum: grad_input = gi_a + gi_b.
            for b in range(BATCH):
                for d in range(Self.IN_DIM):
                    grad_input[b, d] = gi_a[b, d] + gi_b[b, d]
        else:
            self._ensure_scratch_gpu(BATCH)
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](grad_output)
            var grad_input_w  = rebind[TileTensor[DT, LGI, MutAnyOrigin]](grad_input)
            var pa: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_a_dev.value().unsafe_ptr()
            var pb: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.out_b_dev.value().unsafe_ptr()
            var pia: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.gi_a_dev.value().unsafe_ptr()
            var pib: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.gi_b_dev.value().unsafe_ptr()

            # Split grad_output → out_a / out_b scratches.
            comptime layout_a = Layout.row_major(BATCH, Self.OUT_A)
            comptime layout_b = Layout.row_major(BATCH, Self.OUT_B)
            comptime layout_p = Layout.row_major(BATCH, Self.OUT_DIM)
            var p_lt = LayoutTensor[DT, layout_p, MutAnyOrigin](grad_output_w.ptr)
            var a_lt = LayoutTensor[DT, layout_a, MutAnyOrigin](self.out_a_dev.value())
            var b_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](self.out_b_dev.value())
            comptime TPB = 128
            comptime n_blocks_split = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime split_kernel = _parallel_split_kernel[BATCH, Self.OUT_A, Self.OUT_B]
            self.ctx.value().enqueue_function[split_kernel](
                p_lt, a_lt, b_lt,
                grid_dim=n_blocks_split, block_dim=TPB,
            )

            # Per-branch backward into gi_a / gi_b scratches.
            var go_a_tt = TileTensor(pa, row_major[BATCH, Self.OUT_A]())
            var go_b_tt = TileTensor(pb, row_major[BATCH, Self.OUT_B]())
            var gi_a_tt = TileTensor(pia, row_major[BATCH, Self.IN_DIM]())
            var gi_b_tt = TileTensor(pib, row_major[BATCH, Self.IN_DIM]())
            self.branch_a.backward[target, BATCH, POLICY=POLICY](go_a_tt, gi_a_tt)
            self.branch_b.backward[target, BATCH, POLICY=POLICY](go_b_tt, gi_b_tt)

            # Sum: grad_input = gi_a + gi_b.
            comptime layout_in = Layout.row_major(BATCH, Self.IN_DIM)
            var gi_a_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](self.gi_a_dev.value())
            var gi_b_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](self.gi_b_dev.value())
            var gi_out_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](grad_input_w.ptr)
            comptime n_blocks_sum = (BATCH * Self.IN_DIM + TPB - 1) // TPB
            comptime sum_kernel = _elementwise_add_kernel[BATCH, Self.IN_DIM]
            self.ctx.value().enqueue_function[sum_kernel](
                gi_a_lt, gi_b_lt, gi_out_lt,
                grid_dim=n_blocks_sum, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        self.branch_a.for_each_param[target](prefix + sep + "a", visitor)
        self.branch_b.for_each_param[target](prefix + sep + "b", visitor)

    def set_inference(mut self, value: Bool):
        self._inference = value
        self.branch_a.set_inference(value)
        self.branch_b.set_inference(value)
