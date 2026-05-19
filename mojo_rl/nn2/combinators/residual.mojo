"""Residual[Inner] — y = inner(x) + x. Phase 5.5.

Constraint: `Inner.IN_DIM == Inner.OUT_DIM` (enforced at comptime).

Forward:  output[b, d] = inner(input)[b, d] + input[b, d]
Backward: grad_input[b, d] = inner.backward(grad_output)[b, d]
                            + grad_output[b, d]

Both flow grad_output through `Inner.backward` AND directly through
the identity branch — the residual sum's local Jacobian is `I + J_inner`.

Persistent slab: one BATCH × DIM scratch buffer, reused between forward
(holds `inner(x)` before the residual sum) and backward (holds
`inner.backward(grad_y)` before the +grad_y). The two uses don't overlap
in time so a single slab suffices.

Threads POLICY through to Inner. Element-wise add is fp32 regardless
of POLICY (matches LayerNorm/CE force_fp32_input convention).
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


# ──────────────────────────────────────────────────────────────────────────
# Element-wise add kernel: out[i] = a[i] + b[i].
# Used by both forward (mid + input) and backward (inner-grad + grad_out).
# ──────────────────────────────────────────────────────────────────────────


def _elementwise_add_kernel[
    BATCH: Int, DIM: Int,
](
    a: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    # `dst` not `out` — `out` is reserved as an init-param keyword in Mojo.
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var bi = idx // DIM
        var di = idx % DIM
        dst[bi, di] = rebind[Scalar[DT]](a[bi, di]) + rebind[Scalar[DT]](b[bi, di])


# ──────────────────────────────────────────────────────────────────────────
# Residual — owns one child + one slab.
# ──────────────────────────────────────────────────────────────────────────


struct Residual[Inner: Module](Module):
    comptime IN_DIM = Self.Inner.IN_DIM
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner
    var ctx: Optional[DeviceContext]

    # Single persistent scratch slab (BATCH × DIM). Lazy-grown like
    # Sequential's `mid_cpu` / `mid_dev`.
    var mid_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mid_dev: Optional[DeviceBuffer[DT]]
    var mid_cap: Int

    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        """Defaultable form — empty placeholders + UNINIT tag.

        Required so `Residual[Inner]` itself is `Defaultable` and can
        be nested inside a `Sequential[..., Residual[...], ...]`.
        """
        comptime assert (
            Self.Inner.IN_DIM == Self.Inner.OUT_DIM
        ), "Residual requires Inner.IN_DIM == Inner.OUT_DIM"
        self.inner = Self.Inner()
        self.ctx = None
        self.mid_cpu = alloc[Scalar[DT]](1)
        self.mid_dev = None
        self.mid_cap = 0
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "Residual.make[target='gpu', INIT] requires a DeviceContext"
        var r = Self()
        r.inner = Self.Inner.make[target, INIT]()
        r._target_tag = TARGET_CPU
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "Residual.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var r = Self()
        r.inner = Self.Inner.make[target, INIT](ctx)
        r.ctx = ctx
        r.mid_dev = ctx.enqueue_create_buffer[DT](1)
        r._target_tag = TARGET_GPU
        return r^

    def __del__(deinit self):
        self.mid_cpu.free()

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Residual: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_mid_cpu(mut self, needed: Int):
        if self.mid_cap < needed:
            self.mid_cpu.free()
            self.mid_cpu = alloc[Scalar[DT]](needed)
            self.mid_cap = needed

    def _ensure_mid_gpu(mut self, needed: Int) raises:
        if self.mid_cap < needed:
            self.mid_dev = self.ctx.value().enqueue_create_buffer[DT](needed)
            self.mid_cap = needed

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
            self._ensure_mid_cpu(BATCH * Self.IN_DIM)
            var mid = TileTensor(self.mid_cpu, row_major[BATCH, Self.IN_DIM]())
            self.inner.forward[target, BATCH, POLICY=POLICY](input, mid)
            for b in range(BATCH):
                for d in range(Self.IN_DIM):
                    output[b, d] = mid[b, d] + input[b, d]
        else:
            self._ensure_mid_gpu(BATCH * Self.IN_DIM)
            # Launder pointers through MutAnyOrigin (aliasing-analyzer
            # — same trick as Sequential's _forward_gpu).
            var input_w  = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.IN_DIM]())
            self.inner.forward[target, BATCH, POLICY=POLICY](input, mid)
            # output = mid + input
            comptime layout = Layout.row_major(BATCH, Self.IN_DIM)
            var mid_lt = LayoutTensor[DT, layout, MutAnyOrigin](self.mid_dev.value())
            var in_lt  = LayoutTensor[DT, layout, MutAnyOrigin](input_w.ptr)
            var out_lt = LayoutTensor[DT, layout, MutAnyOrigin](output_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.IN_DIM + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.IN_DIM]
            self.ctx.value().enqueue_function[kernel](
                mid_lt, in_lt, out_lt,
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
            self._ensure_mid_cpu(BATCH * Self.IN_DIM)
            var tmp = TileTensor(self.mid_cpu, row_major[BATCH, Self.IN_DIM]())
            self.inner.backward[target, BATCH, POLICY=POLICY](grad_output, tmp)
            for b in range(BATCH):
                for d in range(Self.IN_DIM):
                    grad_input[b, d] = tmp[b, d] + grad_output[b, d]
        else:
            self._ensure_mid_gpu(BATCH * Self.IN_DIM)
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](grad_output)
            var grad_input_w  = rebind[TileTensor[DT, LGI, MutAnyOrigin]](grad_input)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var tmp = TileTensor(mp, row_major[BATCH, Self.IN_DIM]())
            self.inner.backward[target, BATCH, POLICY=POLICY](grad_output, tmp)
            # grad_input = tmp + grad_output
            comptime layout = Layout.row_major(BATCH, Self.IN_DIM)
            var tmp_lt = LayoutTensor[DT, layout, MutAnyOrigin](self.mid_dev.value())
            var go_lt  = LayoutTensor[DT, layout, MutAnyOrigin](grad_output_w.ptr)
            var gi_lt  = LayoutTensor[DT, layout, MutAnyOrigin](grad_input_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.IN_DIM + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.IN_DIM]
            self.ctx.value().enqueue_function[kernel](
                tmp_lt, go_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        # The residual identity branch has no params — only Inner does.
        # Names: "prefix.inner.<inner-leaf-name>".
        self.inner.for_each_param[target](prefix + sep + "inner", visitor)

    def set_inference(mut self, value: Bool):
        self._inference = value
        self.inner.set_inference(value)
