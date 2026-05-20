"""Tanh[DIM] — element-wise hyperbolic tangent. Phase 5.1.

Same shape as ReLU. Cache stores the **output** (y = tanh(x)) rather
than the input, since the backward derivative is `1 - y^2` — saves a
re-evaluation of tanh on backward.

Like ReLU, Tanh is parameterless and ignores POLICY (element-wise op,
runs in DT — bf16 tanh is mantissa-ugly and saves nothing in practice).
"""

from std.math import tanh
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────


def _tanh_forward_kernel[
    BATCH: Int,
    DIM: Int,
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
        var y = tanh(x)
        output[b, d] = y
        cache[b, d] = y


def _tanh_backward_kernel[
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
        var y = rebind[Scalar[DT]](cache[b, d])
        var go = rebind[Scalar[DT]](grad_output[b, d])
        var one: Scalar[DT] = 1.0
        grad_input[b, d] = go * (one - y * y)


# ──────────────────────────────────────────────────────────────────────────
# Tanh — method-level target.
# ──────────────────────────────────────────────────────────────────────────


struct Tanh[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var cache: List[Scalar[DT]]
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int
    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.cache = List[Scalar[DT]]()
        self.cache_dev = None
        self.cache_dev_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. INIT ignored (Tanh is parameterless)."""
        comptime assert (
            target == "cpu"
        ), "Tanh.make[target='gpu', INIT] requires a DeviceContext"
        var t = Self()
        t._target_tag = TARGET_CPU
        return t^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert (
            target == "gpu"
        ), "Tanh.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var t = Self()
        t.cache_dev = ctx.enqueue_create_buffer[DT](1)
        t.cache_dev_n = 0
        t.ctx = ctx
        t._target_tag = TARGET_GPU
        return t^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Tanh: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag="
                + String(Int(self._target_tag))
                + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.DIM
        if len(self.cache) < needed:
            self.cache.resize(needed, 0.0)

    def _ensure_cache_gpu(mut self, needed: Int) raises:
        if self.cache_dev_n < needed:
            self.cache_dev = self.ctx.value().enqueue_create_buffer[DT](needed)
            self.cache_dev_n = needed

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert (
            input.flat_rank == 2
        ), "input must be rank-2 [BATCH, DIM]"
        comptime assert (
            output.flat_rank == 2
        ), "output must be rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # Phase 8.0: SIMD path. `tanh` is lane-wise on SIMD.
            self._ensure_cache_cpu(BATCH)
            var input_w = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var in_p = input_w.ptr
            var out_p = output_w.ptr
            var cache_p = self.cache.unsafe_ptr()
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.load[width=CPU_SIMD_W](k)
                var t = tanh(v)
                cache_p.store(k, t)
                out_p.store(k, t)
                k += CPU_SIMD_W
            while k < N:
                var y = tanh(in_p[k])
                cache_p[k] = y
                out_p[k] = y
                k += 1
        else:
            self._ensure_cache_gpu(BATCH * Self.DIM)
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var input_w = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var input_lt = LayoutTensor[DT, layout, MutAnyOrigin](input_w.ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](output_w.ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _tanh_forward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                input_lt,
                output_lt,
                cache_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # Phase 8.0: SIMD path. grad_in = grad_out * (1 - y^2).
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            var go_p = grad_output_w.ptr
            var gi_p = grad_input_w.ptr
            var c_p = self.cache.unsafe_ptr()
            var one_v = SIMD[DT, CPU_SIMD_W](1)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var y = c_p.load[width=CPU_SIMD_W](k)
                var g = go_p.load[width=CPU_SIMD_W](k)
                gi_p.store(k, g * (one_v - y * y))
                k += CPU_SIMD_W
            while k < N:
                var y = c_p[k]
                gi_p[k] = go_p[k] * (1.0 - y * y)
                k += 1
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                grad_output_w.ptr
            )
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](grad_input_w.ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _tanh_backward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                go_lt,
                cache_lt,
                gi_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        # No params — backward_input is identical to backward.
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        # Tanh has no parameters — nothing to visit.
        pass

    def set_inference(mut self, value: Bool):
        # Tanh forward is deterministic — flag stored for trait
        # conformance but has no behavioral effect.
        self._inference = value
