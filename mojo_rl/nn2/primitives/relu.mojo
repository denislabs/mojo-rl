"""ReLU[DIM] — element-wise rectified linear unit. Phase 2.4: target is
a comptime method param.

Parameterless. Caches input on forward, masks grad by sign(cache) on
backward. At x == 0 the gradient is 0 (matches PyTorch).
"""

from std.gpu import global_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

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


def _relu_forward_kernel[
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
        cache[b, d] = x
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


# ──────────────────────────────────────────────────────────────────────────
# ReLU — method-level target.
# ──────────────────────────────────────────────────────────────────────────


struct ReLU[DIM: Int](Module):
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
        """CPU factory. INIT is ignored (ReLU is parameterless) but accepted
        for uniformity so Sequential.make[target, INIT] can recurse."""
        comptime assert (
            target == "cpu"
        ), "ReLU.make[target='gpu', INIT] requires a DeviceContext"
        var r = Self()
        r._target_tag = TARGET_CPU
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
        r.cache_dev = ctx.enqueue_create_buffer[DT](1)
        r.cache_dev_n = 0
        r.ctx = ctx
        r._target_tag = TARGET_GPU
        return r^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "ReLU: method called with [target='"
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
        # ReLU is element-wise; POLICY is accepted for trait conformance
        # but the implementation stays in DT (fp32). AMPPolicy never
        # downgrades element-wise ops.
        comptime assert (
            input.flat_rank == 2
        ), "input must be rank-2 [BATCH, DIM]"
        comptime assert (
            output.flat_rank == 2
        ), "output must be rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # Phase 8.0: SIMD path. Mojo nightly does NOT autovectorize the
            # scalar `[b, d]` form — manual `load[width=W]` is 3-5x faster.
            self._ensure_cache_cpu(BATCH)
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var cache_p = self.cache.unsafe_ptr()
            var zero_v = SIMD[DT, CPU_SIMD_W](0)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.load[width=CPU_SIMD_W](k)
                cache_p.store(k, v)
                out_p.store(k, v.gt(zero_v).select(v, zero_v))
                k += CPU_SIMD_W
            while k < N:
                var v = in_p[k]
                cache_p[k] = v
                out_p[k] = v if v > 0 else 0
                k += 1
        else:
            self._ensure_cache_gpu(BATCH * Self.DIM)
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var in_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var input_lt = LayoutTensor[DT, layout, MutAnyOrigin](in_ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _relu_forward_kernel[BATCH, Self.DIM]
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
        POLICY: AMPPolicy = NoAMP,
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
        self._assert_tag[target]()

        comptime if target == "cpu":
            # Phase 8.0: SIMD path. grad_input = grad_output where cache > 0.
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var c_p = self.cache.unsafe_ptr()
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
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _relu_backward_kernel[BATCH, Self.DIM]
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
        POLICY: AMPPolicy = NoAMP,
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
        # No params — backward_input is identical to backward.
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        # ReLU has no parameters — nothing to visit.
        pass

    def set_inference(mut self, value: Bool):
        # ReLU forward is deterministic — flag stored for trait
        # conformance but has no behavioral effect.
        self._inference = value
