"""ReLU[DIM] — element-wise rectified linear unit. Phase 2.4: target is
a comptime method param.

Parameterless. Caches input on forward, masks grad by sign(cache) on
backward. At x == 0 the gradient is 0 (matches PyTorch).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Initializer,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────

def _relu_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var x = input.ptr[idx]
        cache.ptr[idx] = x
        var zero: Scalar[DT] = 0.0
        output.ptr[idx] = x if x > zero else zero


def _relu_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var zero: Scalar[DT] = 0.0
        grad_input.ptr[idx] = (
            grad_output.ptr[idx] if cache.ptr[idx] > zero else zero
        )


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

    def __init__(out self):
        self.cache = List[Scalar[DT]]()
        self.cache_dev = None
        self.cache_dev_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. INIT is ignored (ReLU is parameterless) but accepted
        for uniformity so Sequential.make[target, INIT] can recurse."""
        comptime assert target == "cpu", (
            "ReLU.make[target='gpu', INIT] requires a DeviceContext"
        )
        var r = Self()
        r._target_tag = TARGET_CPU
        return r^

    @staticmethod
    def make[target: StaticString, INIT: Initializer](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert target == "gpu", (
            "ReLU.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
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
                "ReLU: method called with [target='" + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
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
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank  == 2, "input must be rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            var cache = TileTensor(self.cache, row_major[BATCH, Self.DIM]())
            var zero: Scalar[DT] = 0.0
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var x = input[b, d]
                    cache[b, d] = x
                    output[b, d] = x if x > zero else zero
        else:
            self._ensure_cache_gpu(BATCH * Self.DIM)
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var input_w  = rebind[TileTensor[DT, LIN,  MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var input_lt  = LayoutTensor[DT, layout, MutAnyOrigin](input_w.ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](output_w.ptr)
            var cache_lt  = LayoutTensor[DT, layout, MutAnyOrigin](self.cache_dev.value())
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _relu_forward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                input_lt, output_lt, cache_lt, grid_dim=n_blocks, block_dim=TPB,
            )

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var cache = TileTensor(self.cache, row_major[BATCH, Self.DIM]())
            var zero: Scalar[DT] = 0.0
            for b in range(BATCH):
                for d in range(Self.DIM):
                    grad_input[b, d] = (
                        grad_output[b, d] if cache[b, d] > zero else zero
                    )
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](grad_output)
            var grad_input_w  = rebind[TileTensor[DT, LGI, MutAnyOrigin]](grad_input)
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](grad_output_w.ptr)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](grad_input_w.ptr)
            var cache_lt = LayoutTensor[DT, layout, MutAnyOrigin](self.cache_dev.value())
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _relu_backward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                go_lt, cache_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        prefix: String,
        mut visitor: V,
    ) raises:
        self._assert_tag[target]()
        # ReLU has no parameters — nothing to visit.
        pass
