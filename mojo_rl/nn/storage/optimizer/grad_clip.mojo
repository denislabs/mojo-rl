"""Global grad-norm clipping over a Module (storage surface, CPU + GPU).

Two-pass walk via `model.for_each_param` (the storage ParamVisitor):
  1. accumulate ‖grad‖² across every Param,
  2. if ‖grad‖ > max_norm, scale every grad in place by max_norm / ‖grad‖.
Returns the pre-clip norm (diagnostics). `max_norm <= 0` is a no-op.

Non-finite norm (any NaN/±inf grad) → scale 0 → every grad is hard-zeroed
(`scale == 0` sentinel; a multiply would leave NaN·0 = NaN), so the optimizer
step becomes a no-op rather than poisoning the moments.

CPU: pure host loops over `grad.data`. GPU: per-Param block-reduction kernel
(comptime `N` layout) writes the param's Σg² into a reusable [1] device scalar,
D2H-accumulated on the host; then a per-Param scale kernel applies the host
scalar. This is storage-clean (no UnsafePointer arrays, no runtime layouts) but
NOT CUDA-graph-capturable (the per-param D2H + host branch); the D2H-free
grouped version lands with the contiguous-arena optimizer (Phase D).

Caller convention: invoke AFTER all backward passes wrote into the params' grads
and BEFORE the optimizer update. Per-optimizer (per-model) clipping — no
cross-model global norm (matches the deep_agents convention).
"""

from std.math import sqrt
from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module


comptime GC_TPB: Int = 128  # single-block reduction width


def _sum_sq_kernel[
    N: Int
](
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    out_sum: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    """Single-block GC_TPB-thread tree reduction of Σ grad²; thread 0 writes
    the total to `out_sum[0]`. (Same block.sum primitive LayerNorm uses.)"""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var g = rebind[Scalar[DT]](grad[k])
        my_sum += g * g
        k += GC_TPB
    var total = block.sum[block_size=GC_TPB, broadcast=False](val=my_sum)
    if t == 0:
        out_sum[0] = total[0]


def _scale_kernel[
    N: Int
](
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    scale: Scalar[DT],
):
    """`grad[i] *= scale`, one thread per element. `scale == 0` hard-writes 0
    (non-finite-norm sentinel)."""
    var i = Int(global_idx.x)
    if i < N:
        var s = scale
        grad[i] = rebind[Scalar[DT]](grad[i]) * s if s != Scalar[DT](
            0.0
        ) else Scalar[DT](0.0)


struct _SumSqCPU(ParamVisitor):
    var sum_sq: Scalar[DT]

    def __init__(out self):
        self.sum_sq = Scalar[DT](0.0)

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            var g = grad.data[i]
            self.sum_sq += g * g


struct _ScaleCPU(ParamVisitor):
    var scale: Scalar[DT]

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.scale == Scalar[DT](0.0):
            for i in range(N):
                grad.data[i] = Scalar[DT](0.0)
        else:
            for i in range(N):
                grad.data[i] = grad.data[i] * self.scale


struct _SumSqGPU(ParamVisitor):
    var sum_sq: Scalar[DT]  # host accumulator across params
    var psum: Tensor  # reusable [1] device scalar

    def __init__(out self):
        self.sum_sq = Scalar[DT](0.0)
        self.psum = Tensor()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var c = ctx.value()
        self.psum.ensure_gpu(c, 1)
        comptime lg = Layout.row_major(N)
        c.enqueue_function[_sum_sq_kernel[N]](
            grad.lt["gpu", lg](),
            self.psum.lt["gpu", Layout.row_major(1)](),
            grid_dim=1,
            block_dim=GC_TPB,
        )
        self.psum.download(c)
        self.sum_sq += self.psum.data[0]


struct _ScaleGPU(ParamVisitor):
    var scale: Scalar[DT]

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var c = ctx.value()
        comptime lg = Layout.row_major(N)
        comptime nblk = (N + TPB - 1) // TPB
        c.enqueue_function[_scale_kernel[N]](
            grad.lt["gpu", lg](),
            self.scale,
            grid_dim=nblk,
            block_dim=TPB,
        )


def _scale_from_norm(
    norm: Scalar[DT], max_norm: Scalar[DT], eps: Scalar[DT]
) -> Scalar[DT]:
    """`min(1, max_norm / max(norm, eps))`; non-finite norm → 0 (NaN guard)."""
    if norm - norm != Scalar[DT](0.0):  # True iff norm is non-finite
        return Scalar[DT](0.0)
    var denom = norm if norm > eps else eps
    var ratio = max_norm / denom
    return ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)


def clip_grad_norm[
    target: StaticString, M: Module
](
    mut model: M,
    max_norm: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
    eps: Scalar[DT] = 1e-6,
) raises -> Scalar[DT]:
    """Clip the global L2 norm of `model`'s grads to `max_norm` in place.
    Returns the pre-clip norm. `max_norm <= 0` → no clip (norm still returned)."""
    var norm: Scalar[DT]
    comptime if target == "cpu":
        var ss = _SumSqCPU()
        model.for_each_param[target](ss, ctx)
        norm = sqrt(ss.sum_sq)
        if max_norm > Scalar[DT](0.0):
            var sc = _ScaleCPU(_scale_from_norm(norm, max_norm, eps))
            if sc.scale < Scalar[DT](1.0):  # scale==1 → no-op skip
                model.for_each_param[target](sc, ctx)
    else:
        var ss = _SumSqGPU()
        model.for_each_param[target](ss, ctx)
        norm = sqrt(ss.sum_sq)
        if max_norm > Scalar[DT](0.0):
            var scale = _scale_from_norm(norm, max_norm, eps)
            if scale < Scalar[DT](1.0):
                var sc = _ScaleGPU(scale)
                model.for_each_param[target](sc, ctx)
    return norm
