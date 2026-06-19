"""Flatten[DIM] — identity passthrough Module (storage surface).

Transformed from legacy `nn.primitives.Flatten`. From the Module's standpoint a
flatten is the identity: `[BATCH, DIM]` in == `[BATCH, DIM]` out (the shape
change is purely in the caller's view layout). No params, no cache; backward is
the same identity copy. Lets `Sequential[Conv2D, ReLU, …, Flatten, Linear, …]`
compose without orchestrator glue.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, CPU_SIMD_W, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer


def _flatten_copy_kernel[
    N: Int
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


struct Flatten[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            var sp = in0.data.unsafe_ptr()
            var dp = out.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= N:
                dp.store(k, sp.load[width=CPU_SIMD_W](k))
                k += CPU_SIMD_W
            while k < N:
                dp[k] = sp[k]
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_flatten_copy_kernel[N]](
                in0.lt["gpu", Layout.row_major(N)](),
                out.lt["gpu", Layout.row_major(N)](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            var sp = grad_output.data.unsafe_ptr()
            var dp = gin.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= N:
                dp.store(k, sp.load[width=CPU_SIMD_W](k))
                k += CPU_SIMD_W
            while k < N:
                dp[k] = sp[k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_flatten_copy_kernel[N]](
                grad_output.lt["gpu", Layout.row_major(N)](),
                gin.lt["gpu", Layout.row_major(N)](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass
