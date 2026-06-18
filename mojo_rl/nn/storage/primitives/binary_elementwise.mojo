"""BinaryElementwise[DIM, OP] — generic 2-input elementwise op (storage surface).

The binary twin of `Elementwise`. Reuses the legacy `BinaryElementOp` trait +
`ops/` structs VERBATIM (DT-only), so the per-lane math is bit-identical. The
storage `vjp` gets BOTH inputs as `forward_input`, so the per-element carry
(`cache`) is RECOMPUTED in backward (`OP.cache_scalar(x,y)`) — no cache field.

    comptime BinaryElemMin[DIM] = BinaryElementwise[DIM, BinaryElemMinOp]   # SAC twin-min
    comptime BinarySub[DIM]     = BinaryElementwise[DIM, BinarySubOp]

Backward: c = OP.cache(x,y) ; gi0 = OP.backward_x(c, go) ; gi1 = OP.backward_y(c, go).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, CPU_SIMD_W, TPB
from mojo_rl.nn.core.binary_element_op import BinaryElementOp
from mojo_rl.nn.primitives.ops.binary_elem_min_op import BinaryElemMinOp
from mojo_rl.nn.primitives.ops.binary_sub_op import BinarySubOp
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor


def _be_fwd_kernel[
    M: Int, OP: BinaryElementOp
](
    a: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < M:
        o[i] = OP.forward_scalar(
            rebind[Scalar[DT]](a[i]), rebind[Scalar[DT]](b[i])
        )


def _be_bwd_kernel[
    M: Int, OP: BinaryElementOp
](
    a: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    gi0: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    gi1: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < M:
        var c = OP.cache_scalar(
            rebind[Scalar[DT]](a[i]), rebind[Scalar[DT]](b[i])
        )
        var g = rebind[Scalar[DT]](go[i])
        gi0[i] = OP.backward_scalar_x(c, g)
        gi1[i] = OP.backward_scalar_y(c, g)


struct BinaryElementwise[DIM_: Int, OP: BinaryElementOp](Module):
    comptime ARITY = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    def __init__(out self):
        pass

    @staticmethod
    def make_cpu() raises -> Self:
        return Self()

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref a = inputs[0]
        ref b = inputs[1]
        comptime if target == "cpu":
            out.ensure(M)
            var ap = a.data.unsafe_ptr()
            var bp = b.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= M:
                op.store(k, Self.OP.forward_simd[CPU_SIMD_W](
                    ap.load[width=CPU_SIMD_W](k), bp.load[width=CPU_SIMD_W](k)))
                k += CPU_SIMD_W
            while k < M:
                op[k] = Self.OP.forward_scalar(ap[k], bp[k])
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_be_fwd_kernel[M, Self.OP]](
                a.lt_gpu[Layout.row_major(M)](), b.lt_gpu[Layout.row_major(M)](),
                out.lt_gpu[Layout.row_major(M)](),
                grid_dim=nblk, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref a = forward_input[0]
        ref b = forward_input[1]
        ref gi0 = grad_inputs[0]
        ref gi1 = grad_inputs[1]
        comptime if target == "cpu":
            gi0.ensure(M)
            gi1.ensure(M)
            var ap = a.data.unsafe_ptr()
            var bp = b.data.unsafe_ptr()
            var gp = grad_output.data.unsafe_ptr()
            var g0 = gi0.data.unsafe_ptr()
            var g1 = gi1.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= M:
                var cc = Self.OP.cache_simd[CPU_SIMD_W](
                    ap.load[width=CPU_SIMD_W](k), bp.load[width=CPU_SIMD_W](k))
                var gv = gp.load[width=CPU_SIMD_W](k)
                g0.store(k, Self.OP.backward_simd_x[CPU_SIMD_W](cc, gv))
                g1.store(k, Self.OP.backward_simd_y[CPU_SIMD_W](cc, gv))
                k += CPU_SIMD_W
            while k < M:
                var cc = Self.OP.cache_scalar(ap[k], bp[k])
                g0[k] = Self.OP.backward_scalar_x(cc, gp[k])
                g1[k] = Self.OP.backward_scalar_y(cc, gp[k])
                k += 1
        else:
            var c = ctx.value()
            gi0.ensure_gpu(c, M)
            gi1.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_be_bwd_kernel[M, Self.OP]](
                a.lt_gpu[Layout.row_major(M)](), b.lt_gpu[Layout.row_major(M)](),
                grad_output.lt_gpu[Layout.row_major(M)](),
                gi0.lt_gpu[Layout.row_major(M)](), gi1.lt_gpu[Layout.row_major(M)](),
                grid_dim=nblk, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass


comptime BinaryElemMin[DIM: Int] = BinaryElementwise[DIM, BinaryElemMinOp]
comptime BinarySub[DIM: Int] = BinaryElementwise[DIM, BinarySubOp]
