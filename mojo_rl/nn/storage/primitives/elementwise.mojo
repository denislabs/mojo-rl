"""Elementwise[DIM, OP] — generic elementwise activation on the storage surface.

The storage-surface twin of legacy `nn.primitives.Elementwise[DIM, OP]`. Reuses
the legacy `ElementOp` trait + `ops/` structs VERBATIM (they depend only on
`DT`), so the per-lane math is bit-identical. Every concrete activation is a
one-line alias:

    comptime ReLU    = Elementwise[DIM, ReLUOp]      # see leaves below
    comptime Tanh    = Elementwise[DIM, TanhOp]
    comptime Sigmoid = Elementwise[DIM, SigmoidOp]

KEY simplification vs legacy: the storage `vjp` receives `forward_input` (x)
explicitly (invariant §3.1), so there is NO cache field and NO
`_cached_input_ptr` alias. For output-cache ops (`owns_cache=True`, e.g. Tanh)
backward recomputes `y = OP.forward(x)` then `OP.backward(y, go)` — bit-
identical to having cached `y`, because `y` is a pure function of `x`. For
input-cache ops (`owns_cache=False`, e.g. ReLU) backward is `OP.backward(x, go)`.

CPU uses the SIMD ops over a tracked `.data.unsafe_ptr()` (origin = the list,
NOT the wildcard); GPU uses one kernel per direction parameterised on `OP`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, CPU_SIMD_W, TPB
from mojo_rl.nn.core.element_op import ElementOp
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer


# ── GPU kernels (OP supplies the math via comptime) ─────────────────────
def _ew_fwd_kernel[
    M: Int, OP: ElementOp
](
    x: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < M:
        o[i] = OP.forward_scalar(rebind[Scalar[DT]](x[i]))


def _ew_bwd_kernel[
    M: Int, OP: ElementOp
](
    x: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    gi: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < M:
        var xv = rebind[Scalar[DT]](x[i])
        var gov = rebind[Scalar[DT]](go[i])
        comptime if OP.owns_cache:
            # output-cache op: recompute y = f(x), then gi = f'(y)·go.
            gi[i] = OP.backward_scalar(OP.forward_scalar(xv), gov)
        else:
            gi[i] = OP.backward_scalar(xv, gov)


struct Elementwise[DIM_: Int, OP: ElementOp](Module):
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
        comptime M = B * Self.DIM_
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(M)
            var xp = in0.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= M:
                op.store(
                    k,
                    Self.OP.forward_simd[CPU_SIMD_W](
                        xp.load[width=CPU_SIMD_W](k)
                    ),
                )
                k += CPU_SIMD_W
            while k < M:
                op[k] = Self.OP.forward_scalar(xp[k])
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_ew_fwd_kernel[M, Self.OP]](
                in0.lt["gpu", Layout.row_major(M)](),
                out.lt["gpu", Layout.row_major(M)](),
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
        comptime M = B * Self.DIM_
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(M)
            var xp = fin.data.unsafe_ptr()
            var gp = grad_output.data.unsafe_ptr()
            var ip = gin.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= M:
                var xv = xp.load[width=CPU_SIMD_W](k)
                var gv = gp.load[width=CPU_SIMD_W](k)
                comptime if Self.OP.owns_cache:
                    ip.store(
                        k,
                        Self.OP.backward_simd[CPU_SIMD_W](
                            Self.OP.forward_simd[CPU_SIMD_W](xv), gv
                        ),
                    )
                else:
                    ip.store(k, Self.OP.backward_simd[CPU_SIMD_W](xv, gv))
                k += CPU_SIMD_W
            while k < M:
                comptime if Self.OP.owns_cache:
                    ip[k] = Self.OP.backward_scalar(
                        Self.OP.forward_scalar(xp[k]), gp[k]
                    )
                else:
                    ip[k] = Self.OP.backward_scalar(xp[k], gp[k])
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_ew_bwd_kernel[M, Self.OP]](
                fin.lt["gpu", Layout.row_major(M)](),
                grad_output.lt["gpu", Layout.row_major(M)](),
                gin.lt["gpu", Layout.row_major(M)](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
