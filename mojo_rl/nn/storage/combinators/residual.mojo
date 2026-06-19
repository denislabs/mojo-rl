"""Residual[Inner] — y = inner(x) + x (storage-passing, CPU + GPU).

The storage twin of legacy `nn.combinators.Residual`. Requires
`Inner.IN_DIMS[0] == Inner.OUT_DIM`. Owns one `mid` scratch Tensor: forward runs
`inner.forward → mid` then `out = mid + x` (SIMD on CPU, one add kernel on GPU);
vjp runs `inner.vjp → mid` (mid = grad wrt inner's input) then
`grad_input = mid + grad_output` (the identity skip's grad). The legacy
`TargetStorage`/`POLICY`/`mode`/name-keyed-param machinery is gone — the storage
surface (`ref`/`mut Tensor`, `TensorRefs`) handles both targets uniformly.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor


def _resid_add_kernel[
    N: Int
](
    a: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](a[i]) + rebind[Scalar[DT]](b[i])


struct Residual[Inner: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    comptime DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner
    var mid: Tensor  # inner's output (fwd) / inner's grad-input (bwd)

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Residual requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.inner = Self.Inner()
        self.mid = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        r.inner = Self.Inner.make[target, INIT](ctx)
        return r^

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        self.inner.forward[target, B](
            TensorRefs[Self.Inner.ARITY](in0), self.mid, ctx
        )
        comptime N = B * Self.DIM
        comptime if target == "cpu":
            out.ensure(N)
            var op = out.data.unsafe_ptr()
            var mp = self.mid.data.unsafe_ptr()
            var ip = in0.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= N:
                op.store(k, mp.load[width=W](k) + ip.load[width=W](k))
                k += W
            while k < N:
                op[k] = mp[k] + ip[k]
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            c.enqueue_function[_resid_add_kernel[N]](
                self.mid.lt["gpu", Layout.row_major(N)](),
                in0.lt["gpu", Layout.row_major(N)](),
                out.lt["gpu", Layout.row_major(N)](),
                grid_dim=(N + TPB - 1) // TPB,
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
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        # mid := grad wrt inner's input; then grad_input = mid + grad_output.
        self.inner.vjp[target, B](
            TensorRefs[Self.Inner.ARITY](fin),
            grad_output,
            TensorRefs[Self.Inner.ARITY](self.mid),
            ctx,
        )
        comptime N = B * Self.DIM
        comptime if target == "cpu":
            gin.ensure(N)
            var gp = gin.data.unsafe_ptr()
            var mp = self.mid.data.unsafe_ptr()
            var gop = grad_output.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= N:
                gp.store(k, mp.load[width=W](k) + gop.load[width=W](k))
                k += W
            while k < N:
                gp[k] = mp[k] + gop[k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            c.enqueue_function[_resid_add_kernel[N]](
                self.mid.lt["gpu", Layout.row_major(N)](),
                grad_output.lt["gpu", Layout.row_major(N)](),
                gin.lt["gpu", Layout.row_major(N)](),
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        self.inner.for_each_param[target](visitor, ctx)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        self.inner.for_each_state[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
