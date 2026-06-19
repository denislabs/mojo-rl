"""ProjectedResidual[Inner, Skip] — y = Inner(x) + Skip(x) (storage surface).

Like `Residual[Inner]` but the skip path is its own parameterised module (so it
can PROJECT the input to match `Inner.OUT_DIM` when shapes change) — the ResNet
downsampling block (`Inner` = 3×3-s2 → BN → ReLU → 3×3-s1 → BN main path,
`Skip` = 1×1-s2 → BN projection; the external ReLU on the sum is applied by
wrapping this in `Sequential[ProjectedResidual[...], ReLU]`).

Constraints: `Inner.IN_DIMS[0] == Skip.IN_DIMS[0]`, `Inner.OUT_DIM == Skip.OUT_DIM`.
Forward sums the two branch outputs; backward feeds BOTH branches the full
grad_output and sums their grad-inputs. (Branch vjps must not mutate grad_output
— true for all real main/skip paths, which end in BN/Conv/Linear, not a gate.)
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from .residual import _resid_add_kernel


struct ProjectedResidual[Inner: Module, Skip: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    comptime IN = Self.Inner.IN_DIMS[0]

    var inner: Self.Inner
    var skip: Self.Skip
    var inner_out: Tensor
    var skip_out: Tensor
    var gi_inner: Tensor
    var gi_skip: Tensor

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Skip.IN_DIMS[0]
        ), "ProjectedResidual requires Inner.IN_DIMS[0] == Skip.IN_DIMS[0]"
        comptime assert (
            Self.Inner.OUT_DIM == Self.Skip.OUT_DIM
        ), "ProjectedResidual requires Inner.OUT_DIM == Skip.OUT_DIM"
        self.inner = Self.Inner()
        self.skip = Self.Skip()
        self.inner_out = Tensor()
        self.skip_out = Tensor()
        self.gi_inner = Tensor()
        self.gi_skip = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        r.inner = Self.Inner.make[target, INIT](ctx)
        r.skip = Self.Skip.make[target, INIT](ctx)
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
            TensorRefs[Self.Inner.ARITY](in0), self.inner_out, ctx
        )
        self.skip.forward[target, B](
            TensorRefs[Self.Skip.ARITY](in0), self.skip_out, ctx
        )
        comptime N = B * Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(N)
            var op = out.data.unsafe_ptr()
            var ap = self.inner_out.data.unsafe_ptr()
            var bp = self.skip_out.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= N:
                op.store(k, ap.load[width=W](k) + bp.load[width=W](k))
                k += W
            while k < N:
                op[k] = ap[k] + bp[k]
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            c.enqueue_function[_resid_add_kernel[N]](
                self.inner_out.lt["gpu", Layout.row_major(N)](),
                self.skip_out.lt["gpu", Layout.row_major(N)](),
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
        self.inner.vjp[target, B](
            TensorRefs[Self.Inner.ARITY](fin),
            grad_output,
            TensorRefs[Self.Inner.ARITY](self.gi_inner),
            ctx,
        )
        self.skip.vjp[target, B](
            TensorRefs[Self.Skip.ARITY](fin),
            grad_output,
            TensorRefs[Self.Skip.ARITY](self.gi_skip),
            ctx,
        )
        comptime NIN = B * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            var gp = gin.data.unsafe_ptr()
            var ap = self.gi_inner.data.unsafe_ptr()
            var bp = self.gi_skip.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= NIN:
                gp.store(k, ap.load[width=W](k) + bp.load[width=W](k))
                k += W
            while k < NIN:
                gp[k] = ap[k] + bp[k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, NIN)
            c.enqueue_function[_resid_add_kernel[NIN]](
                self.gi_inner.lt["gpu", Layout.row_major(NIN)](),
                self.gi_skip.lt["gpu", Layout.row_major(NIN)](),
                gin.lt["gpu", Layout.row_major(NIN)](),
                grid_dim=(NIN + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        self.inner.for_each_param[target](visitor, ctx)
        self.skip.for_each_param[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)
        self.skip.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
        self.skip.polyak_from[target](src.skip, tau, ctx)
