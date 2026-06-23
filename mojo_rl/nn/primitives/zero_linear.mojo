"""ZeroLinear[IN, OUT] — a Linear whose weight AND bias start at zero (storage).

Transformed from legacy `nn.primitives.ZeroLinear` (surface-only change). Owns a
storage `Linear`, zero-fills it after construction (ignoring INIT), and delegates
forward / vjp / param-walks to the inner Linear.

For AdaLN-zero conditioning: the modulation projection must output 0 at init so
the conditional block is the identity. A graph applies one `INIT` to all nodes,
so we can't selectively zero a single Linear via `make`'s INIT; this wrapper
zero-fills regardless of INIT, then trains normally once gradients ramp it up.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .linear import Linear


struct ZeroLinear[IN_: Int, OUT_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_

    var inner: Linear[Self.IN_, Self.OUT_]

    def __init__(out self):
        self.inner = Linear[Self.IN_, Self.OUT_]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var z = Self()
        z.inner = Linear[Self.IN_, Self.OUT_].make[target, INIT](ctx)
        # Override INIT: zero weight + bias for AdaLN-zero identity.
        comptime if target == "cpu":
            for i in range(Self.IN_ * Self.OUT_):
                z.inner.weight.val.data[i] = Scalar[DT](0.0)
            for i in range(Self.OUT_):
                z.inner.bias.val.data[i] = Scalar[DT](0.0)
        else:
            z.inner.weight.val.dev.value().enqueue_fill(Scalar[DT](0.0))
            z.inner.bias.val.dev.value().enqueue_fill(Scalar[DT](0.0))
        return z^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.forward[target, B, POLICY=POLICY](inputs, out, ctx)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
