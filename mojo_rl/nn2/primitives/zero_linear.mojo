"""ZeroLinear[IN, OUT] — a Linear whose weight AND bias start at zero.

For AdaLN-zero conditioning: the modulation projection must output 0 at
init so the conditional block is the identity (shift/scale=0 ⇒ Modulate is
identity; gate=0 ⇒ Gate drops the branch). A graph applies one `INIT` to
all nodes, so we can't selectively zero a single Linear via `make`'s INIT;
this wrapper owns a `Linear` and zero-fills it after construction, ignoring
INIT. Delegates forward / vjp / for_each_param to the inner Linear, so it
trains normally once gradients ramp it up.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for
from .linear import Linear


struct ZeroLinear[IN: Int, OUT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = Self.OUT

    @staticmethod
    def display_label() -> String:
        return String("ZeroLinear")

    var inner: Linear[Self.IN, Self.OUT]
    var ts: TargetStorage

    def __init__(out self):
        self.inner = Linear[Self.IN, Self.OUT]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var z = Self()
        z.inner = Linear[Self.IN, Self.OUT].make[target=target, INIT=INIT](
            ctx=ctx
        )
        # Override INIT: zero weight + bias for AdaLN-zero identity.
        comptime if target == "cpu":
            var w = z.inner.weight.value_unsafe_ptr_cpu()
            for i in range(Self.IN * Self.OUT):
                w[i] = Scalar[DT](0.0)
            var b = z.inner.bias.value_unsafe_ptr_cpu()
            for i in range(Self.OUT):
                b[i] = Scalar[DT](0.0)
            z.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = ctx.value()
            z.inner.weight.val.dev.value().enqueue_fill(0.0)
            z.inner.bias.val.dev.value().enqueue_fill(0.0)
            z.ts = TargetStorage.make_gpu(ctx_v)
        return z^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        self.inner.forward[target, BATCH, POLICY=POLICY](
            inputs[0], output=output
        )

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        self.inner.vjp[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, grad_inputs[0]
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        self.inner.for_each_param[target, V](prefix, visitor)

    def for_each_state[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        self.inner.for_each_state[target, V](prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        self.inner.zero_grad[target]()
