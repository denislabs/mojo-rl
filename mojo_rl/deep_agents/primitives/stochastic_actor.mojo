"""StochasticActor — actor net: feature trunk + (mu, log_std) heads (storage).

  obs → Sequential[*TRUNK] → (B × HIDDEN) → Parallel[Linear, Linear]
                                          → packed [mu | log_std]  (B × 2·ACT)

A plain storage `Module` (ARITY 1). The trunk→heads intermediate is the owned
`_mid` Tensor (+ `_mid_grad` for the backward), mirroring how storage Sequential
caches per-child activations: forward writes `_mid` (trunk out) then heads read it;
vjp uses the cached `_mid` as heads' forward-input and routes the mid-grad into the
trunk's vjp. RSample is applied SEPARATELY by the SAC actor block (this net stops
at [mu | log_std]).

STORAGE migration (Stage 5): legacy TargetStorage/mptr/TileTensor/`mode`/
typed_view gone; the trunk+heads are gated storage combinators so correctness is
inherited. Walkers recurse into trunk + heads.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.core.walkers import join_name
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.parallel import Parallel
from mojo_rl.nn.primitives.linear import Linear


struct StochasticActor[
    OBS_DIM: Int,
    ACT_DIM: Int,
    *TRUNK: Module,
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.OBS_DIM)
    comptime OUT_DIM = 2 * Self.ACT_DIM
    comptime N_TRUNK = Self.TRUNK.length
    comptime HIDDEN = Self.TRUNK[Self.N_TRUNK - 1].OUT_DIM
    comptime Heads = Parallel[
        Linear[Self.HIDDEN, Self.ACT_DIM],
        Linear[Self.HIDDEN, Self.ACT_DIM],
    ]

    var trunk: Sequential[*Self.TRUNK]
    var heads: Self.Heads
    var _mid: Tensor       # trunk output (= heads input), cached for vjp
    var _mid_grad: Tensor  # grad wrt _mid, routed trunk-ward in vjp

    def __init__(out self):
        comptime assert (
            Self.N_TRUNK >= 1
        ), "StochasticActor requires at least one TRUNK module"
        comptime assert (
            Self.TRUNK[0].IN_DIMS[0] == Self.OBS_DIM
        ), "StochasticActor: TRUNK[0].IN_DIM must equal OBS_DIM"
        self.trunk = Sequential[*Self.TRUNK]()
        self.heads = Self.Heads()
        self._mid = Tensor()
        self._mid_grad = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "StochasticActor: target must be 'cpu' or 'gpu'"
        var a = Self()
        a.trunk = Sequential[*Self.TRUNK].make[target, INIT](ctx)
        a.heads = Self.Heads.make[target, INIT](ctx)
        return a^

    def forward[
        target: StaticString, B: Int, o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        call_forward[target, B, POLICY=POLICY](
            self.trunk, inputs, self._mid, ctx
        )
        call_forward[target, B, POLICY=POLICY](
            self.heads, TensorRefs[1](self._mid), out, ctx
        )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        call_vjp[target, B, POLICY=POLICY](
            self.heads,
            TensorRefs[1](self._mid),
            grad_output,
            TensorRefs[1](self._mid_grad),
            ctx,
        )
        call_vjp[target, B, POLICY=POLICY](
            self.trunk, forward_input, self._mid_grad, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.trunk.for_each_param[target](
            visitor, ctx, join_name(prefix, String("trunk"))
        )
        self.heads.for_each_param[target](
            visitor, ctx, join_name(prefix, String("heads"))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.trunk.for_each_state[target](
            visitor, ctx, join_name(prefix, String("trunk"))
        )
        self.heads.for_each_state[target](
            visitor, ctx, join_name(prefix, String("heads"))
        )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        self.trunk.zero_grad[target](ctx)
        self.heads.zero_grad[target](ctx)

    def polyak_from[target: StaticString](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.trunk.polyak_from[target](src.trunk, tau, ctx)
        self.heads.polyak_from[target](src.heads, tau, ctx)
