"""InitWith[INNER, INIT_OVR] — override the propagated initializer for a subtree.

A name-transparent, forward/vjp-transparent wrapper whose ONLY effect is at
construction: `make[target, INIT]` discards the propagated `INIT` and builds
`INNER` with `INIT_OVR` instead. Everything else (forward, vjp, param/state
walks, polyak, attrs, display) delegates verbatim.

This replaces brittle post-hoc init surgery (walk the net, match a positional
param path like `"1.0.4.weight"`, scale it in place — which fails SILENTLY when
a refactor shifts the path). Instead the special init is declared structurally,
at the definition site, where the "which layer" knowledge belongs:

    Sequential[ Conv2D[...], ReLU[...], Linear[H, FC], ReLU[FC],
                InitWith[Linear[FC, ACT], Zero] ]   # ← zero-init output head

Reorder/rename/resize the surrounding layers and the init moves WITH the layer —
nothing to keep in sync, and no global "scale the last Linear" heuristic (which
can't even express a two-head `Parallel`).

Discipline: the override applies to the WHOLE `INNER` subtree, so wrap ONLY the
single leaf you want re-initialized (e.g. the output `Linear`) — wrapping a
multi-layer head in `Zero` would zero its hidden layers too.

Name transparency: `for_each_param`/`for_each_state` pass `prefix` THROUGH
UNCHANGED (no extra segment), so the inner `Linear`'s params keep the exact names
they'd have without the wrapper. Checkpoint load and params-only promotion
(`hard_copy_params`) stay byte-compatible.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.amp import AMPPolicy, NoAMP
from ..core.graph_visitor import DisplayStep


struct InitWith[INNER: Module, INIT_OVR: Initializer](Module):
    # Fully transparent surface: arity, dims and activation dtype are INNER's.
    comptime ARITY = Self.INNER.ARITY
    comptime IN_DIMS = Self.INNER.IN_DIMS
    comptime OUT_DIM = Self.INNER.OUT_DIM
    comptime ACT_DT = Self.INNER.ACT_DT

    var inner: Self.INNER

    def __init__(out self):
        self.inner = Self.INNER()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        # The ONE behavior: ignore the propagated `INIT`, build INNER with the
        # override. (INNER threads INIT_OVR down to its own leaves as usual.)
        var s = Self()
        s.inner = Self.INNER.make[target, Self.INIT_OVR](ctx)
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.forward[target, B, POLICY=POLICY](inputs, out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        # Name-transparent: NO added path segment.
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
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)

    def set_attr_buf[ATTR: StaticString](mut self, buf: DeviceBuffer[DT]):
        self.inner.set_attr_buf[ATTR](buf)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)

    @staticmethod
    def display_label() -> String:
        return Self.INNER.display_label()

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        return Self.INNER.display_steps()
