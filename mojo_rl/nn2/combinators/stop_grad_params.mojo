"""StopGradParams[Inner] — Module wrapper that freezes the inner's params
through the *backward* path. Phase 8.2.

Forward     : passthrough to `inner.forward(...)`.
Backward    : routes to `inner.backward_input(...)` — computes grad_input
              only, does **not** accumulate grad_w / grad_b on Inner.
backward_input: same as backward (already param-free).
for_each_param: passthrough to `inner.for_each_param(...)` — Inner's
              params are still visible to the optimizer, so other loss
              paths (e.g. the critic loss in SAC) can still update them.

Contrast with `nn2.primitives.StopGrad[DIM]`: that one *severs* the
gradient at this point in the network (zeroes grad_input). StopGradParams
*lets the gradient flow through* (so an actor upstream of frozen critics
still gets non-zero grads), but blocks param updates on the wrapped
Module via the backward path.

Use case (Phase 8.2): SAC actor update goes through the twin critics. We
want grad_action via the chain rule (for the policy gradient), but we
**don't** want the critic params to change during the actor step. Wrap
each critic in `StopGradParams` and the backward chain takes care of it.
Equivalent to TF/PyTorch's `with torch.no_grad():` over the critic
weights, except we explicitly want grad_input through them.

Use case (Phase 8.4+): frozen encoder during policy training, frozen
reward heads during dynamics warmup — same primitive, different inners.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Initializer,
    AMPPolicy, NoAMP,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


struct StopGradParams[Inner: Module](Module):
    comptime IN_DIM = Self.Inner.IN_DIM
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8
    var _inference: Bool

    # ------------------------------------------------------------------
    # Defaultable — empty Inner + UNINIT tag.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.inner = Self.Inner()
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    def __init__(out self, var inner: Self.Inner):
        """CPU wrap constructor — takes ownership of a pre-built inner."""
        self.inner = inner^
        self.ctx = None
        self._target_tag = TARGET_CPU
        self._inference = False

    def __init__(out self, var inner: Self.Inner, ctx: DeviceContext):
        """GPU wrap constructor — takes ownership of a pre-built inner."""
        self.inner = inner^
        self.ctx = ctx
        self._target_tag = TARGET_GPU
        self._inference = False

    # ------------------------------------------------------------------
    # make[target, INIT] — recursive build for trait conformance. Useful
    # mostly for tests; production code wraps existing trained Inners via
    # the constructors above.
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "StopGradParams.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s.inner = Self.Inner.make[target, INIT]()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "StopGradParams.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.inner = Self.Inner.make[target, INIT](ctx)
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "StopGradParams: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    # ------------------------------------------------------------------
    # Forward — passthrough.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        self._assert_tag[target]()
        self.inner.forward[target, BATCH, POLICY=POLICY](input, output)

    # ------------------------------------------------------------------
    # Backward — routes to inner.backward_input. This is THE point.
    # ------------------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        self._assert_tag[target]()
        self.inner.backward_input[target, BATCH, POLICY=POLICY](
            grad_output, grad_input
        )

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        self._assert_tag[target]()
        self.inner.backward_input[target, BATCH, POLICY=POLICY](
            grad_output, grad_input
        )

    # ------------------------------------------------------------------
    # for_each_param — passthrough. Inner's params are still walked by
    # the optimizer; this combinator only severs the *backward-through-
    # this-loss* path. Updates from other losses still apply.
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target](prefix + sep + "inner", visitor)

    def set_inference(mut self, value: Bool):
        self._inference = value
        self.inner.set_inference(value)
