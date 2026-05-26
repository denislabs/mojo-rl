"""StopGradParams[Inner] — Module wrapper that freezes the inner's params
through the backward path. Phase F0c migration to Module.

Forward     : passthrough to `inner.forward(...)`.
Backward    : routes to `inner.vjp[..., mode="input_only"]` —
              computes grad_input only, does NOT accumulate grad_w / grad_b
              on Inner. With `Module.vjp[mode]`, this is now a
              one-liner — no separate `backward_input` method needed.
for_each_param: passthrough — Inner's params are still visible to the
              optimizer, so other loss paths can still update them.

Contrast with `nn2.primitives.StopGrad[DIM]` (which zeros grad_input
entirely). StopGradParams lets the gradient flow through but blocks
param updates via the backward path.

NOTE (Phase 3, 2026-05-22): For stop-grad references INSIDE a
`ComputeGraph`, prefer `ExternalNode[NAME, M, "src", MODE="input_only"]`
over `Node[NAME, StopGradParams[M], "src"]`. The ExternalNode
form (a) avoids owning a separate Module copy inside the graph,
(b) plumbs `mode="input_only"` directly into the referenced module's
backward, and (c) keeps the trainer as the canonical owner of the
underlying network. StopGradParams remains useful for compositions
OUTSIDE a graph context — e.g. inside `Sequential[..., StopGradParams[Linear[...]], ...]`.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


struct StopGradParams[Inner: Module](Module):
    comptime ARITY: Int = 1
    comptime IN_DIM = Self.Inner.IN_DIMS[0]
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner
    var ts: TargetStorage

    def __init__(out self):
        self.inner = Self.Inner()
        self.ts = TargetStorage.make_uninit()

    def __init__(out self, var inner: Self.Inner):
        """CPU wrap constructor — takes ownership of a pre-built inner."""
        self.inner = inner^
        self.ts = TargetStorage.make_cpu()

    def __init__(out self, var inner: Self.Inner, ctx: DeviceContext):
        """GPU wrap constructor — takes ownership of a pre-built inner."""
        self.inner = inner^
        self.ts = TargetStorage.make_gpu(ctx)

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "StopGradParams.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s.inner = Self.Inner.make[target, INIT]()
        s.ts = TargetStorage.make_cpu()
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
        s.ts = TargetStorage.make_gpu(ctx)
        return s^

    # ----- Forward (passthrough) ------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
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
        assert_tag_for["StopGradParams", target](self.ts.target_tag)
        self.inner.forward[target, BATCH, POLICY=POLICY](inputs[0], output=output)

    # ----- Backward — always input_only on Inner --------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
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
        """Always calls `inner.vjp[mode="input_only"]` regardless of
        the `mode` arg from the caller — that's the whole point of
        StopGradParams: never accumulate Inner's param grads via this
        loss path."""
        assert_tag_for["StopGradParams", target](self.ts.target_tag)
        self.inner.vjp[
            target, BATCH, POLICY=POLICY, mode="input_only",
        ](grad_output, grad_inputs[0])

    # ----- Walkers --------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["StopGradParams", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target, V](prefix + sep + "inner", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["StopGradParams", target](self.ts.target_tag)
        self.inner.zero_grad[target]()
