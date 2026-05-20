"""Module trait — uniform tree-walk API for leaves and combinators.

Phase 2.4: `target` is a comptime method param. Modules carry a runtime
`_target_tag` set by `make[target, INIT]`, asserted by every method.

Tensor args use generic `MutOrigin` so callers can pass `TileTensor`
views built directly from `DeviceBuffer` (narrow origin) without an
intermediate `MutAnyOrigin` widening step. Impl bodies that pipe pointers
into kernels accept the generic origin and rebind to `MutAnyOrigin` only
at the kernel-launch boundary.

Trait requirements:
  - `Defaultable`: zero-arg `__init__()` yields empty placeholders.
  - `IN_DIM`, `OUT_DIM`: comptime ints.
  - `make[target, INIT]()` / `make[target, INIT](ctx)`: static factories.
  - `forward`, `backward`, `for_each_param`: see signatures below.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from .param_visitor import ParamVisitor
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP


trait Module(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString, INIT: Initializer](ctx: DeviceContext) raises -> Self:
        ...

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
        ...

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
        ...

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
        """Backward computing grad_input only — does NOT accumulate
        grad_w / grad_b. Used by `StopGradParams` and by inline frozen-
        network paths (e.g. SAC actor update through the twin critics).

        For param-less primitives (ReLU/Tanh/StopGrad/...) this is just
        a delegate to `backward`. For Linear/LayerNorm it skips the
        param-grad kernels. For combinators (Sequential/Residual/Parallel)
        it chains `backward_input` over children so no inner Module
        writes its grad_w.
        """
        ...

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        prefix: String,
        mut visitor: V,
    ) raises:
        ...

    def set_inference(mut self, value: Bool):
        """Set inference mode on this module and recurse into children.

        When `value=True`, forward should skip side effects that are
        training-only — dropout sampling, BatchNorm running-stats
        updates, NoisyLinear noise injection, etc. Current leaves
        (Linear, ReLU, Tanh, LayerNorm, StopGrad) just store the flag
        for downstream layers that need it.

        Combinators (Sequential, Residual, Parallel, ...) propagate
        the flag to every child.
        """
        ...

    # ──────────────────────────────────────────────────────────────────
    # Phase 10A — Module-owned output / grad buffers.
    #
    # Each Module owns three List[Scalar[DT]] buffers that ComputeGraph
    # v2 (Phase 10D) uses for inter-node wiring:
    #   - _out_buf      [BATCH, OUT_DIM]   forward writes here
    #   - _grad_in_buf  [BATCH, IN_DIM]    backward writes here
    #   - _grad_out_buf [BATCH, OUT_DIM]   graph zeros + consumers
    #                                       scatter-add into; backward
    #                                       reads this as grad_output
    #
    # Lazy-grown on first `ensure_buffers[BATCH]()` call. The Module's
    # existing forward/backward signatures (with explicit `output` /
    # `grad_input` args) stay unchanged — the graph wraps `out_ptr()`
    # and `grad_in_ptr()` into TileTensors and passes them as those
    # args. Existing direct callers (tests, hand-orchestrated loss
    # blocks) keep working unchanged.
    #
    # CPU-only contract for now. GPU buffers (DeviceBuffer mirrors)
    # land alongside CG v2's GPU path in a later phase.
    # ──────────────────────────────────────────────────────────────────

    def ensure_buffers[BATCH: Int](mut self) raises:
        """Lazy-grow internal out / grad_in / grad_out buffers to BATCH
        samples (each sized BATCH × {OUT_DIM, IN_DIM, OUT_DIM}).
        Idempotent. Called once per BATCH by the graph before forward.

        Default impl: no-op. Modules that participate in ComputeGraph v2
        (Phase 10D) must override to grow their owned buffers."""
        pass

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer into the owned output buffer [BATCH, OUT_DIM].
        Valid after `ensure_buffers[BATCH]()` has been called.

        Default impl: returns null. Modules that participate in
        ComputeGraph v2 must override."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer into the owned grad-input buffer [BATCH, IN_DIM].
        Backward writes its grad_input here; graph reads to forward
        to predecessors' grad_out_bufs via scatter-add.

        Default impl: returns null."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer into the owned grad-output buffer [BATCH, OUT_DIM].
        Graph zeros at start of backward; downstream consumers scatter-
        add their grad_input pieces here; Module's backward reads it
        as grad_output.

        Default impl: returns null."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
