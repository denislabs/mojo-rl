"""NaryModule — Phase 4.6 unified trait surface.

[FOUNDATION — coexists with `Module` / `BinaryModule` / `TernaryModule`.
Next session's Phase 4.6b big-bang retires the three old traits and
migrates every leaf + combinator to conform here.]

`NaryModule` collapses the three near-identical traits into a single
trait surface with:

  - `comptime ARITY: Int`            (1, 2, 3, ... — today bounded by 3)
  - `comptime OUT_DIM: Int`
  - Variadic `forward[BATCH](*inputs, mut output)` with origin-erased
    TileTensor packs.
  - Variadic `vjp[BATCH, mode](grad_output, mut *grad_inputs)` with
    mutable origin-erased packs.
  - Provided defaults for `for_each_param`, `zero_grad`, `set_attr`.

The unblock: pinning `origin=MutAnyOrigin` on the variadic TileTensor
parameter homogenizes the pack (Mojo nightly otherwise rejects variadic
TileTensors because per-source `MutOrigin`s don't unify). See spike
`tests/nn2/spikes/spike_variadic_origin_erased.mojo` and feedback
memory `feedback_mojo_variadic_tiletensor_blocked` for the diagnosis.

Caller-side: orchestrators must pass TileTensors whose origin slot
resolves to `MutAnyOrigin`. nn2 orchestrators already construct
internal TileTensors from `UnsafePointer[Scalar[DT], MutAnyOrigin]`
slabs (Sequential.mid_cpu, ComputeGraph node buffers, …) so the
pinning is natural. External callers (test code, agent training loops)
rebind once with `rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
buf.unsafe_ptr())`.

Body-side: `inputs[k]` returns a TileTensor whose LayoutType is opaque
from inside the body, so direct indexing (`inputs[k][b, d]`) fails the
flat_rank constraint. Workaround: extract `.ptr` (already
origin-erased) and rebuild a typed view via `typed_view[BATCH, DIM]`.
For leaves with known `comptime ARITY`, prefer `comptime if Self.ARITY
== N:` dispatch with per-branch typed views — no per-iteration cost.

Coexistence rule (Foundation only):

  - New leaves (NaryBinarySub demo + future Phase 4.6b conformers)
    implement NaryModule.
  - Existing leaves (Linear, BinaryElementwise, …) keep conforming to
    Module/BinaryModule/TernaryModule until Phase 4.6b migration.
  - Combinators (Sequential, Parallel, …) still bound on Module
    (unary). Mixed graphs aren't supported across the transition —
    deliberate; the big-bang next session resolves it.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from layout import TileTensor, row_major

from ..constants import DT
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP
from .param_visitor import ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# typed_view — rebuild a typed rank-2 TileTensor view from a variadic
# element whose LayoutType is opaque inside the body.
#
# Usage (inside a leaf forward body):
#   var in0 = typed_view[BATCH, Self.IN0_DIM](inputs[0])
#   var in1 = typed_view[BATCH, Self.IN1_DIM](inputs[1])
#   var out = typed_view[BATCH, Self.OUT_DIM](output)
#
# All three carry `origin=MutAnyOrigin` so kernel/SIMD dispatch picks
# the right addressing.
# ──────────────────────────────────────────────────────────────────────


def typed_view[
    BATCH: Int,
    DIM: Int,
](
    t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) -> TileTensor[
    DT,
    type_of(row_major[BATCH, DIM]()),
    MutAnyOrigin,
]:
    """Rebuild a typed rank-2 [BATCH, DIM] view from an origin-erased
    variadic TileTensor element. Zero-cost: just pointer + layout
    rebinding — no copy."""
    var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.ptr)
    return TileTensor(p, row_major[BATCH, DIM]())


def typed_view_mut[
    BATCH: Int,
    DIM: Int,
](
    mut t: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) -> TileTensor[
    DT,
    type_of(row_major[BATCH, DIM]()),
    MutAnyOrigin,
]:
    """Mutable variant of `typed_view`. Mojo's mut-tracking is
    preserved through the rebind via `MutAnyOrigin`."""
    var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.ptr)
    return TileTensor(p, row_major[BATCH, DIM]())


# ──────────────────────────────────────────────────────────────────────
# NaryModule — the new unified trait. Same shape (Defaultable + Movable
# + ImplicitlyDestructible) as the three old traits.
# ──────────────────────────────────────────────────────────────────────


trait NaryModule(Defaultable & Movable & ImplicitlyDestructible):
    """Unified N-ary module trait. Leaves declare their own `comptime
    ARITY` (1, 2, 3, ...). Forward/vjp use origin-erased variadic
    TileTensor packs.

    Phase 4.6a Foundation: NaryModule exists alongside the three legacy
    traits. Phase 4.6b will retire the legacy traits and have every leaf
    + combinator conform here.

    Required: ARITY, OUT_DIM, make factories, forward, vjp.
    Provided: for_each_param (no-op), zero_grad (no-op), set_attr
    (no-op). Parameterised leaves override; param-less leaves inherit."""

    comptime ARITY: Int
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: DeviceContext) raises -> Self:
        ...

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
        """N-ary forward. Leaf body must:
          1. Assert `len(inputs)` matches `Self.ARITY` (compile-time
             via `comptime assert`, runtime via guard if dispatching
             dynamically).
          2. `comptime if Self.ARITY == K:` branch with per-arity typed
             view extraction via `typed_view[BATCH, IN<i>_DIM](
             inputs[i])`.
          3. Existing kernel/SIMD body unchanged once views are typed."""
        ...

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
        """N-ary vector-Jacobian product.

        `mode = "all"` (default): writes all grad_inputs AND accumulates
        param grads (if any).
        `mode = "input_only"`: writes grad_inputs ONLY; skips param-grad
        work. Used by StopGradParams + SAC actor flow through twin
        critics. Param-less leaves ignore `mode`.

        BACKWARD-ORDER INVARIANT: leaves that alias forward inputs by
        pointer (Linear's cached_input_ptr) must compute param grads
        BEFORE writing grad_inputs[i] — clobbering the cache mid-read
        breaks the gradient. Same invariant as legacy Module/BinaryModule.
        """
        ...

    # ──────────────────────────────────────────────────────────────────
    # Provided defaults — mirror legacy traits. Parameterless leaves
    # inherit the no-op; parameterised leaves override to walk Params.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Default: no params. Parameterised leaves override to call
        `for_each_param_auto[Self, V, target]` from `walkers.mojo`."""
        pass

    def zero_grad[target: StaticString](mut self) raises:
        """Default: no params. Override on param-bearing leaves."""
        pass

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Per-call runtime attribute mutation. Default no-op. Modules
        with mutable runtime state (e.g. Scale.multiplier, Clamp.min_val)
        override and comptime-branch on `ATTR`."""
        pass
