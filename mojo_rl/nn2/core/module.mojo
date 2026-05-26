"""Module — Phase 4.6b unified N-ary trait.

Replaces the three near-identical traits (`Module` / `BinaryModule` /
`TernaryModule`) with a single `Module` trait carrying `comptime ARITY`
and a variadic forward / vjp surface. ARITY = 1 → legacy unary.
ARITY = 2 → legacy binary. ARITY = 3 → legacy ternary. ARITY ≥ 4 is
forward-compatible — no leaf needs it today.

Trait properties (mirror the slim trait properties from before Phase
4.6):

  1. **No buffer surface.** Orchestrators (`Sequential`, `ComputeGraph`,
     …) own every inter-module slab. Leaves that need an input cache for
     backward alias the orchestrator's input slab via a pointer field —
     no copy.

  2. **`vjp[mode]` collapses backward + backward_input.** A comptime
     `mode = "all" | "input_only"` param skips param-grad work when only
     grad_inputs are needed (e.g. SAC actor flow through twin critics).
     Param-less leaves ignore `mode`. Param-bearing leaves gate their
     param-grad kernels on `comptime if mode == "all"`.

  3. **Default no-op `for_each_param` / `zero_grad` / `set_attr`.**
     Parameter-less leaves (ReLU/Tanh/Sub/Concat/…) auto-inherit.
     Parameterised leaves override (typically via
     `for_each_param_auto[Self, V, target]` from `walkers.mojo`).
     Combinators override to recurse over children.

Variadic+origin-erasure unblock (Mojo nightly): pinning
`origin=MutAnyOrigin` on every variadic TileTensor parameter homogenises
the pack — without it, Mojo rejects variadic TileTensors because per-
source `MutOrigin`s don't unify. Caller-side: orchestrators construct
TileTensors from `UnsafePointer[Scalar[DT], MutAnyOrigin]` slabs (heap
allocs + DeviceBuffer derefs are already MutAnyOrigin); external callers
rebind once with `rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](...)`.

Body-side: a variadic element `inputs[k]` has an opaque LayoutType from
inside the body, so direct `inputs[k][b, d]` fails the flat_rank
constraint. Use `typed_view[BATCH, DIM](inputs[k])` to rebuild a typed
rank-2 view. For known `comptime ARITY`, `comptime if Self.ARITY == N:`
dispatch chooses the typed-view branch — no per-iteration overhead.
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
# typed_view / typed_view_mut — rebuild a typed rank-2 TileTensor view
# from a variadic element whose LayoutType is opaque inside the body.
#
# Usage (inside a leaf forward body):
#   var in0 = typed_view[BATCH, Self.IN0_DIM](inputs[0])
#   var in1 = typed_view[BATCH, Self.IN1_DIM](inputs[1])
#   var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
#
# Both carry `origin=MutAnyOrigin`, so kernel + SIMD dispatch see the
# right addressing.
# ──────────────────────────────────────────────────────────────────────


def typed_view[
    BATCH: Int,
    DIM: Int,
](
    t: TileTensor[
        dtype=DT,
        address_space=AddressSpace.GENERIC,
        element_size=1,
        origin=MutAnyOrigin,
        ...,
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
        mut=True,
        dtype=DT,
        address_space=AddressSpace.GENERIC,
        element_size=1,
        origin=MutAnyOrigin,
        ...,
    ],
) -> TileTensor[
    DT,
    type_of(row_major[BATCH, DIM]()),
    MutAnyOrigin,
]:
    """Mutable variant of `typed_view`. Mojo's mut-tracking is preserved
    through the rebind via `MutAnyOrigin`."""
    var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.ptr)
    return TileTensor(p, row_major[BATCH, DIM]())


# ──────────────────────────────────────────────────────────────────────
# Module — the unified N-ary trait. Conformers declare their `comptime
# ARITY` (1, 2, 3, ...) and `comptime OUT_DIM`. Per-input dims are
# struct-level fields (`IN_DIM` for unary, `IN0_DIM` / `IN1_DIM` / ...
# for multi-arity). Combinators that chain unary leaves enforce
# `comptime assert M.ARITY == 1` and access `M.IN_DIM` directly on
# concrete child types — not via the trait — so the trait stays minimal.
# ──────────────────────────────────────────────────────────────────────


trait Module(Defaultable & Movable & ImplicitlyDestructible):
    """Unified N-ary module trait. Leaves declare their own `comptime
    ARITY` (1, 2, 3, ...). Forward/vjp use origin-erased variadic
    TileTensor packs.

    Required: ARITY, IN_DIM, IN1_DIM, IN2_DIM, OUT_DIM, make factories,
    forward, vjp. Provided: for_each_param (no-op), zero_grad (no-op),
    set_attr (no-op). Parameterised leaves override; param-less leaves
    inherit.

    Per-input dim semantics:
      - `IN_DIM` — first input dim. For unary leaves this is the sole
        input dim. For binary/ternary/quaternary, `IN_DIM == IN0_DIM`.
      - `IN1_DIM` — second input dim. `0` for unary leaves.
      - `IN2_DIM` — third input dim. `0` for unary and binary leaves.
      - `IN3_DIM` — fourth input dim. `0` for unary/binary/ternary leaves.

    Unary leaves set `IN1_DIM = 0` / `IN2_DIM = 0` / `IN3_DIM = 0` by
    convention. Binary leaves override `IN1_DIM` only; ternary override
    `IN1_DIM + IN2_DIM`; quaternary override all four. This shape lets
    graph-node wrappers (`Node`, externals) read input dims through the
    trait surface without sub-traits.

    Phase I.2.5 added `IN3_DIM` so quaternary loss leaves (e.g.
    `PPOObjective4(actor_out, action, old_log_prob, advantage)`) can
    declare their fourth input dim natively, retiring the aux-packing
    workaround that PPOActorLossCG used at I.2 landing time.

    Phase I.2.6.h: `IN_DIMS: InlineArray[Int, Self.ARITY]` is the
    SOLE per-input dim member. The legacy `IN_DIM` / `IN1_DIM` /
    `IN2_DIM` / `IN3_DIM` ladder is fully removed. Conformers declare
    `IN_DIMS` directly (typically `InlineArray[Int, ARITY](fill=D)`
    for homogeneous leaves; a `_build_in_dims()` static helper for
    heterogeneous leaves like PPOObjective `[2*ACT, ACT, 1, 1]`).
    No arity cap on this surface — extending to ARITY=5+ for DreamerV3
    imagination losses requires zero trait changes."""

    comptime ARITY: Int
    comptime IN_DIMS: InlineArray[Int, Self.ARITY]
    comptime OUT_DIM: Int

    @staticmethod
    def make[
        target: StaticString,
        INIT: Initializer,
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU
        (impls raise at runtime if missing)."""
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
        mut output: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        """N-ary forward. Leaf body must:
          1. Use `comptime if Self.ARITY == K:` to branch on arity.
          2. Inside each branch, rebuild typed views with
             `typed_view[BATCH, IN<i>_DIM](inputs[i])` and
             `typed_view_mut[BATCH, Self.OUT_DIM](output)`.
          3. Existing kernel/SIMD bodies follow once views are typed.

        Callers pass `output` as a keyword arg (`output=...`) — required
        by Mojo to disambiguate from the variadic pack."""
        ...

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
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
        breaks the gradient."""
        ...

    # ──────────────────────────────────────────────────────────────────
    # Provided defaults — parameterless leaves auto-inherit no-ops.
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
