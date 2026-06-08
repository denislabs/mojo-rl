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
from .graph_visitor import DisplayStep
from .walkers import for_each_param_auto, zero_grad_auto
from .state import for_each_state_auto
from .tensor_pack import TensorPack


# ──────────────────────────────────────────────────────────────────────
# mptr — THE origin-erasure chokepoint (S2′, 2026-06-08).
#
# The codebase erases pointer origins to `MutAnyOrigin` constantly (the
# variadic-TileTensor limitation is irreducible — see audit §B0). Before
# this helper that meant ~800 inline copies of the verbose
#   rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](view.ptr)
# drowning the actual math. `mptr` collapses each to `mptr(view.ptr)` (or
# `mptr(view)` straight from a TileTensor). Dtype-generic, so it also
# absorbs the bf16 AMP rebinds. The unsafe step now lives in ONE place.
# ──────────────────────────────────────────────────────────────────────


@always_inline
def mptr[
    dt: DType, o: Origin
](p: UnsafePointer[Scalar[dt], o]) -> UnsafePointer[Scalar[dt], MutAnyOrigin]:
    """Erase a `Scalar[dt]` pointer's origin to `MutAnyOrigin`. Replaces
    the inline `rebind[UnsafePointer[Scalar[dt], MutAnyOrigin]](p)` dance."""
    return rebind[UnsafePointer[Scalar[dt], MutAnyOrigin]](p)


@always_inline
def mptr(
    t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Erased base pointer of a TileTensor view — `mptr(view)` instead of
    `rebind[...](view.ptr)`."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.ptr)


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
    workaround that PPOActorLoss used at I.2 landing time.

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
        inputs: TensorPack[Self.ARITY],
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
        grad_inputs: TensorPack[Self.ARITY],
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
    # Two-phase vjp (S7, 2026-06-07) — structurally enforces the
    # backward-aliasing order (A2/A3) at the ORCHESTRATOR instead of
    # trusting each leaf's internal `grad_b → grad_w → grad_input`
    # ordering. Orchestrators (`Sequential`/`ComputeGraph`) call
    # `vjp_param_grads` (reads the cached input + grad_output) BEFORE
    # `vjp_grad_input` (writes grad_inputs — the same slab the cache may
    # alias). A leaf that splits these two phases physically cannot
    # interleave them wrong: the order is fixed by the caller.
    #
    # Both carry defaults, so the split is INCREMENTAL — only leaves that
    # cache a forward input by pointer (Linear, Conv2D, NoisyLinear, …)
    # need to override. Every other leaf inherits:
    #   • `vjp_param_grads` → no-op (param-less leaves have nothing to do;
    #     non-split param leaves still do all their work in the combined
    #     `vjp`, reached via the `vjp_grad_input` default below).
    #   • `vjp_grad_input` → delegates to the combined `vjp` (which the
    #     leaf already orders correctly internally). Bit-identical to the
    #     single-call path because `vjp_param_grads` was the no-op.
    # ──────────────────────────────────────────────────────────────────

    def vjp_param_grads[
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
    ) raises:
        """Phase 1 of the two-phase vjp: accumulate PARAM grads only.
        Reads `grad_output` (+ any cached forward input), writes the
        leaf's own `Param` grads, touches NO grad_inputs slab. Skipped
        entirely under `mode == "input_only"`. Default no-op — only
        cached-input leaves override; everything else does its param work
        inside the combined `vjp` (reached via `vjp_grad_input`)."""
        pass

    def vjp_grad_input[
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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        """Phase 2 of the two-phase vjp: write grad_inputs (the
        predecessor slab). Default: delegate to the combined `vjp`. For a
        NON-split leaf this runs its full backward (params + inputs) here
        — correct, because the orchestrator's `vjp_param_grads` call was
        the no-op default. A split leaf (Linear) overrides to do ONLY the
        grad_input computation, relying on `vjp_param_grads` having
        already run."""
        self.vjp[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, grad_inputs
        )

    # ──────────────────────────────────────────────────────────────────
    # Provided defaults — parameterless leaves auto-inherit no-ops.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Default: reflection-walk every `IsParam` field of the concrete
        leaf and dispatch the visitor (S1, 2026-06-07). Param-less leaves
        reflect to a no-op (no IsParam fields). Param-bearing leaves no
        longer need to override — forgetting the override can no longer
        silently skip params in checkpoint/optimizer walks. Combinators +
        wrapper leaves (children are Module-typed, not IsParam) still
        override to recurse into children."""
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        """Default: reflection-walk every `IsParam` field of the concrete
        leaf and zero its grad (S1, 2026-06-07). Param-less leaves reflect
        to a no-op. Param-bearing leaves no longer need to override —
        forgetting it can no longer silently accumulate grads under
        `Sequential`. Combinators + wrapper leaves still override to
        recurse into children."""
        zero_grad_auto[Self, target](self)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Default: reflection-walk every `IsState` field of the concrete
        leaf and dispatch the visitor (S5 Stage 3, 2026-06-07). The
        checkpoint path runs this right after `for_each_param`, so State
        fields (e.g. BatchNorm running stats) are persisted; the optimizer
        path (`for_each_param`) never reaches them. State-less leaves
        reflect to a no-op. Combinators override to recurse into children."""
        for_each_state_auto[Self, V, target](self, prefix, visitor)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Per-call runtime attribute mutation. Default no-op. Modules
        with mutable runtime state (e.g. Scale.multiplier, Clamp.min_val)
        override and comptime-branch on `ATTR`."""
        pass

    def set_attr_ptr[ATTR: StaticString](
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Bind a device-resident attribute source (CUDA-graph capture).
        Default no-op. Modules whose runtime attribute can live in a
        device buffer mutated by another kernel (e.g. `Scale.multiplier`
        ← SAC's on-device α) override and comptime-branch on `ATTR`.
        Distinct from `set_attr` (host scalar baked into the kernel arg):
        a pointer set here is read on-device each forward so the value
        can change between captured replays without re-baking."""
        pass

    # ──────────────────────────────────────────────────────────────────
    # Display surface — read by `ComputeGraph.describe` exporters. Both
    # carry defaults, so existing conformers need no change; leaves
    # override `display_label` with their type name, and containers
    # (Sequential) override `display_steps` to expand into their children.
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def display_label() -> String:
        """Short display name for graph exporters. Default generic;
        leaves override with their type name (e.g. "Linear")."""
        return String("module")

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        """Inner display steps for container modules — one per child,
        each `(child_label, child_out_dim)`. Default empty = atomic leaf;
        `Sequential` overrides to expand its chain."""
        return List[DisplayStep]()
