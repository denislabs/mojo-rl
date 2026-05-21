"""Slim Module trait (NN2_AUDIT retrofit, Follow-up #7).

Successor to `nn2/core/module.mojo`. Differences:

  1. **No Phase 10A buffer surface.** The `out_ptr/grad_in_ptr/grad_out_ptr/
     ensure_buffers` default-null methods on the existing trait are gone.
     Orchestrators (Sequential, ComputeGraph) own all inter-module slabs
     (audit Spike #1).

  2. **`backward_input` collapsed into `backward[mode]`.** A
     `BackwardMode = "all" | "input_only"` comptime param replaces the
     separate `backward_input` method. Leaves dispatch on `comptime if
     mode == "all"` to skip param-grad work when only `grad_input` is
     wanted. Halves the per-leaf method count.

  3. **for_each_param / set_inference / zero_grad become provided
     (trait default) methods.** The trait body declares them; leaves
     opt in by being conformers, opt out (override) only for combinators
     whose internal structure isn't reachable through reflection (e.g.
     Sequential, which holds children in a Tuple).

Coexistence: `module.mojo` (the original Module trait) is untouched.
Existing leaves keep working. New leaves implement `ModuleV2`. When
the retrofit completes, `module.mojo` and its conformers get deleted
and `ModuleV2` gets renamed to `Module`.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP
from .param_visitor import ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# ModuleV2 — the slim trait.
#
# The `backward` mode comptime param uses StaticString for ergonomics
# (`"all"` / `"input_only"`) matching the existing `target` convention.
# ──────────────────────────────────────────────────────────────────────


trait ModuleV2(Defaultable & Movable & ImplicitlyDestructible):
    """Slim Module trait. Required: `make` factories, `forward`,
    `backward`. Provided: nothing yet (reflection-derived `zero_grad`
    etc. ship as free functions in `walkers.mojo`; the trait stays
    minimal until the Mojo nightly limitation around `conforms_to`-
    dispatch in trait bodies is lifted — see audit Spike #6 caveat)."""

    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        ...

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        """Backward with comptime `mode`:
          - `"all"` (default): writes grad_input AND accumulates param grads.
          - `"input_only"`: writes grad_input ONLY; skips param-grad work.
            Used by `StopGradParams` and SAC actor update through twin
            critics (audit Spike #1 — replaces the separate
            `backward_input` method on the old trait).

        Param-less leaves (ReLU/Tanh/StopGrad/...) ignore `mode`.
        Param-bearing leaves (Linear/LayerNorm/GaussianHead/...) gate
        their param-grad kernels on `comptime if mode == "all"`.

        BACKWARD-ORDER INVARIANT (audit Spike #1): when this leaf
        aliases its forward input by pointer (the unified-buffer
        design), `grad_input.ptr` may equal the cache pointer. Param
        grads MUST be computed before grad_input is written, or the
        cache is clobbered mid-read."""
        ...

    # ──────────────────────────────────────────────────────────────────
    # Provided (default) methods — leaves opt in by being conformers.
    # Parameterless leaves (ReLU/Tanh/StopGrad/Scale/Slice/...) inherit
    # the no-op default. Parameterised leaves and combinators override
    # to recurse over their Params / children.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Default: no params. Parameterised leaves override to
        call `for_each_param_auto[Self, V, target]` from `walkers.mojo`;
        combinators override to recurse over children."""
        pass

    def zero_grad[target: StaticString](mut self) raises:
        """Default: no params. Parameterised leaves override to call
        `zero_grad_auto[Self, target]`; combinators override to recurse."""
        pass
