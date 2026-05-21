"""Slim Module trait (NN2_AUDIT retrofit).

Three properties distinguish this trait from the pre-retrofit shape:

  1. **No buffer surface.** No `out_ptr/grad_in_ptr/grad_out_ptr/
     ensure_buffers` methods. Orchestrators (`Sequential`,
     `ComputeGraph`) own every inter-module slab (audit Spike #1, the
     unified-buffer design). Leaves that need an input cache for
     backward alias the orchestrator's input slab via a pointer field —
     no copy.

  2. **`backward[mode]` collapses backward + backward_input.** A
     comptime `mode = "all" | "input_only"` param replaces the separate
     `backward_input` method. Leaves dispatch on `comptime if mode ==
     "all"` to skip param-grad work when only `grad_input` is needed
     (e.g. through twin critics during SAC actor update).

  3. **for_each_param / zero_grad have no-op default impls.**
     Parameterless leaves (ReLU/Tanh/Scale/Slice/Sub/...) auto-inherit.
     Parameterised leaves override to call `for_each_param_auto[Self, V,
     target]` from `walkers.mojo` (reflection over `Param[NAME, DECAY,
     SIZE]` fields). Combinators override to recurse over children.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP
from .param_visitor import ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# Module — the slim trait.
#
# The `backward` mode comptime param uses StaticString for ergonomics
# (`"all"` / `"input_only"`) matching the existing `target` convention.
# ──────────────────────────────────────────────────────────────────────


trait Module(Defaultable & Movable & ImplicitlyDestructible):
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
