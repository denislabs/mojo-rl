"""Slim BinaryModule trait (NN2_AUDIT retrofit).

Same shape as `Module` (Defaultable + Movable + ImplicitlyDestructible)
but for 2-input → 1-output ops. Scaffold trims:

  1. No Phase 10A buffer surface (`ensure_buffers`, `out_ptr`,
     `grad_in0_ptr`, `grad_in1_ptr`, `grad_out_ptr` removed).
     Orchestrators own all inter-module slabs.
  2. `backward_input` collapsed into `backward[mode]`. A
     `mode = "all" | "input_only"` comptime param replaces the separate
     method. Param-less leaves ignore `mode`; param-bearing binary
     leaves (none today, but the slot exists for symmetry) gate their
     param-grad work on `comptime if mode == "all"`.
  3. `for_each_param` / `zero_grad` are provided as default no-op
     methods on the trait — today's binaries (Sub/ElemMin/Concat) are
     all parameter-less and auto-inherit; future param-bearing binaries
     override to walk their `Param[NAME, DECAY, SIZE]` fields via
     reflection (same pattern as `Module`).
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP
from .param_visitor import ParamVisitor


trait BinaryModule(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN0_DIM: Int
    comptime IN1_DIM: Int
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString, INIT: Initializer](
        ctx: DeviceContext,
    ) raises -> Self:
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        in1: TileTensor[
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
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        """Backward with comptime `mode`:
          - `"all"` (default): writes both grad inputs AND accumulates
            param grads (if any).
          - `"input_only"`: writes both grad inputs ONLY; skips param-grad
            work. For param-less binaries (Sub/ElemMin/Concat) this is
            identical to `"all"`."""
        ...

    # ──────────────────────────────────────────────────────────────────
    # Provided (default) methods — mirror `Module`'s pattern. Today's
    # binaries (BinarySub / BinaryElemMin / BinaryConcat) are all
    # parameter-less, so the no-op default fits. Future param-bearing
    # binaries override to walk their Params.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Default: no params. Override on param-bearing binaries."""
        pass

    def zero_grad[target: StaticString](mut self) raises:
        """Default: no params. Override on param-bearing binaries."""
        pass
