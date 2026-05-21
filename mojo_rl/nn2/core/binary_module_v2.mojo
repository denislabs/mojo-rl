"""Slim BinaryModule trait (NN2_AUDIT retrofit, Phase B).

Successor to `nn2/core/binary_module.mojo`. Same shape (2-input → 1-output)
but mirrors `ModuleV2`'s scaffold trims:

  1. No Phase 10A buffer surface (`ensure_buffers`, `out_ptr`,
     `grad_in0_ptr`, `grad_in1_ptr`, `grad_out_ptr` removed).
     Orchestrators own all inter-module slabs.
  2. `backward_input` collapsed into `backward[mode]`. A
     `mode = "all" | "input_only"` comptime param replaces the separate
     method. Param-less leaves ignore `mode`; param-bearing binary
     leaves (none today, but the slot exists for symmetry) gate their
     param-grad work on `comptime if mode == "all"`.
  3. `set_inference` / `for_each_param` deferred to free walker funcs
     (see `core/walkers.mojo`) — same approach as `ModuleV2`.

Coexistence: the original `binary_module.mojo` is untouched. v1
binaries (`BinarySub`, `BinaryElemMin`, `BinaryConcat`) keep working.
v2 binaries implement `BinaryModuleV2`. Both versions live side-by-side
until Phase F deletes the v1 files.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP


trait BinaryModuleV2(Defaultable & Movable & ImplicitlyDestructible):
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
