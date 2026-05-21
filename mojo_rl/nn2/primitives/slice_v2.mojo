"""Slice[IN, START, END] — retrofit (Phase A).

Extracts column range `[START, END)` from input. Zero-fills the rest
of grad_input on backward so CG v2's scatter-add into a shared
predecessor `_grad_out_buf` interleaves correctly (see v1 docstring).

`ModuleV2` conformance INTENTIONALLY drops the Phase 10A buffer
surface (`_out_buf` / `_grad_in_buf` / `_grad_out_buf` fields,
`ensure_buffers` / `out_ptr` / `grad_in_ptr` / `grad_out_ptr` methods).
**Slice's CG v2 consumers (`binary_concat`, `sac_actor_loss_cg`) still
expect those methods on the v1 type and must be migrated to the
orchestrator-owns-slabs model (Phase C/F) before SliceV2 can replace
Slice in CG-v2 graphs.** Until then, both versions coexist:

  - `Slice` (v1): used by CG v2 today.
  - `SliceV2` (this file): used by anyone NOT yet on CG v2 — direct
    forward/backward callers.

No params. `mode` accepted on backward, has no effect.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module_v2 import ModuleV2
from ..core.target_storage import TargetStorage, assert_tag_for


struct SliceV2[IN: Int, START: Int, END: Int](ModuleV2):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.END - Self.START

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "SliceV2.make[target='gpu', INIT] requires a DeviceContext"
        )
        comptime assert Self.START >= 0, "Slice.START must be >= 0"
        comptime assert Self.END > Self.START, "Slice.END must be > START"
        comptime assert Self.END <= Self.IN, "Slice.END must be <= IN_DIM"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "SliceV2.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        comptime assert Self.START >= 0, "Slice.START must be >= 0"
        comptime assert Self.END > Self.START, "Slice.END must be > START"
        comptime assert Self.END <= Self.IN, "Slice.END must be <= IN_DIM"
        var s = Self()
        s.ts = TargetStorage.make_gpu(ctx)
        return s^

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
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, IN_DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, OUT_DIM]"
        assert_tag_for["SliceV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    output[b, j] = input[b, Self.START + j]
        else:
            raise Error("SliceV2: GPU path not yet implemented")

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
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["SliceV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            # Zero whole grad_input first; scatter the slice in afterward.
            # See v1 docstring for why zeros are required for CG v2 scatter-add.
            for b in range(BATCH):
                for k in range(Self.IN_DIM):
                    grad_input[b, k] = Scalar[DT](0.0)
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    grad_input[b, Self.START + j] = grad_output[b, j]
        else:
            raise Error("SliceV2: GPU backward not yet implemented")
