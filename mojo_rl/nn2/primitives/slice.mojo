"""Slice[IN, START, END] — extracts column range `[START, END)` from input.

Zero-fills the rest of grad_input on backward so that ComputeGraph's
scatter-add into a shared predecessor `_grad_out_buf` interleaves
correctly with parallel slicers (e.g. the q1/q2/log_prob unpack in
`SACActorLossCG`).

No params. Conforms to `Module`. Orchestrator owns slabs;
`backward[mode]` accepted, has no effect (no params to skip).
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


struct Slice[IN: Int, START: Int, END: Int](Module):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.END - Self.START

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Slice.make[target='gpu', INIT] requires a DeviceContext"
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
            "Slice.make[target='cpu', INIT](ctx) — drop ctx for CPU"
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
        assert_tag_for["Slice", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    output[b, j] = input[b, Self.START + j]
        else:
            raise Error("Slice: GPU path not yet implemented")

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
        assert_tag_for["Slice", target](self.ts.target_tag)

        comptime if target == "cpu":
            # Zero whole grad_input first; scatter the slice in afterward.
            # Zeros required for ComputeGraph scatter-add: when multiple
        # slicers share a predecessor, each writes its slice range and
        # leaves the rest at 0 so the scatter-add sums correctly.
            for b in range(BATCH):
                for k in range(Self.IN_DIM):
                    grad_input[b, k] = Scalar[DT](0.0)
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    grad_input[b, Self.START + j] = grad_output[b, j]
        else:
            raise Error("Slice: GPU backward not yet implemented")
