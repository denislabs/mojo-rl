"""Reduction Modules — retrofit (Phase B).

Sum[DIM] and Mean[DIM] across the feature axis. Same algorithm as v1,
just the scaffold collapses: `ts: TargetStorage` replaces the per-leaf
tag/inference/ctx triplet, `backward[mode]` collapses `backward` +
`backward_input`, and Phase 10A buffer surface is dropped.

No params on either struct → no Param wrappers; `for_each_param` is
a no-op trait conformance.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module_v2 import ModuleV2
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# SumV2
# ──────────────────────────────────────────────────────────────────────


struct SumV2[DIM: Int](ModuleV2):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "SumV2.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "SumV2.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
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
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, 1]"
        assert_tag_for["SumV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc
        else:
            raise Error("SumV2: GPU path not yet implemented")

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
        assert_tag_for["SumV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var go = grad_output[b, 0]
                for d in range(Self.DIM):
                    grad_input[b, d] = go
        else:
            raise Error("SumV2: GPU backward not yet implemented")


# ──────────────────────────────────────────────────────────────────────
# MeanV2
# ──────────────────────────────────────────────────────────────────────


struct MeanV2[DIM: Int](ModuleV2):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1
    comptime _INV_DIM: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.DIM)

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "MeanV2.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "MeanV2.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var m = Self()
        m.ts = TargetStorage.make_gpu(ctx)
        return m^

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
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, 1]"
        assert_tag_for["MeanV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc * Self._INV_DIM
        else:
            raise Error("MeanV2: GPU path not yet implemented")

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
        assert_tag_for["MeanV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                var go_inv = grad_output[b, 0] * Self._INV_DIM
                for d in range(Self.DIM):
                    grad_input[b, d] = go_inv
        else:
            raise Error("MeanV2: GPU backward not yet implemented")
