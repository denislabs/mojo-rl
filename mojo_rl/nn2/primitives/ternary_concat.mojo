"""TernaryConcat[D0, D1, D2] — 3-input horizontal stack (Block D-7).

  output[b, d]                  = in0[b, d]                   d ∈ [0, D0)
  output[b, D0 + d]             = in1[b, d]                   d ∈ [0, D1)
  output[b, D0 + D1 + d]        = in2[b, d]                   d ∈ [0, D2)

Backward routes the slice of grad_output back to each input. CPU only
(DreamerV3-style dynamics doesn't yet need GPU; mirror BinaryConcat
GPU pattern when called for).
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.ternary_module import TernaryModule
from ..core.target_storage import TargetStorage, assert_tag_for


struct TernaryConcat[D0_: Int, D1_: Int, D2_: Int](TernaryModule):
    comptime IN0_DIM = Self.D0_
    comptime IN1_DIM = Self.D1_
    comptime IN2_DIM = Self.D2_
    comptime OUT_DIM = Self.D0_ + Self.D1_ + Self.D2_

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", "TernaryConcat: CPU only"
        var c = Self()
        c.ts = TargetStorage.make_cpu()
        return c^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "cpu", "TernaryConcat: CPU only"
        var c = Self()
        c.ts = TargetStorage.make_cpu()
        return c^

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
        in2: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert in0.flat_rank == 2, "in0 rank-2"
        comptime assert in1.flat_rank == 2, "in1 rank-2"
        comptime assert in2.flat_rank == 2, "in2 rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        comptime assert target == "cpu", "TernaryConcat: CPU only"
        assert_tag_for["TernaryConcat", target](self.ts.target_tag)

        for b in range(BATCH):
            for d in range(Self.IN0_DIM):
                output[b, d] = in0[b, d]
            for d in range(Self.IN1_DIM):
                output[b, Self.IN0_DIM + d] = in1[b, d]
            for d in range(Self.IN2_DIM):
                output[b, Self.IN0_DIM + Self.IN1_DIM + d] = in2[b, d]

    def vjp[
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
        mut grad_in2: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_in0.flat_rank == 2, "grad_in0 rank-2"
        comptime assert grad_in1.flat_rank == 2, "grad_in1 rank-2"
        comptime assert grad_in2.flat_rank == 2, "grad_in2 rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        comptime assert target == "cpu", "TernaryConcat: CPU only"
        assert_tag_for["TernaryConcat", target](self.ts.target_tag)

        for b in range(BATCH):
            for d in range(Self.IN0_DIM):
                grad_in0[b, d] = grad_output[b, d]
            for d in range(Self.IN1_DIM):
                grad_in1[b, d] = grad_output[b, Self.IN0_DIM + d]
            for d in range(Self.IN2_DIM):
                grad_in2[b, d] = grad_output[b, Self.IN0_DIM + Self.IN1_DIM + d]
