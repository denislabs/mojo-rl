"""BinaryConcat[IN0_DIM, IN1_DIM] — retrofit (Phase B).

Horizontal stack of two `[BATCH, *]` tiles into `[BATCH, IN0_DIM+IN1_DIM]`.

  output[b, d]              = in0[b, d]                     d in [0, IN0_DIM)
  output[b, IN0_DIM + d]    = in1[b, d]                     d in [0, IN1_DIM)
  grad_in0[b, d]            = grad_output[b, d]             d in [0, IN0_DIM)
  grad_in1[b, d]            = grad_output[b, IN0_DIM + d]   d in [0, IN1_DIM)

No params. Conforms to `BinaryModule`.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.binary_module import BinaryModule
from ..core.target_storage import TargetStorage, assert_tag_for


struct BinaryConcat[IN0_DIM_: Int, IN1_DIM_: Int](BinaryModule):
    comptime IN0_DIM = Self.IN0_DIM_
    comptime IN1_DIM = Self.IN1_DIM_
    comptime OUT_DIM = Self.IN0_DIM_ + Self.IN1_DIM_

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryConcat.make[target='gpu', INIT] requires a DeviceContext"
        )
        var c = Self()
        c.ts = TargetStorage.make_cpu()
        return c^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryConcat.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var c = Self()
        c.ts = TargetStorage.make_gpu(ctx)
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
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert in0.flat_rank == 2, "in0 rank-2 [BATCH, IN0_DIM]"
        comptime assert in1.flat_rank == 2, "in1 rank-2 [BATCH, IN1_DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, OUT_DIM]"
        assert_tag_for["BinaryConcat", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.IN0_DIM):
                    output[b, d] = in0[b, d]
                for d in range(Self.IN1_DIM):
                    output[b, Self.IN0_DIM + d] = in1[b, d]
        else:
            raise Error("BinaryConcat: GPU path not yet implemented")

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
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_in0.flat_rank == 2, "grad_in0 rank-2"
        comptime assert grad_in1.flat_rank == 2, "grad_in1 rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["BinaryConcat", target](self.ts.target_tag)

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.IN0_DIM):
                    grad_in0[b, d] = grad_output[b, d]
                for d in range(Self.IN1_DIM):
                    grad_in1[b, d] = grad_output[b, Self.IN0_DIM + d]
        else:
            raise Error("BinaryConcat: GPU backward not yet implemented")
