"""BinaryElemMin[DIM] — retrofit (Phase B).

  output[b, d]   = min(in0[b, d], in1[b, d])
  grad_in0[b, d] = grad_output[b, d] if in0 wins, else 0
  grad_in1[b, d] = grad_output[b, d] if in1 wins, else 0   (ties → in0)

Cache: mask byte per output element (1.0 = in0 won, 0.0 = in1 won),
stored as `Scalar[DT]`. Leaf-owned, no aliasing concern.

No params. Conforms to `BinaryModule`.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.binary_module import BinaryModule
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer,
)


struct BinaryElemMin[DIM: Int](BinaryModule):
    comptime IN0_DIM = Self.DIM
    comptime IN1_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var mask: List[Scalar[DT]]   # [BATCH, DIM] win-mask cache
    var ts: TargetStorage

    def __init__(out self):
        self.mask = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryElemMin.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryElemMin.make[target='cpu', INIT](ctx) — drop ctx for CPU"
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
        comptime assert in0.flat_rank == 2, "in0 rank-2 [BATCH, DIM]"
        comptime assert in1.flat_rank == 2, "in1 rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        assert_tag_for["BinaryElemMin", target](self.ts.target_tag)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.mask, BATCH * Self.DIM)
            var m_p = self.mask.unsafe_ptr()
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var a = in0[b, d]
                    var bv = in1[b, d]
                    if a < bv:
                        output[b, d] = a
                        m_p[b * Self.DIM + d] = Scalar[DT](1.0)
                    else:
                        output[b, d] = bv
                        m_p[b * Self.DIM + d] = Scalar[DT](0.0)
        else:
            raise Error("BinaryElemMin: GPU path not yet implemented")

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
        assert_tag_for["BinaryElemMin", target](self.ts.target_tag)

        comptime if target == "cpu":
            var m_p = self.mask.unsafe_ptr()
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var mask_v = m_p[b * Self.DIM + d]
                    var go = grad_output[b, d]
                    if mask_v > Scalar[DT](0.5):
                        grad_in0[b, d] = go
                        grad_in1[b, d] = Scalar[DT](0.0)
                    else:
                        grad_in0[b, d] = Scalar[DT](0.0)
                        grad_in1[b, d] = go
        else:
            raise Error("BinaryElemMin: GPU backward not yet implemented")
