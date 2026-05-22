"""TernaryFusedAdd[DIM] — element-wise 3-input sum (Block D-7).

  output[b, d]    = in0[b, d] + in1[b, d] + in2[b, d]
  grad_in0[b, d]  = grad_output[b, d]
  grad_in1[b, d]  = grad_output[b, d]
  grad_in2[b, d]  = grad_output[b, d]

Used by Dreamer-style RSSM fusion of (deterministic state, stochastic
state, action embedding) before a downstream nonlinearity. CPU only.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.ternary_module import TernaryModule
from ..core.target_storage import TargetStorage, assert_tag_for


struct TernaryFusedAdd[DIM_: Int](TernaryModule):
    comptime IN0_DIM = Self.DIM_
    comptime IN1_DIM = Self.DIM_
    comptime IN2_DIM = Self.DIM_
    comptime OUT_DIM = Self.DIM_

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", "TernaryFusedAdd: CPU only"
        var a = Self()
        a.ts = TargetStorage.make_cpu()
        return a^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "cpu", "TernaryFusedAdd: CPU only"
        var a = Self()
        a.ts = TargetStorage.make_cpu()
        return a^

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
        comptime assert target == "cpu", "TernaryFusedAdd: CPU only"
        assert_tag_for["TernaryFusedAdd", target](self.ts.target_tag)

        var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in0.ptr)
        var i1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in1.ptr)
        var i2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in2.ptr)
        var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)

        comptime N = BATCH * Self.DIM_
        var k = 0
        while k + CPU_SIMD_W <= N:
            var v = (
                i0_p.load[width=CPU_SIMD_W](k)
                + i1_p.load[width=CPU_SIMD_W](k)
                + i2_p.load[width=CPU_SIMD_W](k)
            )
            o_p.store(k, v)
            k += CPU_SIMD_W
        while k < N:
            o_p[k] = i0_p[k] + i1_p[k] + i2_p[k]
            k += 1

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
        comptime assert target == "cpu", "TernaryFusedAdd: CPU only"
        assert_tag_for["TernaryFusedAdd", target](self.ts.target_tag)

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
        var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in0.ptr)
        var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in1.ptr)
        var gi2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in2.ptr)

        comptime N = BATCH * Self.DIM_
        var k = 0
        while k + CPU_SIMD_W <= N:
            var g = go_p.load[width=CPU_SIMD_W](k)
            gi0_p.store(k, g)
            gi1_p.store(k, g)
            gi2_p.store(k, g)
            k += CPU_SIMD_W
        while k < N:
            var g = go_p[k]
            gi0_p[k] = g
            gi1_p[k] = g
            gi2_p[k] = g
            k += 1
