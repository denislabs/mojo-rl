"""Scale[DIM] — multiply by a runtime scalar. Phase 8.4.

Trivial elementwise primitive — exists so that the SAC actor loss
(`α · log_prob - min_q`) can be expressed as a Module chain rather
than inline arithmetic. The multiplier is a public runtime field
(`scale.multiplier = current_alpha`), settable per-step to track
moving values like SAC's auto-tuned α.

    out[b, d]      = multiplier · input[b, d]
    grad_in[b, d]  = multiplier · grad_out[b, d]

No cache (multiplier needed on backward, but it's a field, not derived).
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace

from layout import TileTensor

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


struct Scale[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var multiplier: Scalar[DT]
    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.multiplier = Scalar[DT](1.0)
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Scale.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Scale.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Scale: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

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
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                out_p.store(k, in_p.load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                out_p[k] = in_p[k] * self.multiplier
                k += 1
        else:
            raise Error("Scale: GPU path not yet implemented (Phase 8.4 CPU only)")

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
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
        self._assert_tag[target]()

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                gi_p.store(k, go_p.load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                gi_p[k] = go_p[k] * self.multiplier
                k += 1
        else:
            raise Error("Scale: GPU backward not yet implemented (Phase 8.4 CPU only)")

    def backward_input[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
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
        # No parameters — backward_input ≡ backward.
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        pass

    def set_inference(mut self, value: Bool):
        self._inference = value
