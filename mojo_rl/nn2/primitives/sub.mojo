"""Sub[DIM] — elementwise subtract of two packed inputs. Phase 8.4.

Multi-input via the packed-tensor convention (same pattern as ElemMin):
caller packs both operands side-by-side into a single `[BATCH, 2*DIM]`
tile `[a | b]`, Sub outputs `a - b` shape `[BATCH, DIM]`.

    output[b, d]            = input[b, d] - input[b, DIM + d]

Backward:
    grad_input[b, d]        =  grad_output[b, d]
    grad_input[b, DIM + d]  = -grad_output[b, d]

Used in SAC composed actor loss for `α·log_prob - min_q`.
"""

from std.gpu.host import DeviceContext

from layout import TileTensor, TensorLayout

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


struct Sub[DIM: Int](Module):
    comptime IN_DIM = 2 * Self.DIM
    comptime OUT_DIM = Self.DIM

    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Sub.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Sub.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Sub: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, 2*DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    output[b, d] = input[b, d] - input[b, Self.DIM + d]
        else:
            raise Error("Sub: GPU path not yet implemented (Phase 8.4 CPU only)")

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var go = grad_output[b, d]
                    grad_input[b, d] = go
                    grad_input[b, Self.DIM + d] = -go
        else:
            raise Error("Sub: GPU backward not yet implemented (Phase 8.4 CPU only)")

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        pass

    def set_inference(mut self, value: Bool):
        self._inference = value
