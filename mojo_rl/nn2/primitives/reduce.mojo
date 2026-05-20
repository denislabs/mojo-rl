"""Reduction Modules — Sum[DIM] and Mean[DIM]. Phase 8.4.

Reduce across the feature axis (column-wise). BATCH dim is preserved so
combinators like Sequential can still chain afterward.

Sum[DIM]:
    output[b, 0]     = Σ_d input[b, d]
    grad_input[b, d] = grad_output[b, 0]

Mean[DIM]:
    output[b, 0]     = (1/DIM) Σ_d input[b, d]
    grad_input[b, d] = grad_output[b, 0] / DIM

For a batch-wise scalar reduction (e.g. SAC's final per-batch mean to
produce the scalar loss), the training loop sums the [BATCH, 1] output
and divides by BATCH outside the Module chain — Mean here reduces the
*feature* axis, not the batch axis. This deliberately keeps a single
clean convention: Modules always have a BATCH-preserving signature.
"""

from std.gpu.host import DeviceContext

from layout import TileTensor, TensorLayout

from ..constants import DT
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


struct Sum[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1

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
            "Sum.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Sum.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Sum: method called with [target='"
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
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, 1]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc
        else:
            raise Error("Sum: GPU path not yet implemented (Phase 8.4 CPU only)")

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
                var go = grad_output[b, 0]
                for d in range(Self.DIM):
                    grad_input[b, d] = go
        else:
            raise Error("Sum: GPU backward not yet implemented (Phase 8.4 CPU only)")

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


struct Mean[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = 1
    comptime _INV_DIM: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.DIM)

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
            "Mean.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m._target_tag = TARGET_CPU
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Mean.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var m = Self()
        m.ctx = ctx
        m._target_tag = TARGET_GPU
        return m^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Mean: method called with [target='"
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
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, 1]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    acc += input[b, d]
                output[b, 0] = acc * Self._INV_DIM
        else:
            raise Error("Mean: GPU path not yet implemented (Phase 8.4 CPU only)")

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
                var go_inv = grad_output[b, 0] * Self._INV_DIM
                for d in range(Self.DIM):
                    grad_input[b, d] = go_inv
        else:
            raise Error("Mean: GPU backward not yet implemented (Phase 8.4 CPU only)")

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
