"""ElemMin[DIM] — elementwise minimum of two packed inputs. Phase 8.4.

Multi-input via the packed-tensor convention: caller packs both inputs
side-by-side into a single `[BATCH, 2*DIM]` tile (`[a | b]`), ElemMin
outputs `min(a, b)` shape `[BATCH, DIM]`.

    output[b, d]   = min(input[b, d], input[b, DIM + d])

Backward (subgradient at ties picks `a`):
    grad_input[b, d]        = grad_output[b, d] if a wins, else 0
    grad_input[b, DIM + d]  = grad_output[b, d] if b wins, else 0

Cache: a mask byte per element (1=a_won, 0=b_won). Stored as Scalar[DT]
for simplicity (vs an explicit Bool list); 1.0 or 0.0.

Use case in SAC composed actor loss:
    pack [q1 | q2] → [BATCH, 2]
    ElemMin[1].forward → min_q [BATCH, 1]
    ElemMin[1].backward(grad_min_q) → packed [grad_q1 | grad_q2]
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


struct ElemMin[DIM: Int](Module):
    comptime IN_DIM = 2 * Self.DIM
    comptime OUT_DIM = Self.DIM

    var mask: List[Scalar[DT]]
    var cache_n_batch: Int

    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.mask = List[Scalar[DT]]()
        self.cache_n_batch = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "ElemMin.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m._target_tag = TARGET_CPU
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "ElemMin.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var m = Self()
        m.ctx = ctx
        m._target_tag = TARGET_GPU
        return m^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "ElemMin: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def _ensure_mask_cpu(mut self, batch: Int):
        if self.cache_n_batch < batch:
            self.mask.resize(batch * Self.DIM, Scalar[DT](0.0))
            self.cache_n_batch = batch

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
            self._ensure_mask_cpu(BATCH)
            var m_p = self.mask.unsafe_ptr()
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var a = input[b, d]
                    var bv = input[b, Self.DIM + d]
                    if a < bv:
                        output[b, d] = a
                        m_p[b * Self.DIM + d] = Scalar[DT](1.0)
                    else:
                        output[b, d] = bv
                        m_p[b * Self.DIM + d] = Scalar[DT](0.0)
        else:
            raise Error("ElemMin: GPU path not yet implemented (Phase 8.4 CPU only)")

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
            var m_p = self.mask.unsafe_ptr()
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var mask_v = m_p[b * Self.DIM + d]
                    var go = grad_output[b, d]
                    if mask_v > Scalar[DT](0.5):
                        grad_input[b, d] = go
                        grad_input[b, Self.DIM + d] = Scalar[DT](0.0)
                    else:
                        grad_input[b, d] = Scalar[DT](0.0)
                        grad_input[b, Self.DIM + d] = go
        else:
            raise Error("ElemMin: GPU backward not yet implemented (Phase 8.4 CPU only)")

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
