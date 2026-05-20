"""BinaryElemMin[DIM] — two-input elementwise min. Phase 10C.

Sibling of packed `ElemMin[DIM]`; two separate `[BATCH, DIM]` tiles
instead of a packed `[BATCH, 2*DIM]` input.

    output[b, d]   = min(in0[b, d], in1[b, d])

Backward (subgradient: ties go to `in0`):
    grad_in0[b, d] = grad_output[b, d] if in0 wins, else 0
    grad_in1[b, d] = grad_output[b, d] if in1 wins, else 0

Cache: one mask byte per output element (1=in0_won, 0=in1_won), stored
as Scalar[DT] (1.0 / 0.0) to match the packed ElemMin convention.

Use case: SAC twin-critic min — `BinaryElemMin(q1, q2)` instead of
packing `[q1 | q2]` and calling packed ElemMin.
"""

from std.gpu.host import DeviceContext

from layout import TileTensor, TensorLayout

from ..constants import DT
from ..core import (
    BinaryModule,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


struct BinaryElemMin[DIM: Int](BinaryModule):
    comptime IN0_DIM = Self.DIM
    comptime IN1_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var mask: List[Scalar[DT]]
    var cache_n_batch: Int

    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    # Phase 10A buffer surface (CG v2 wiring).
    var _out_buf: List[Scalar[DT]]
    var _grad_in0_buf: List[Scalar[DT]]
    var _grad_in1_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    var _n_batch_buf: Int

    def __init__(out self):
        self.mask = List[Scalar[DT]]()
        self.cache_n_batch = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False
        self._out_buf = List[Scalar[DT]]()
        self._grad_in0_buf = List[Scalar[DT]]()
        self._grad_in1_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._n_batch_buf = 0

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryElemMin.make[target='gpu', INIT] requires a DeviceContext"
        )
        var m = Self()
        m._target_tag = TARGET_CPU
        return m^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryElemMin.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var m = Self()
        m.ctx = ctx
        m._target_tag = TARGET_GPU
        return m^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "BinaryElemMin: method called with [target='"
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
        L0: TensorLayout,
        L1: TensorLayout,
        LOUT: TensorLayout,
        O0: MutOrigin,
        O1: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0: TileTensor[DT, L0, O0],
        in1: TileTensor[DT, L1, O1],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert in0.flat_rank == 2, "in0 rank-2 [BATCH, DIM]"
        comptime assert in1.flat_rank == 2, "in1 rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_mask_cpu(BATCH)
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
            raise Error("BinaryElemMin: GPU path not yet implemented (Phase 10C CPU only)")

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LG0: TensorLayout,
        LG1: TensorLayout,
        OGO: MutOrigin,
        OG0: MutOrigin,
        OG1: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_in0: TileTensor[DT, LG0, OG0],
        mut grad_in1: TileTensor[DT, LG1, OG1],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_in0.flat_rank == 2, "grad_in0 rank-2"
        comptime assert grad_in1.flat_rank == 2, "grad_in1 rank-2"
        self._assert_tag[target]()

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
            raise Error("BinaryElemMin: GPU backward not yet implemented (Phase 10C CPU only)")

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LG0: TensorLayout,
        LG1: TensorLayout,
        OGO: MutOrigin,
        OG0: MutOrigin,
        OG1: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_in0: TileTensor[DT, LG0, OG0],
        mut grad_in1: TileTensor[DT, LG1, OG1],
    ) raises:
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_in0, grad_in1)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        pass

    def set_inference(mut self, value: Bool):
        self._inference = value

    # ── Phase 10A buffer surface ──────────────────────────────────────

    def ensure_buffers[BATCH: Int](mut self) raises:
        if self._n_batch_buf < BATCH:
            self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
            self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
            self._grad_in1_buf.resize(BATCH * Self.IN1_DIM, Scalar[DT](0.0))
            self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
            self._n_batch_buf = BATCH

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._out_buf.unsafe_ptr()
        )

    def grad_in0_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_in0_buf.unsafe_ptr()
        )

    def grad_in1_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_in1_buf.unsafe_ptr()
        )

    def grad_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_out_buf.unsafe_ptr()
        )
