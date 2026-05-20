"""BinarySub[DIM] — two-input elementwise subtract. Phase 10C.

Sibling of the packed `Sub[DIM]` (single `[a | b]` input of width
`2*DIM`); this version takes two separate `[BATCH, DIM]` tiles, which
is the natural shape for ComputeGraph v2.

    output[b, d]        = in0[b, d] - in1[b, d]
    grad_in0[b, d]      =  grad_output[b, d]
    grad_in1[b, d]      = -grad_output[b, d]

Use case (post-Phase-10E): SAC composed actor loss's `α·log_prob - min_q`
will flow as `BinarySub(α·log_prob, min_q)` through CG v2 instead of
packing into `[α·log_prob | min_q]` and calling packed Sub.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace

from layout import TileTensor

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


struct BinarySub[DIM: Int](BinaryModule):
    comptime IN0_DIM = Self.DIM
    comptime IN1_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

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
            "BinarySub.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinarySub.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "BinarySub: method called with [target='"
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
        in0: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        in1: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        comptime assert in0.flat_rank == 2, "in0 rank-2 [BATCH, DIM]"
        comptime assert in1.flat_rank == 2, "in1 rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    output[b, d] = in0[b, d] - in1[b, d]
        else:
            raise Error("BinarySub: GPU path not yet implemented (Phase 10C CPU only)")

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_in0.flat_rank == 2, "grad_in0 rank-2"
        comptime assert grad_in1.flat_rank == 2, "grad_in1 rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var go = grad_output[b, d]
                    grad_in0[b, d] = go
                    grad_in1[b, d] = -go
        else:
            raise Error("BinarySub: GPU backward not yet implemented (Phase 10C CPU only)")

    def backward_input[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
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
