"""BinaryConcat[IN0_DIM, IN1_DIM] — two-input feature concatenation. Cleanup 13.

Sibling of `BinarySub` / `BinaryElemMin`; packs two separate
`[BATCH, IN0_DIM]` + `[BATCH, IN1_DIM]` tiles into a single
`[BATCH, IN0_DIM + IN1_DIM]` output by horizontal stack.

    output[b, d]              = in0[b, d]                  d in [0, IN0_DIM)
    output[b, IN0_DIM + d]    = in1[b, d]                  d in [0, IN1_DIM)
    grad_in0[b, d]            = grad_output[b, d]          d in [0, IN0_DIM)
    grad_in1[b, d]            = grad_output[b, IN0_DIM + d] d in [0, IN1_DIM)

Use cases:
  - SAC: `concat(s, action) → sa` for critic input, used currently as an
    inline `_concat_sa` free function. `BinaryConcat` lets the same op
    live as a node inside a CG v2 graph when gradients must flow back
    through both inputs.
  - Dreamer / TD-MPC2: `concat(latent, action)` for dynamics input.
  - Any DAG with a fan-in concat node.

Lazy-grown Phase 10A buffer surface — CG v2 wiring ready.
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


struct BinaryConcat[IN0_DIM_: Int, IN1_DIM_: Int](BinaryModule):
    comptime IN0_DIM = Self.IN0_DIM_
    comptime IN1_DIM = Self.IN1_DIM_
    comptime OUT_DIM = Self.IN0_DIM_ + Self.IN1_DIM_

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
            "BinaryConcat.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryConcat.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "BinaryConcat: method called with [target='"
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
        comptime assert in0.flat_rank == 2, "in0 rank-2 [BATCH, IN0_DIM]"
        comptime assert in1.flat_rank == 2, "in1 rank-2 [BATCH, IN1_DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, OUT_DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.IN0_DIM):
                    output[b, d] = in0[b, d]
                for d in range(Self.IN1_DIM):
                    output[b, Self.IN0_DIM + d] = in1[b, d]
        else:
            raise Error("BinaryConcat: GPU path not yet implemented (CPU only)")

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
                for d in range(Self.IN0_DIM):
                    grad_in0[b, d] = grad_output[b, d]
                for d in range(Self.IN1_DIM):
                    grad_in1[b, d] = grad_output[b, Self.IN0_DIM + d]
        else:
            raise Error("BinaryConcat: GPU backward not yet implemented (CPU only)")

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
