"""Slice[IN_DIM, START, END] — extract a column range. Phase 10E.

Pure structural op: forward copies `input[b, START:END]` into the
output; backward writes `grad_output` into `grad_input[b, START:END]`
and **zeros the rest of `grad_input`**. The zero-fill is what makes
ComputeGraph v2's scatter-add semantics work: when two Slice nodes
source from the same predecessor (e.g. `action` cols `[0, ACT)` +
`log_prob` col `[ACT, ACT+1)` both sourcing from rsample), each
scatter-adds its full-width grad-in tile into the predecessor's
`_grad_out_buf`. The zeros outside each slice's range mean the two
contributions interleave correctly into `[grad_action | grad_lp]`.

Forward:
    output[b, j] = input[b, START + j]            j ∈ [0, OUT_DIM)

Backward:
    grad_input[b, START + j] = grad_output[b, j]  j ∈ [0, OUT_DIM)
    grad_input[b, k]         = 0                  otherwise

Cache: none (slice is value-independent — backward needs only the index
arithmetic). No parameters.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace

from layout import TileTensor

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


struct Slice[IN: Int, START: Int, END: Int](Module):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.END - Self.START

    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    # Phase 10A buffer surface (CG v2 wiring).
    var _out_buf: List[Scalar[DT]]
    var _grad_in_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    var _n_batch_buf: Int

    def __init__(out self):
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False
        self._out_buf = List[Scalar[DT]]()
        self._grad_in_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._n_batch_buf = 0

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Slice.make[target='gpu', INIT] requires a DeviceContext"
        )
        comptime assert Self.START >= 0, "Slice.START must be >= 0"
        comptime assert Self.END > Self.START, "Slice.END must be > START"
        comptime assert Self.END <= Self.IN, "Slice.END must be <= IN_DIM"
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Slice.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        comptime assert Self.START >= 0, "Slice.START must be >= 0"
        comptime assert Self.END > Self.START, "Slice.END must be > START"
        comptime assert Self.END <= Self.IN, "Slice.END must be <= IN_DIM"
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Slice: method called with [target='"
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
        comptime assert input.flat_rank == 2, "input rank-2 [BATCH, IN_DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, OUT_DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    output[b, j] = input[b, Self.START + j]
        else:
            raise Error("Slice: GPU path not yet implemented (Phase 10E CPU only)")

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
            # Zero whole grad_input first so the scatter-add into a
            # shared predecessor `_grad_out_buf` interleaves correctly.
            for b in range(BATCH):
                for k in range(Self.IN_DIM):
                    grad_input[b, k] = Scalar[DT](0.0)
            for b in range(BATCH):
                for j in range(Self.OUT_DIM):
                    grad_input[b, Self.START + j] = grad_output[b, j]
        else:
            raise Error("Slice: GPU backward not yet implemented (Phase 10E CPU only)")

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

    # ── Phase 10A buffer surface ──────────────────────────────────────

    def ensure_buffers[BATCH: Int](mut self) raises:
        if self._n_batch_buf < BATCH:
            self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
            self._grad_in_buf.resize(BATCH * Self.IN_DIM, Scalar[DT](0.0))
            self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
            self._n_batch_buf = BATCH

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._out_buf.unsafe_ptr()
        )

    def grad_in_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_in_buf.unsafe_ptr()
        )

    def grad_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_out_buf.unsafe_ptr()
        )
