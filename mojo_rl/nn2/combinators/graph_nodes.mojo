"""InputSlot / UnaryNode / BinaryNode / ExternalUnaryNode / ExternalBinaryNode — GraphNode wrappers.

Each wrapper owns its underlying op instance (or none, for InputSlot /
External*Node) plus the buffers used by ComputeGraph for inter-node
wiring.

`InputSlot[NAME, DIM]`
    `KIND = 0`. Represents one named external input to the graph. No
    compute; its `out_ptr_via()` returns a pointer set per-call by the
    graph's `set_input[NAME]` method. Its `grad_out_buf` is the
    accumulator for the input-gradient flowing back from consumers
    (`get_grad_input[NAME]` / `grad_input_ptr[NAME]` on the graph).

`UnaryNode[NAME, M, IN0_NAME]`
    wraps a `M: Module` (1→1). `KIND = 1`, `IN1_NAME = ""`, `IN1_DIM = 0`.
    `forward_via(in0, _)` ignores the second input; `vjp_via` writes
    only `_grad_in0_buf`.

`BinaryNode[NAME, BM, IN0_NAME, IN1_NAME]`
    wraps a `BM: BinaryModule` (2→1). `KIND = 2`. `forward_via(in0, in1)`
    uses both pointers; `vjp_via` writes both `_grad_in0_buf` and
    `_grad_in1_buf`.

`ExternalUnaryNode[NAME, M, IN0_NAME, MODE="all"]`
    `KIND = 1` (same scatter-add path as UnaryNode). Like UnaryNode but
    does NOT own its op instance — instead holds `_module_ptr:
    UnsafePointer[M, MutAnyOrigin]` set per-call by the graph's
    `set_external[NAME](mut module)` method. The module lives elsewhere
    (typically the trainer). `MODE` is plumbed into `M.vjp[mode]`
    so a single declaration can express stop-grad-style references —
    e.g. `MODE="input_only"` for the actor-loss view of a critic, which
    skips param-grad accumulation on that path. Phase 3.

`ExternalBinaryNode[NAME, M, IN0_NAME, IN1_NAME, MODE="all"]`
    `KIND = 2`. BinaryModule sibling of ExternalUnaryNode.

Backward contract: the wrapper reads its own `_grad_out_buf` as the
incoming gradient (graph zeros + scatter-adds into this before the
call), and writes `_grad_in*_buf`. The graph scatter-adds those into
predecessors' grad_out_bufs after the call returns. For InputSlot,
vjp_via is a no-op — `_grad_out_buf` already holds the final
input-gradient by the time it's reached in the reverse-topo walk.

CPU vs GPU storage: each node carries `ts: TargetStorage` (matches the
rest of nn2). CPU nodes own `List[Scalar[DT]]` for each buffer; GPU
nodes own `Optional[DeviceBuffer[DT]]`. `*_ptr_via()` dispatches on
`ts.target_tag` and returns whichever pointer is live; `ensure_buffers_via`
likewise grows the appropriate storage.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    GraphNode,
    Module,
    BinaryModule,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
)
from ..core.target_tag import TARGET_GPU
from ..core.target_storage import (
    TargetStorage,
    ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# InputSlot — external input to the graph (KIND=0). Block B.
#
# No compute, no Op. Holds an externally-set out_ptr (cached via
# set_input_via) and a grad_out_buf that accumulates the gradient
# flowing back to this input from all consumer nodes. Looks identical
# to a regular producer node from the graph's name-resolution loop;
# the only behavioral difference is that forward_via / vjp_via
# are no-ops (graph zeros grad_out_buf at backward start; predecessors'
# scatter-add does the rest).
# ──────────────────────────────────────────────────────────────────────


struct InputSlot[
    slot_name: StaticString,
    DIM_: Int,
](GraphNode):
    comptime NAME = Self.slot_name
    comptime IN0_NAME = ""
    comptime IN1_NAME = ""
    comptime IN0_DIM = 0
    comptime IN1_DIM = 0
    comptime OUT_DIM = Self.DIM_
    comptime KIND = 0

    # CPU/GPU grad accumulator. `_grad_out_buf` for CPU,
    # `_grad_out_buf_dev` for GPU; matches Unary/BinaryNode layout.
    var _grad_out_buf: List[Scalar[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev_n: Int

    # Block C: cached pointers.
    #   _out_ptr        — set externally per call via `set_input_via`.
    #   _grad_out_ptr   — points at the slot's grad accumulator (resolved
    #                     in ensure_buffers_via, stable thereafter).
    #   _grad_in0/1_ptr — null; InputSlot has no inputs to receive grad
    #                     into. Required by GraphNode trait.
    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_out_buf_dev = None
        self._grad_out_buf_dev_n = 0
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_in0_ptr = null_p
        self._grad_in1_ptr = null_p
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "InputSlot.make_via[target='gpu', INIT] requires a DeviceContext"
        )
        var n = Self()
        n.ts = TargetStorage.make_cpu()
        return n^

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "InputSlot.make_via[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var n = Self()
        n.ts = TargetStorage.make_gpu(ctx)
        return n^

    def ensure_buffers_via[BATCH: Int](mut self) raises:
        if self.ts.target_tag == TARGET_GPU:
            if self._n_batch_buf < BATCH:
                var ctx = self.ts.ctx.value()
                ensure_gpu_buffer(
                    self._grad_out_buf_dev, self._grad_out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                self._n_batch_buf = BATCH

    def set_input_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self._out_ptr = ptr

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in0_ptr  # null — InputSlot has no inputs

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in1_ptr  # null — InputSlot has no inputs

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        # No compute: the slot's out_ptr is whatever the caller set via
        # graph.set_input[NAME](tile). in0_ptr / in1_ptr are unused.
        pass

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        # No compute: the grad already sits in _grad_out_buf, accumulated
        # by consumers via scatter-add during the reverse-topo walk.
        pass

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        # No params.
        pass


# ──────────────────────────────────────────────────────────────────────
# UnaryNode — wraps a Module (1→1).
# ──────────────────────────────────────────────────────────────────────


struct UnaryNode[
    node_name: StaticString,
    Op: Module,
    in0_name: StaticString = "input",
](GraphNode):
    comptime NAME = Self.node_name
    comptime IN0_NAME = Self.in0_name
    comptime IN1_NAME = ""
    comptime IN0_DIM = Self.Op.IN_DIM
    comptime IN1_DIM = 0
    comptime OUT_DIM = Self.Op.OUT_DIM
    comptime KIND = 1

    var op: Self.Op

    # CPU buffers (used when ts.target_tag == TARGET_CPU).
    var _out_buf: List[Scalar[DT]]       # [BATCH, OUT_DIM]
    var _grad_out_buf: List[Scalar[DT]]  # [BATCH, OUT_DIM]
    var _grad_in0_buf: List[Scalar[DT]]  # [BATCH, IN0_DIM]
    var _grad_in1_buf: List[Scalar[DT]]  # length-1 stub for unary nodes

    # GPU buffers (used when ts.target_tag == TARGET_GPU).
    var _out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in0_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in1_buf_dev: Optional[DeviceBuffer[DT]]  # null; never sized
    var _out_buf_dev_n: Int
    var _grad_out_buf_dev_n: Int
    var _grad_in0_buf_dev_n: Int

    # Block C: cached pointers resolved at `ensure_buffers_via` time. Eliminates
    # the per-call `target_tag` branch + Optional.value() + rebind inside
    # `*_ptr_via()` (called O(N) times per forward + O(N) per backward by the
    # ComputeGraph name-resolution loops; ~30 calls/train_step for SAC).
    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]  # null for unary

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        self.op = Self.Op()
        self._out_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_in0_buf = List[Scalar[DT]]()
        self._grad_in1_buf = List[Scalar[DT]]()
        self._out_buf_dev = None
        self._grad_out_buf_dev = None
        self._grad_in0_buf_dev = None
        self._grad_in1_buf_dev = None
        self._out_buf_dev_n = 0
        self._grad_out_buf_dev_n = 0
        self._grad_in0_buf_dev_n = 0
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_in0_ptr = null_p
        self._grad_in1_ptr = null_p
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "UnaryNode.make_via[target='gpu', INIT] requires a DeviceContext"
        )
        var n = Self()
        n.op = Self.Op.make[target, INIT]()
        n.ts = TargetStorage.make_cpu()
        return n^

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "UnaryNode.make_via[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var n = Self()
        n.op = Self.Op.make[target, INIT](ctx)
        n.ts = TargetStorage.make_gpu(ctx)
        return n^

    def ensure_buffers_via[BATCH: Int](mut self) raises:
        if self.ts.target_tag == TARGET_GPU:
            if self._n_batch_buf < BATCH:
                var ctx = self.ts.ctx.value()
                ensure_gpu_buffer(
                    self._out_buf_dev, self._out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_out_buf_dev, self._grad_out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_in0_buf_dev, self._grad_in0_buf_dev_n,
                    BATCH * Self.IN0_DIM, ctx,
                )
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf_dev.value().unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf_dev.value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
                if len(self._grad_in1_buf) < 1:
                    self._grad_in1_buf.resize(1, Scalar[DT](0.0))
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf.unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf.unsafe_ptr())
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in0_ptr

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        # Unary node: no in1 buffer. Null pointer; callers must
        # not dereference (they check IN1_NAME == "" first).
        return self._grad_in1_ptr

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var in0_t = TileTensor(in0_ptr, row_major[BATCH, Self.IN0_DIM]())
        var out_p = self.out_ptr_via()
        var out_t = TileTensor(out_p, row_major[BATCH, Self.OUT_DIM]())
        self.op.forward[target, BATCH, POLICY=POLICY](in0_t, out_t)

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var go_p = self.grad_out_ptr_via()
        var gi0_p = self.grad_in0_ptr_via()
        var go_t = TileTensor(go_p, row_major[BATCH, Self.OUT_DIM]())
        var gi0_t = TileTensor(gi0_p, row_major[BATCH, Self.IN0_DIM]())
        self.op.vjp[target, BATCH, POLICY=POLICY](go_t, gi0_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self.op.for_each_param[target, V](prefix, visitor)

    def set_op_attr_via[ATTR: StaticString](
        mut self, value: Scalar[DT],
    ):
        self.op.set_attr[ATTR](value)


# ──────────────────────────────────────────────────────────────────────
# BinaryNode — wraps a BinaryModule (2→1).
# ──────────────────────────────────────────────────────────────────────


struct BinaryNode[
    node_name: StaticString,
    Op: BinaryModule,
    in0_name: StaticString = "input",
    in1_name: StaticString = "input",
](GraphNode):
    comptime NAME = Self.node_name
    comptime IN0_NAME = Self.in0_name
    comptime IN1_NAME = Self.in1_name
    comptime IN0_DIM = Self.Op.IN0_DIM
    comptime IN1_DIM = Self.Op.IN1_DIM
    comptime OUT_DIM = Self.Op.OUT_DIM
    comptime KIND = 2

    var op: Self.Op

    var _out_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    var _grad_in0_buf: List[Scalar[DT]]
    var _grad_in1_buf: List[Scalar[DT]]

    var _out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in0_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in1_buf_dev: Optional[DeviceBuffer[DT]]
    var _out_buf_dev_n: Int
    var _grad_out_buf_dev_n: Int
    var _grad_in0_buf_dev_n: Int
    var _grad_in1_buf_dev_n: Int

    # Block C: cached pointers resolved at `ensure_buffers_via` time.
    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        self.op = Self.Op()
        self._out_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_in0_buf = List[Scalar[DT]]()
        self._grad_in1_buf = List[Scalar[DT]]()
        self._out_buf_dev = None
        self._grad_out_buf_dev = None
        self._grad_in0_buf_dev = None
        self._grad_in1_buf_dev = None
        self._out_buf_dev_n = 0
        self._grad_out_buf_dev_n = 0
        self._grad_in0_buf_dev_n = 0
        self._grad_in1_buf_dev_n = 0
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_in0_ptr = null_p
        self._grad_in1_ptr = null_p
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "BinaryNode.make_via[target='gpu', INIT] requires a DeviceContext"
        )
        var n = Self()
        n.op = Self.Op.make[target, INIT]()
        n.ts = TargetStorage.make_cpu()
        return n^

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "BinaryNode.make_via[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var n = Self()
        n.op = Self.Op.make[target, INIT](ctx)
        n.ts = TargetStorage.make_gpu(ctx)
        return n^

    def ensure_buffers_via[BATCH: Int](mut self) raises:
        if self.ts.target_tag == TARGET_GPU:
            if self._n_batch_buf < BATCH:
                var ctx = self.ts.ctx.value()
                ensure_gpu_buffer(
                    self._out_buf_dev, self._out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_out_buf_dev, self._grad_out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_in0_buf_dev, self._grad_in0_buf_dev_n,
                    BATCH * Self.IN0_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_in1_buf_dev, self._grad_in1_buf_dev_n,
                    BATCH * Self.IN1_DIM, ctx,
                )
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf_dev.value().unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf_dev.value().unsafe_ptr())
                self._grad_in1_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in1_buf_dev.value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
                self._grad_in1_buf.resize(BATCH * Self.IN1_DIM, Scalar[DT](0.0))
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf.unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf.unsafe_ptr())
                self._grad_in1_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in1_buf.unsafe_ptr())
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in0_ptr

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in1_ptr

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var in0_t = TileTensor(in0_ptr, row_major[BATCH, Self.IN0_DIM]())
        var in1_t = TileTensor(in1_ptr, row_major[BATCH, Self.IN1_DIM]())
        var out_p = self.out_ptr_via()
        var out_t = TileTensor(out_p, row_major[BATCH, Self.OUT_DIM]())
        self.op.forward[target, BATCH, POLICY=POLICY](in0_t, in1_t, out_t)

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var go_p = self.grad_out_ptr_via()
        var gi0_p = self.grad_in0_ptr_via()
        var gi1_p = self.grad_in1_ptr_via()
        var go_t = TileTensor(go_p, row_major[BATCH, Self.OUT_DIM]())
        var gi0_t = TileTensor(gi0_p, row_major[BATCH, Self.IN0_DIM]())
        var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN1_DIM]())
        self.op.vjp[target, BATCH, POLICY=POLICY](go_t, gi0_t, gi1_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self.op.for_each_param[target, V](prefix, visitor)

    def set_op_attr_via[ATTR: StaticString](
        mut self, value: Scalar[DT],
    ):
        self.op.set_attr[ATTR](value)


# ──────────────────────────────────────────────────────────────────────
# ExternalUnaryNode — Phase 3. Like UnaryNode but does NOT own its op.
#
# The node holds `_module_ptr: UnsafePointer[M, MutAnyOrigin]` to a
# Module instance owned by the caller (typically the trainer). The
# pointer is set per-call via `graph.set_external[NAME](mut module)`;
# `forward_via` / `vjp_via` dereference and invoke the standard
# `Module.forward` / `Module.backward` methods.
#
# `MODE` plumbs into `M.vjp[mode]` at comptime — `MODE="input_only"`
# expresses stop-grad references inline without needing `StopGradParams`.
# Param accumulation stays on the external instance; the graph never
# walks the module's params. The trainer's own optimizer step does
# (when `MODE="all"`).
# ──────────────────────────────────────────────────────────────────────


struct ExternalUnaryNode[
    node_name: StaticString,
    M: Module,
    in0_name: StaticString = "input",
    MODE: StaticString = "all",
](GraphNode):
    comptime NAME = Self.node_name
    comptime IN0_NAME = Self.in0_name
    comptime IN1_NAME = ""
    comptime IN0_DIM = Self.M.IN_DIM
    comptime IN1_DIM = 0
    comptime OUT_DIM = Self.M.OUT_DIM
    comptime KIND = 1  # same scatter-add path as UnaryNode

    # Type-erased so the GraphNode trait can carry a uniform
    # `set_external_via` method. Rebound to UnsafePointer[Self.M] at
    # every dispatch site (forward_via / vjp_via).
    var _module_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _out_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    var _grad_in0_buf: List[Scalar[DT]]
    var _grad_in1_buf: List[Scalar[DT]]

    var _out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in0_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in1_buf_dev: Optional[DeviceBuffer[DT]]
    var _out_buf_dev_n: Int
    var _grad_out_buf_dev_n: Int
    var _grad_in0_buf_dev_n: Int

    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        self._module_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_in0_buf = List[Scalar[DT]]()
        self._grad_in1_buf = List[Scalar[DT]]()
        self._out_buf_dev = None
        self._grad_out_buf_dev = None
        self._grad_in0_buf_dev = None
        self._grad_in1_buf_dev = None
        self._out_buf_dev_n = 0
        self._grad_out_buf_dev_n = 0
        self._grad_in0_buf_dev_n = 0
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_in0_ptr = null_p
        self._grad_in1_ptr = null_p
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "ExternalUnaryNode.make_via[target='gpu', INIT] requires a DeviceContext"
        )
        var n = Self()
        n.ts = TargetStorage.make_cpu()
        return n^

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "ExternalUnaryNode.make_via[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var n = Self()
        n.ts = TargetStorage.make_gpu(ctx)
        return n^

    def set_external_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Bind the external Module instance whose `forward`/`backward`
        this node dispatches to (ExternalUnaryNode variant). Pointer is
        type-erased at the trait surface; rebound to
        `UnsafePointer[Self.M]` at dispatch time."""
        self._module_ptr = ptr

    def ensure_buffers_via[BATCH: Int](mut self) raises:
        if self.ts.target_tag == TARGET_GPU:
            if self._n_batch_buf < BATCH:
                var ctx = self.ts.ctx.value()
                ensure_gpu_buffer(
                    self._out_buf_dev, self._out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_out_buf_dev, self._grad_out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_in0_buf_dev, self._grad_in0_buf_dev_n,
                    BATCH * Self.IN0_DIM, ctx,
                )
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf_dev.value().unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf_dev.value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
                if len(self._grad_in1_buf) < 1:
                    self._grad_in1_buf.resize(1, Scalar[DT](0.0))
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf.unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf.unsafe_ptr())
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in0_ptr

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in1_ptr  # null — external unary has no in1

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var in0_t = TileTensor(in0_ptr, row_major[BATCH, Self.IN0_DIM]())
        var out_p = self.out_ptr_via()
        var out_t = TileTensor(out_p, row_major[BATCH, Self.OUT_DIM]())
        var typed_ptr = rebind[UnsafePointer[Self.M, MutAnyOrigin]](
            self._module_ptr
        )
        typed_ptr[].forward[target, BATCH, POLICY=POLICY](in0_t, out_t)

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var go_p = self.grad_out_ptr_via()
        var gi0_p = self.grad_in0_ptr_via()
        var go_t = TileTensor(go_p, row_major[BATCH, Self.OUT_DIM]())
        var gi0_t = TileTensor(gi0_p, row_major[BATCH, Self.IN0_DIM]())
        var typed_ptr = rebind[UnsafePointer[Self.M, MutAnyOrigin]](
            self._module_ptr
        )
        typed_ptr[].vjp[
            target, BATCH, POLICY=POLICY, mode=Self.MODE,
        ](go_t, gi0_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        pass

    def set_op_attr_via[ATTR: StaticString](
        mut self, value: Scalar[DT],
    ):
        pass


# ──────────────────────────────────────────────────────────────────────
# ExternalBinaryNode — BinaryModule sibling of ExternalUnaryNode.
# ──────────────────────────────────────────────────────────────────────


struct ExternalBinaryNode[
    node_name: StaticString,
    M: BinaryModule,
    in0_name: StaticString = "input",
    in1_name: StaticString = "input",
    MODE: StaticString = "all",
](GraphNode):
    comptime NAME = Self.node_name
    comptime IN0_NAME = Self.in0_name
    comptime IN1_NAME = Self.in1_name
    comptime IN0_DIM = Self.M.IN0_DIM
    comptime IN1_DIM = Self.M.IN1_DIM
    comptime OUT_DIM = Self.M.OUT_DIM
    comptime KIND = 2

    # Type-erased so the GraphNode trait can carry a uniform
    # `set_external_via` method. Rebound to UnsafePointer[Self.M] at
    # every dispatch site (forward_via / vjp_via).
    var _module_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _out_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    var _grad_in0_buf: List[Scalar[DT]]
    var _grad_in1_buf: List[Scalar[DT]]

    var _out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in0_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_in1_buf_dev: Optional[DeviceBuffer[DT]]
    var _out_buf_dev_n: Int
    var _grad_out_buf_dev_n: Int
    var _grad_in0_buf_dev_n: Int
    var _grad_in1_buf_dev_n: Int

    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        self._module_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_in0_buf = List[Scalar[DT]]()
        self._grad_in1_buf = List[Scalar[DT]]()
        self._out_buf_dev = None
        self._grad_out_buf_dev = None
        self._grad_in0_buf_dev = None
        self._grad_in1_buf_dev = None
        self._out_buf_dev_n = 0
        self._grad_out_buf_dev_n = 0
        self._grad_in0_buf_dev_n = 0
        self._grad_in1_buf_dev_n = 0
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_in0_ptr = null_p
        self._grad_in1_ptr = null_p
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "ExternalBinaryNode.make_via[target='gpu', INIT] requires a DeviceContext"
        )
        var n = Self()
        n.ts = TargetStorage.make_cpu()
        return n^

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "ExternalBinaryNode.make_via[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var n = Self()
        n.ts = TargetStorage.make_gpu(ctx)
        return n^

    def set_external_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self._module_ptr = ptr

    def ensure_buffers_via[BATCH: Int](mut self) raises:
        if self.ts.target_tag == TARGET_GPU:
            if self._n_batch_buf < BATCH:
                var ctx = self.ts.ctx.value()
                ensure_gpu_buffer(
                    self._out_buf_dev, self._out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_out_buf_dev, self._grad_out_buf_dev_n,
                    BATCH * Self.OUT_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_in0_buf_dev, self._grad_in0_buf_dev_n,
                    BATCH * Self.IN0_DIM, ctx,
                )
                ensure_gpu_buffer(
                    self._grad_in1_buf_dev, self._grad_in1_buf_dev_n,
                    BATCH * Self.IN1_DIM, ctx,
                )
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf_dev.value().unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf_dev.value().unsafe_ptr())
                self._grad_in1_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in1_buf_dev.value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
                self._grad_in1_buf.resize(BATCH * Self.IN1_DIM, Scalar[DT](0.0))
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf.unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                self._grad_in0_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in0_buf.unsafe_ptr())
                self._grad_in1_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_in1_buf.unsafe_ptr())
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in0_ptr

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_in1_ptr

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var in0_t = TileTensor(in0_ptr, row_major[BATCH, Self.IN0_DIM]())
        var in1_t = TileTensor(in1_ptr, row_major[BATCH, Self.IN1_DIM]())
        var out_p = self.out_ptr_via()
        var out_t = TileTensor(out_p, row_major[BATCH, Self.OUT_DIM]())
        var typed_ptr = rebind[UnsafePointer[Self.M, MutAnyOrigin]](
            self._module_ptr
        )
        typed_ptr[].forward[
            target, BATCH, POLICY=POLICY,
        ](in0_t, in1_t, out_t)

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var go_p = self.grad_out_ptr_via()
        var gi0_p = self.grad_in0_ptr_via()
        var gi1_p = self.grad_in1_ptr_via()
        var go_t = TileTensor(go_p, row_major[BATCH, Self.OUT_DIM]())
        var gi0_t = TileTensor(gi0_p, row_major[BATCH, Self.IN0_DIM]())
        var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN1_DIM]())
        var typed_ptr = rebind[UnsafePointer[Self.M, MutAnyOrigin]](
            self._module_ptr
        )
        typed_ptr[].vjp[
            target, BATCH, POLICY=POLICY, mode=Self.MODE,
        ](go_t, gi0_t, gi1_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        pass

    def set_op_attr_via[ATTR: StaticString](
        mut self, value: Scalar[DT],
    ):
        pass

