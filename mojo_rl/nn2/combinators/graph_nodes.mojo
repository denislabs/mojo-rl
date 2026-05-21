"""UnaryNode / BinaryNode — GraphNode wrappers around Module / BinaryModule.

Phase 10D + Block A (GPU). Each wrapper owns its underlying op instance,
plus four buffers (out, grad_out, grad_in0, grad_in1) used by
ComputeGraph v2 for inter-node wiring.

`UnaryNode[NAME, M, IN0_NAME]`
    wraps a `M: Module` (1→1). `KIND = 1`, `IN1_NAME = ""`, `IN1_DIM = 0`.
    `forward_via(in0, _)` ignores the second input; `backward_via` writes
    only `_grad_in0_buf`.

`BinaryNode[NAME, BM, IN0_NAME, IN1_NAME]`
    wraps a `BM: BinaryModule` (2→1). `KIND = 2`. `forward_via(in0, in1)`
    uses both pointers; `backward_via` writes both `_grad_in0_buf` and
    `_grad_in1_buf`.

Backward contract: the wrapper reads its own `_grad_out_buf` as the
incoming gradient (graph zeros + scatter-adds into this before the
call), and writes `_grad_in*_buf`. The graph scatter-adds those into
predecessors' grad_out_bufs after the call returns.

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
    TARGET_GPU,
)
from ..core.target_storage import (
    TargetStorage,
    ensure_gpu_buffer,
)


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
            self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
                if len(self._grad_in1_buf) < 1:
                    self._grad_in1_buf.resize(1, Scalar[DT](0.0))
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._out_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._out_buf.unsafe_ptr()
        )

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._grad_out_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_out_buf.unsafe_ptr()
        )

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._grad_in0_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_in0_buf.unsafe_ptr()
        )

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        # Unary node: no in1 buffer. Return a null pointer; callers must
        # not dereference (they check IN1_NAME == "" first).
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

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

    def backward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var go_p = self.grad_out_ptr_via()
        var gi0_p = self.grad_in0_ptr_via()
        var go_t = TileTensor(go_p, row_major[BATCH, Self.OUT_DIM]())
        var gi0_t = TileTensor(gi0_p, row_major[BATCH, Self.IN0_DIM]())
        self.op.backward[target, BATCH, POLICY=POLICY](go_t, gi0_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self.op.for_each_param[target, V](prefix, visitor)


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
            self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_in0_buf.resize(BATCH * Self.IN0_DIM, Scalar[DT](0.0))
                self._grad_in1_buf.resize(BATCH * Self.IN1_DIM, Scalar[DT](0.0))
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._out_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._out_buf.unsafe_ptr()
        )

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._grad_out_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_out_buf.unsafe_ptr()
        )

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._grad_in0_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_in0_buf.unsafe_ptr()
        )

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._grad_in1_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_in1_buf.unsafe_ptr()
        )

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

    def backward_via[
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
        self.op.backward[target, BATCH, POLICY=POLICY](go_t, gi0_t, gi1_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self.op.for_each_param[target, V](prefix, visitor)
