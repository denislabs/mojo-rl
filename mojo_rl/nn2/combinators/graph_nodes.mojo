"""InputSlot / Node / ExternalNode — GraphNode wrappers (Phase 4.6e).

Three structs (down from five) — the unary/binary split is gone:

`InputSlot[NAME, DIM]`
    `KIND = 0`. Represents one named external input to the graph. No
    compute; its `out_ptr_via()` returns a pointer set per-call by the
    graph's `set_input[NAME]` method. Its `grad_out_buf` is the
    accumulator for the input-gradient flowing back from consumers
    (`grad_input_ptr[NAME]` on the graph).

`Node[NAME, Op, IN0_NAME, IN1_NAME=""]`
    Wraps an owned `Op: Module`. `KIND = Op.ARITY` (1 = unary, 2 = binary,
    forward-compat for higher arities). `forward_via` / `vjp_via`
    comptime-branch on `Self.Op.ARITY` to dispatch with the right input
    count. For ARITY=1, `IN1_NAME` defaults to "" and `_grad_in1_buf`
    stays length-0.

`ExternalNode[NAME, M, IN0_NAME, IN1_NAME="", MODE="all"]`
    Like Node but does NOT own its op — holds `_module_ptr:
    UnsafePointer[Scalar[DT], MutAnyOrigin]` (type-erased at the trait
    surface, rebound to `UnsafePointer[Self.M]` at dispatch). Bound by
    `graph.set_external[NAME](mut module)`. `MODE` plumbs into `M.vjp[mode]`
    so a single declaration can express stop-grad-style references (e.g.
    `MODE="input_only"` for the actor-loss view of a critic — skips
    param-grad accumulation on that path).

Backward contract: the wrapper reads its own `_grad_out_buf` as the
incoming gradient (graph zeros + scatter-adds into this before the
call), and writes `_grad_in*_buf`. The graph scatter-adds those into
predecessors' grad_out_bufs after the call returns. For InputSlot,
vjp_via is a no-op — `_grad_out_buf` already holds the final
input-gradient by the time it's reached in the reverse-topo walk.

CPU vs GPU storage: each node carries `ts: TargetStorage` (matches the
rest of nn2). CPU nodes own `List[Scalar[DT]]` for each buffer; GPU
nodes own `Optional[DeviceBuffer[DT]]`. `*_ptr_via()` returns whichever
pointer was cached by `ensure_buffers_via` at the configured target.

Hetero-binary variadic workaround (Phase 4.6c): the Module trait's
variadic forward/vjp surface unifies layouts symbolically — even when
two binary inputs have different IN0_DIM / IN1_DIM values, the variadic
pack requires the same Layout *type*. Node/ExternalNode forward_via
constructs `in1_t` with `row_major[BATCH, Self.IN0_DIM]()` (matching
`in0_t`), and the leaf's `typed_view[BATCH, IN<i>_DIM]` rebuilds the
correctly-typed view from `.ptr`. The Layout carried by the variadic
TileTensor is dead metadata once unpacked.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    GraphNode,
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
)
from ..core.graph_node import names_to_inline_array
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
# flowing back to this input from all consumer nodes.
# ──────────────────────────────────────────────────────────────────────


struct InputSlot[
    slot_name: StaticString,
    DIM_: Int,
](GraphNode):
    comptime NAME = Self.slot_name
    # I.2.6.h/i — InputSlot has no inputs (KIND=0 → empty IN_DIMS + IN_NAMES).
    comptime IN_DIMS = InlineArray[Int, 0]()
    comptime IN_NAMES = InlineArray[StaticString, 0]()
    comptime OUT_DIM = Self.DIM_
    comptime KIND = 0

    # CPU/GPU grad accumulator.
    var _grad_out_buf: List[Scalar[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev_n: Int

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
    def make_via[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "InputSlot: target must be 'cpu' or 'gpu'"
        )
        var n = Self()
        comptime if target == "cpu":
            n.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("InputSlot.make_via[target='gpu']: ctx required")
            n.ts = TargetStorage.make_gpu(ctx.value())
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
        in_ptrs: InlineArray[
            UnsafePointer[Scalar[DT], MutAnyOrigin], Self.KIND,
        ],
    ) raises:
        # No compute: KIND=0 → empty in_ptrs; the slot's out_ptr is
        # whatever the caller set via graph.set_input[NAME](tile).
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
        pass


# ──────────────────────────────────────────────────────────────────────
# Node — wraps an owned Op: Module of any arity (1..4). I.2.5 raised
# the cap from 2 → 4 + switched the user-facing declaration to a
# variadic `*in_names: StaticString` (the spike-de-risked pattern).
# IN0_NAME / IN1_NAME / IN2_NAME / IN3_NAME are derived from `in_names`
# at comptime with safe "" defaults past ARITY-1.
# ──────────────────────────────────────────────────────────────────────


struct Node[
    node_name: StaticString,
    Op: Module,
    *in_names: StaticString,
](GraphNode):
    comptime NAME = Self.node_name
    # I.2.6.h/i — IN_DIMS / IN_NAMES come straight from Op + the variadic
    # struct param. IN0_DIM kept as a legacy accessor used by Node's own
    # forward_via / vjp_via for the IN0_DIM-shaped variadic-unification trick.
    comptime IN_DIMS = Self.Op.IN_DIMS
    comptime IN_NAMES = names_to_inline_array[
        Self.KIND, *Self.in_names,
    ]()
    comptime IN0_DIM = (
        Self.Op.IN_DIMS[0] if Self.Op.ARITY > 0 else 0
    )
    comptime OUT_DIM = Self.Op.OUT_DIM
    comptime KIND = Self.Op.ARITY

    var op: Self.Op

    var _out_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    # I.2.6: per-input grad bufs collapsed from 4 named fields into one
    # List-of-List indexed by k ∈ [0, ARITY). Sizing driven by IN_DIMS[k]
    # via comptime-for in ensure_buffers_via. Initialised with ARITY
    # empty inner Lists; lazy-grown on first ensure_buffers_via call.
    var _grad_ins_buf: List[List[Scalar[DT]]]

    var _out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_ins_buf_dev: List[Optional[DeviceBuffer[DT]]]
    var _out_buf_dev_n: Int
    var _grad_out_buf_dev_n: Int
    var _grad_ins_buf_dev_n: List[Int]

    # Cached pointers resolved by ensure_buffers_via — stable thereafter.
    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_ins_ptr: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.in_names.size == Self.Op.ARITY, (
            "Node: number of in_names must match Op.ARITY"
        )
        self.op = Self.Op()
        self._out_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_ins_buf = List[List[Scalar[DT]]]()
        self._out_buf_dev = None
        self._grad_out_buf_dev = None
        self._grad_ins_buf_dev = List[Optional[DeviceBuffer[DT]]]()
        self._out_buf_dev_n = 0
        self._grad_out_buf_dev_n = 0
        self._grad_ins_buf_dev_n = List[Int]()
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_ins_ptr = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        # Pre-populate the List-of-List slots with ARITY entries (one
        # per input slot). Unary leaves still get 1 entry; ARITY≥5
        # nodes get the right count, no cap. ARITY=0 (impossible for
        # Node — it's a compute node) would give empty lists, harmless.
        comptime for _ in range(Self.Op.ARITY):
            self._grad_ins_buf.append(List[Scalar[DT]]())
            self._grad_ins_buf_dev.append(None)
            self._grad_ins_buf_dev_n.append(0)
            self._grad_ins_ptr.append(null_p)
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Node: target must be 'cpu' or 'gpu'"
        )
        var n = Self()
        n.op = Self.Op.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            n.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Node.make_via[target='gpu']: ctx required")
            n.ts = TargetStorage.make_gpu(ctx.value())
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
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf_dev.value().unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                # I.2.6: uniform comptime-for over ARITY input slots.
                comptime for k in range(Self.Op.ARITY):
                    comptime in_dim_k = Self.Op.IN_DIMS[k]
                    ensure_gpu_buffer(
                        self._grad_ins_buf_dev[k],
                        self._grad_ins_buf_dev_n[k],
                        BATCH * in_dim_k, ctx,
                    )
                    self._grad_ins_ptr[k] = rebind[
                        UnsafePointer[Scalar[DT], MutAnyOrigin]
                    ](self._grad_ins_buf_dev[k].value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf.unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                comptime for k in range(Self.Op.ARITY):
                    comptime in_dim_k = Self.Op.IN_DIMS[k]
                    self._grad_ins_buf[k].resize(BATCH * in_dim_k, Scalar[DT](0.0))
                    self._grad_ins_ptr[k] = rebind[
                        UnsafePointer[Scalar[DT], MutAnyOrigin]
                    ](self._grad_ins_buf[k].unsafe_ptr())
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    # The grad_inK_ptr_via accessors read from the consolidated List
    # at slot K. They return a null pointer when ARITY ≤ K (which
    # callers must not dereference; ComputeGraph guards via kind >= K).
    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[0] if Self.Op.ARITY >= 1 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[1] if Self.Op.ARITY >= 2 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def grad_in2_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[2] if Self.Op.ARITY >= 3 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def grad_in3_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[3] if Self.Op.ARITY >= 4 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in_ptrs: InlineArray[
            UnsafePointer[Scalar[DT], MutAnyOrigin], Self.KIND,
        ],
    ) raises:
        # Hetero-variadic workaround: all input tiles share IN0_DIM
        # Layout for pack unification; leaves recover real per-input
        # shape via typed_view[BATCH, Self.IN_DIMS[k]]. See module.mojo.
        var in0_t = TileTensor(in_ptrs[0], row_major[BATCH, Self.IN0_DIM]())
        var out_p = self.out_ptr_via()
        var out_t = TileTensor(out_p, row_major[BATCH, Self.OUT_DIM]())
        comptime if Self.Op.ARITY == 1:
            self.op.forward[target, BATCH, POLICY=POLICY](in0_t, output=out_t)
        elif Self.Op.ARITY == 2:
            var in1_t = TileTensor(in_ptrs[1], row_major[BATCH, Self.IN0_DIM]())
            self.op.forward[target, BATCH, POLICY=POLICY](
                in0_t, in1_t, output=out_t,
            )
        elif Self.Op.ARITY == 3:
            var in1_t = TileTensor(in_ptrs[1], row_major[BATCH, Self.IN0_DIM]())
            var in2_t = TileTensor(in_ptrs[2], row_major[BATCH, Self.IN0_DIM]())
            self.op.forward[target, BATCH, POLICY=POLICY](
                in0_t, in1_t, in2_t, output=out_t,
            )
        else:  # ARITY == 4 (further arities mechanical: add another branch)
            var in1_t = TileTensor(in_ptrs[1], row_major[BATCH, Self.IN0_DIM]())
            var in2_t = TileTensor(in_ptrs[2], row_major[BATCH, Self.IN0_DIM]())
            var in3_t = TileTensor(in_ptrs[3], row_major[BATCH, Self.IN0_DIM]())
            self.op.forward[target, BATCH, POLICY=POLICY](
                in0_t, in1_t, in2_t, in3_t, output=out_t,
            )

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var go_p = self.grad_out_ptr_via()
        var gi0_p = self.grad_in0_ptr_via()
        var go_t = TileTensor(go_p, row_major[BATCH, Self.OUT_DIM]())
        var gi0_t = TileTensor(gi0_p, row_major[BATCH, Self.IN0_DIM]())
        comptime if Self.Op.ARITY == 1:
            self.op.vjp[target, BATCH, POLICY=POLICY](go_t, gi0_t)
        elif Self.Op.ARITY == 2:
            # Hetero-binary variadic workaround (see forward_via).
            var gi1_p = self.grad_in1_ptr_via()
            var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN0_DIM]())
            self.op.vjp[target, BATCH, POLICY=POLICY](go_t, gi0_t, gi1_t)
        elif Self.Op.ARITY == 3:
            var gi1_p = self.grad_in1_ptr_via()
            var gi2_p = self.grad_in2_ptr_via()
            var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN0_DIM]())
            var gi2_t = TileTensor(gi2_p, row_major[BATCH, Self.IN0_DIM]())
            self.op.vjp[target, BATCH, POLICY=POLICY](
                go_t, gi0_t, gi1_t, gi2_t,
            )
        else:  # ARITY == 4
            var gi1_p = self.grad_in1_ptr_via()
            var gi2_p = self.grad_in2_ptr_via()
            var gi3_p = self.grad_in3_ptr_via()
            var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN0_DIM]())
            var gi2_t = TileTensor(gi2_p, row_major[BATCH, Self.IN0_DIM]())
            var gi3_t = TileTensor(gi3_p, row_major[BATCH, Self.IN0_DIM]())
            self.op.vjp[target, BATCH, POLICY=POLICY](
                go_t, gi0_t, gi1_t, gi2_t, gi3_t,
            )

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
# ExternalNode — wraps an externally-owned Module of any arity.
#
# Same buffer layout as Node, but op is type-erased to a raw pointer
# bound per-call via `graph.set_external[NAME](mut module)`. MODE plumbs
# into M.vjp[mode] so stop-grad-style references can be expressed
# inline (e.g. MODE="input_only" for actor-loss view of critics).
# ──────────────────────────────────────────────────────────────────────


struct ExternalNode[
    node_name: StaticString,
    M: Module,
    *in_names: StaticString,
    MODE: StaticString = "all",
](GraphNode):
    """I.2.5 — `MODE` is now keyword-only (lives after `*in_names` in
    the struct param list). Existing call sites that pass `MODE="input_only"`
    continue to compile; positional-style `MODE` calls would conflict
    with the variadic and were never used in the codebase."""

    comptime NAME = Self.node_name
    # I.2.6.h/i — IN_DIMS / IN_NAMES come straight from M + the variadic
    # struct param.
    comptime IN_DIMS = Self.M.IN_DIMS
    comptime IN_NAMES = names_to_inline_array[
        Self.KIND, *Self.in_names,
    ]()
    comptime IN0_DIM = (
        Self.M.IN_DIMS[0] if Self.M.ARITY > 0 else 0
    )
    comptime OUT_DIM = Self.M.OUT_DIM
    comptime KIND = Self.M.ARITY

    # Type-erased so GraphNode.set_external_via carries a uniform
    # signature. Rebound to UnsafePointer[Self.M] at every dispatch site.
    var _module_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var _out_buf: List[Scalar[DT]]
    var _grad_out_buf: List[Scalar[DT]]
    # I.2.6: per-input grad bufs collapsed from 4 named fields into one
    # List-of-List indexed by k ∈ [0, ARITY). See Node above for the
    # same rationale.
    var _grad_ins_buf: List[List[Scalar[DT]]]

    var _out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_out_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_ins_buf_dev: List[Optional[DeviceBuffer[DT]]]
    var _out_buf_dev_n: Int
    var _grad_out_buf_dev_n: Int
    var _grad_ins_buf_dev_n: List[Int]

    var _out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _grad_ins_ptr: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    var _n_batch_buf: Int
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.in_names.size == Self.M.ARITY, (
            "ExternalNode: number of in_names must match M.ARITY"
        )
        self._module_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_buf = List[Scalar[DT]]()
        self._grad_out_buf = List[Scalar[DT]]()
        self._grad_ins_buf = List[List[Scalar[DT]]]()
        self._out_buf_dev = None
        self._grad_out_buf_dev = None
        self._grad_ins_buf_dev = List[Optional[DeviceBuffer[DT]]]()
        self._out_buf_dev_n = 0
        self._grad_out_buf_dev_n = 0
        self._grad_ins_buf_dev_n = List[Int]()
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._out_ptr = null_p
        self._grad_out_ptr = null_p
        self._grad_ins_ptr = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        comptime for _ in range(Self.M.ARITY):
            self._grad_ins_buf.append(List[Scalar[DT]]())
            self._grad_ins_buf_dev.append(None)
            self._grad_ins_buf_dev_n.append(0)
            self._grad_ins_ptr.append(null_p)
        self._n_batch_buf = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "ExternalNode: target must be 'cpu' or 'gpu'"
        )
        var n = Self()
        comptime if target == "cpu":
            n.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("ExternalNode.make_via[target='gpu']: ctx required")
            n.ts = TargetStorage.make_gpu(ctx.value())
        return n^

    def set_external_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Bind the external Module instance. Pointer is type-erased at the
        trait surface; rebound to UnsafePointer[Self.M] at dispatch time."""
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
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf_dev.value().unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf_dev.value().unsafe_ptr())
                # I.2.6: uniform comptime-for over ARITY input slots.
                comptime for k in range(Self.M.ARITY):
                    comptime in_dim_k = Self.M.IN_DIMS[k]
                    ensure_gpu_buffer(
                        self._grad_ins_buf_dev[k],
                        self._grad_ins_buf_dev_n[k],
                        BATCH * in_dim_k, ctx,
                    )
                    self._grad_ins_ptr[k] = rebind[
                        UnsafePointer[Scalar[DT], MutAnyOrigin]
                    ](self._grad_ins_buf_dev[k].value().unsafe_ptr())
                self._n_batch_buf = BATCH
        else:
            if self._n_batch_buf < BATCH:
                self._out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._grad_out_buf.resize(BATCH * Self.OUT_DIM, Scalar[DT](0.0))
                self._out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._out_buf.unsafe_ptr())
                self._grad_out_ptr = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self._grad_out_buf.unsafe_ptr())
                comptime for k in range(Self.M.ARITY):
                    comptime in_dim_k = Self.M.IN_DIMS[k]
                    self._grad_ins_buf[k].resize(BATCH * in_dim_k, Scalar[DT](0.0))
                    self._grad_ins_ptr[k] = rebind[
                        UnsafePointer[Scalar[DT], MutAnyOrigin]
                    ](self._grad_ins_buf[k].unsafe_ptr())
                self._n_batch_buf = BATCH

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._out_ptr

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_out_ptr

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[0] if Self.M.ARITY >= 1 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[1] if Self.M.ARITY >= 2 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def grad_in2_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[2] if Self.M.ARITY >= 3 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def grad_in3_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self._grad_ins_ptr[3] if Self.M.ARITY >= 4 else UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in_ptrs: InlineArray[
            UnsafePointer[Scalar[DT], MutAnyOrigin], Self.KIND,
        ],
    ) raises:
        var in0_t = TileTensor(in_ptrs[0], row_major[BATCH, Self.IN0_DIM]())
        var out_p = self.out_ptr_via()
        var out_t = TileTensor(out_p, row_major[BATCH, Self.OUT_DIM]())
        var typed_ptr = rebind[UnsafePointer[Self.M, MutAnyOrigin]](
            self._module_ptr
        )
        comptime if Self.M.ARITY == 1:
            typed_ptr[].forward[target, BATCH, POLICY=POLICY](
                in0_t, output=out_t,
            )
        elif Self.M.ARITY == 2:
            var in1_t = TileTensor(in_ptrs[1], row_major[BATCH, Self.IN0_DIM]())
            typed_ptr[].forward[target, BATCH, POLICY=POLICY](
                in0_t, in1_t, output=out_t,
            )
        elif Self.M.ARITY == 3:
            var in1_t = TileTensor(in_ptrs[1], row_major[BATCH, Self.IN0_DIM]())
            var in2_t = TileTensor(in_ptrs[2], row_major[BATCH, Self.IN0_DIM]())
            typed_ptr[].forward[target, BATCH, POLICY=POLICY](
                in0_t, in1_t, in2_t, output=out_t,
            )
        else:  # ARITY == 4
            var in1_t = TileTensor(in_ptrs[1], row_major[BATCH, Self.IN0_DIM]())
            var in2_t = TileTensor(in_ptrs[2], row_major[BATCH, Self.IN0_DIM]())
            var in3_t = TileTensor(in_ptrs[3], row_major[BATCH, Self.IN0_DIM]())
            typed_ptr[].forward[target, BATCH, POLICY=POLICY](
                in0_t, in1_t, in2_t, in3_t, output=out_t,
            )

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
        comptime if Self.M.ARITY == 1:
            typed_ptr[].vjp[
                target, BATCH, POLICY=POLICY, mode=Self.MODE,
            ](go_t, gi0_t)
        elif Self.M.ARITY == 2:
            var gi1_p = self.grad_in1_ptr_via()
            var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN0_DIM]())
            typed_ptr[].vjp[
                target, BATCH, POLICY=POLICY, mode=Self.MODE,
            ](go_t, gi0_t, gi1_t)
        elif Self.M.ARITY == 3:
            var gi1_p = self.grad_in1_ptr_via()
            var gi2_p = self.grad_in2_ptr_via()
            var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN0_DIM]())
            var gi2_t = TileTensor(gi2_p, row_major[BATCH, Self.IN0_DIM]())
            typed_ptr[].vjp[
                target, BATCH, POLICY=POLICY, mode=Self.MODE,
            ](go_t, gi0_t, gi1_t, gi2_t)
        else:  # ARITY == 4
            var gi1_p = self.grad_in1_ptr_via()
            var gi2_p = self.grad_in2_ptr_via()
            var gi3_p = self.grad_in3_ptr_via()
            var gi1_t = TileTensor(gi1_p, row_major[BATCH, Self.IN0_DIM]())
            var gi2_t = TileTensor(gi2_p, row_major[BATCH, Self.IN0_DIM]())
            var gi3_t = TileTensor(gi3_p, row_major[BATCH, Self.IN0_DIM]())
            typed_ptr[].vjp[
                target, BATCH, POLICY=POLICY, mode=Self.MODE,
            ](go_t, gi0_t, gi1_t, gi2_t, gi3_t)

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        pass

    # set_op_attr_via inherits trait default (no-op) — external module's
    # attrs are managed by its owner, not the graph.
