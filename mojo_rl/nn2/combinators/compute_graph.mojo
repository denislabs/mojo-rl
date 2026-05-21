"""ComputeGraph v2 — name-based DAG over GraphNode variadic. Phase 10D + Block A.

Builds a named DAG by composing `UnaryNode` / `BinaryNode` wrappers in
topological order. Each node carries `NAME` + predecessor names; the
graph resolves names at compile time via a `comptime for` double-loop.

```mojo
comptime SACActorGraph = ComputeGraph[
    OBS_DIM, 1,
    UnaryNode["actor",     ActorModel,         "input"],
    UnaryNode["rsample",   RSample[ACT_DIM],   "actor"],
    UnaryNode["q1",        CriticModel,        "input"],   # also takes action
    # ... etc
]
```

Inputs to the graph come in as the magic name `"input"` (graph external
input). The last node in `*NODES` is the graph's external output.

Memory: each node owns four buffers (out / grad_out / grad_in0 /
grad_in1). The graph owns one external `_grad_input_buf` for the
gradient flowing back to `"input"` (scatter-add target for any node
whose IN0/IN1 references `"input"`).

Forward (topo order):
  - comptime for each node i:
      - resolve IN0_NAME → either external `input_t.ptr` or NODES[j].out_ptr
      - resolve IN1_NAME similarly (or null for unary nodes)
      - call node_i.forward_via(in0_ptr, in1_ptr)
  - copy `nodes[N-1].out_ptr` → `output_t`

Backward (reverse topo):
  - zero all `grad_out_buf`s + the external `_grad_input_buf`
  - copy `grad_output_t` → `nodes[N-1].grad_out_buf`
  - comptime for each node i in reverse:
      - call node_i.backward_via (reads its grad_out_buf, writes
        grad_in0/1_buf)
      - scatter-add `node_i.grad_in0_buf` into predecessor's grad_out_buf
        (or external `_grad_input_buf` if IN0_NAME == "input")
      - same for grad_in1_buf if KIND == 2
  - copy `_grad_input_buf` → `grad_input_t`

Fan-out is handled by `+=` accumulation: when one producer feeds two
consumers, each consumer's grad_in_* writes scatter-add into the same
producer.grad_out_buf, naturally summing the gradient contributions.

Block A (Phase A1, 2026-05-21): GPU path. The external `_grad_input_buf`
gains an Optional[DeviceBuffer] sibling, sized lazily on first call.
`_forward_gpu` / `_backward_gpu` mirror the CPU bodies; scatter-add and
zero are GPU kernels.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    GraphNode,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_GPU,
)
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — buffer-wide zero / scatter-add / copy. Used to implement
# the inter-node wiring in `_forward_gpu` / `_backward_gpu`. Free
# module-level functions so `enqueue_function` can bind them.
# ──────────────────────────────────────────────────────────────────────


def _zero_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = Scalar[DT](0.0)


def _add_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


def _copy_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


def _enqueue_zero[N: Int](
    ctx: DeviceContext,
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)
    comptime TPB = 128
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _zero_kernel[N]
    ctx.enqueue_function[kernel](lt, grid_dim=n_blocks, block_dim=TPB)


def _enqueue_add[N: Int](
    ctx: DeviceContext,
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var dst_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](dst)
    var src_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](src)
    comptime TPB = 128
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _add_kernel[N]
    ctx.enqueue_function[kernel](
        dst_lt, src_lt, grid_dim=n_blocks, block_dim=TPB,
    )


def _enqueue_copy[N: Int](
    ctx: DeviceContext,
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var dst_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](dst)
    var src_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](src)
    comptime TPB = 128
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _copy_kernel[N]
    ctx.enqueue_function[kernel](
        dst_lt, src_lt, grid_dim=n_blocks, block_dim=TPB,
    )


# ──────────────────────────────────────────────────────────────────────
# ComputeGraph
# ──────────────────────────────────────────────────────────────────────


struct ComputeGraph[
    IN_DIM_: Int,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](Movable & ImplicitlyDestructible):
    comptime IN_DIM = Self.IN_DIM_
    comptime OUT_DIM = Self.OUT_DIM_
    comptime N = Self.NODES.size

    var nodes: Tuple[*Self.NODES]
    var ts: TargetStorage

    # External grad sink: scatter-add target for nodes referencing "input"
    # as IN0 or IN1. Sized [BATCH, IN_DIM] on first ensure_buffers.
    var _grad_input_buf: List[Scalar[DT]]
    var _grad_input_buf_dev: Optional[DeviceBuffer[DT]]
    var _grad_input_buf_dev_n: Int
    var _n_batch_grad_input: Int

    def __init__(out self):
        """Defaultable form — empty placeholders, tag=UNINIT."""
        comptime assert Self.N >= 1, "ComputeGraph requires at least one node"
        comptime assert Self.NODES[Self.N - 1].OUT_DIM == Self.OUT_DIM, (
            "ComputeGraph: last node OUT_DIM must equal graph OUT_DIM"
        )
        self.nodes = Tuple[*Self.NODES]()
        self.ts = TargetStorage.make_uninit()
        self._grad_input_buf = List[Scalar[DT]]()
        self._grad_input_buf_dev = None
        self._grad_input_buf_dev_n = 0
        self._n_batch_grad_input = 0

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory — recursive over nodes."""
        comptime assert target == "cpu", (
            "ComputeGraph.make[target='gpu', INIT] requires a DeviceContext"
        )
        var g = Self()
        comptime for i in range(Self.N):
            g.nodes[i] = Self.NODES[i].make_via[target, INIT]()
        g.ts = TargetStorage.make_cpu()
        return g^

    @staticmethod
    def make[target: StaticString, INIT: Initializer](
        ctx: DeviceContext
    ) raises -> Self:
        """GPU factory — recursive over nodes."""
        comptime assert target == "gpu", (
            "ComputeGraph.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var g = Self()
        comptime for i in range(Self.N):
            g.nodes[i] = Self.NODES[i].make_via[target, INIT](ctx)
        g.ts = TargetStorage.make_gpu(ctx)
        return g^

    def _ensure_all_buffers[BATCH: Int](mut self) raises:
        comptime for i in range(Self.N):
            self.nodes[i].ensure_buffers_via[BATCH]()
        if self.ts.target_tag == TARGET_GPU:
            ensure_gpu_buffer(
                self._grad_input_buf_dev, self._grad_input_buf_dev_n,
                BATCH * Self.IN_DIM, self.ts.ctx.value(),
            )
            self._n_batch_grad_input = BATCH
        else:
            if self._n_batch_grad_input < BATCH:
                self._grad_input_buf.resize(BATCH * Self.IN_DIM, Scalar[DT](0.0))
                self._n_batch_grad_input = BATCH

    # ──────────────────────────────────────────────────────────────────
    # Forward — topological walk, comptime name resolution.
    # ──────────────────────────────────────────────────────────────────

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
        comptime assert input.flat_rank == 2, "input must be rank-2"
        comptime assert output.flat_rank == 2, "output must be rank-2"
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        self._ensure_all_buffers[BATCH]()

        comptime if target == "cpu":
            _forward_cpu[
                target, BATCH, POLICY=POLICY,
            ](self, input, output)
        else:
            _forward_gpu[
                target, BATCH, POLICY=POLICY,
            ](self, input, output)

    # ──────────────────────────────────────────────────────────────────
    # Backward — reverse topo + scatter-add.
    #
    # The `mode` comptime param mirrors `Module.backward[mode]` (audit
    # Follow-up #7) for parity with the slim trait, but for a graph it's
    # not a meaningful distinction: each node decides what to accumulate
    # on its own. Callers that want frozen-params behaviour should wrap
    # individual sub-nets in `StopGradParams`. `mode` is accepted and
    # forwarded to every child so leaves can gate their own param-grad
    # work uniformly.
    # ──────────────────────────────────────────────────────────────────

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
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
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        self._ensure_all_buffers[BATCH]()

        comptime if target == "cpu":
            _backward_cpu[
                target, BATCH, POLICY=POLICY,
            ](self, grad_output, grad_input)
        else:
            _backward_gpu[
                target, BATCH, POLICY=POLICY,
            ](self, grad_output, grad_input)

    # ──────────────────────────────────────────────────────────────────
    # for_each_param — recurse with `node_name.` prefix.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.nodes[i].for_each_param_via[target, V](
                prefix + sep + String(Self.NODES[i].NAME),
                visitor,
            )

    def _grad_input_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.ts.target_tag == TARGET_GPU:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._grad_input_buf_dev.value().unsafe_ptr()
            )
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_input_buf.unsafe_ptr()
        )


# ──────────────────────────────────────────────────────────────────────
# Free-function forward / backward bodies (so the comptime double loops
# don't blow up the inliner the way an inline body would — same lesson
# as LeWM trainer extraction in 2026-05-14).
# ──────────────────────────────────────────────────────────────────────


def _forward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    IN_DIM_: Int,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[IN_DIM_, OUT_DIM_, *NODES],
    input: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime N = NODES.size

    # External input pointer, widened to MutAnyOrigin so it satisfies
    # the GraphNode trait's forward_via signature uniformly.
    var input_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)

    # Null sentinel for unary nodes' unused in1 slot.
    var null_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    comptime for i in range(N):
        comptime src0 = NODES[i].IN0_NAME
        comptime src1 = NODES[i].IN1_NAME
        comptime kind = NODES[i].KIND

        # Resolve in0_ptr.
        var in0_ptr = null_ptr
        comptime if src0 == "input":
            in0_ptr = input_ptr
        else:
            comptime for j in range(N):
                comptime if NODES[j].NAME == src0:
                    in0_ptr = g.nodes[j].out_ptr_via()

        # Resolve in1_ptr (only for binary nodes).
        var in1_ptr = null_ptr
        comptime if kind == 2:
            comptime if src1 == "input":
                in1_ptr = input_ptr
            else:
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src1:
                        in1_ptr = g.nodes[j].out_ptr_via()

        g.nodes[i].forward_via[target, BATCH, POLICY=POLICY](in0_ptr, in1_ptr)

    # Copy last node's out_buf into the external output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
    var last_out_ptr = g.nodes[N - 1].out_ptr_via()
    var total = BATCH * LAST_OUT_DIM
    for k in range(total):
        out_p[k] = last_out_ptr[k]


def _backward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    IN_DIM_: Int,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[IN_DIM_, OUT_DIM_, *NODES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mut grad_input: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime N = NODES.size

    # Zero all node grad_out_bufs (scatter-add targets) + the external
    # grad_input_buf.
    comptime for i in range(N):
        comptime OUT_DIM_i = NODES[i].OUT_DIM
        var go_p_i = g.nodes[i].grad_out_ptr_via()
        var total_i = BATCH * OUT_DIM_i
        for k in range(total_i):
            go_p_i[k] = Scalar[DT](0.0)
    var ext_gi_p = g._grad_input_ptr()
    var ext_total = BATCH * IN_DIM_
    for k in range(ext_total):
        ext_gi_p[k] = Scalar[DT](0.0)

    # Seed last node's grad_out_buf from the external grad_output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    var ext_go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
    var last_go_p = g.nodes[N - 1].grad_out_ptr_via()
    var last_total = BATCH * LAST_OUT_DIM
    for k in range(last_total):
        last_go_p[k] = ext_go_p[k]

    # Reverse topological backward.
    comptime for ridx in range(N):
        comptime i = N - 1 - ridx
        comptime src0 = NODES[i].IN0_NAME
        comptime src1 = NODES[i].IN1_NAME
        comptime kind = NODES[i].KIND
        comptime IN0_DIM_i = NODES[i].IN0_DIM
        comptime IN1_DIM_i = NODES[i].IN1_DIM

        # Run backward: reads node_i.grad_out_buf, writes grad_in0/1_buf.
        g.nodes[i].backward_via[target, BATCH, POLICY=POLICY]()

        # Scatter-add grad_in0_buf into predecessor's grad_out_buf
        # (or external _grad_input_buf if IN0_NAME == "input").
        var gi0_p = g.nodes[i].grad_in0_ptr_via()
        var total0 = BATCH * IN0_DIM_i
        comptime if src0 == "input":
            for k in range(total0):
                ext_gi_p[k] += gi0_p[k]
        else:
            comptime for j in range(N):
                comptime if NODES[j].NAME == src0:
                    var pred_go_p = g.nodes[j].grad_out_ptr_via()
                    for k in range(total0):
                        pred_go_p[k] += gi0_p[k]

        # Scatter-add grad_in1_buf (binary nodes only).
        comptime if kind == 2:
            var gi1_p = g.nodes[i].grad_in1_ptr_via()
            var total1 = BATCH * IN1_DIM_i
            comptime if src1 == "input":
                for k in range(total1):
                    ext_gi_p[k] += gi1_p[k]
            else:
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src1:
                        var pred_go_p = g.nodes[j].grad_out_ptr_via()
                        for k in range(total1):
                            pred_go_p[k] += gi1_p[k]

    # Copy external grad_input_buf into the caller's grad_input tile.
    var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
    for k in range(ext_total):
        gi_p[k] = ext_gi_p[k]


# ──────────────────────────────────────────────────────────────────────
# GPU forward / backward — mirror the CPU bodies, swap scalar loops for
# `_enqueue_*` kernel calls. Buffers come from each node's GPU storage
# (chosen via `*_ptr_via()` based on `ts.target_tag`).
# ──────────────────────────────────────────────────────────────────────


def _forward_gpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    IN_DIM_: Int,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[IN_DIM_, OUT_DIM_, *NODES],
    input: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime N = NODES.size
    var ctx = g.ts.ctx.value()
    var input_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
    var null_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    comptime for i in range(N):
        comptime src0 = NODES[i].IN0_NAME
        comptime src1 = NODES[i].IN1_NAME
        comptime kind = NODES[i].KIND

        var in0_ptr = null_ptr
        comptime if src0 == "input":
            in0_ptr = input_ptr
        else:
            comptime for j in range(N):
                comptime if NODES[j].NAME == src0:
                    in0_ptr = g.nodes[j].out_ptr_via()

        var in1_ptr = null_ptr
        comptime if kind == 2:
            comptime if src1 == "input":
                in1_ptr = input_ptr
            else:
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src1:
                        in1_ptr = g.nodes[j].out_ptr_via()

        g.nodes[i].forward_via[target, BATCH, POLICY=POLICY](in0_ptr, in1_ptr)

    # Copy last node's out_buf into the external output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    comptime last_total = BATCH * LAST_OUT_DIM
    var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
    var last_out_ptr = g.nodes[N - 1].out_ptr_via()
    _enqueue_copy[last_total](ctx, out_p, last_out_ptr)


def _backward_gpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    IN_DIM_: Int,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[IN_DIM_, OUT_DIM_, *NODES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mut grad_input: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime N = NODES.size
    var ctx = g.ts.ctx.value()

    # Zero all node grad_out_bufs + the external grad_input_buf.
    comptime for i in range(N):
        comptime OUT_DIM_i = NODES[i].OUT_DIM
        comptime total_i = BATCH * OUT_DIM_i
        var go_p_i = g.nodes[i].grad_out_ptr_via()
        _enqueue_zero[total_i](ctx, go_p_i)
    var ext_gi_p = g._grad_input_ptr()
    comptime ext_total = BATCH * IN_DIM_
    _enqueue_zero[ext_total](ctx, ext_gi_p)

    # Seed last node's grad_out_buf from the external grad_output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    comptime last_total = BATCH * LAST_OUT_DIM
    var ext_go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
    var last_go_p = g.nodes[N - 1].grad_out_ptr_via()
    _enqueue_copy[last_total](ctx, last_go_p, ext_go_p)

    # Reverse topological backward.
    comptime for ridx in range(N):
        comptime i = N - 1 - ridx
        comptime src0 = NODES[i].IN0_NAME
        comptime src1 = NODES[i].IN1_NAME
        comptime kind = NODES[i].KIND
        comptime IN0_DIM_i = NODES[i].IN0_DIM
        comptime IN1_DIM_i = NODES[i].IN1_DIM

        g.nodes[i].backward_via[target, BATCH, POLICY=POLICY]()

        var gi0_p = g.nodes[i].grad_in0_ptr_via()
        comptime total0 = BATCH * IN0_DIM_i
        comptime if src0 == "input":
            _enqueue_add[total0](ctx, ext_gi_p, gi0_p)
        else:
            comptime for j in range(N):
                comptime if NODES[j].NAME == src0:
                    var pred_go_p = g.nodes[j].grad_out_ptr_via()
                    _enqueue_add[total0](ctx, pred_go_p, gi0_p)

        comptime if kind == 2:
            var gi1_p = g.nodes[i].grad_in1_ptr_via()
            comptime total1 = BATCH * IN1_DIM_i
            comptime if src1 == "input":
                _enqueue_add[total1](ctx, ext_gi_p, gi1_p)
            else:
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src1:
                        var pred_go_p = g.nodes[j].grad_out_ptr_via()
                        _enqueue_add[total1](ctx, pred_go_p, gi1_p)

    # Copy external grad_input_buf into the caller's grad_input tile.
    var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
    _enqueue_copy[ext_total](ctx, gi_p, ext_gi_p)
