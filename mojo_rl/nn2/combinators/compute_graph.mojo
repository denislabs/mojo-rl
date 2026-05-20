"""ComputeGraph v2 — name-based DAG over GraphNode variadic. Phase 10D.

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

CPU only (Phase 10D). GPU lands when CG v2 is the actual SAC path
(Phase 10F).
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    GraphNode,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


struct ComputeGraph[
    IN_DIM_: Int,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](Movable & ImplicitlyDestructible):
    comptime IN_DIM = Self.IN_DIM_
    comptime OUT_DIM = Self.OUT_DIM_
    comptime N = Self.NODES.size

    var nodes: Tuple[*Self.NODES]
    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    # External grad sink: scatter-add target for nodes referencing "input"
    # as IN0 or IN1. Sized [BATCH, IN_DIM] on first ensure_buffers.
    var _grad_input_buf: List[Scalar[DT]]
    var _n_batch_grad_input: Int

    def __init__(out self):
        """Defaultable form — empty placeholders, tag=UNINIT."""
        comptime assert Self.N >= 1, "ComputeGraph requires at least one node"
        comptime assert Self.NODES[Self.N - 1].OUT_DIM == Self.OUT_DIM, (
            "ComputeGraph: last node OUT_DIM must equal graph OUT_DIM"
        )
        self.nodes = Tuple[*Self.NODES]()
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False
        self._grad_input_buf = List[Scalar[DT]]()
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
        g._target_tag = TARGET_CPU
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
        g.ctx = ctx
        g._target_tag = TARGET_GPU
        return g^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "ComputeGraph: method called with [target='"
                + String(target)
                + "'] but graph was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def _ensure_all_buffers[BATCH: Int](mut self) raises:
        comptime for i in range(Self.N):
            self.nodes[i].ensure_buffers_via[BATCH]()
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
        self._assert_tag[target]()
        self._ensure_all_buffers[BATCH]()

        comptime if target == "cpu":
            _forward_cpu[
                target, BATCH, POLICY=POLICY,
            ](self, input, output)
        else:
            raise Error(
                "ComputeGraph: GPU forward not yet implemented (Phase 10D CPU only)"
            )

    # ──────────────────────────────────────────────────────────────────
    # Backward — reverse topo + scatter-add.
    # ──────────────────────────────────────────────────────────────────

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
        self._ensure_all_buffers[BATCH]()

        comptime if target == "cpu":
            _backward_cpu[
                target, BATCH, POLICY=POLICY,
            ](self, grad_output, grad_input)
        else:
            raise Error(
                "ComputeGraph: GPU backward not yet implemented (Phase 10D CPU only)"
            )

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
        # ComputeGraph's nodes own their own params; backward already
        # accumulates only what each node opts in to. backward_input is
        # not a meaningful operation for a graph as a whole — delegate
        # to backward. Callers that want frozen-params behavior should
        # wrap individual sub-nets in StopGradParams.
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    # ──────────────────────────────────────────────────────────────────
    # for_each_param — recurse with `node_name.` prefix.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.nodes[i].for_each_param_via[target](
                prefix + sep + String(Self.NODES[i].NAME),
                visitor,
            )

    def set_inference(mut self, value: Bool):
        self._inference = value
        comptime for i in range(Self.N):
            self.nodes[i].set_inference_via(value)


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
    var ext_gi_p = g._grad_input_buf.unsafe_ptr()
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
