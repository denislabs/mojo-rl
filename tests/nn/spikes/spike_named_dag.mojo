"""Spike DR.4 — name-based DAG (v1 ComputeGraph style) under nn Module trait.

Goal: confirm that v1's `*NODES: GraphNode` variadic + comptime name
resolution pattern still works in current Mojo nightly, ported to the
nn Module trait (instead of v1's stateless Model trait).

The shape we want to validate:

    struct Graph[*NODES: GraphNode](...):
        var nodes: Tuple[*Self.NODES]  # owns one instance per node

        def forward[BATCH](input_t):
            comptime for i in range(N):
                # resolve IN0_NAME at compile time:
                comptime src0 = Self.NODES[i].IN0_NAME
                comptime if src0 == "input":
                    use input_t
                else:
                    comptime for j in range(N):
                        comptime if Self.NODES[j].NAME == src0:
                            use self.nodes[j].out_buf
                self.nodes[i].forward(...)

The interesting question isn't whether comptime-for + comptime-if work
(they do — Sequential proves it). It's whether:
  1. A GraphNode struct can wrap an arbitrary `Op: Module` while carrying
     name strings as comptime params.
  2. The comptime double-loop name resolution (find node j with NAME == src)
     compiles cleanly + folds at compile time.
  3. We can route activations through the variadic Tuple based on comptime
     name lookups.

Minimal validation: a 3-node chain `n0 → n1 → n2` (single-input) with
arbitrary node names, where node n2 sources from "n1" by name.
"""

from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn.constants import DT


# ──────────────────────────────────────────────────────────────────────
# Tiny DT-only Module-style trait. Stripped-down for spike.
# ──────────────────────────────────────────────────────────────────────


trait MiniModule(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    def forward[
        BATCH: Int, LIN: TensorLayout, OIN: MutOrigin,
    ](mut self, input: TileTensor[DT, LIN, OIN]) raises:
        ...

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...


# ──────────────────────────────────────────────────────────────────────
# Concrete MiniModule: bias-add. Owns out_buf.
# ──────────────────────────────────────────────────────────────────────


struct BiasAdd[IN: Int, BIAS: Float64](MiniModule):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.IN

    var out_buf: List[Scalar[DT]]
    var n_batch: Int

    def __init__(out self):
        self.out_buf = List[Scalar[DT]]()
        self.n_batch = 0

    def forward[
        BATCH: Int, LIN: TensorLayout, OIN: MutOrigin,
    ](mut self, input: TileTensor[DT, LIN, OIN]) raises:
        comptime assert input.flat_rank == 2, "BiasAdd: input must be rank-2"
        if self.n_batch < BATCH:
            self.out_buf.resize(BATCH * Self.IN, Scalar[DT](0.0))
            self.n_batch = BATCH
        var out_p = self.out_buf.unsafe_ptr()
        var b_v: Scalar[DT] = Scalar[DT](Self.BIAS)
        for b in range(BATCH):
            for j in range(Self.IN):
                out_p[b * Self.IN + j] = input[b, j] + b_v

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.out_buf.unsafe_ptr()
        )


# ──────────────────────────────────────────────────────────────────────
# GraphNode trait: wraps a MiniModule, carries NAME + IN0_NAME (single
# input only for the spike; multi-input is a separate question).
# ──────────────────────────────────────────────────────────────────────


trait GraphNode(Defaultable & Movable & ImplicitlyDestructible):
    comptime NAME: StaticString
    comptime IN0_NAME: StaticString
    comptime OP_IN_DIM: Int
    comptime OP_OUT_DIM: Int

    def node_forward[
        BATCH: Int, LIN: TensorLayout, OIN: MutOrigin,
    ](mut self, input: TileTensor[DT, LIN, OIN]) raises:
        ...

    def node_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...


struct GNode[
    node_name: StaticString,
    Op: MiniModule,
    in0_name: StaticString = "input",
](GraphNode):
    comptime NAME = Self.node_name
    comptime IN0_NAME = Self.in0_name
    comptime OP_IN_DIM = Self.Op.IN_DIM
    comptime OP_OUT_DIM = Self.Op.OUT_DIM

    var op: Self.Op

    def __init__(out self):
        self.op = Self.Op()

    def node_forward[
        BATCH: Int, LIN: TensorLayout, OIN: MutOrigin,
    ](mut self, input: TileTensor[DT, LIN, OIN]) raises:
        self.op.forward[BATCH](input)

    def node_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.op.out_ptr()


# ──────────────────────────────────────────────────────────────────────
# Mini Graph: walks the variadic nodes in topo order.
# Comptime name resolution via double-loop comptime for + comptime if.
# ──────────────────────────────────────────────────────────────────────


struct MiniGraph[*NODES: GraphNode](Movable & ImplicitlyDestructible):
    comptime N = Self.NODES.size

    var nodes: Tuple[*Self.NODES]

    def __init__(out self):
        self.nodes = Tuple[*Self.NODES]()

    def forward[
        BATCH: Int, LIN: TensorLayout, OIN: MutOrigin,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
    ) raises:
        """Topological forward. For each node i, look up its IN0_NAME:
        either 'input' (route the graph input) or a sibling node's name."""
        comptime for i in range(Self.N):
            comptime src0 = Self.NODES[i].IN0_NAME
            comptime if src0 == "input":
                self.nodes[i].node_forward[BATCH](input)
            else:
                # find the node whose NAME == src0
                comptime for j in range(Self.N):
                    comptime if Self.NODES[j].NAME == src0:
                        comptime PRED_OUT = Self.NODES[j].OP_OUT_DIM
                        var pred_view = TileTensor(
                            self.nodes[j].node_out_ptr(),
                            row_major[BATCH, PRED_OUT](),
                        )
                        self.nodes[i].node_forward[BATCH](pred_view)


# ──────────────────────────────────────────────────────────────────────
# Smoke driver: 3-node chain n0 → n1 → n2, each BiasAdd[3, k].
# ──────────────────────────────────────────────────────────────────────


def smoke_chain() raises:
    print("--- spike: 3-node name-resolved chain ---")
    var g = MiniGraph[
        GNode["n0", BiasAdd[3, 1.0], "input"],
        GNode["n1", BiasAdd[3, 2.0], "n0"],
        GNode["n2", BiasAdd[3, 4.0], "n1"],
    ]()

    var input_buf = List[Scalar[DT]](length=6, fill=Scalar[DT](0.0))
    for k in range(6):
        input_buf[k] = Scalar[DT](Float64(k))
    var input_t = TileTensor(input_buf.unsafe_ptr(), row_major[2, 3]())

    g.forward[2](input_t)

    # Each BiasAdd adds its BIAS. Total bias = 1 + 2 + 4 = 7.
    # So output[i] = input[i] + 7.
    var out_view = TileTensor(
        g.nodes[2].node_out_ptr(), row_major[2, 3]()
    )
    print("  out[0, 0]=", out_view[0, 0], " expected ", 0.0 + 7.0)
    print("  out[0, 1]=", out_view[0, 1], " expected ", 1.0 + 7.0)
    print("  out[0, 2]=", out_view[0, 2], " expected ", 2.0 + 7.0)
    print("  out[1, 0]=", out_view[1, 0], " expected ", 3.0 + 7.0)
    var ok = (
        out_view[0, 0] == 7.0
        and out_view[0, 1] == 8.0
        and out_view[0, 2] == 9.0
        and out_view[1, 0] == 10.0
    )
    if ok:
        print("  smoke chain: PASSED — name-based DAG resolution works")
    else:
        print("  smoke chain: FAILED")


def main() raises:
    print("=" * 70)
    print("DR.4 — name-based DAG (v1 ComputeGraph style) spike")
    print("=" * 70)
    smoke_chain()
    print("=" * 70)
