"""ComputeGraph[NUM_IN, *NODES] — typed multi-input DAG over the storage design.

The keystone orchestrator for agent loss/target graphs (SAC actor-loss,
target-y, twin critic). Generalises the single-input `graph.mojo` prototype:

  - `NUM_IN` EXTERNAL input slots (0 .. NUM_IN-1) — the graph's leaves (state,
    action, reward, a reparam sample, …), seeded each forward.
  - `*NODES: Module` in topological order; node `i` writes pool slot
    `NUM_IN + i`. `edges[i]` lists the slots feeding node `i` (each in
    `0 .. NUM_IN+i-1`), so any node can read an external input OR an earlier
    node's output (skips / fan-out).
  - ALL activations live in ONE owning `TensorPack` pool → every edge is
    same-origin (§B0), no raw pointers (cf. legacy `GraphNode`'s out_ptr/
    grad_out_ptr raw-pointer model that the storage design replaces).
  - Backward walks reverse-topo; each node's `vjp` scatters its grad_inputs
    into `tmp`, which the graph ACCUMULATES into the fed slots' grad buffers
    (fan-out summation). `vjp` writes the external-input grads back into
    `grad_inputs` for chaining.
  - `node_output(i)` exposes a node's forward output (the `node_out_ptr`
    equivalent — for diagnostics / reading an intermediate like `min_q`).

CPU path (validates the builder API + unblocks SAC blocks; SAC has a CPU
training path). GPU graph execution (device pool-seed / grad-accumulate
kernels) is the mechanical next step — the leaves already run on GPU.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor


struct ComputeGraph[NUM_IN: Int, *NODES: Module](
    Movable & ImplicitlyDeletable
):
    comptime N = Self.NODES.size
    comptime NSLOT = Self.NUM_IN + Self.N
    comptime OUT_DIM = Self.NODES[Self.N - 1].OUT_DIM
    var children: Tuple[*Self.NODES]
    var pool: TensorPack[Self.NSLOT]    # 0..NUM_IN-1 = inputs, then node outs
    var gpool: TensorPack[Self.NSLOT]   # grads (zeroed + fan-out-accumulated)
    var tmp: TensorPack[2]              # grad_inputs scratch (max arity 2)
    var slot_n: List[Int]               # logical elem count per slot (forward)

    def __init__(out self):
        comptime assert Self.N >= 1, "ComputeGraph needs >= 1 node"
        comptime assert Self.NUM_IN >= 1, "ComputeGraph needs >= 1 input"
        self.children = Tuple[*Self.NODES]()
        self.pool = TensorPack[Self.NSLOT]()
        self.gpool = TensorPack[Self.NSLOT]()
        self.tmp = TensorPack[2]()
        self.slot_n = List[Int](length=Self.NSLOT, fill=0)

    @staticmethod
    def make_cpu() raises -> Self:
        var g = Self()
        comptime for i in range(Self.N):
            g.children[i] = Self.NODES[i].make_cpu()
        return g^

    def node_output(mut self, node_i: Int) raises -> ref [MutAnyOrigin] Tensor:
        """The forward output of node `node_i` (pool slot NUM_IN+node_i) — the
        storage `node_out_ptr` equivalent for diagnostics / intermediate reads.
        """
        return self.pool[Self.NUM_IN + node_i]

    def forward[
        B: Int
    ](
        mut self,
        edges: List[List[Int]],
        mut inputs: TensorPack[Self.NUM_IN],
        mut out: Tensor,
    ) raises:
        # Seed external input slots.
        for k in range(Self.NUM_IN):
            var nk = inputs[k].n
            self.slot_n[k] = nk
            self.pool[k].ensure(nk)
            for q in range(nk):
                self.pool[k].data[q] = inputs[k].data[q]
        # Topological forward; gather each node's inputs from the pool by slot.
        comptime for i in range(Self.N):
            self.slot_n[Self.NUM_IN + i] = B * Self.NODES[i].OUT_DIM
            comptime if Self.NODES[i].ARITY == 1:
                self.children[i].forward["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of1(self.pool[edges[i][0]]),
                    self.pool[Self.NUM_IN + i], None,
                )
            elif Self.NODES[i].ARITY == 2:
                self.children[i].forward["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of2(
                        self.pool[edges[i][0]], self.pool[edges[i][1]]
                    ),
                    self.pool[Self.NUM_IN + i], None,
                )
        var dN = B * Self.OUT_DIM
        out.ensure(dN)
        for q in range(dN):
            out.data[q] = self.pool[Self.NSLOT - 1].data[q]

    def vjp[
        B: Int
    ](
        mut self,
        edges: List[List[Int]],
        mut grad_out: Tensor,
        mut grad_inputs: TensorPack[Self.NUM_IN],
    ) raises:
        # Size + ZERO every grad slot (forward-recorded slot_n), then seed the
        # output slot's grad = grad_out.
        for k in range(Self.NSLOT):
            var dk = self.slot_n[k]
            self.gpool[k].ensure(dk)
            for q in range(dk):
                self.gpool[k].data[q] = 0
        var dN = B * Self.OUT_DIM
        for q in range(dN):
            self.gpool[Self.NSLOT - 1].data[q] = grad_out.data[q]
        # Reverse-topo: each node scatters grad_inputs into tmp; the graph
        # ACCUMULATES tmp into the fed slots' grad buffers (fan-out sum).
        comptime for jj in range(Self.N):
            comptime i = Self.N - 1 - jj
            comptime if Self.NODES[i].ARITY == 1:
                var a0 = B * Self.NODES[i].IN_DIMS[0]
                self.tmp[0].ensure(a0)
                self.children[i].vjp["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of1(self.pool[edges[i][0]]),
                    self.gpool[Self.NUM_IN + i],
                    TensorRefs[Self.NODES[i].ARITY].of1(self.tmp[0]),
                    None,
                )
                var e0 = edges[i][0]
                for q in range(a0):
                    self.gpool[e0].data[q] += self.tmp[0].data[q]
            elif Self.NODES[i].ARITY == 2:
                var a0 = B * Self.NODES[i].IN_DIMS[0]
                var a1 = B * Self.NODES[i].IN_DIMS[1]
                self.tmp[0].ensure(a0)
                self.tmp[1].ensure(a1)
                self.children[i].vjp["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of2(
                        self.pool[edges[i][0]], self.pool[edges[i][1]]
                    ),
                    self.gpool[Self.NUM_IN + i],
                    TensorRefs[Self.NODES[i].ARITY].of2(self.tmp[0], self.tmp[1]),
                    None,
                )
                var e0 = edges[i][0]
                var e1 = edges[i][1]
                for q in range(a0):
                    self.gpool[e0].data[q] += self.tmp[0].data[q]
                for q in range(a1):
                    self.gpool[e1].data[q] += self.tmp[1].data[q]
        # Write external-input grads back (slots 0..NUM_IN-1) for chaining.
        for k in range(Self.NUM_IN):
            var dk = self.slot_n[k]
            grad_inputs[k].ensure(dk)
            for q in range(dk):
                grad_inputs[k].data[q] = self.gpool[k].data[q]

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update every node toward `src`'s matching node (target ← online)."""
        comptime for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)
