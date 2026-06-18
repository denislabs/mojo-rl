"""Graph[*NODES] — a GENERAL ComputeGraph executor over the storage design.

Generalises the hand-wired residual: declare any list of `ModuleS` nodes
(unary or binary) plus an `edges` table, and the graph runs the DAG. Nodes are
given in topological order; node `i` writes pool slot `i+1` (slot 0 = the graph
input), and `edges[i]` lists the pool slots feeding node `i` (in `0..i`). All
activations live in ONE `TensorPack` pool → every edge is same-origin (§B0).

Backward walks reverse-topo; each node's `vjp` writes its grad_inputs into a
shared `tmp` pack, which the graph then ACCUMULATES into the input slots' grad
buffers — so a slot feeding several nodes (a skip / any fan-out) sums all its
contributions. CPU prototype (the leaf GPU path is proven separately).

Run: pixi run mojo run -I . mojo_rl/nn/storage/graph.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.tensor import Tensor
from mojo_rl.nn.storage.tensor_refs import TensorRefs
from mojo_rl.nn.storage.tensor_pack import TensorPack
from mojo_rl.nn.storage.module import ModuleS
from mojo_rl.nn.storage.leaves import LinS, ReLUS, AddS


struct Graph[*NODES: ModuleS](Movable & ImplicitlyDeletable):
    comptime N = Self.NODES.size
    var children: Tuple[*Self.NODES]
    var pool: TensorPack[Self.N + 1]   # slot 0 = input, 1..N = node outputs
    var gpool: TensorPack[Self.N + 1]  # grads (zeroed + fan-out-accumulated)
    var tmp: TensorPack[2]             # grad_inputs scratch (max arity 2)

    def __init__(out self):
        self.children = Tuple[*Self.NODES]()
        self.pool = TensorPack[Self.N + 1]()
        self.gpool = TensorPack[Self.N + 1]()
        self.tmp = TensorPack[2]()

    @staticmethod
    def make_cpu() raises -> Self:
        var g = Self()
        comptime for i in range(Self.N):
            g.children[i] = Self.NODES[i].make_cpu()
        return g^

    def _slot_dim[k: Int](self) -> Int:
        comptime if k == 0:
            return Self.NODES[0].IN_DIMS[0]
        else:
            return Self.NODES[k - 1].OUT_DIM

    def forward[
        B: Int
    ](mut self, edges: List[List[Int]], ref inp: Tensor, mut out: Tensor) raises:
        # slot 0 = the graph input.
        var d0 = B * Self.NODES[0].IN_DIMS[0]
        self.pool[0].ensure(d0)
        for q in range(d0):
            self.pool[0].data[q] = inp.data[q]
        # run nodes in topo order; gather inputs from the pool by edge slot.
        comptime for i in range(Self.N):
            comptime if Self.NODES[i].ARITY == 1:
                self.children[i].forward["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of1(self.pool[edges[i][0]]),
                    self.pool[i + 1], None,
                )
            elif Self.NODES[i].ARITY == 2:
                self.children[i].forward["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of2(
                        self.pool[edges[i][0]], self.pool[edges[i][1]]
                    ),
                    self.pool[i + 1], None,
                )
        var dN = B * Self.NODES[Self.N - 1].OUT_DIM
        out.ensure(dN)
        for q in range(dN):
            out.data[q] = self.pool[Self.N].data[q]

    def vjp[
        B: Int
    ](
        mut self, edges: List[List[Int]], ref inp: Tensor, mut grad_out: Tensor
    ) raises:
        # size + ZERO every grad slot, then seed gpool[N] = grad_out.
        comptime for k in range(Self.N + 1):
            var dk = B * self._slot_dim[k]()
            self.gpool[k].ensure(dk)
            for q in range(dk):
                self.gpool[k].data[q] = 0
        var dN = B * Self.NODES[Self.N - 1].OUT_DIM
        for q in range(dN):
            self.gpool[Self.N].data[q] = grad_out.data[q]
        # reverse topo: each node scatters its grad_inputs into tmp, then the
        # graph ACCUMULATES tmp into the input slots (fan-out summation).
        comptime for jj in range(Self.N):
            comptime i = Self.N - 1 - jj
            comptime if Self.NODES[i].ARITY == 1:
                var a0 = B * Self.NODES[i].IN_DIMS[0]
                self.tmp[0].ensure(a0)
                self.children[i].vjp["cpu", B](
                    TensorRefs[Self.NODES[i].ARITY].of1(self.pool[edges[i][0]]),
                    self.gpool[i + 1],
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
                    self.gpool[i + 1],
                    TensorRefs[Self.NODES[i].ARITY].of2(self.tmp[0], self.tmp[1]),
                    None,
                )
                var e0 = edges[i][0]
                var e1 = edges[i][1]
                for q in range(a0):
                    self.gpool[e0].data[q] += self.tmp[0].data[q]
                for q in range(a1):
                    self.gpool[e1].data[q] += self.tmp[1].data[q]
        # input grad = slot 0.
        var d0 = B * Self.NODES[0].IN_DIMS[0]
        grad_out.ensure(d0)
        for q in range(d0):
            grad_out.data[q] = self.gpool[0].data[q]


def main() raises:
    comptime B = 2
    comptime IN = 3
    comptime H = 4

    # Residual block expressed GENERALLY: Lin1 → ReLU → Lin2 → Add(·, input).
    #   node0=Lin1[IN,H]  node1=ReLU[H]  node2=Lin2[H,IN]  node3=Add[IN]
    #   edges:           [0]            [1]               [2]   [3, 0]
    # Slot 0 (the input) feeds BOTH node0 and node3 → backward fans out.
    var g = Graph[
        LinS[IN, H], ReLUS[H], LinS[H, IN], AddS[IN]
    ].make_cpu()
    var edges = List[List[Int]]()
    edges.append([0])
    edges.append([1])
    edges.append([2])
    edges.append([3, 0])

    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](i + 1) * 0.5
    var out = Tensor.alloc(B * IN)
    g.forward[B](edges, x, out)
    print("x:  ", x.data[0], x.data[1], x.data[5])
    print("out:", out.data[0], out.data[1], out.data[5])

    var go = Tensor.alloc(B * IN)
    for i in range(B * IN):
        go.data[i] = Scalar[DT](1)
    g.vjp[B](edges, x, go)
    print("grad_input (fan-out summed):", go.data[0], go.data[1])
    print("GENERAL GRAPH OK — arbitrary DAG from an edge table + fan-out")
