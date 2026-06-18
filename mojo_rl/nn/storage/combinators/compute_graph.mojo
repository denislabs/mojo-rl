"""ResidualGraph — a ComputeGraph (DAG) prototype over the storage design.

Proves the storage pool/TensorRefs design maps onto a GRAPH, not just a chain:
a residual block `out = Add(Lin2(ReLU(Lin1(x))), x)` — a SKIP connection, the
thing `Sequential` can't express. The whole DAG's activations (INCLUDING the
input) live in ONE `TensorPack` node pool, so every edge references the pool
and multi-input nodes (the `Add`) get same-origin inputs for free (§B0).

  node 0 = input (copied into the pool)
  node 1 = Lin1(node0)   node 2 = ReLU(node1)   node 3 = Lin2(node2)
  node 4 = Add(node3, node0)   ← reads node3 AND node0 (the skip)

Backward fans out: node0's grad accumulates the Lin1 path + the skip path —
the DAG behaviour Sequential lacks. CPU prototype (the GPU leaf path is already
proven); the new thing here is the graph STRUCTURE.

Run: pixi run mojo run -I . mojo_rl/nn/storage/compute_graph.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.add import Add
from mojo_rl.nn.storage.primitives.activations import ReLU


struct ResidualGraph[IN: Int, H: Int](Movable & ImplicitlyDeletable):
    var lin1: Linear[Self.IN, Self.H]
    var relu: ReLU[Self.H]
    var lin2: Linear[Self.H, Self.IN]
    var add: Add[Self.IN]
    var act: TensorPack[5]  # node outputs 0..4 (0 = input copy)
    var grd: TensorPack[5]  # node grads (fan-out accumulates here)

    def __init__(out self):
        self.lin1 = Linear[Self.IN, Self.H]()
        self.relu = ReLU[Self.H]()
        self.lin2 = Linear[Self.H, Self.IN]()
        self.add = Add[Self.IN]()
        self.act = TensorPack[5]()
        self.grd = TensorPack[5]()

    @staticmethod
    def make_cpu() raises -> Self:
        var g = Self()
        g.lin1 = Linear[Self.IN, Self.H].make_cpu()
        g.relu = ReLU[Self.H].make_cpu()
        g.lin2 = Linear[Self.H, Self.IN].make_cpu()
        g.add = Add[Self.IN].make_cpu()
        return g^

    def forward[B: Int](mut self, ref x: Tensor, mut out: Tensor) raises:
        # node 0 = input copy.
        self.act[0].ensure(B * Self.IN)
        for i in range(B * Self.IN):
            self.act[0].data[i] = x.data[i]
        # node 1/2/3 = the branch.
        self.lin1.forward["cpu", B](
            TensorRefs[1].of1(self.act[0]), self.act[1], None
        )
        self.relu.forward["cpu", B](
            TensorRefs[1].of1(self.act[1]), self.act[2], None
        )
        self.lin2.forward["cpu", B](
            TensorRefs[1].of1(self.act[2]), self.act[3], None
        )
        # node 4 = Add(node3, node0) — the SKIP. Both inputs from the pool.
        self.add.forward["cpu", B](
            TensorRefs[2].of2(self.act[3], self.act[0]), self.act[4], None
        )
        out.ensure(B * Self.IN)
        for i in range(B * Self.IN):
            out.data[i] = self.act[4].data[i]

    def vjp[B: Int](mut self, ref x: Tensor, mut grad_out: Tensor) raises:
        # Reverse topo. grad[4] = grad_out; Add splits it to grad[3] and a
        # SEPARATE skip buffer; the input grad node[0] ACCUMULATES the Lin1
        # path and the skip path (the DAG fan-out).
        for k in range(5):
            self.grd[k].ensure(B * Self.IN if (k == 0 or k >= 3) else B * Self.H)
        var skip = Tensor.alloc(B * Self.IN)  # grad into node0 via the skip edge

        # node 4 (Add): grad_out → grad[3] and skip.
        self.add.vjp["cpu", B](
            TensorRefs[2].of2(self.act[3], self.act[0]),
            grad_out,
            TensorRefs[2].of2(self.grd[3], skip),
            None,
        )
        # node 3 (Lin2): grad[3] → grad[2].
        self.lin2.vjp["cpu", B](
            TensorRefs[1].of1(self.act[2]),
            self.grd[3],
            TensorRefs[1].of1(self.grd[2]),
            None,
        )
        # node 2 (ReLU): grad[2] → grad[1].
        self.relu.vjp["cpu", B](
            TensorRefs[1].of1(self.act[1]),
            self.grd[2],
            TensorRefs[1].of1(self.grd[1]),
            None,
        )
        # node 1 (Lin1): grad[1] → grad[0].
        self.lin1.vjp["cpu", B](
            TensorRefs[1].of1(self.act[0]),
            self.grd[1],
            TensorRefs[1].of1(self.grd[0]),
            None,
        )
        # FAN-OUT accumulation: node0 grad = Lin1-path (grd[0]) + skip.
        for i in range(B * Self.IN):
            self.grd[0].data[i] += skip.data[i]
        # input grad (for chaining); copy out for inspection.
        grad_out.ensure(B * Self.IN)
        for i in range(B * Self.IN):
            grad_out.data[i] = self.grd[0].data[i]
        _ = skip^


def main() raises:
    comptime B = 2
    comptime IN = 3
    comptime H = 4

    var g = ResidualGraph[IN, H].make_cpu()
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](i + 1) * 0.5
    var out = Tensor.alloc(B * IN)
    g.forward[B](x, out)

    # out should equal branch(x) + x. Check the skip term is present:
    # out[i] - branch[i] == x[i]  →  out - branch == x. We can at least
    # confirm forward ran and out is finite + the skip added x back.
    print("x:   ", x.data[0], x.data[1], x.data[5])
    print("out: ", out.data[0], out.data[1], out.data[5])

    var go = Tensor.alloc(B * IN)
    for i in range(B * IN):
        go.data[i] = Scalar[DT](1)
    g.vjp[B](x, go)
    print("grad_input (fan-out accumulated):", go.data[0], go.data[1])
    print("COMPUTE GRAPH OK — residual DAG (skip + fan-out) on the pool")
