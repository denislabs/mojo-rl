"""ComputeGraph builder gate — a 2-input DAG trained with Adam (CPU).

Graph (NUM_IN=2):  x0, x1  →  Add(x0,x1)  →  Linear  →  ReLU  →  Linear  →  out
  slots: 0=x0 1=x1 | 2=Add 3=Linear 4=ReLU 5=Linear(out)
Exercises: multiple external inputs, a binary node, params across nodes,
forward + fan-out vjp, for_each_param/zero_grad, node_output access, training.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_compute_graph.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.add import Add
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.loss.mse import mse_forward, mse_backward
from mojo_rl.nn.core.initializer import Deterministic


def main() raises:
    comptime B = 4
    comptime IN = 3
    comptime H = 6
    comptime OUT = 2

    var g = ComputeGraph[
        InputSlot["x0", IN],
        InputSlot["x1", IN],
        Node["add", Add[IN], "x0", "x1"],      # Add(x0, x1)
        Node["l1", Linear[IN, H], "add"],      # Linear(Add)
        Node["relu", ReLU[H], "l1"],           # ReLU(Linear)
        Node["l2", Linear[H, OUT], "relu"],    # Linear(ReLU) → out
    ].make["cpu", Deterministic]()

    var x0 = Tensor.alloc(B * IN)
    var x1 = Tensor.alloc(B * IN)
    var tgt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x0.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
        x1.data[i] = Scalar[DT]((i % 3) - 1) * 0.2
    for i in range(B * OUT):
        tgt.data[i] = Scalar[DT](1) if (i % 2 == 0) else Scalar[DT](-1)
    g.set_input["x0", B](x0)
    g.set_input["x1", B](x1)

    # node_output sanity: after a forward, the Add node == x0 + x1.
    var out = Tensor.alloc(B * OUT)
    g.forward[B](out)
    var add_ok = True
    for i in range(B * IN):
        ref add_out = g.node_output["add"]()
        if abs(add_out.data[i] - (x0.data[i] + x1.data[i])) > 1e-6:
            add_ok = False
    print("node_output('add') == x0+x1:", "OK" if add_ok else "FAIL")

    var opt = Adam(lr=0.05)
    var grad = Tensor.alloc(B * OUT)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(60):
        g.zero_grad["cpu"](None)
        g.forward[B](out)
        var loss = mse_forward[B, OUT](out, tgt)
        if step == 0:
            first = loss
        last = loss
        if step % 12 == 0:
            print("step", step, " mse", loss)
        mse_backward["cpu", B, OUT](out, tgt, grad)
        g.vjp[B](grad)
        opt.begin_step()
        g.for_each_param["cpu"](opt, None)

    print("final mse", last)
    if add_ok and last < first * 0.1:
        print("COMPUTE GRAPH OK — 2-input DAG trains; loss", first, "->", last)
    else:
        print("COMPUTE GRAPH FAIL")
