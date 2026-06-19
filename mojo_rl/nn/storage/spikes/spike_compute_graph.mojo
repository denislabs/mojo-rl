"""ComputeGraph builder gate — a 2-input DAG trained with Adam (CPU).

Graph (NUM_IN=2):  x0, x1  →  Add(x0,x1)  →  Linear  →  ReLU  →  Linear  →  out
  slots: 0=x0 1=x1 | 2=Add 3=Linear 4=ReLU 5=Linear(out)
Exercises: multiple external inputs, a binary node, params across nodes,
forward + fan-out vjp, for_each_param/zero_grad, node_output access, training.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_compute_graph.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.add import Add
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.loss.mse import mse_forward, mse_backward
from mojo_rl.nn.storage.core.initializer import Deterministic


def main() raises:
    comptime B = 4
    comptime IN = 3
    comptime H = 6
    comptime OUT = 2

    var g = ComputeGraph[
        2, Add[IN], Linear[IN, H], ReLU[H], Linear[H, OUT]
    ].make["cpu", Deterministic]()
    var edges = List[List[Int]]()
    edges.append([0, 1])   # Add(x0, x1)
    edges.append([2])      # Linear(Add)
    edges.append([3])      # ReLU(Linear)
    edges.append([4])      # Linear(ReLU)

    var inp = TensorPack[2]()
    inp[0].ensure(B * IN)
    inp[1].ensure(B * IN)
    var tgt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        inp[0].data[i] = Scalar[DT]((i % 5) - 2) * 0.3
        inp[1].data[i] = Scalar[DT]((i % 3) - 1) * 0.2
    for i in range(B * OUT):
        tgt.data[i] = Scalar[DT](1) if (i % 2 == 0) else Scalar[DT](-1)

    # node_output sanity: after a forward, node 0 (Add) == x0 + x1.
    var out = Tensor.alloc(B * OUT)
    g.forward[B](edges, inp, out)
    var add_ok = True
    for i in range(B * IN):
        ref add_out = g.node_output(0)
        if abs(add_out.data[i] - (inp[0].data[i] + inp[1].data[i])) > 1e-6:
            add_ok = False
    print("node_output(0) == x0+x1:", "OK" if add_ok else "FAIL")

    var opt = Adam(lr=0.05)
    var grad = Tensor.alloc(B * OUT)
    var gin = TensorPack[2]()
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(60):
        g.zero_grad["cpu"](None)
        g.forward[B](edges, inp, out)
        var loss = mse_forward[B, OUT](out, tgt)
        if step == 0:
            first = loss
        last = loss
        if step % 12 == 0:
            print("step", step, " mse", loss)
        mse_backward["cpu", B, OUT](out, tgt, grad)
        g.vjp[B](edges, grad, gin)
        opt.begin_step()
        g.for_each_param["cpu"](opt, None)

    print("final mse", last)
    if add_ok and last < first * 0.1:
        print("COMPUTE GRAPH OK — 2-input DAG trains; loss", first, "->", last)
    else:
        print("COMPUTE GRAPH FAIL")
