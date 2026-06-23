"""Adam end-to-end: Sequential[Linear, ReLU, Linear] trains MSE with Adam,
driven through `for_each_param` (begin_step bumps the optimizer each step).
Confirms the stateful optimizer composes via the param walk.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_train_adam.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.loss.mse import mse_forward, mse_backward
from mojo_rl.nn.core.initializer import Deterministic


def main() raises:
    comptime B = 4
    comptime IN = 3
    comptime H = 6
    comptime OUT = 2

    var model = Sequential[
        Linear[IN, H], ReLU[H], Linear[H, OUT]
    ].make["cpu", Deterministic]()
    var opt = Adam(lr=0.05)

    var x = Tensor.alloc(B * IN)
    var tgt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.5
    for i in range(B * OUT):
        tgt.data[i] = Scalar[DT](1) if (i % 2 == 0) else Scalar[DT](-1)

    var pred = Tensor.alloc(B * OUT)
    var grad = Tensor.alloc(B * OUT)
    var gi = Tensor.alloc(B * IN)

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(60):
        model.zero_grad["cpu"](None)
        model.forward["cpu", B](TensorRefs[1](x), pred, None)
        var loss = mse_forward[B, OUT](pred, tgt)
        if step == 0:
            first = loss
        last = loss
        if step % 12 == 0:
            print("step", step, " mse", loss)
        mse_backward["cpu", B, OUT](pred, tgt, grad)
        model.vjp["cpu", B](TensorRefs[1](x), grad, TensorRefs[1](gi), None)
        opt.begin_step()
        model.for_each_param["cpu"](opt, None)

    print("final mse", last)
    if last < first * 0.1:
        print("ADAM TRAIN OK — loss", first, "->", last)
    else:
        print("ADAM TRAIN WEAK (", first, "->", last, ")")
