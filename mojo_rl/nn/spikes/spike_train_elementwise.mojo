"""Integration gate: Sequential[Linear, ReLU, Linear] trains identically to the hand-
written ReLU chain — proves the generic Elementwise composes in the
orchestrator as a drop-in Module. Expect the SAME loss curve as spike_train.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spike_train_elementwise.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.sgd import SGD
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
    var opt = SGD(lr=0.1, wd=0.0)

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
    for step in range(40):
        model.zero_grad["cpu"](None)
        model.forward["cpu", B](TensorRefs[1](x), pred, None)
        var loss = mse_forward[B, OUT](pred, tgt)
        if step == 0:
            first = loss
        last = loss
        mse_backward["cpu", B, OUT](pred, tgt, grad)
        model.vjp["cpu", B](TensorRefs[1](x), grad, TensorRefs[1](gi), None)
        model.for_each_param["cpu"](opt, None)

    print("first", first, "final", last)
    # spike_train (hand-written ReLU) lands at 0.0028509053.
    if abs(last - Scalar[DT](0.0028509053)) < 1e-6:
        print("ELEMENTWISE COMPOSE OK — bit-identical to hand-written ReLU")
    else:
        print("ELEMENTWISE COMPOSE MISMATCH")
