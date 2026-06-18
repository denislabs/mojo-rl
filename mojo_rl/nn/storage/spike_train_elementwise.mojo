"""Integration gate: SeqS[LinS, ReLUE, LinS] trains identically to the hand-
written ReLUS chain — proves the generic ElementwiseS composes in the
orchestrator as a drop-in ModuleS. Expect the SAME loss curve as spike_train.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spike_train_elementwise.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.tensor import Tensor
from mojo_rl.nn.storage.tensor_refs import TensorRefs
from mojo_rl.nn.storage.leaves import LinS
from mojo_rl.nn.storage.activations import ReLUE
from mojo_rl.nn.storage.sequential import SeqS
from mojo_rl.nn.storage.optim_loss import SGDS, mse_forward, mse_backward


def main() raises:
    comptime B = 4
    comptime IN = 3
    comptime H = 6
    comptime OUT = 2

    var model = SeqS[LinS[IN, H], ReLUE[H], LinS[H, OUT]].make_cpu()
    var opt = SGDS(lr=0.1, wd=0.0)

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
        model.forward["cpu", B](TensorRefs[1].of1(x), pred, None)
        var loss = mse_forward[B, OUT](pred, tgt)
        if step == 0:
            first = loss
        last = loss
        mse_backward["cpu", B, OUT](pred, tgt, grad)
        model.vjp["cpu", B](
            TensorRefs[1].of1(x), grad, TensorRefs[1].of1(gi), None
        )
        model.for_each_param["cpu"](opt, None)

    print("first", first, "final", last)
    # spike_train (hand-written ReLUS) lands at 0.0028509053.
    if abs(last - Scalar[DT](0.0028509053)) < 1e-6:
        print("ELEMENTWISE COMPOSE OK — bit-identical to hand-written ReLUS")
    else:
        print("ELEMENTWISE COMPOSE MISMATCH")
