"""End-to-end GPU training (storage slice): Linear → ReLU → Linear, SGD on MSE,
Apple Metal. SAME code as spike_train.mojo with target="gpu" + a ctx; device
params (Param), device buffers, kernels. Loss monitor downloads `pred`.

Should converge to the SAME loss as the CPU run (deterministic init/data,
matching naive math) — a CPU/GPU parity check on the full training loop.

Run: pixi run -e apple mojo run -I . \
    mojo_rl/nn/storage/spike_train_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.optimizer.sgd import SGD
from mojo_rl.nn.storage.loss.mse import mse_forward, mse_backward
from mojo_rl.nn.storage.core.initializer import Deterministic


def main() raises:
    comptime B = 4
    comptime IN = 3
    comptime H = 6
    comptime OUT = 2
    var ctx = DeviceContext()

    var model = Sequential[
        Linear[IN, H], ReLU[H], Linear[H, OUT]
    ].make["gpu", Deterministic](Optional(ctx))
    var opt = SGD(lr=0.1, wd=0.0)

    var x = Tensor.alloc(B * IN)
    var tgt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.5
    for i in range(B * OUT):
        tgt.data[i] = Scalar[DT](1) if (i % 2 == 0) else Scalar[DT](-1)
    x.upload(ctx)
    tgt.upload(ctx)

    var pred = Tensor.alloc_gpu(ctx, B * OUT)
    var grad = Tensor.alloc_gpu(ctx, B * OUT)
    var gi = Tensor.alloc_gpu(ctx, B * IN)

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(40):
        model.zero_grad["gpu"](ctx)
        model.forward["gpu", B](TensorRefs[1](x), pred, ctx)

        pred.download(ctx)  # monitor only
        var loss = mse_forward[B, OUT](pred, tgt)
        if step == 0:
            first = loss
        last = loss
        if step % 8 == 0:
            print("step", step, " mse", loss)

        mse_backward["gpu", B, OUT](pred, tgt, grad, ctx)
        model.vjp["gpu", B](TensorRefs[1](x), grad, TensorRefs[1](gi), ctx)
        model.for_each_param["gpu"](opt, ctx)

    print("final mse", last)
    if last < first:
        print("GPU LIGHTHOUSE OK — loss", first, "->", last)
    else:
        print("GPU LIGHTHOUSE FAIL")
