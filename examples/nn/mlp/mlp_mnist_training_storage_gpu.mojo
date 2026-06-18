"""MLP on MNIST (GPU) — storage-surface (nn.storage) port.

The GPU twin of mlp_mnist_training_storage_cpu.mojo. Same storage Trainer /
Sequential[Linear, ReLU, ...] / CrossEntropyLoss / Adam / Kaiming, with
target="gpu": the Trainer uploads each batch and runs the model + loss + Adam on
device. The caller owns the DeviceContext and reuses it for make/train/eval.

Run (Apple):  pixi run -e apple mojo run -I . examples/nn/mlp/mlp_mnist_training_storage_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/mlp/mlp_mnist_training_storage_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.training.trainer import Trainer


def main() raises:
    comptime IN_DIM = 784
    comptime H1 = 256
    comptime H2 = 128
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 5
    comptime TARGET_ACC: Float64 = 0.97

    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var c = DeviceContext()

    comptime Net = Sequential[
        LinearReLU[IN_DIM, H1],
        LinearReLU[H1, H2],
        Linear[H2, NC],
    ]
    print("initializing network (GPU)...")
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=MNIST.N_TRAIN * NC, fill=0.0)
    for i in range(MNIST.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    var best_acc: Float64 = 0.0
    var t0 = perf_counter_ns()
    for epoch in range(N_EPOCHS):
        var loss = trainer.train_epoch[MNIST.N_TRAIN](
            ds.train_images, train_y, Optional(c)
        )
        var acc = trainer.eval_top1[MNIST.N_TEST](
            ds.test_images, ds.test_labels, Optional(c)
        )
        if acc > best_acc:
            best_acc = acc
        print("epoch", epoch, " loss", loss, " test_top1", acc * 100.0, "%")
    var total_s = Float64(perf_counter_ns() - t0) / 1e9

    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")
    print("total wall time: " + String(total_s) + "s")
    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%",
    )
    print("DONE")
