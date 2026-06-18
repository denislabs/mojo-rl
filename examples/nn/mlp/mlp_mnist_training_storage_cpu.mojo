"""MLP on MNIST (CPU) — storage-surface (nn.storage) port.

The nn.storage twin of mlp_mnist_training_cpu.mojo. Uses the storage Trainer
(batch loop + top-1 eval), Sequential[Linear, ReLU, ...] (unfused — the storage
surface has Linear + Elementwise ReLU, not a fused LinearReLU yet),
CrossEntropyLoss, Adam, and Kaiming init via the `reinit` walk.

Architecture: Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)
Data:         MNIST via mojo_rl.nn.datasets.MNIST (framework-agnostic loader)

Run:
    pixi run mojo run -I . examples/nn/mlp/mlp_mnist_training_storage_cpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns

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

    comptime Net = Sequential[
        LinearReLU[IN_DIM, H1],
        LinearReLU[H1, H2],
        Linear[H2, NC],
    ]
    print("initializing network...")
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "cpu"].make[Kaiming](lr=1e-3)

    # One-hot the training labels (images already flat [N, 784]).
    var train_y = List[Scalar[DT]](length=MNIST.N_TRAIN * NC, fill=0.0)
    for i in range(MNIST.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    var best_acc: Float64 = 0.0
    var t0 = perf_counter_ns()
    for epoch in range(N_EPOCHS):
        var loss = trainer.train_epoch[MNIST.N_TRAIN](
            ds.train_images, train_y, None
        )
        var acc = trainer.eval_top1[MNIST.N_TEST](
            ds.test_images, ds.test_labels, None
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
