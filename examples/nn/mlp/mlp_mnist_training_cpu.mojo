"""MLP on MNIST (CPU).

Port of `examples/nn/mlp/mlp_mnist_training_cpu.mojo` to the nn framework.

Uses the `trainer.train_cpu[N_TRAIN, N_TEST]` whole-dataset API (the CPU
mirror of `train_gpu`): build the flat train + test sets once, then the
trainer slices batches by pointer offset and runs the forward / backward /
step loop with per-epoch top-1 eval internally.

Architecture: LinearReLU(784→256) → LinearReLU(256→128) → Linear(128→10)
Loss:         CrossEntropyLoss[10] on one-hot targets
Optimizer:    Adam (lr=1e-3)
Data:         MNIST via `mojo_rl.nn.datasets.MNIST` (framework-agnostic loader)

Run:
    pixi run mojo run -I . examples/nn/mlp/mlp_mnist_training_cpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer import Kaiming


def main() raises:
    comptime IN_DIM = 784
    comptime H1 = 256
    comptime H2 = 128
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 5
    comptime TARGET_ACC: Float64 = 0.97

    seed(42)
    print("loading MNIST...")
    var ds = MNIST()

    print("initializing network...")
    comptime Net = Sequential[
        LinearReLU[IN_DIM, H1],
        LinearReLU[H1, H2],
        Linear[H2, N_CLASSES],
    ]

    var trainer = Trainer[
        Net,
        Adam,
        CrossEntropyLoss[N_CLASSES],
        BATCH,
        target="cpu",
    ].make[INIT=Kaiming]()

    # One-hot the training labels; images are already flat [N, 784].
    var train_y = List[Scalar[DT]](length=MNIST.N_TRAIN * N_CLASSES, fill=0.0)
    for i in range(MNIST.N_TRAIN):
        train_y[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0

    var t0 = perf_counter_ns()
    var result = trainer.train_cpu[MNIST.N_TRAIN, MNIST.N_TEST](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        epochs=N_EPOCHS,
    )
    var total_s = Float64(perf_counter_ns() - t0) / 1e9

    var best_acc: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best_acc:
            best_acc = a
    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")
    print("total wall time: " + String(total_s) + "s")
    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= "
        + String(TARGET_ACC * 100.0)
        + "%, got "
        + String(best_acc * 100.0)
        + "%",
    )
    print("DONE")
