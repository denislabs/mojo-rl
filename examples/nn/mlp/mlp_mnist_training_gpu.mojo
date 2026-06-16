"""MLP on MNIST (GPU).

Port of `examples/nn/mlp/mlp_mnist_training_gpu.mojo` to nn.

Uses the `trainer.train_gpu[N_TRAIN, N_TEST]` whole-dataset API: upload
the full train + test sets to the GPU once, then the trainer slices
batches by pointer offset (no host copies in the inner loop).

Run (Apple Metal):
    pixi run -e apple mojo run -I . examples/nn/mlp/mlp_mnist_training_gpu.mojo
Run (NVIDIA CUDA):
    pixi run -e nvidia mojo run -I . examples/nn/mlp/mlp_mnist_training_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
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
    comptime LR: Scalar[DT] = 0.001
    comptime TARGET_ACC: Float64 = 0.97

    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    print("initializing network on GPU...")
    comptime Net = Sequential[
        Linear[IN_DIM, H1],
        ReLU[H1],
        Linear[H1, H2],
        ReLU[H2],
        Linear[H2, N_CLASSES],
    ]

    var trainer = Trainer[
        Net,
        Adam,
        CrossEntropyLoss[N_CLASSES],
        BATCH,
        target="gpu",
    ].make[INIT=Kaiming](ctx)

    print("uploading dataset to GPU...")
    var train_x_host = ctx.enqueue_create_host_buffer[DT](
        MNIST.N_TRAIN * IN_DIM
    )
    var train_y_host = ctx.enqueue_create_host_buffer[DT](
        MNIST.N_TRAIN * N_CLASSES
    )
    var test_x_host = ctx.enqueue_create_host_buffer[DT](MNIST.N_TEST * IN_DIM)
    ctx.synchronize()
    for i in range(MNIST.N_TRAIN * IN_DIM):
        train_x_host[i] = ds.train_images[i]
    for i in range(MNIST.N_TRAIN * N_CLASSES):
        train_y_host[i] = 0.0
    for i in range(MNIST.N_TRAIN):
        train_y_host[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0
    for i in range(MNIST.N_TEST * IN_DIM):
        test_x_host[i] = ds.test_images[i]

    var train_x_dev = ctx.enqueue_create_buffer[DT](MNIST.N_TRAIN * IN_DIM)
    var train_y_dev = ctx.enqueue_create_buffer[DT](MNIST.N_TRAIN * N_CLASSES)
    var test_x_dev = ctx.enqueue_create_buffer[DT](MNIST.N_TEST * IN_DIM)
    ctx.enqueue_copy(train_x_dev, train_x_host)
    ctx.enqueue_copy(train_y_dev, train_y_host)
    ctx.enqueue_copy(test_x_dev, test_x_host)
    ctx.synchronize()

    var train_x = TileTensor(train_x_dev, row_major[MNIST.N_TRAIN, IN_DIM]())
    var train_y = TileTensor(train_y_dev, row_major[MNIST.N_TRAIN, N_CLASSES]())
    var test_x = TileTensor(test_x_dev, row_major[MNIST.N_TEST, IN_DIM]())

    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[MNIST.N_TRAIN, MNIST.N_TEST](
        train_x,
        train_y,
        test_x,
        ds.test_labels,
        epochs=N_EPOCHS,
        shuffle=True,
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
