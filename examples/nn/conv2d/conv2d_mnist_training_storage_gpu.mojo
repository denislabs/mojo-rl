"""LeNet-style CNN on MNIST (GPU) — storage-surface (nn.storage) port.

Exercises the storage Conv2D + Flatten + Linear stack end-to-end via the
storage `Trainer.train_gpu` whole-dataset API (resident upload + on-device
per-epoch shuffle). No BatchNorm, so the net is train/eval mode-agnostic.

    Conv2D(1→16, 5×5 s2) → ReLU   : 28→12
    Conv2D(16→32, 5×5 s2) → ReLU  : 12→4
    Flatten(32·4·4=512) → Linear(512→10)

Run (Apple):  pixi run -e apple mojo run -I . examples/nn/conv2d/conv2d_mnist_training_storage_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/conv2d/conv2d_mnist_training_storage_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.training.trainer import Trainer


def main() raises:
    comptime IN_DIM = 1 * 28 * 28
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 3
    comptime TARGET_ACC: Float64 = 0.97

    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var c = DeviceContext()

    comptime Net = Sequential[
        Conv2D[1, 16, 5, 2, 0, 28, 28],
        ReLU[16 * 12 * 12],
        Conv2D[16, 32, 5, 2, 0, 12, 12],
        ReLU[32 * 4 * 4],
        Flatten[32 * 4 * 4],
        Linear[32 * 4 * 4, NC],
    ]
    print("initializing network (GPU)...")
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=MNIST.N_TRAIN * NC, fill=0.0)
    for i in range(MNIST.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[MNIST.N_TRAIN, MNIST.N_TEST](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        Optional(c),
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
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%",
    )
    print("DONE")
