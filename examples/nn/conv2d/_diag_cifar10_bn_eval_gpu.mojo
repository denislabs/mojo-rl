"""DIAGNOSTIC (throwaway): CIFAR VGG — split eval-mode vs train-mode top-1.

Mirrors conv2d_cifar10_training_storage_gpu.mojo, but after training it reports
the test top-1 TWICE:
  - eval-mode  (BatchNorm running stats)  ← what the real eval uses
  - train-mode (BatchNorm per-batch stats)

On Apple these match (running stats are correct). If on NVIDIA eval-mode is much
LOWER than train-mode, the BN running-stat/eval path is the culprit in the full
net. If BOTH are low (~20%), eval is fine and the trained features themselves
don't classify the test set (not a BN-eval issue).

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/conv2d/_diag_cifar10_bn_eval_gpu.mojo

Delete after debugging.
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CIFAR10
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.training.trainer import Trainer


def main() raises:
    comptime IN_DIM = 3 * 32 * 32
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 15

    seed(42)
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var c = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
        Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
        MaxPool2D[32, 2, 2, 0, 32, 32],
        Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16],
        Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16],
        MaxPool2D[64, 2, 2, 0, 16, 16],
        Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8],
        Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8],
        MaxPool2D[128, 2, 2, 0, 8, 8],
        Flatten[128 * 4 * 4],
        Linear[128 * 4 * 4, 128],
        ReLU[128],
        Linear[128, NC],
    ]
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=CIFAR10.N_TRAIN * NC, fill=0.0)
    for i in range(CIFAR10.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    _ = trainer.train_gpu[CIFAR10.N_TRAIN, CIFAR10.N_TEST](
        ds.train_images, train_y, ds.test_images, ds.test_labels,
        Optional(c), epochs=N_EPOCHS, shuffle=True,
    )

    print("\n=== post-train eval comparison ===")
    # train_gpu leaves the model in eval mode (training=0).
    var acc_eval = trainer.eval_top1[CIFAR10.N_TEST](
        ds.test_images, ds.test_labels, Optional(c)
    )
    print("eval-mode  (running stats) top1 = " + String(acc_eval * 100.0) + "%")

    trainer.model.set_attr["training"](Scalar[DT](1.0))
    var acc_train = trainer.eval_top1[CIFAR10.N_TEST](
        ds.test_images, ds.test_labels, Optional(c)
    )
    print("train-mode (batch stats)   top1 = " + String(acc_train * 100.0) + "%")
    print("DONE")
