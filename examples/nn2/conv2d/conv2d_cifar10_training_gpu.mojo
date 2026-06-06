"""nn2 VGG-style CNN (with BatchNorm) on CIFAR-10 (GPU).

Port of `examples/nn/conv2d/conv2d_cifar10_training_gpu.mojo` to nn2.

Demonstrates the `Conv2DBatchNormReLU` composite + MaxPool2D + a manual
per-epoch loop that toggles BatchNorm train/eval mode via
`net.set_attr["training"](...)` — propagated through Sequential to every
BatchNorm2D leaf. Training uses batch stats; evaluation uses the running
stats accumulated during training.

Architecture (3 VGG blocks, channels 32→64→128):
    [Conv-BN-ReLU ×2, MaxPool] ×3  → 128·4·4 = 2048
    Flatten → Linear(2048→128) → ReLU → Linear(128→10)

Note: no data augmentation here (the legacy example adds random
crop+flip). Bump N_EPOCHS and add augmentation to push past ~75%+. This
example validates the conv/BN stack trains; a few epochs already clears
the accuracy gate.

Run (Apple Metal):
    pixi run -e apple mojo run -I . examples/nn2/conv2d/conv2d_cifar10_training_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.datasets import CIFAR10
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.composites import Conv2DBatchNormReLU
from mojo_rl.nn2.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


def main() raises:
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 15
    comptime LR: Scalar[DT] = 0.001
    comptime TARGET_ACC: Float64 = 0.65

    seed(42)
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
        Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
        MaxPool2D[32, 2, 2, 0, 32, 32],                 # → 32×16×16
        Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16],
        Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16],
        MaxPool2D[64, 2, 2, 0, 16, 16],                 # → 64×8×8
        Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8],
        Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8],
        MaxPool2D[128, 2, 2, 0, 8, 8],                  # → 128×4×4 = 2048
        Flatten[128 * 4 * 4],
        Linear[128 * 4 * 4, 128], ReLU[128],
        Linear[128, N_CLASSES],
    ]

    print("initializing network on GPU...")
    var trainer = Trainer[
        Net, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu",
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = LR

    print("uploading dataset to GPU...")
    comptime IN_DIM = 3 * 32 * 32
    var train_x_host = ctx.enqueue_create_host_buffer[DT](CIFAR10.N_TRAIN * IN_DIM)
    var train_y_host = ctx.enqueue_create_host_buffer[DT](CIFAR10.N_TRAIN * N_CLASSES)
    var test_x_host = ctx.enqueue_create_host_buffer[DT](CIFAR10.N_TEST * IN_DIM)
    ctx.synchronize()
    for i in range(CIFAR10.N_TRAIN * IN_DIM):
        train_x_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(CIFAR10.N_TRAIN * N_CLASSES):
        train_y_host.unsafe_ptr()[i] = 0.0
    for i in range(CIFAR10.N_TRAIN):
        train_y_host.unsafe_ptr()[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0
    for i in range(CIFAR10.N_TEST * IN_DIM):
        test_x_host.unsafe_ptr()[i] = ds.test_images[i]

    var test_labels_host: UnsafePointer[Int32, MutAnyOrigin] = (
        ctx.enqueue_create_host_buffer[DType.int32](CIFAR10.N_TEST).unsafe_ptr()
    )
    for i in range(CIFAR10.N_TEST):
        test_labels_host[i] = Int32(ds.test_labels[i])

    var train_x_dev = ctx.enqueue_create_buffer[DT](CIFAR10.N_TRAIN * IN_DIM)
    var train_y_dev = ctx.enqueue_create_buffer[DT](CIFAR10.N_TRAIN * N_CLASSES)
    var test_x_dev = ctx.enqueue_create_buffer[DT](CIFAR10.N_TEST * IN_DIM)
    ctx.enqueue_copy(train_x_dev, train_x_host)
    ctx.enqueue_copy(train_y_dev, train_y_host)
    ctx.enqueue_copy(test_x_dev, test_x_host)
    ctx.synchronize()

    var train_x = TileTensor(train_x_dev, row_major[CIFAR10.N_TRAIN, IN_DIM]())
    var train_y = TileTensor(train_y_dev, row_major[CIFAR10.N_TRAIN, N_CLASSES]())
    var test_x = TileTensor(test_x_dev, row_major[CIFAR10.N_TEST, IN_DIM]())

    var best_acc: Float64 = 0.0
    for epoch in range(N_EPOCHS):
        var t0 = perf_counter_ns()
        # BN in training mode → uses + updates batch stats.
        trainer.net.set_attr["training"](Scalar[DT](1.0))
        var r = trainer.train_gpu[CIFAR10.N_TRAIN](
            train_x, train_y, epochs=1, print_progress=False,
            shuffle=True, rng_seed=UInt64(42 + epoch),
        )
        var train_s = Float64(perf_counter_ns() - t0) / 1e9

        # BN in eval mode → uses running stats.
        trainer.net.set_attr["training"](Scalar[DT](0.0))
        var acc = trainer.eval_top1_gpu[CIFAR10.N_TEST](test_x, test_labels_host)
        if acc > best_acc:
            best_acc = acc
        print(
            "epoch " + String(epoch)
            + " | train_loss=" + String(r.epoch_train_loss[0])
            + " | test_acc=" + String(acc * 100.0) + "%"
            + " | train=" + String(train_s) + "s"
        )

    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")
    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%",
    )
    print("DONE")
