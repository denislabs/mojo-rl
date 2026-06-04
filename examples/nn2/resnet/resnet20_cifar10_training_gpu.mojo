"""nn2 ResNet-20 on CIFAR-10 (GPU).

Port of `examples/nn/resnet/resnet20_cifar10_training_gpu.mojo` to nn2.

ResNet-20 (He et al. 2016, CIFAR variant; 6n+2 with n=3): a 3×3 BN stem
followed by 3 stages of 3 residual blocks (channels 16→32→64), with the
first block of stages 2 and 3 downsampling (stride-2 main path + 1×1
projection skip), then global average pool + linear head.

Built entirely from the nn2 composites added for the zero series:
  - `ResBlockConv2DBN`      — identity-skip block (Residual)
  - `ResBlockDownsampleBN`  — projection-skip block (ProjectedResidual)

BatchNorm train/eval is toggled per epoch via `set_attr["training"]`,
which propagates through Sequential/Residual/ProjectedResidual to every
BatchNorm2D leaf.

Note: no augmentation here; full convergence (~91%) needs SGD+momentum,
augmentation and ~80 epochs. This example validates the residual stack
trains end-to-end. ResNet-20 is deep — expect a long compile.

Run (Apple Metal):
    pixi run -e apple mojo run -I . examples/nn2/resnet/resnet20_cifar10_training_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.datasets import CIFAR10
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.composites import Conv2DBatchNormReLU, ResBlockConv2DBN, ResBlockDownsampleBN
from mojo_rl.nn2.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


def main() raises:
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 20
    comptime LR: Scalar[DT] = 0.001
    comptime TARGET_ACC: Float64 = 0.60

    seed(42)
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 16, 3, 1, 1, 32, 32],     # stem → 16×32×32
        # Stage 1: 16 channels, 32×32
        ResBlockConv2DBN[16, 3, 1, 32, 32],
        ResBlockConv2DBN[16, 3, 1, 32, 32],
        ResBlockConv2DBN[16, 3, 1, 32, 32],
        # Stage 2: 16→32, 32×32 → 16×16
        ResBlockDownsampleBN[16, 32, 3, 1, 32, 32],
        ResBlockConv2DBN[32, 3, 1, 16, 16],
        ResBlockConv2DBN[32, 3, 1, 16, 16],
        # Stage 3: 32→64, 16×16 → 8×8
        ResBlockDownsampleBN[32, 64, 3, 1, 16, 16],
        ResBlockConv2DBN[64, 3, 1, 8, 8],
        ResBlockConv2DBN[64, 3, 1, 8, 8],
        # Head: global avg pool → 64, linear
        AvgPool2D[64, 8, 8, 0, 8, 8],
        Flatten[64],
        Linear[64, N_CLASSES],
    ]

    print("initializing ResNet-20 on GPU (this compile is long)...")
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
        trainer.net.set_attr["training"](Scalar[DT](1.0))
        var r = trainer.train_gpu[CIFAR10.N_TRAIN](
            train_x, train_y, epochs=1, print_progress=False,
            shuffle=True, rng_seed=UInt64(42 + epoch),
        )
        var train_s = Float64(perf_counter_ns() - t0) / 1e9

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
