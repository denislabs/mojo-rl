"""Trainer augmentation smoke (GPU): train a small BN conv net on a CIFAR
subset via train_gpu with CIFAR10CropFlipAugmenter. Validates the
augmenter kernel + per-epoch aug buffer + BatchNorm train/eval toggle
end-to-end; asserts it learns above chance and stays finite.
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.datasets import CIFAR10
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.composites import Conv2DBatchNormReLU
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer, CIFAR10CropFlipAugmenter
from mojo_rl.nn2.initializer import Kaiming


def main() raises:
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime NTR = 2000
    comptime NTE = 1000
    comptime IN_DIM = 3 * 32 * 32
    comptime EPOCHS = 4

    seed(42)
    print("test_trainer_augment_gpu_smoke ...")
    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 16, 3, 2, 1, 32, 32],     # → 16×16×16
        Conv2DBatchNormReLU[16, 32, 3, 2, 1, 16, 16],    # → 32×8×8 = 2048
        Flatten[32 * 8 * 8],
        Linear[32 * 8 * 8, N_CLASSES],
    ]
    var trainer = Trainer[
        Net, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu",
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = Scalar[DT](0.001)

    var xh = ctx.enqueue_create_host_buffer[DT](NTR * IN_DIM)
    var yh = ctx.enqueue_create_host_buffer[DT](NTR * N_CLASSES)
    var txh = ctx.enqueue_create_host_buffer[DT](NTE * IN_DIM)
    ctx.synchronize()
    for i in range(NTR * IN_DIM):
        xh.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(NTR * N_CLASSES):
        yh.unsafe_ptr()[i] = 0.0
    for i in range(NTR):
        yh.unsafe_ptr()[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0
    for i in range(NTE * IN_DIM):
        txh.unsafe_ptr()[i] = ds.test_images[i]
    var tlbl: UnsafePointer[Int32, MutAnyOrigin] = (
        ctx.enqueue_create_host_buffer[DType.int32](NTE).unsafe_ptr()
    )
    for i in range(NTE):
        tlbl[i] = Int32(ds.test_labels[i])

    var xd = ctx.enqueue_create_buffer[DT](NTR * IN_DIM)
    var yd = ctx.enqueue_create_buffer[DT](NTR * N_CLASSES)
    var txd = ctx.enqueue_create_buffer[DT](NTE * IN_DIM)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(yd, yh)
    ctx.enqueue_copy(txd, txh)
    ctx.synchronize()

    var train_x = TileTensor(xd, row_major[NTR, IN_DIM]())
    var train_y = TileTensor(yd, row_major[NTR, N_CLASSES]())
    var test_x = TileTensor(txd, row_major[NTE, IN_DIM]())

    var result = trainer.train_gpu[NTR, NTE, AUGMENTER=CIFAR10CropFlipAugmenter](
        train_x, train_y, test_x, tlbl,
        epochs=EPOCHS, shuffle=True, rng_seed=UInt64(42), aug_seed=UInt64(1000),
    )

    var best: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best:
            best = a
    var last_loss = result.epoch_train_loss[EPOCHS - 1]
    print("  best_acc =", best * 100.0, "%  last_train_loss =", last_loss)
    assert_true(last_loss == last_loss, "train loss NaN")
    assert_true(best > 0.20, "augmented training should clear chance (>20%)")
    print("  ok")
