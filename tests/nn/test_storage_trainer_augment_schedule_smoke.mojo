"""Storage `Trainer.train_gpu` augment + LR-schedule smoke (GPU).

Exercises the Tier-B run features together on a tiny synthetic CIFAR-shaped
dataset (IN = 3·32·32 = 3072 so `CIFAR10CropFlipAugmenter`'s assert holds):
per-epoch on-device crop+flip augmentation, `WarmupCosineSchedule` LR scaling,
and on-device shuffle, all through one `train_gpu` call.

Each class is encoded as an intensity over the central 24×24 region (rows/cols
4..28) of every channel — a dense signal that survives the pad-4 crop (±4 keeps
the bulk in-frame) and the horizontal flip (symmetric about column 16), so a
linear head learns it quickly and the run climbs well above 10% chance.

Run: pixi run -e apple mojo run -I . tests/nn/test_storage_trainer_augment_schedule_smoke.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.training.augmenter import CIFAR10CropFlipAugmenter
from mojo_rl.nn.optimizer.lr_scheduler import WarmupCosineSchedule


comptime IN = 3 * 32 * 32  # 3072 (required by CIFAR augmenter)
comptime NC = 10
comptime BATCH = 40
comptime N_TRAIN = 200
comptime N_TEST = 40
comptime N_EPOCHS = 80


def _make_set(n: Int, mut x: List[Scalar[DT]], mut y_oh: List[Scalar[DT]], mut labels: List[Int32]):
    comptime H = 32
    comptime W = 32
    comptime CHAN = H * W
    for i in range(n):
        var cls = i % NC
        var v = Scalar[DT](Float64(cls + 1) * 0.1)  # central-region intensity
        for ch in range(3):
            for row in range(4, 28):
                for col in range(4, 28):
                    x[i * IN + ch * CHAN + row * W + col] = v
        y_oh[i * NC + cls] = 1.0
        labels[i] = Int32(cls)


def main() raises:
    print("=" * 60)
    print("storage Trainer.train_gpu augment + schedule smoke")
    print("=" * 60)
    seed(0)
    var c = DeviceContext()

    var train_x = List[Scalar[DT]](length=N_TRAIN * IN, fill=0.0)
    var train_y = List[Scalar[DT]](length=N_TRAIN * NC, fill=0.0)
    var train_lbl = List[Int32](length=N_TRAIN, fill=0)
    _make_set(N_TRAIN, train_x, train_y, train_lbl)
    var test_x = List[Scalar[DT]](length=N_TEST * IN, fill=0.0)
    var test_y = List[Scalar[DT]](length=N_TEST * NC, fill=0.0)
    var test_lbl = List[Int32](length=N_TEST, fill=0)
    _make_set(N_TEST, test_x, test_y, test_lbl)

    # Convex linear softmax classifier — no hidden ReLU to die; the smoke
    # validates the train_gpu augment/schedule/shuffle loop, not net depth.
    comptime Net = Linear[IN, NC]
    var trainer = Trainer[Net, NC, IN, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-2
    )

    var result = trainer.train_gpu[
        N_TRAIN,
        N_TEST,
        CIFAR10CropFlipAugmenter,
        WarmupCosineSchedule[2, 0.1],
    ](
        train_x, train_y, test_x, test_lbl, Optional(c),
        epochs=N_EPOCHS, shuffle=True, print_progress=False,
    )

    var best: Float64 = 0.0
    var last_finite = True
    for i in range(len(result.epoch_test_top1)):
        var a = result.epoch_test_top1[i]
        if a != a:
            last_finite = False
        if a > best:
            best = a
    var loss0 = result.epoch_train_loss[0]
    var lossN = result.epoch_train_loss[len(result.epoch_train_loss) - 1]
    print("  epochs run:", len(result.epoch_test_top1))
    print("  train_loss:", loss0, "->", lossN)
    print("  best test top-1:", best * 100.0, "%")
    # Machinery checks: the augment/schedule/shuffle path ran every epoch with
    # finite results and the optimizer made real progress (loss fell well below
    # the uniform ln(10) ≈ 2.30 baseline), climbing above 10% chance.
    assert_true(last_finite, "all epoch accuracies finite")
    assert_true(len(result.epoch_test_top1) == N_EPOCHS, "ran all epochs")
    assert_true(lossN < loss0 - 0.5, "train loss decreased substantially")
    assert_true(best > 0.3, "augment+schedule run climbs above chance")
    print("ALL PASSED")
