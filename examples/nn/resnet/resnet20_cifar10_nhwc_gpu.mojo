"""ResNet-20 on CIFAR-10 — NCHW vs NHWC convergence A/B (GPU).

Channels-last (NHWC) convergence validator for the conv-stack layout migration.
The per-op parity gates already prove conv/BN/pool are numerically bit-identical
across layouts; this proves the WHOLE pipeline (stem → 3 residual stages → global
avg-pool → linear head, with per-epoch crop+flip aug, BN train/eval, cosine LR)
LEARNS identically under NHWC — catching any integration bug a per-op gate can't.
On NVIDIA the NHWC BN runs the 2D occupancy kernels (USE_2D_NHWC), so this also
exercises those in a real converging run, beyond the parity gate.

Flip `USE_NHWC` and run both; the best test accuracies should match (the NCHW arm
reproduces resnet20_cifar10_training_storage_gpu.mojo's ~80%+). NHWC needs no
transpose tax in the net: the CIFAR images are transposed to NHWC ONCE at load,
the augmenter emits NHWC, and the head's global avg-pool collapses spatial to 1×1
so Flatten→Linear is layout-invariant (no retrain-permutation at the head).

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/resnet/resnet20_cifar10_nhwc_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . examples/nn/resnet/resnet20_cifar10_nhwc_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CIFAR10
from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.models.resnet import (
    ResBlockConv2DBN,
    ResBlockDownsampleBN,
)
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.repeat import Repeat
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.training.augmenter import (
    CIFAR10CropFlipAugmenter,
    CIFAR10CropFlipAugmenterNHWC,
)
from mojo_rl.nn.optimizer.lr_scheduler import WarmupCosineSchedule


# ── A/B toggle: True = channels-last (NHWC), False = channels-first (NCHW) ──
comptime USE_NHWC = False
comptime LAYOUT = LAYOUT_NHWC if USE_NHWC else LAYOUT_NCHW


# Reorder one logical [N, 3*32*32] image set from NCHW (c*HW + y*W + x) to NHWC
# ((y*W+x)*C + c). One-time host transpose at load; values unchanged (the per-
# channel normalization was already applied by the loader in NCHW order).
def _nchw_to_nhwc(src: List[Scalar[DT]], N: Int) -> List[Scalar[DT]]:
    comptime C = 3
    comptime HW = 32 * 32
    comptime IMG = C * HW
    var dst = List[Scalar[DT]](length=len(src), fill=Scalar[DT](0.0))
    for n in range(N):
        var base = n * IMG
        for c in range(C):
            for s in range(HW):
                dst[base + s * C + c] = src[base + c * HW + s]
    return dst^


def main() raises:
    comptime IN_DIM = 3 * 32 * 32
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 50  # match the NCHW baseline recipe (~80%+)
    comptime TARGET_ACC: Float64 = 0.80

    seed(42)
    print(
        "ResNet-20 CIFAR-10 ["
        + ("NHWC channels-last" if USE_NHWC else "NCHW channels-first")
        + "]"
    )
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var c = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 16, 3, 1, 1, 32, 32, LAYOUT=LAYOUT],  # stem
        # Stage 1: 3 identity blocks @ 16ch, 32×32
        Repeat[
            3, ResBlockConv2DBN[16, 3, 1, 32, 32, LAYOUT=LAYOUT], shared=False
        ],
        # Stage 2: downsample 16→32 (32×32→16×16) + 2 identity blocks
        ResBlockDownsampleBN[16, 32, 3, 1, 32, 32, LAYOUT=LAYOUT],
        Repeat[
            2, ResBlockConv2DBN[32, 3, 1, 16, 16, LAYOUT=LAYOUT], shared=False
        ],
        # Stage 3: downsample 32→64 (16×16→8×8) + 2 identity blocks
        ResBlockDownsampleBN[32, 64, 3, 1, 16, 16, LAYOUT=LAYOUT],
        Repeat[
            2, ResBlockConv2DBN[64, 3, 1, 8, 8, LAYOUT=LAYOUT], shared=False
        ],
        # Head: global avg pool → 64 (spatial collapses to 1×1 → layout-invariant)
        AvgPool2D[64, 8, 8, 0, 8, 8, LAYOUT=LAYOUT],
        Flatten[64],
        Linear[64, NC],
    ]

    print("initializing ResNet-20 on GPU (this compile is long)...")
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=CIFAR10.N_TRAIN * NC, fill=0.0)
    for i in range(CIFAR10.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    comptime Sched = WarmupCosineSchedule[5, 0.01]
    var best_acc: Float64 = 0.0
    # The augmenter type differs by layout (no type-ternary in Mojo), so the run
    # call is branched; everything else is identical.
    comptime if USE_NHWC:
        print("transposing CIFAR images NCHW→NHWC (one-time)...")
        var train_x = _nchw_to_nhwc(ds.train_images, CIFAR10.N_TRAIN)
        var test_x = _nchw_to_nhwc(ds.test_images, CIFAR10.N_TEST)
        var result = trainer.train_gpu[
            CIFAR10.N_TRAIN,
            CIFAR10.N_TEST,
            CIFAR10CropFlipAugmenterNHWC,
            Sched,
        ](
            train_x,
            train_y,
            test_x,
            ds.test_labels,
            Optional(c),
            epochs=N_EPOCHS,
            shuffle=True,
            rng_seed=UInt64(42),
            aug_seed=UInt64(1000),
        )
        for a in result.epoch_test_top1:
            if a > best_acc:
                best_acc = a
    else:
        var result = trainer.train_gpu[
            CIFAR10.N_TRAIN,
            CIFAR10.N_TEST,
            CIFAR10CropFlipAugmenter,
            Sched,
        ](
            ds.train_images,
            train_y,
            ds.test_images,
            ds.test_labels,
            Optional(c),
            epochs=N_EPOCHS,
            shuffle=True,
            rng_seed=UInt64(42),
            aug_seed=UInt64(1000),
        )
        for a in result.epoch_test_top1:
            if a > best_acc:
                best_acc = a

    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")
    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= "
        + String(TARGET_ACC * 100.0)
        + "%, got "
        + String(best_acc * 100.0)
        + "%",
    )
    print("DONE")
