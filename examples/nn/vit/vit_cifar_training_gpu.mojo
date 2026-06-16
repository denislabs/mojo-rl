"""ViT CIFAR-10 training — nn GPU (real-dataset parity run).

nn port of `examples/nn/vit/vit_cifar_training_gpu.mojo`. Trains the nn
`ViT` composite (PatchEmbed → pos BiasAdd → non-causal TransformerBlock×N →
LayerNorm → TokenMean → head) on CIFAR-10 via the stateful nn
`Trainer.train_gpu` (on-device shuffle + CIFAR10CropFlipAugmenter +
WarmupCosineSchedule + per-epoch top-1 eval).

Config note: the DEFAULT below is the *production* config (≥70% top-1 target,
NVIDIA) — larger model, 100 epochs, BATCH=128 — and OOMs on an M1. For Apple
Metal / CI smoke runs, flip to the DEV config in the comment block (small
model, 8 epochs).

Deferred vs gen-1 (convergence refinements, not architecture): nanoGPT-style
`Normal(0,0.02)` init + 1/√(2L) c_proj scaled-init are not applied (nn uses
Kaiming); see docs/NN2_TRANSFORMER_PORT.md.

Run:
    pixi run -e apple  mojo run -I . examples/nn/vit/vit_cifar_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/nn/vit/vit_cifar_training_gpu.mojo
"""

from std.memory import alloc
from std.random import seed
from std.time import perf_counter_ns
from std.math import log
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.datasets import CIFAR10
from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.vit import ViT
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.training import Trainer, WarmupCosineSchedule
from mojo_rl.nn.training.augmenter import CIFAR10CropFlipAugmenter
from mojo_rl.nn.initializer import Kaiming


# ── PRODUCTION config (NVIDIA, ≥70% target) ────────────────────────────
# DEV / Apple-smoke (runs on M1): PATCH=8, N_PATCHES=16, EMBED=64, HEADS=4,
#   LAYERS=3, FF_MULT=2, BATCH=64, EPOCHS=8, WARMUP=2.
comptime IN_CHANNELS = 3
comptime IMG_H = 32
comptime IMG_W = 32
comptime PATCH = 4  # 8×8 = 64 patches
comptime N_PATCHES = (IMG_H // PATCH) * (IMG_W // PATCH)
comptime EMBED = 192
comptime HEADS = 6  # head_dim = 32
comptime LAYERS = 6
comptime FF_MULT = 4
comptime N_CLASSES = 10

comptime BATCH = 128
comptime EPOCHS = 100
comptime WARMUP_EPOCHS = 5

comptime BASE_LR: Scalar[DT] = 3e-4
comptime WD: Scalar[DT] = 0.05

# Drop trailing partial batch (train_gpu asserts BATCH-divisibility).
comptime N_TRAIN = (CIFAR10.N_TRAIN // BATCH) * BATCH
comptime N_TEST = (CIFAR10.N_TEST // BATCH) * BATCH

comptime VIT_MODEL = ViT[
    IN_CHANNELS,
    IMG_H,
    IMG_W,
    PATCH,
    EMBED,
    HEADS,
    LAYERS,
    N_PATCHES,
    N_CLASSES,
    FF_MULT,
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("ViT CIFAR-10 training — nn GPU")
    print("=" * 70)
    print(
        "  patch="
        + String(PATCH)
        + " n_patches="
        + String(N_PATCHES)
        + " embed="
        + String(EMBED)
        + " heads="
        + String(HEADS)
        + " layers="
        + String(LAYERS)
        + " ff_mult="
        + String(FF_MULT)
    )
    print(
        "  batch="
        + String(BATCH)
        + " epochs="
        + String(EPOCHS)
        + " base_lr="
        + String(BASE_LR)
        + " wd="
        + String(WD)
        + " | random baseline ≈ "
        + String(1.0 / Float64(N_CLASSES))
    )

    print("\n[data] loading CIFAR-10...")
    var ds = CIFAR10()
    var ctx = DeviceContext()

    print("[init] building nn ViT on GPU...")
    var net = VIT_MODEL.make["gpu", INIT=Kaiming](ctx)
    var loss_fn = CrossEntropyLoss[N_CLASSES].make["gpu"](ctx)
    var optim = AdamW.make["gpu"](net, ctx)
    optim.lr = BASE_LR
    optim.weight_decay = WD
    optim.beta2 = 0.999

    var trainer = Trainer[
        BATCH=BATCH,
        target="gpu",
    ].make_from(net^, optim^, loss_fn^, ctx)

    # ── Upload train set (images + one-hot targets) ──
    var tr_img_h = ctx.enqueue_create_host_buffer[DT](
        N_TRAIN * CIFAR10.IMG_SIZE
    )
    var tr_tgt_h = ctx.enqueue_create_host_buffer[DT](N_TRAIN * N_CLASSES)
    for i in range(N_TRAIN * CIFAR10.IMG_SIZE):
        tr_img_h[i] = ds.train_images[i]
    for i in range(N_TRAIN * N_CLASSES):
        tr_tgt_h[i] = 0.0
    for i in range(N_TRAIN):
        tr_tgt_h[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0
    var tr_img_d = ctx.enqueue_create_buffer[DT](N_TRAIN * CIFAR10.IMG_SIZE)
    var tr_tgt_d = ctx.enqueue_create_buffer[DT](N_TRAIN * N_CLASSES)
    ctx.enqueue_copy(tr_img_d, tr_img_h)
    ctx.enqueue_copy(tr_tgt_d, tr_tgt_h)

    var train_x = TileTensor(tr_img_d, row_major[N_TRAIN, CIFAR10.IMG_SIZE]())
    var train_y = TileTensor(tr_tgt_d, row_major[N_TRAIN, N_CLASSES]())

    # ── Upload test images; labels stay host-side (int32) for eval ──
    var te_img_h = ctx.enqueue_create_host_buffer[DT](N_TEST * CIFAR10.IMG_SIZE)
    for i in range(N_TEST * CIFAR10.IMG_SIZE):
        te_img_h[i] = ds.test_images[i]
    var te_img_d = ctx.enqueue_create_buffer[DT](N_TEST * CIFAR10.IMG_SIZE)
    ctx.enqueue_copy(te_img_d, te_img_h)

    var test_x = TileTensor(te_img_d, row_major[N_TEST, CIFAR10.IMG_SIZE]())

    ctx.synchronize()

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[
        N_TRAIN,
        N_TEST,
        CIFAR10CropFlipAugmenter,
        WarmupCosineSchedule[WARMUP_EPOCHS, 0.1],
    ](
        train_x,
        train_y,
        test_x,
        ds.test_labels,
        epochs=EPOCHS,
        print_progress=True,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s")

    var n = len(result.epoch_test_top1)
    var final_top1 = result.epoch_test_top1[n - 1]
    print("\n── Final ──")
    print("  top1=" + String(final_top1 * 100.0)[byte=:6] + "%")

    print("=" * 70)
    # Production target: from-scratch ViT + crop/flip aug + cosine should reach
    # ~70% on this config. Threshold set at 60% to tolerate seed variance.
    # (For the dev/Apple-smoke config, ≥30% is the right floor.)
    if final_top1 >= 0.60:
        print("PASS — nn ViT CIFAR-10 (top1 ≥ 60%, production config)")
    else:
        raise Error("ViT under target: top1=" + String(final_top1))
