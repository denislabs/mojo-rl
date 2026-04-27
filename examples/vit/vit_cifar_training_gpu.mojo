"""ViT CIFAR-10 training — GPU.

Validates the ViT composite (PatchEmbed + Transpose2D + non-causal
TransformerBlock × N + final LN + TokenMean + LM head) against CIFAR-10
classification. Doc target (Phase B): ≥ 70 % top-1 with the default config
(D=192, H=6, N=6, patch=4, n_patches=64).

Uses `Trainer.train_gpu_minibatch_full` for the full training loop:
  - CosineWarmupSchedule for the LR, applied per-epoch on device via
    `state.set_lr_scale`.
  - `CIFAR10CropFlipAugmenter` re-augments the training set each epoch
    (random pad-4 crop + horizontal flip — same recipe as the resnet20 test).
  - Per-epoch top-1 + CE-loss on the test set, computed on-device by the
    Trainer's eval kernels (no host argmax loops).

Default config is sized for NVIDIA GPUs. On the M1 Pro development hardware
the config OOMs — for local iteration shrink EMBED, LAYERS, BATCH, and EPOCHS
to the values commented in the constants block.

Run on NVIDIA (production target):
    pixi run -e nvidia mojo run -I . examples/vit/vit_cifar_training_gpu.mojo
Run on Apple Metal (dev iteration only — shrink config first):
    pixi run -e apple mojo run -I . examples/vit/vit_cifar_training_gpu.mojo
"""

from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from std.math import log

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.composites import ViT
from mojo_rl.nn.training import (
    GPUNetworkState,
    Trainer,
    Augmenter,
    CosineWarmupSchedule,
)
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets import CIFAR10
from layout import Layout, LayoutTensor


# =============================================================================
# Hyperparameters — full ViT config targeting ≥70% top-1 on CIFAR-10.
# Matches docs/TRANSFORMER_VIT.md Phase B defaults. Designed for NVIDIA GPUs;
# on Apple M1 Pro shrink EMBED→64, LAYERS→4, BATCH→32, EPOCHS→10.
# =============================================================================
comptime IN_CHANNELS = 3
comptime IMG_H = 32
comptime IMG_W = 32
comptime PATCH = 4                 # 8×8 = 64 patches at 32×32 input
comptime N_PATCHES = (IMG_H // PATCH) * (IMG_W // PATCH)
comptime EMBED = 192               # transformer width
comptime HEADS = 6                 # head_dim = 32
comptime LAYERS = 6                # transformer blocks
comptime FF_MULT = 4
comptime N_CLASSES = 10

comptime BATCH = 128

comptime BASE_LR = 3e-4
comptime BETA1 = 0.9
comptime BETA2 = 0.999             # vision: canonical 0.999 (vs 0.95 for LM)
comptime WD = 0.05

# 50k train images / 128 batch = 391 iters/epoch. 100 epochs ≈ 39000 steps.
# Cosine spans the full window (post-warmup), so doubling EPOCHS doesn't just
# add tail steps at lr_scale=0.1 — it stretches the whole schedule, giving
# the model ~2× more time at lr_scale > 0.5.
comptime EPOCHS = 100
comptime WARMUP_EPOCHS = 5         # ~5 epochs of linear warmup
# Per-epoch eval over the test set; the Trainer's `N_VAL_BATCHES =
# N_VAL // BATCH` slicing drops the final partial batch automatically
# (10_000 / 128 = 78 batches, last 16 images skipped).
comptime N_VAL = CIFAR10.N_TEST


# =============================================================================
# Aliases for readability.
# =============================================================================
comptime VIT_MODEL = ViT[
    IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
    N_CLASSES, FF_MULT,
]
comptime VIT_OPT = AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD]
comptime VIT_TRAINER = Trainer[VIT_MODEL, VIT_OPT, CrossEntropyLoss]
comptime VIT_SCHEDULER = CosineWarmupSchedule[WARMUP_EPOCHS, 0.1]


# =============================================================================
# CIFAR augmentation kernel — random crop pad-4 + horizontal flip per sample.
# Identical recipe to tests/nn/test_resnet20_cifar10.mojo.
# Grid: (N,), Block: (TPB,). One block per sample; threads parallelize the
# 3072 output pixels. All threads in a block derive dx/dy/flip from
# PhiloxRandom(epoch_seed, b) identically — out-of-bounds pixels get 0.
# =============================================================================
def _cifar_augment_kernel[
    N: Int,
    dtype: DType = DType.float32,
](
    aug: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    raw: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    epoch_seed: Scalar[DType.uint64],
):
    var b = Int(block_idx.x)
    if b >= N:
        return
    var tid = Int(thread_idx.x)

    comptime C = 3
    comptime H = 32
    comptime W = 32
    comptime CHAN = H * W
    comptime IMG_SIZE = C * CHAN

    var rng = PhiloxRandom(seed=UInt64(epoch_seed), offset=UInt64(b))
    var r = rng.step_uniform()
    var dx = Int(Scalar[DType.float32](r[0]) * 9.0) - 4  # [-4, 4]
    var dy = Int(Scalar[DType.float32](r[1]) * 9.0) - 4  # [-4, 4]
    var flip = Scalar[DType.float32](r[2]) > 0.5

    var idx = tid
    while idx < IMG_SIZE:
        var c = idx // CHAN
        var yx = idx % CHAN
        var oy = yx // W
        var ox = yx % W
        var src_y = oy + dy
        var vx = ox + dx
        var val = Scalar[dtype](0.0)
        if src_y >= 0 and src_y < H and vx >= 0 and vx < W:
            var src_x = (W - 1 - vx) if flip else vx
            val = rebind[Scalar[dtype]](raw[b, c * CHAN + src_y * W + src_x])
        aug[b, idx] = val
        idx += TPB


# =============================================================================
# Augmenter wrapper around the kernel — fed to Trainer as a comptime
# parameter. Trainer calls `augment` before each epoch with a fresh seed.
# =============================================================================
struct CIFAR10CropFlipAugmenter(Augmenter):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def augment[N: Int, IN_DIM: Int, ddtype: DType](
        ctx: DeviceContext,
        aug: LayoutTensor[ddtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        raw: LayoutTensor[ddtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        epoch: Int,
        base_seed: UInt64,
    ) raises:
        comptime aug_k = _cifar_augment_kernel[N, ddtype]
        ctx.enqueue_function[aug_k, aug_k](
            aug,
            raw,
            Scalar[DType.uint64](base_seed + UInt64(epoch)),
            grid_dim=(N,),
            block_dim=(TPB,),
        )


# =============================================================================
# Driver
# =============================================================================
def main() raises:
    seed(42)

    print("=" * 70)
    print("ViT CIFAR-10 training (GPU) — Phase B target ≥70% top-1")
    print("=" * 70)
    print(
        "  in_ch="
        + String(IN_CHANNELS)
        + " img="
        + String(IMG_H)
        + "x"
        + String(IMG_W)
        + " patch="
        + String(PATCH)
        + " n_patches="
        + String(N_PATCHES)
    )
    print(
        "  embed="
        + String(EMBED)
        + " heads="
        + String(HEADS)
        + " layers="
        + String(LAYERS)
        + " ff_mult="
        + String(FF_MULT)
        + " n_classes="
        + String(N_CLASSES)
    )
    print(
        "  batch="
        + String(BATCH)
        + " base_lr="
        + String(BASE_LR)
        + " wd="
        + String(WD)
        + " epochs="
        + String(EPOCHS)
        + " warmup_ep="
        + String(WARMUP_EPOCHS)
        + " n_val="
        + String(N_VAL)
    )
    print(
        "  PARAM_SIZE="
        + String(VIT_MODEL.PARAM_SIZE)
        + " CACHE/sample="
        + String(VIT_MODEL.CACHE_SIZE)
        + " WS/sample="
        + String(VIT_MODEL.WORKSPACE_SIZE_PER_SAMPLE)
    )

    # ---------- Data ----------
    print("\n[data] loading CIFAR-10...")
    var ds = CIFAR10()
    var ctx = DeviceContext()

    # ---------- Initialize network on GPU ----------
    var state = VIT_TRAINER.init_state_gpu[Xavier[]](ctx)

    # ---------- Upload full train set (raw images + one-hot targets) ----------
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    for i in range(CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = 0
    for i in range(CIFAR10.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.train_labels[i])
        ] = 1

    var train_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var train_img_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE), MutAnyOrigin
    ](train_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.NUM_CLASSES),
        MutAnyOrigin,
    ](train_tgt_buf)

    # ---------- Upload test set + int32 labels (no augmentation here) ----------
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    var test_lbl_host = ctx.enqueue_create_host_buffer[DType.int32](
        CIFAR10.N_TEST
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    for i in range(CIFAR10.N_TEST):
        test_lbl_host.unsafe_ptr()[i] = ds.test_labels[i]

    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    var test_lbl_buf = ctx.enqueue_create_buffer[DType.int32](CIFAR10.N_TEST)
    ctx.enqueue_copy(test_img_buf, test_img_host)
    ctx.enqueue_copy(test_lbl_buf, test_lbl_host)

    var test_img_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TEST, CIFAR10.IMG_SIZE), MutAnyOrigin
    ](test_img_buf.unsafe_ptr())
    var test_lbl_lt = LayoutTensor[
        DType.int32, Layout.row_major(CIFAR10.N_TEST), MutAnyOrigin
    ](test_lbl_buf.unsafe_ptr())

    print(
        "  random baseline ≈ ln(C)="
        + String(log(Float64(N_CLASSES)))
        + " / 1/C="
        + String(1.0 / Float64(N_CLASSES))
    )

    # ---------- Train + per-epoch eval in one call ----------
    print("\n── Training ──")
    var t_start = perf_counter_ns()

    var result = VIT_TRAINER.train_gpu_minibatch_full[
        BATCH, CIFAR10.N_TRAIN, N_VAL,
        VIT_SCHEDULER, CIFAR10CropFlipAugmenter,
    ](
        state, ctx,
        train_img_lt, train_tgt_lt,
        test_img_lt, test_lbl_lt,
        epochs=EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
        show_progress=True,
        eval_every_epochs=1,
        progress_label="ViT-CIFAR10",
    )

    var t_end = perf_counter_ns()
    print(
        "\n  training time: "
        + String(Float64(t_end - t_start) / 1e9)[byte=:6]
        + " s"
    )

    # ---------- Final report ----------
    var n_evals = len(result.val_top1_history)
    var final_top1 = result.val_top1_history[n_evals - 1]
    var final_loss = result.val_loss_history[n_evals - 1]
    print("\n── Final evaluation (full test set) ──")
    print(
        "  test_loss="
        + String(final_loss)
        + " top1="
        + String(final_top1)
    )
    if final_top1 >= 0.70:
        print("  PASS: top-1 accuracy ≥ 70 % — Phase B target hit")
    elif final_top1 > 0.55:
        print(
            "  PARTIAL: top-1 > 55 % — clearly learning, may need more"
            + " epochs / stronger aug for 70 %"
        )
    else:
        print("  WARN: top-1 ≤ 55 % — increase EPOCHS or check")
    print("=" * 70)
