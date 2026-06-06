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
    pixi run -e nvidia mojo run -I . examples/nn/vit/vit_cifar_training_gpu.mojo
Run on Apple Metal (dev iteration only — shrink config first):
    pixi run -e apple mojo run -I . examples/nn/vit/vit_cifar_training_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed
from std.time import perf_counter_ns
from std.math import log, sqrt

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import ViT
from mojo_rl.nn.training import (
    NetworkState,
    GPUNetworkState,
    Trainer,
    CosineWarmupSchedule,
)
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.initializer import Normal
from mojo_rl.nn2.datasets import CIFAR10, CIFAR10CropFlipAugmenter
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
# Per-block layout (TransformerBlock inside the ViT chain) — used by the
# c_proj scaled-init pass below. Each block lays out as:
#
#   LN1 (Tokenwise[seq, LayerNorm[D]])    : 2*D
#   Linear[D, 3D]   (QKV proj)            : 3*D² + 3*D
#   ScaledDotProductAttention              : 0
#   Linear[D, D]    (attn out, c_proj)    : D² + D     ← W scaled by 1/√(2L)
#   LN2                                    : 2*D
#   Linear[D, F]    (FFN first)           : D*F + F
#   GELU                                   : 0
#   Linear[F, D]    (FFN out, c_proj)     : F*D + D    ← W scaled by 1/√(2L)
#
# Linear layout inside the fused block is [W_flat | b], so each c_proj W is
# the first D² (attn-out) or F*D (FFN-out) entries of its layer.
#
# BLOCKS_BASE in the full param vector = PatchEmbed (Conv2D + Transpose2D)
# params + position BiasAdd params:
#   PatchEmbed.PARAM_SIZE = embed * (in_channels * patch² + 1)
#   BiasAdd.PARAM_SIZE    = n_patches * embed
# (Transpose2D contributes 0.)
# =============================================================================
comptime FFDIM = FF_MULT * EMBED
comptime LN_SIZE = 2 * EMBED
comptime QKV_SIZE = 3 * EMBED * EMBED + 3 * EMBED
comptime ATTN_OUT_SIZE = EMBED * EMBED + EMBED
comptime FFN1_SIZE = EMBED * FFDIM + FFDIM
comptime FFN2_SIZE = FFDIM * EMBED + EMBED
comptime BLOCK_SIZE = (
    LN_SIZE + QKV_SIZE + ATTN_OUT_SIZE + LN_SIZE + FFN1_SIZE + FFN2_SIZE
)

# Offsets within a block.
comptime OFF_LN1 = 0
comptime OFF_QKV = OFF_LN1 + LN_SIZE
comptime OFF_ATTN_OUT = OFF_QKV + QKV_SIZE
comptime OFF_LN2 = OFF_ATTN_OUT + ATTN_OUT_SIZE
comptime OFF_FFN1 = OFF_LN2 + LN_SIZE
comptime OFF_FFN2 = OFF_FFN1 + FFN1_SIZE

# First block starts after PatchEmbed (Conv2D weights + bias) + position BiasAdd.
comptime PATCH_EMBED_PARAM_SIZE = EMBED * (IN_CHANNELS * PATCH * PATCH + 1)
comptime POS_EMBED_SIZE = N_PATCHES * EMBED
comptime BLOCKS_BASE = PATCH_EMBED_PARAM_SIZE + POS_EMBED_SIZE


def _apply_c_proj_scaled_init(
    mut p: LayoutTensor[
        dtype, Layout.row_major(VIT_MODEL.PARAM_SIZE), MutAnyOrigin
    ],
) raises:
    """Scale attn-output-proj W and FFN-output-proj W per block by 1/√(2L).

    Matches nanoGPT's GPT-2-style scaled init for residual output projections.
    Run on CPU after `cpu.initialize[Normal[0, 0.02]]()` and before the
    `gpu.upload_from(cpu, ctx)` that uploads weights to the GPU.

    Takes the raw param `LayoutTensor` (not the wrapping NetworkState) to
    keep the function's mangled name short — the deep `ViT[…]` chain in
    the type would otherwise push the Apple `ld` symbol-name limit.
    """
    var scale = Scalar[dtype](1.0 / sqrt(Float64(2 * LAYERS)))

    for b in range(LAYERS):
        var block_off = BLOCKS_BASE + b * BLOCK_SIZE

        # Attention output proj W (Linear[D, D]) — first D² entries of its block.
        var attn_w_off = block_off + OFF_ATTN_OUT
        for i in range(EMBED * EMBED):
            p[attn_w_off + i] = (
                rebind[Scalar[dtype]](p[attn_w_off + i]) * scale
            )

        # FFN output proj W (Linear[F, D]) — first F*D entries of its block.
        var mlp_w_off = block_off + OFF_FFN2
        for i in range(FFDIM * EMBED):
            p[mlp_w_off + i] = (
                rebind[Scalar[dtype]](p[mlp_w_off + i]) * scale
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
    # nanoGPT-style transformer init: N(0, 0.02) on every Linear / Conv2D
    # weight, plus 1/√(2L) post-init scaling on every attention-output and
    # FFN-output projection (the GPT-2 "scaled init"). Mirrors the recipe in
    # examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo —
    # weight tying doesn't apply here (no embedding↔head pair to share).
    var cpu = NetworkState[VIT_MODEL, VIT_OPT]()
    cpu.initialize[Normal[0.0, 0.02]]()
    var cpu_params = cpu.params_view()
    _apply_c_proj_scaled_init(cpu_params)
    var state = GPUNetworkState[VIT_MODEL, VIT_OPT](ctx)
    state.upload_from(cpu, ctx)

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
