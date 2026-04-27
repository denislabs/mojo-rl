"""ViT CIFAR-10 training — GPU.

Validates the ViT composite (PatchEmbed + Transpose2D + non-causal
TransformerBlock × N + final LN + TokenMean + LM head) against CIFAR-10
classification. Doc target (Phase B): ≥ 70 % top-1 with the default config
(D=192, H=6, N=6, patch=4, n_patches=64).

Mirrors the structure of `tests/nn/test_resnet20_cifar10.mojo` so results are
directly comparable:
  - Same per-epoch augmentation kernel (random crop pad-4 + horizontal flip).
  - Same `Trainer.train_gpu_minibatch[BATCH, N_TRAIN]` driver (Fisher-Yates
    shuffle, CUDA-graph-captured).
  - One-pass training image upload, in-place re-augmentation each epoch.
  - One-hot target upload once, reused every epoch.
Differences vs resnet20 test:
  - AdamW (with weight decay) instead of plain Adam — standard ViT recipe.
  - Cosine LR schedule with warmup, applied per-epoch via `state.set_lr_scale`.
  - Per-epoch test top-1 evaluation so progress is visible before the run ends.

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
from std.math import cos, log, exp

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.composites import ViT
from mojo_rl.nn.training import GPUNetworkState, Trainer
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

# 50k train images / 128 batch = 391 iters/epoch. 50 epochs ≈ 19500 steps.
comptime EPOCHS = 50
comptime WARMUP_EPOCHS = 5         # ~5 epochs of linear warmup
comptime EVAL_BATCHES = 32          # ~32 * 128 = 4096 test samples per eval


# =============================================================================
# Aliases for readability.
# =============================================================================
comptime VIT_MODEL = ViT[
    IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
    N_CLASSES, FF_MULT,
]
comptime VIT_OPT = AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD]
comptime VIT_TRAINER = Trainer[VIT_MODEL, VIT_OPT, CrossEntropyLoss]


struct EvalResult(Movable):
    """Test-set evaluation result — average CE loss + top-1 accuracy."""
    var loss: Float64
    var top1: Float64

    def __init__(out self, loss: Float64, top1: Float64):
        self.loss = loss
        self.top1 = top1

    def __init__(out self, deinit existing: Self):
        self.loss = existing.loss
        self.top1 = existing.top1


# =============================================================================
# CIFAR augmentation kernel — random crop pad-4 + horizontal flip per sample.
# Identical recipe to tests/nn/test_resnet20_cifar10.mojo so the numbers can
# be compared apples-to-apples between ViT and ResNet-20.
# =============================================================================
def _cifar_augment_kernel[
    N: Int,
    dtype: DType = DType.float32,
](
    aug: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    raw: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    epoch_seed: Scalar[DType.uint64],
):
    """Random crop (pad 4) + random horizontal flip, per sample.

    Grid: (N,), Block: (TPB,). One block per sample; threads parallelize
    the 3072 output pixels. All threads in a block derive dx/dy/flip from
    PhiloxRandom(epoch_seed, b) identically — out-of-bounds pixels get 0.
    """
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
# LR schedule (per-epoch granularity, applied via state.set_lr_scale before
# train_gpu_minibatch).
# =============================================================================
def lr_scale(epoch: Int, warmup_epochs: Int, total_epochs: Int) -> Float64:
    if epoch < warmup_epochs:
        return Float64(epoch + 1) / Float64(warmup_epochs)
    var progress = Float64(epoch - warmup_epochs) / Float64(
        max(1, total_epochs - warmup_epochs)
    )
    if progress > 1.0:
        progress = 1.0
    var c = 0.5 * (1.0 + cos(progress * 3.141592653589793))
    return 0.1 + 0.9 * c


# =============================================================================
# Test-set evaluation: top-1 accuracy + average CE loss on `n_batches`
# contiguous batches starting at offset 0.
# =============================================================================
def _eval_topk(
    ctx: DeviceContext,
    state: GPUNetworkState[VIT_MODEL, VIT_OPT],
    test_img_buf: DeviceBuffer[dtype],
    test_labels: List[Int32],
    n_batches: Int,
    output_buf: DeviceBuffer[dtype],
    workspace_buf: DeviceBuffer[dtype],
    output_host: HostBuffer[dtype],
) raises -> EvalResult:
    """avg CE loss + top-1 over n_batches consecutive test batches."""
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin
    ](output_buf.unsafe_ptr())

    var n_correct = 0
    var n_total = 0
    var total_logloss: Float64 = 0
    for batch_idx in range(n_batches):
        var batch_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, VIT_MODEL.IN_DIM), MutAnyOrigin
        ](test_img_buf.unsafe_ptr() + batch_idx * BATCH * VIT_MODEL.IN_DIM)

        VIT_MODEL.forward_gpu_no_cache[BATCH](
            ctx, output_lt, batch_input, p_view, s_view, workspace_buf,
        )
        ctx.enqueue_copy(output_host, output_buf)
        ctx.synchronize()

        for b in range(BATCH):
            var best_idx = 0
            var best_val = Float64(output_host.unsafe_ptr()[b * N_CLASSES + 0])
            var max_val = best_val
            for c in range(1, N_CLASSES):
                var v = Float64(output_host.unsafe_ptr()[b * N_CLASSES + c])
                if v > best_val:
                    best_val = v
                    best_idx = c
                if v > max_val:
                    max_val = v
            # Negative log-likelihood at the true class (numerically stable).
            var true_label = Int(test_labels[batch_idx * BATCH + b])
            var sum_exp: Float64 = 0
            for c in range(N_CLASSES):
                var v = Float64(output_host.unsafe_ptr()[b * N_CLASSES + c])
                sum_exp += exp(v - max_val)
            var true_logit = Float64(
                output_host.unsafe_ptr()[b * N_CLASSES + true_label]
            )
            total_logloss += (max_val + log(sum_exp)) - true_logit
            if best_idx == true_label:
                n_correct += 1
            n_total += 1
    return EvalResult(
        total_logloss / Float64(n_total),
        Float64(n_correct) / Float64(n_total),
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

    var raw_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var aug_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(raw_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var raw_img_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE), MutAnyOrigin
    ](raw_img_buf)
    var aug_img_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE), MutAnyOrigin
    ](aug_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.NUM_CLASSES),
        MutAnyOrigin,
    ](train_tgt_buf)

    # ---------- Upload test set (no augmentation, no labels here) ----------
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_buf, test_img_host)

    # ---------- Eval scratch buffers (reused per epoch) ----------
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * VIT_MODEL.OUT_DIM)
    var ws_buf = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * VIT_MODEL.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var out_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * VIT_MODEL.OUT_DIM
    )

    # ---------- Initial test eval (random init baseline) ----------
    var ev0 = _eval_topk(
        ctx, state, test_img_buf, ds.test_labels, EVAL_BATCHES,
        out_buf, ws_buf, out_host,
    )
    print(
        "\n[epoch 0] initial test_loss="
        + String(ev0.loss)
        + " top1="
        + String(ev0.top1)
        + "  (random ≈ ln(C)="
        + String(log(Float64(N_CLASSES)))
        + " / 1/C="
        + String(1.0 / Float64(N_CLASSES))
        + ")"
    )

    # ---------- Training loop ----------
    print("\n── Training ──")
    var t_start = perf_counter_ns()
    comptime aug_k = _cifar_augment_kernel[CIFAR10.N_TRAIN, dtype]
    for epoch in range(EPOCHS):
        # Per-epoch LR scale (linear warmup → cosine decay to 10% of peak).
        var lr_s = lr_scale(epoch, WARMUP_EPOCHS, EPOCHS)
        state.set_lr_scale(lr_s)

        # Re-augment the training set (fresh crop+flip seeds per epoch).
        var aug_seed = Scalar[DType.uint64](UInt64(1000) + UInt64(epoch))
        ctx.enqueue_function[aug_k, aug_k](
            aug_img_lt, raw_img_lt, aug_seed,
            grid_dim=(CIFAR10.N_TRAIN,),
            block_dim=(TPB,),
        )

        # One full pass over augmented training set, shuffled.
        var result = VIT_TRAINER.train_gpu_minibatch[
            BATCH, CIFAR10.N_TRAIN, USE_CUDA_GRAPH=False
        ](
            state, ctx, aug_img_lt, train_tgt_lt,
            epochs=1, print_every_batches=0, shuffle=True,
            rng_seed=UInt64(42 + epoch),
        )
        ctx.synchronize()

        # Test eval every epoch.
        var ev = _eval_topk(
            ctx, state, test_img_buf, ds.test_labels, EVAL_BATCHES,
            out_buf, ws_buf, out_host,
        )
        print(
            "  epoch "
            + String(epoch + 1)
            + "/"
            + String(EPOCHS)
            + "  train_loss="
            + String(Float32(result.final_loss))
            + "  test_loss="
            + String(ev.loss)
            + "  top1="
            + String(ev.top1)
            + "  lr_scale="
            + String(lr_s)
        )
    var t_end = perf_counter_ns()
    print(
        "\n  training time: "
        + String(Float64(t_end - t_start) / 1e9)[byte=:6]
        + " s"
    )

    # ---------- Final eval on the entire test set ----------
    print("\n── Final evaluation (full test set) ──")
    comptime FULL_TEST_BATCHES = CIFAR10.N_TEST // BATCH
    var ev_final = _eval_topk(
        ctx, state, test_img_buf, ds.test_labels, FULL_TEST_BATCHES,
        out_buf, ws_buf, out_host,
    )
    print(
        "  test_loss="
        + String(ev_final.loss)
        + " top1="
        + String(ev_final.top1)
        + "  (start "
        + String(ev0.top1)
        + ")"
    )
    if ev_final.top1 >= 0.70:
        print("  PASS: top-1 accuracy ≥ 70 % — Phase B target hit")
    elif ev_final.top1 > 0.55:
        print(
            "  PARTIAL: top-1 > 55 % — clearly learning, may need more"
            + " epochs / stronger aug for 70 %"
        )
    else:
        print("  WARN: top-1 ≤ 55 % — increase EPOCHS or check")
    print("=" * 70)
