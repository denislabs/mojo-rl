"""ViT CIFAR-10 training — GPU.

Validates the ViT composite (PatchEmbed + Transpose2D + non-causal
TransformerBlock × N + final LN + TokenMean + LM head) against CIFAR-10
classification. Doc target (Phase B): ≥ 70 % top-1 with the default config
(D=192, H=6, N=6, patch=4, n_patches=64).

Default config is sized for NVIDIA GPUs. On the M1 Pro development hardware
the config OOMs — for local iteration shrink EMBED, LAYERS, and N_STEPS to
the values commented in the constants block.

All compute lives on device:
- Native GPU kernels for Conv2D (PatchEmbed), Transpose2DOp, BiasAdd
  (position embedding), MatMul/BiasAdd (AutoFused), LayerNorm, GELU,
  ScaledDotProductAttention (forward + 4-stage backward), TokenMean.
- Per-image cross-entropy via `CrossEntropyLoss.forward_gpu` /
  `.backward_gpu` over (BATCH, n_classes) logits + (BATCH, n_classes)
  one-hot targets.
- AdamW step + on-device step counter, gradient clipping.
- Eval = top-1 accuracy on the test set.

Run on NVIDIA (production target):
    pixi run -e nvidia mojo run -I . examples/vit/vit_cifar_training_gpu.mojo
Run on Apple Metal (dev iteration only — shrink config first):
    pixi run -e apple mojo run -I . examples/vit/vit_cifar_training_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed, random_si64, random_float64
from std.math import cos, log

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import ViT
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets import CIFAR10
from layout import Layout, LayoutTensor


struct EvalResult(Movable):
    var loss: Float64
    var top1: Float64

    def __init__(out self, loss: Float64, top1: Float64):
        self.loss = loss
        self.top1 = top1

    def __init__(out self, deinit existing: Self):
        self.loss = existing.loss
        self.top1 = existing.top1


struct SampledBatch(Movable):
    var inputs: List[Scalar[dtype]]
    var labels: List[Int]

    def __init__(out self, var inputs: List[Scalar[dtype]], var labels: List[Int]):
        self.inputs = inputs^
        self.labels = labels^

    def __init__(out self, deinit existing: Self):
        self.inputs = existing.inputs^
        self.labels = existing.labels^


# =============================================================================
# Hyperparameters — full ViT config targeting ≥70% top-1 on CIFAR-10.
# Matches docs/TRANSFORMER_VIT.md Phase B defaults. Designed for NVIDIA GPUs;
# on Apple M1 Pro shrink EMBED→64, LAYERS→4, BATCH→32, N_STEPS→2000.
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
comptime BETA2 = 0.999             # vision: 0.999 is canonical (vs 0.95 for LM)
comptime WD = 0.05
comptime GRAD_CLIP = 1.0

# 50k train images / 128 batch = 391 iters/epoch. 50 epochs ≈ 19500 steps.
comptime N_STEPS = 20000
comptime WARMUP_STEPS = 500
comptime PRINT_EVERY = 100
comptime EVAL_EVERY = 1000
comptime EVAL_BATCHES = 16


# =============================================================================
# LR schedule: linear warmup then cosine decay to 10 % of peak
# =============================================================================
def lr_scale(step: Int, warmup: Int, total: Int) -> Float64:
    if step < warmup:
        return Float64(step + 1) / Float64(warmup)
    var progress = Float64(step - warmup) / Float64(max(1, total - warmup))
    if progress > 1.0:
        progress = 1.0
    var c = 0.5 * (1.0 + cos(progress * 3.141592653589793))
    return 0.1 + 0.9 * c


# =============================================================================
# Sample a minibatch by random indices (with replacement). Returns flat input
# (BATCH * IMG_SIZE) and label list (BATCH).
# =============================================================================
def _sample_batch(
    images: List[Scalar[dtype]],
    labels: List[Int32],
    n_samples: Int,
    batch_size: Int,
    img_size: Int,
) raises -> SampledBatch:
    var inp = List[Scalar[dtype]](
        length=batch_size * img_size, fill=Scalar[dtype](0)
    )
    var lbl = List[Int](length=batch_size, fill=0)
    var max_idx = Int64(n_samples - 1)
    for b in range(batch_size):
        var idx = Int(random_si64(Int64(0), max_idx))
        for k in range(img_size):
            inp[b * img_size + k] = images[idx * img_size + k]
        lbl[b] = Int(labels[idx])
    return SampledBatch(inp^, lbl^)


def _labels_to_one_hot(
    labels: List[Int], batch_size: Int, n_classes: Int
) -> List[Scalar[dtype]]:
    var oh = List[Scalar[dtype]](
        length=batch_size * n_classes, fill=Scalar[dtype](0)
    )
    for b in range(batch_size):
        var c = labels[b]
        if 0 <= c and c < n_classes:
            oh[b * n_classes + c] = Scalar[dtype](1.0)
    return oh^


# =============================================================================
# Cross-entropy on (BATCH, n_classes). Standard form — averaged across batch.
# =============================================================================
def _ce_loss_and_grad_gpu(
    ctx: DeviceContext,
    output_dev: DeviceBuffer[dtype],
    target_dev: DeviceBuffer[dtype],
    grad_dev: DeviceBuffer[dtype],
    loss_dev: DeviceBuffer[dtype],
) raises:
    var output_v = LayoutTensor[
        dtype, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin
    ](output_dev.unsafe_ptr())
    var target_v = LayoutTensor[
        dtype, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin
    ](target_dev.unsafe_ptr())
    var grad_v = LayoutTensor[
        dtype, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin
    ](grad_dev.unsafe_ptr())
    var loss_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](loss_dev.unsafe_ptr())
    CrossEntropyLoss.forward_gpu[BATCH, N_CLASSES, dtype](
        ctx, loss_t, output_v, target_v
    )
    CrossEntropyLoss.backward_gpu[BATCH, N_CLASSES, dtype](
        ctx, grad_v, output_v, target_v
    )


# =============================================================================
# Training step: forward → CE → backward → grad clip → AdamW step
# =============================================================================
def _train_step_gpu(
    ctx: DeviceContext,
    mut state: GPUNetworkState[
        ViT[
            IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
            N_CLASSES, FF_MULT,
        ],
        AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD],
    ],
    inp_dev: DeviceBuffer[dtype],
    tgt_dev: DeviceBuffer[dtype],
    out_dev: DeviceBuffer[dtype],
    cache_dev: DeviceBuffer[dtype],
    gin_dev: DeviceBuffer[dtype],
    gout_dev: DeviceBuffer[dtype],
    ws_dev: DeviceBuffer[dtype],
    loss_dev: DeviceBuffer[dtype],
    loss_host: HostBuffer[dtype],
) raises -> Float64:
    comptime Model = ViT[
        IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
        N_CLASSES, FF_MULT,
    ]
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_dev.unsafe_ptr())
    var gin_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](gin_dev.unsafe_ptr())
    var gout_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](gout_dev.unsafe_ptr())

    Model.forward_gpu[BATCH, dtype](
        ctx, out_t, inp_t, p_view, s_view, cache_t, ws_dev
    )
    _ce_loss_and_grad_gpu(ctx, out_dev, tgt_dev, gout_dev, loss_dev)

    state.zero_grads(ctx)
    var grads_view = state.grads_view()
    Model.backward_gpu[BATCH, dtype](
        ctx, gin_t, gout_t, p_view, s_view, cache_t, grads_view, ws_dev
    )
    state.clip_grads(ctx, Scalar[dtype](GRAD_CLIP))
    state.optimizer_step(ctx)

    ctx.enqueue_copy(loss_host, loss_dev)
    ctx.synchronize()
    return Float64(loss_host[0])


# =============================================================================
# Evaluation: top-1 accuracy + average CE on a sampled subset of test data.
# =============================================================================
def _eval_gpu(
    ctx: DeviceContext,
    state: GPUNetworkState[
        ViT[
            IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
            N_CLASSES, FF_MULT,
        ],
        AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD],
    ],
    test_images: List[Scalar[dtype]],
    test_labels: List[Int32],
    n_test: Int,
    n_batches: Int,
    inp_dev: DeviceBuffer[dtype],
    tgt_dev: DeviceBuffer[dtype],
    out_dev: DeviceBuffer[dtype],
    cache_dev: DeviceBuffer[dtype],
    ws_dev: DeviceBuffer[dtype],
    loss_dev: DeviceBuffer[dtype],
    inp_host: HostBuffer[dtype],
    tgt_host: HostBuffer[dtype],
    loss_host: HostBuffer[dtype],
    out_host: HostBuffer[dtype],
) raises -> EvalResult:
    """Returns avg_loss + top-1 accuracy over n_batches sampled test batches."""
    comptime Model = ViT[
        IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
        N_CLASSES, FF_MULT,
    ]
    var p_view = state.params_view()
    var s_view = state.model_state_view()
    var img_size = IN_CHANNELS * IMG_H * IMG_W

    var total_loss: Float64 = 0
    var n_correct: Int = 0
    var n_total: Int = 0
    for _ in range(n_batches):
        var batch = _sample_batch(test_images, test_labels, n_test, BATCH, img_size)
        var tgt_data = _labels_to_one_hot(batch.labels, BATCH, N_CLASSES)
        for i in range(BATCH * img_size):
            inp_host[i] = batch.inputs[i]
        for i in range(BATCH * N_CLASSES):
            tgt_host[i] = tgt_data[i]
        ctx.enqueue_copy(inp_dev, inp_host)
        ctx.enqueue_copy(tgt_dev, tgt_host)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
        ](inp_dev.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](out_dev.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](cache_dev.unsafe_ptr())

        Model.forward_gpu[BATCH, dtype](
            ctx, out_t, inp_t, p_view, s_view, cache_t, ws_dev
        )
        var out_v = LayoutTensor[
            dtype, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin
        ](out_dev.unsafe_ptr())
        var tgt_v = LayoutTensor[
            dtype, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin
        ](tgt_dev.unsafe_ptr())
        var loss_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](loss_dev.unsafe_ptr())
        CrossEntropyLoss.forward_gpu[BATCH, N_CLASSES, dtype](
            ctx, loss_t, out_v, tgt_v
        )
        ctx.enqueue_copy(loss_host, loss_dev)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        total_loss += Float64(loss_host[0])
        for b in range(BATCH):
            var best_v = Float64(out_host[b * N_CLASSES + 0])
            var best_c = 0
            for c in range(1, N_CLASSES):
                var v = Float64(out_host[b * N_CLASSES + c])
                if v > best_v:
                    best_v = v
                    best_c = c
            if best_c == batch.labels[b]:
                n_correct += 1
            n_total += 1
    return EvalResult(total_loss / Float64(n_batches), Float64(n_correct) / Float64(n_total))


# =============================================================================
# Driver
# =============================================================================
def main() raises:
    seed(42)
    comptime Model = ViT[
        IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES,
        N_CLASSES, FF_MULT,
    ]
    comptime Opt = AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD]

    print("=" * 70)
    print("ViT CIFAR-10 training (GPU)")
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
        + " grad_clip="
        + String(GRAD_CLIP)
    )
    print(
        "  steps="
        + String(N_STEPS)
        + " warmup="
        + String(WARMUP_STEPS)
    )
    print(
        "  PARAM_SIZE="
        + String(Model.PARAM_SIZE)
        + " CACHE/sample="
        + String(Model.CACHE_SIZE)
        + " WS/sample="
        + String(Model.WORKSPACE_SIZE_PER_SAMPLE)
    )

    # ---------- Data ----------
    print("\n[data] loading CIFAR-10...")
    var ds = CIFAR10()
    print(
        "  train="
        + String(ds.num_train)
        + " test="
        + String(ds.num_test)
    )

    # ---------- Device + state ----------
    var ctx = DeviceContext()
    var state = GPUNetworkState[Model, Opt](ctx)

    var cpu = NetworkState[Model, Opt]()
    cpu.initialize[Xavier[]]()
    state.upload_from(cpu, ctx)

    # ---------- Pre-allocated buffers ----------
    var inp_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var tgt_dev = ctx.enqueue_create_buffer[dtype](BATCH * N_CLASSES)
    var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.CACHE_SIZE)
    var gin_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var gout_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var ws_dev = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * Model.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var loss_dev = ctx.enqueue_create_buffer[dtype](1)
    var inp_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.IN_DIM)
    var tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * N_CLASSES)
    var loss_host = ctx.enqueue_create_host_buffer[dtype](1)
    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.OUT_DIM)

    var img_size = Model.IN_DIM

    # ---------- Initial eval (random init baseline) ----------
    var eval0 = _eval_gpu(
        ctx, state, ds.test_images, ds.test_labels, ds.num_test, EVAL_BATCHES,
        inp_dev, tgt_dev, out_dev, cache_dev, ws_dev, loss_dev,
        inp_host, tgt_host, loss_host, out_host,
    )
    print(
        "\n[step 0] initial test_loss="
        + String(eval0.loss)
        + " top1="
        + String(eval0.top1)
        + "  (random ≈ ln(C)="
        + String(log(Float64(N_CLASSES)))
        + " / 1/C="
        + String(1.0 / Float64(N_CLASSES))
        + ")"
    )

    # ---------- Training loop ----------
    var loss_running: Float64 = 0
    var loss_count: Int = 0
    for step in range(N_STEPS):
        var s = lr_scale(step, WARMUP_STEPS, N_STEPS)
        state.set_lr_scale(s)

        var batch = _sample_batch(ds.train_images, ds.train_labels, ds.num_train, BATCH, img_size)
        var tgt_data = _labels_to_one_hot(batch.labels, BATCH, N_CLASSES)
        for i in range(BATCH * img_size):
            inp_host[i] = batch.inputs[i]
        for i in range(BATCH * N_CLASSES):
            tgt_host[i] = tgt_data[i]
        ctx.enqueue_copy(inp_dev, inp_host)
        ctx.enqueue_copy(tgt_dev, tgt_host)

        var loss = _train_step_gpu(
            ctx, state, inp_dev, tgt_dev, out_dev, cache_dev,
            gin_dev, gout_dev, ws_dev, loss_dev, loss_host,
        )
        loss_running += loss
        loss_count += 1

        if (step + 1) % PRINT_EVERY == 0:
            var avg = loss_running / Float64(loss_count)
            print(
                "[step "
                + String(step + 1)
                + "] train_loss="
                + String(avg)
                + " lr_scale="
                + String(s)
            )
            loss_running = 0
            loss_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            var ev = _eval_gpu(
                ctx, state, ds.test_images, ds.test_labels, ds.num_test, EVAL_BATCHES,
                inp_dev, tgt_dev, out_dev, cache_dev, ws_dev, loss_dev,
                inp_host, tgt_host, loss_host, out_host,
            )
            print("           test_loss=" + String(ev.loss) + " top1=" + String(ev.top1))

    # ---------- Final eval on a larger subset ----------
    var ev_final = _eval_gpu(
        ctx, state, ds.test_images, ds.test_labels, ds.num_test, EVAL_BATCHES * 4,
        inp_dev, tgt_dev, out_dev, cache_dev, ws_dev, loss_dev,
        inp_host, tgt_host, loss_host, out_host,
    )
    print(
        "\n[final] test_loss="
        + String(ev_final.loss)
        + " top1="
        + String(ev_final.top1)
        + "  (start "
        + String(eval0.top1)
        + ")"
    )
    if ev_final.top1 >= 0.70:
        print("  PASS: top-1 accuracy ≥ 70 % — Phase B target hit")
    elif ev_final.top1 > 0.5:
        print("  PARTIAL: top-1 > 50 % — clearly learning, may need more steps or aug for 70 %")
    else:
        print("  WARN: top-1 ≤ 50 % — increase N_STEPS or check")
    print("=" * 70)
