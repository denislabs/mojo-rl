"""TinyShakespeare char-GPT training — GPU.

GPU training script targeting the docs/TRANSFORMER_VIT.md Phase A goal:
val loss ≤ 1.5 nats/char on the default nanoGPT-class config.

Same high-level structure as `examples/nn/mlp/mlp_mnist_training_gpu.mojo`
and `examples/nn/vit/vit_cifar_training_gpu.mojo`:
  - Pre-sample the train/val datasets once and upload to device.
  - Per-epoch: apply a `CosineWarmupSchedule` LR scale, train one pass,
    run validation, print one line.
  - Final sampling at the end.

Two intentional differences from MLP/ViT:
  1. The training loop is written inline (manual forward → CE → backward →
     clip → optimizer step) instead of going through
     `Trainer.train_gpu_minibatch` / `train_gpu_minibatch_full`. The deeply
     nested GPT generic chain (three `Tokenwise[256, …]` wrappers around
     Embedding, LayerNorm, and the LM head) overflows Apple `ld`'s symbol-
     name length limit when wrapped in the Trainer's per-epoch closure.
     Calling `GPT_MODEL.forward_gpu` / `backward_gpu` directly keeps every
     mangled name short enough to link on macOS.
  2. Validation is sequence-CE (mean per-token nats, computed via
     `CrossEntropyLoss.forward_gpu` on a `(BATCH*SEQ, VOCAB)` reinterpret),
     not the framework's classification eval — the latter assumes one
     int32 label per sample, which doesn't fit `(BATCH, SEQ*VOCAB)` outputs.

Default config is sized for NVIDIA GPUs (~RTX 4090 / 5090). On the M1 Pro
development hardware shrink SEQ→64, EMBED→64, LAYERS→4, EPOCHS→4,
N_TRAIN_WINDOWS→1024 to fit memory and iterate quickly.

Run on NVIDIA (production target):
    pixi run -e nvidia mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo
Run on Apple Metal (dev iteration only — shrink config first):
    pixi run -e apple mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed, random_float64
from std.math import log, exp
from std.time import perf_counter_ns

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import GPTDrop
from mojo_rl.nn.training import (
    NetworkState,
    GPUNetworkState,
    CosineWarmupSchedule,
)
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets import (
    CharTokenizer,
    load_text,
    train_val_split,
    make_batch,
    to_one_hot,
)
from layout import Layout, LayoutTensor


# =============================================================================
# Hyperparameters — full nanoGPT-class config targeting val loss ≤ 1.5 nats.
# =============================================================================
comptime VOCAB = 65          # TinyShakespeare unique chars
comptime SEQ = 256           # context length
comptime EMBED = 192         # transformer width
comptime HEADS = 6           # head_dim = 32
comptime LAYERS = 6          # transformer blocks
comptime FF_MULT = 4         # FFN inner dim = 4 * EMBED = 768

comptime BATCH = 16          # ~150 MB activations at this config; lower if OOM

comptime BASE_LR = 3e-4
comptime BETA1 = 0.9
comptime BETA2 = 0.95        # LM canonical (vs 0.999 vision)
comptime WD = 0.1
comptime GRAD_CLIP = 1.0     # max-abs clip on params grads each step

# Pre-sampled dataset:
#   N_TRAIN_WINDOWS = distinct training windows sampled once and iterated
#     each epoch (no per-epoch reshuffle to avoid the Trainer dependency
#     that overflows the Apple linker; pre-sampling already gives random
#     starting positions, so within-epoch order doesn't add much).
#   N_VAL_WINDOWS = held-out validation windows.
#   Total steps ≈ EPOCHS × (N_TRAIN_WINDOWS / BATCH).
#
# Sized to ~20 k steps (10 × 2048 = 20 480) but with 4× more unique windows
# than the original recipe — each window now appears 10× during training
# instead of 40×, reducing the overfit-to-fixed-corpus failure mode that
# made the previous run report low val loss + degenerate samples.
# Memory: 32768 × 256 × 65 × 4 B ≈ 2.2 GB train_inp + 2.2 GB train_tgt
# on device; same again on host during upload. Fits comfortably on a 24 GB
# 4090. If host RAM is tight, dial both down by 2× (16384 × 20).
comptime N_TRAIN_WINDOWS = 32768
comptime N_VAL_WINDOWS = 256          # 16 batches × BATCH=16
# Early-stop test: previous 10-epoch run hit val ≈ 1.65 at epoch 2 then
# fell off a cliff to val ≈ 0.86 by epoch 10 with degenerate samples.
# Stopping at 4 keeps us in nanoGPT's reported regime (val ≈ 1.47).
comptime EPOCHS = 4
comptime WARMUP_EPOCHS = 1            # 1 epoch of linear warmup, then cosine

comptime N_TRAIN_BATCHES = N_TRAIN_WINDOWS // BATCH
comptime N_VAL_BATCHES = N_VAL_WINDOWS // BATCH

# Dropout — fixed at 0.2 to match nanoGPT's char-Shakespeare config. The
# Model-level `Dropout[…, training=True]` carries an on-device PRNG counter
# that gets bumped per forward, so the mask is fresh every step.
# Inference paths (`forward_gpu_no_cache`) bypass to identity regardless of
# the `training` flag, so we don't need a separate eval-mode model.
comptime DROPOUT_P = 0.2
comptime DROP_SEED_BASE = UInt64(0xC0FFEE)


# =============================================================================
# Model alias — full GPT chain lives in `mojo_rl.nn.composites.GPTDrop`.
# Defining it inside the package (rather than inline here) keeps the deeply
# nested generic specialization in a unit the compiler processes once,
# instead of redoing it every time this script changes.
# =============================================================================
comptime GPT_MODEL = GPTDrop[
    VOCAB,
    SEQ,
    EMBED,
    HEADS,
    LAYERS,
    FF_MULT,
    True,  # causal
    DROPOUT_P,
    DROP_SEED_BASE,
]
comptime GPT_OPT = AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD]
comptime GPT_SCHEDULER = CosineWarmupSchedule[WARMUP_EPOCHS, 0.1]


# =============================================================================
# Sequence-level validation: mean per-token CE in nats over pre-uploaded
# (N_VAL_WINDOWS, SEQ*VOCAB) one-hot windows. Uses the same forward + CE
# kernels as training; the (BATCH, SEQ*VOCAB) output is reinterpreted as
# (BATCH*SEQ, VOCAB) for the per-token CE.
# =============================================================================
def _eval_loss_seq_gpu(
    ctx: DeviceContext,
    state: GPUNetworkState[GPT_MODEL, GPT_OPT],
    val_input: LayoutTensor[
        dtype, Layout.row_major(N_VAL_WINDOWS, GPT_MODEL.IN_DIM), MutAnyOrigin
    ],
    val_target: LayoutTensor[
        dtype, Layout.row_major(N_VAL_WINDOWS, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ],
    output_buf: DeviceBuffer[dtype],
    cache_buf: DeviceBuffer[dtype],
    ws_buf: DeviceBuffer[dtype],
    loss_buf: DeviceBuffer[dtype],
    loss_host: HostBuffer[dtype],
) raises -> Float64:
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var output_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var loss_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](loss_buf.unsafe_ptr())

    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var total: Float64 = 0.0
    for batch_idx in range(N_VAL_BATCHES):
        var batch_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, GPT_MODEL.IN_DIM), MutAnyOrigin
        ](val_input.ptr + batch_idx * BATCH * GPT_MODEL.IN_DIM)
        var target_v = LayoutTensor[
            dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
        ](val_target.ptr + batch_idx * BATCH * GPT_MODEL.OUT_DIM)

        # forward_gpu_no_cache → Dropout falls back to identity (eval mode).
        GPT_MODEL.forward_gpu_no_cache[BATCH, dtype](
            ctx, output_t, batch_input, p_view, s_view, ws_buf
        )
        CrossEntropyLoss.forward_gpu[BATCH * SEQ, VOCAB, dtype](
            ctx, loss_t, output_v, target_v
        )
        ctx.enqueue_copy(loss_host, loss_buf)
        ctx.synchronize()
        total += Float64(loss_host[0])

    return total / Float64(N_VAL_BATCHES)


# =============================================================================
# Diagnostic: per-token top-1 argmax accuracy on the val windows.
#
# Same forward pass as `_eval_loss_seq_gpu`, but instead of CE we argmax
# the logits at every (sample, position) pair and compare to the integer
# target id. Tells us whether the training loss is *consistent* with
# good next-token prediction:
#
#   val_loss = 0.55 nats  ⇒  e^-0.55 ≈ 58 % avg true-class probability,
#   so a non-pathological model should land near 55–65 % top-1. If we
#   instead see ~5–15 % we know the loss is artifactually low (forward-
#   pass leak somewhere) rather than the model genuinely predicting well
#   in-context.
# =============================================================================
def _eval_topk_accuracy_gpu(
    ctx: DeviceContext,
    state: GPUNetworkState[GPT_MODEL, GPT_OPT],
    val_input: LayoutTensor[
        dtype, Layout.row_major(N_VAL_WINDOWS, GPT_MODEL.IN_DIM), MutAnyOrigin
    ],
    val_target_ids: List[Int],
    output_buf: DeviceBuffer[dtype],
    cache_buf: DeviceBuffer[dtype],
    ws_buf: DeviceBuffer[dtype],
) raises -> Float64:
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var output_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * GPT_MODEL.OUT_DIM
    )

    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var total_correct: Int = 0
    var total_count: Int = 0
    for batch_idx in range(N_VAL_BATCHES):
        var batch_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, GPT_MODEL.IN_DIM), MutAnyOrigin
        ](val_input.ptr + batch_idx * BATCH * GPT_MODEL.IN_DIM)

        # forward_gpu_no_cache → Dropout falls back to identity (eval mode).
        GPT_MODEL.forward_gpu_no_cache[BATCH, dtype](
            ctx, output_t, batch_input, p_view, s_view, ws_buf
        )
        ctx.enqueue_copy(output_host, output_buf)
        ctx.synchronize()

        # Per-(sample, position) argmax over VOCAB logits, vs integer target.
        for b in range(BATCH):
            var b_off = b * SEQ * VOCAB
            for t in range(SEQ):
                var row_off = b_off + t * VOCAB
                var best_v = Float64(output_host[row_off])
                var best_idx = 0
                for v in range(1, VOCAB):
                    var x = Float64(output_host[row_off + v])
                    if x > best_v:
                        best_v = x
                        best_idx = v
                var tgt = val_target_ids[
                    batch_idx * BATCH * SEQ + b * SEQ + t
                ]
                if best_idx == tgt:
                    total_correct += 1
                total_count += 1

    return Float64(total_correct) / Float64(total_count)


# =============================================================================
# Sampling on device — nanoGPT-style.
#
# Slow path: one BATCH=1 forward per generated token (no KV cache). The
# context window is FRONT-anchored: the running sequence sits at positions
# 0..n_eff-1 and any unused tail (positions n_eff..SEQ-1) is filled with
# pad_id but never reaches the read position, because causal attention
# means output at position p only depends on inputs 0..p. Logits are
# pulled at position read_pos = n_eff - 1 (the last real token), so the
# model is always asked to predict the next token from an in-distribution
# Shakespeare prefix — exactly the regime it was trained on. Once the
# sequence overflows SEQ tokens we slide the window to keep the last SEQ
# tokens and read at SEQ-1, matching nanoGPT's `idx[:, -block_size:]`
# pattern in `sample.py`.
# =============================================================================
def _sample_token(
    logits_row: List[Scalar[dtype]],
    vocab: Int,
    temperature: Float64,
    top_k: Int,
) -> Int:
    """Greedy if `temperature <= 0`, else sample from softmax(logits/T)
    optionally restricted to the top_k logits (top_k <= 0 disables the
    filter)."""
    if temperature <= 0.0:
        var best_v = Float64(logits_row[0])
        var best_idx = 0
        for v in range(1, vocab):
            var x = Float64(logits_row[v])
            if x > best_v:
                best_v = x
                best_idx = v
        return best_idx

    var inv_t = 1.0 / temperature
    var scaled = List[Float64](capacity=vocab)
    for v in range(vocab):
        scaled.append(Float64(logits_row[v]) * inv_t)

    # Top-k filter via repeated argmax (vocab is small, ~65, so O(k*vocab)
    # is cheap). Marks the top_k indices as keep=True; rest stay False.
    var keep = List[Bool](capacity=vocab)
    if top_k > 0 and top_k < vocab:
        for _ in range(vocab):
            keep.append(False)
        var work = List[Float64](capacity=vocab)
        for v in range(vocab):
            work.append(scaled[v])
        for _ in range(top_k):
            var bv: Float64 = -1e30
            var bi = 0
            for v in range(vocab):
                if work[v] > bv:
                    bv = work[v]
                    bi = v
            keep[bi] = True
            work[bi] = -1e30
    else:
        for _ in range(vocab):
            keep.append(True)

    # Numerically-stable softmax over kept entries, then categorical sample.
    var max_l: Float64 = -1e30
    for v in range(vocab):
        if keep[v] and scaled[v] > max_l:
            max_l = scaled[v]
    var sum_exp: Float64 = 0.0
    var exps = List[Float64](capacity=vocab)
    for v in range(vocab):
        if keep[v]:
            var e = exp(scaled[v] - max_l)
            exps.append(e)
            sum_exp += e
        else:
            exps.append(0.0)
    var u = random_float64(0.0, 1.0) * sum_exp
    var acc = 0.0
    for v in range(vocab):
        acc += exps[v]
        if u < acc:
            return v
    return vocab - 1


def _generate_text_gpu(
    ctx: DeviceContext,
    state: GPUNetworkState[GPT_MODEL, GPT_OPT],
    tok: CharTokenizer,
    prompt: String,
    n_tokens: Int,
    temperature: Float64 = 0.8,
    top_k: Int = 10,
    pad_id: Int = 0,
) raises -> String:
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var all_ids = tok.encode(prompt)
    var prompt_len = len(all_ids)
    if prompt_len == 0:
        raise Error("generate_text: prompt is empty after tokenization")

    var inp_host = ctx.enqueue_create_host_buffer[dtype](GPT_MODEL.IN_DIM)
    var inp_dev = ctx.enqueue_create_buffer[dtype](GPT_MODEL.IN_DIM)
    var out_dev = ctx.enqueue_create_buffer[dtype](GPT_MODEL.OUT_DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](GPT_MODEL.CACHE_SIZE)
    var ws_dev = ctx.enqueue_create_buffer[dtype](
        max(1, GPT_MODEL.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var out_host = ctx.enqueue_create_host_buffer[dtype](GPT_MODEL.OUT_DIM)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(1, GPT_MODEL.IN_DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(1, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(1, GPT_MODEL.CACHE_SIZE), MutAnyOrigin
    ](cache_dev.unsafe_ptr())

    for _ in range(n_tokens):
        # Front-anchored window:
        #   positions 0..n_eff-1 = real ids (last SEQ when sequence overflows)
        #   positions n_eff..SEQ-1 = pad_id (does not affect read position
        #     thanks to causal attention)
        # Read logits at position read_pos = n_eff - 1.
        for i in range(GPT_MODEL.IN_DIM):
            inp_host[i] = 0
        var n_have = len(all_ids)
        var n_eff = n_have if n_have <= SEQ else SEQ
        var first_real = 0 if n_have <= SEQ else n_have - SEQ
        for t in range(SEQ):
            var tid: Int
            if t < n_eff:
                tid = all_ids[first_real + t]
            else:
                tid = pad_id
            if tid < 0 or tid >= VOCAB:
                continue
            inp_host[t * VOCAB + tid] = Scalar[dtype](1.0)
        ctx.enqueue_copy(inp_dev, inp_host)

        # forward_gpu_no_cache → Dropout falls back to identity (eval mode).
        GPT_MODEL.forward_gpu_no_cache[1, dtype](
            ctx, out_t, inp_t, p_view, s_view, ws_dev
        )
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        var read_pos = n_eff - 1
        var last_row = List[Scalar[dtype]](capacity=VOCAB)
        for v in range(VOCAB):
            last_row.append(out_host[read_pos * VOCAB + v])
        var next_id = _sample_token(last_row, VOCAB, temperature, top_k)
        all_ids.append(next_id)

    var gen_only = List[Int](capacity=n_tokens)
    for i in range(prompt_len, len(all_ids)):
        gen_only.append(all_ids[i])
    return tok.decode(gen_only)


# =============================================================================
# Driver
# =============================================================================
def main() raises:
    seed(42)

    print("=" * 70)
    print("TinyShakespeare GPT training (GPU)")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB)
        + " seq=" + String(SEQ)
        + " embed=" + String(EMBED)
        + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS)
    )
    print(
        "  batch=" + String(BATCH)
        + " base_lr=" + String(BASE_LR)
        + " wd=" + String(WD)
        + " grad_clip=" + String(GRAD_CLIP)
    )
    print(
        "  n_train_windows=" + String(N_TRAIN_WINDOWS)
        + " n_val_windows=" + String(N_VAL_WINDOWS)
        + " epochs=" + String(EPOCHS)
        + " warmup_ep=" + String(WARMUP_EPOCHS)
    )
    print(
        "  PARAM_SIZE=" + String(GPT_MODEL.PARAM_SIZE)
        + " CACHE/sample=" + String(GPT_MODEL.CACHE_SIZE)
        + " WS/sample=" + String(GPT_MODEL.WORKSPACE_SIZE_PER_SAMPLE)
    )

    # ---------- Data ----------
    print("\n[data] loading TinyShakespeare...")
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error(
            "vocab mismatch: tokenizer found "
            + String(tok.vocab_size)
            + " unique chars, expected VOCAB="
            + String(VOCAB)
        )
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  total tokens=" + String(len(ids))
        + " train=" + String(len(split.train))
        + " val=" + String(len(split.val))
    )

    # ---------- Pre-sample windows once on host ----------
    print(
        "\n[data] pre-sampling " + String(N_TRAIN_WINDOWS)
        + " train windows + " + String(N_VAL_WINDOWS) + " val windows..."
    )
    var train_batch = make_batch(split.train, N_TRAIN_WINDOWS, SEQ)
    var train_inp_data = to_one_hot(
        train_batch.inputs, VOCAB, N_TRAIN_WINDOWS, SEQ
    )
    var train_tgt_data = to_one_hot(
        train_batch.targets, VOCAB, N_TRAIN_WINDOWS, SEQ
    )
    var val_batch = make_batch(split.val, N_VAL_WINDOWS, SEQ)
    var val_inp_data = to_one_hot(
        val_batch.inputs, VOCAB, N_VAL_WINDOWS, SEQ
    )
    var val_tgt_data = to_one_hot(
        val_batch.targets, VOCAB, N_VAL_WINDOWS, SEQ
    )

    # ---------- Device + state ----------
    var ctx = DeviceContext()
    var state = GPUNetworkState[GPT_MODEL, GPT_OPT](ctx)
    var cpu = NetworkState[GPT_MODEL, GPT_OPT]()
    cpu.initialize[Xavier[]]()
    state.upload_from(cpu, ctx)

    # ---------- Upload train + val datasets once ----------
    print("[data] uploading datasets to device...")
    var train_inp_host = ctx.enqueue_create_host_buffer[dtype](
        N_TRAIN_WINDOWS * GPT_MODEL.IN_DIM
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        N_TRAIN_WINDOWS * GPT_MODEL.OUT_DIM
    )
    for i in range(N_TRAIN_WINDOWS * GPT_MODEL.IN_DIM):
        train_inp_host.unsafe_ptr()[i] = train_inp_data[i]
        train_tgt_host.unsafe_ptr()[i] = train_tgt_data[i]
    var train_inp_buf = ctx.enqueue_create_buffer[dtype](
        N_TRAIN_WINDOWS * GPT_MODEL.IN_DIM
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        N_TRAIN_WINDOWS * GPT_MODEL.OUT_DIM
    )
    ctx.enqueue_copy(train_inp_buf, train_inp_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)
    var train_inp_lt = LayoutTensor[
        dtype, Layout.row_major(N_TRAIN_WINDOWS, GPT_MODEL.IN_DIM), MutAnyOrigin
    ](train_inp_buf.unsafe_ptr())
    var train_tgt_lt = LayoutTensor[
        dtype, Layout.row_major(N_TRAIN_WINDOWS, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](train_tgt_buf.unsafe_ptr())

    var val_inp_host = ctx.enqueue_create_host_buffer[dtype](
        N_VAL_WINDOWS * GPT_MODEL.IN_DIM
    )
    var val_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        N_VAL_WINDOWS * GPT_MODEL.OUT_DIM
    )
    for i in range(N_VAL_WINDOWS * GPT_MODEL.IN_DIM):
        val_inp_host.unsafe_ptr()[i] = val_inp_data[i]
        val_tgt_host.unsafe_ptr()[i] = val_tgt_data[i]
    var val_inp_buf = ctx.enqueue_create_buffer[dtype](
        N_VAL_WINDOWS * GPT_MODEL.IN_DIM
    )
    var val_tgt_buf = ctx.enqueue_create_buffer[dtype](
        N_VAL_WINDOWS * GPT_MODEL.OUT_DIM
    )
    ctx.enqueue_copy(val_inp_buf, val_inp_host)
    ctx.enqueue_copy(val_tgt_buf, val_tgt_host)
    var val_inp_lt = LayoutTensor[
        dtype, Layout.row_major(N_VAL_WINDOWS, GPT_MODEL.IN_DIM), MutAnyOrigin
    ](val_inp_buf.unsafe_ptr())
    var val_tgt_lt = LayoutTensor[
        dtype, Layout.row_major(N_VAL_WINDOWS, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](val_tgt_buf.unsafe_ptr())

    # ---------- Per-batch training scratch (allocated once, reused) ----------
    var output_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * GPT_MODEL.OUT_DIM
    )
    var cache_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * GPT_MODEL.CACHE_SIZE
    )
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * GPT_MODEL.IN_DIM
    )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * GPT_MODEL.OUT_DIM
    )
    var ws_buf = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * GPT_MODEL.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var loss_buf = ctx.enqueue_create_buffer[dtype](1)
    var loss_host = ctx.enqueue_create_host_buffer[dtype](1)

    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.CACHE_SIZE), MutAnyOrigin
    ](cache_buf.unsafe_ptr())
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.IN_DIM), MutAnyOrigin
    ](grad_in_buf.unsafe_ptr())
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.OUT_DIM), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())
    var output_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var grad_out_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())
    var loss_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](loss_buf.unsafe_ptr())

    # ---------- Initial val loss ----------
    var val_init = _eval_loss_seq_gpu(
        ctx, state, val_inp_lt, val_tgt_lt,
        output_buf, cache_buf, ws_buf, loss_buf, loss_host,
    )
    print(
        "\n[epoch 0] initial val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    # ---------- Per-epoch loop: scheduler → batches → eval ----------
    print("\n── Training ──")
    var t_start = perf_counter_ns()
    var final_loss: Float64 = 0.0
    for epoch in range(EPOCHS):
        var lr_s = GPT_SCHEDULER.lr_scale_at(epoch, EPOCHS)
        state.set_lr_scale(lr_s, ctx)

        var p_view = state.params_view()
        var s_view = state.model_state_view()

        for batch_idx in range(N_TRAIN_BATCHES):
            var batch_input = LayoutTensor[
                dtype, Layout.row_major(BATCH, GPT_MODEL.IN_DIM), MutAnyOrigin
            ](train_inp_lt.ptr + batch_idx * BATCH * GPT_MODEL.IN_DIM)
            var target_v = LayoutTensor[
                dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
            ](train_tgt_lt.ptr + batch_idx * BATCH * GPT_MODEL.OUT_DIM)

            GPT_MODEL.forward_gpu[BATCH, dtype](
                ctx, output_t, batch_input, p_view, s_view, cache_t, ws_buf
            )
            CrossEntropyLoss.forward_gpu[BATCH * SEQ, VOCAB, dtype](
                ctx, loss_t, output_v, target_v
            )
            CrossEntropyLoss.backward_gpu[BATCH * SEQ, VOCAB, dtype](
                ctx, grad_out_v, output_v, target_v
            )

            state.zero_grads(ctx)
            var grads_view = state.grads_view()
            GPT_MODEL.backward_gpu[BATCH, dtype](
                ctx, grad_in_t, grad_out_t, p_view, s_view, cache_t,
                grads_view, ws_buf,
            )
            # Per-element max-abs grad clip — cheap insurance against the
            # rare-spike instability common at this depth, standard in nanoGPT.
            state.clip_grads(ctx, Scalar[dtype](GRAD_CLIP))
            state.optimizer_step(ctx)

        # Read the last batch's loss as a cheap epoch summary.
        ctx.enqueue_copy(loss_host, loss_buf)
        ctx.synchronize()
        final_loss = Float64(loss_host[0])

        var v = _eval_loss_seq_gpu(
            ctx, state, val_inp_lt, val_tgt_lt,
            output_buf, cache_buf, ws_buf, loss_buf, loss_host,
        )
        print(
            "  epoch " + String(epoch + 1) + "/" + String(EPOCHS)
            + "  train_loss(last_batch)=" + String(Float32(final_loss))
            + "  val_loss=" + String(v)
            + "  lr_scale=" + String(lr_s)
        )

    var t_end = perf_counter_ns()
    print(
        "\n  training time: "
        + String(Float64(t_end - t_start) / 1e9)[byte=:6]
        + " s"
    )

    # ---------- Final eval ----------
    var val_final = _eval_loss_seq_gpu(
        ctx, state, val_inp_lt, val_tgt_lt,
        output_buf, cache_buf, ws_buf, loss_buf, loss_host,
    )
    print(
        "\n[final] val_loss=" + String(val_final)
        + " (start " + String(val_init) + ")"
    )
    if val_final < val_init - 0.1:
        print("  PASS: validation loss decreased by > 0.1 nats")
    else:
        print(
            "  WARN: validation loss did not improve substantially —"
            + " increase EPOCHS or check"
        )

    # ---------- Diagnostic: per-token top-1 accuracy on val ----------
    # Distinguishes "loss is real, generator is broken" from "loss is
    # artifactually low (forward-pass leak)". Expected with val_loss≈0.55:
    #   genuine model     → ~55–65 % top-1 (e^-0.55 ≈ 58 %)
    #   leak/cheat        → ~5–15 % top-1 (loss decoupled from prediction)
    var val_acc = _eval_topk_accuracy_gpu(
        ctx, state, val_inp_lt, val_batch.targets,
        output_buf, cache_buf, ws_buf,
    )
    print(
        "[diagnostic] val per-token top-1 accuracy="
        + String(val_acc * 100.0)[byte=:5]
        + "%  (random ≈ "
        + String(100.0 / Float64(VOCAB))[byte=:4]
        + "%, expected from loss≈"
        + String(val_final)[byte=:5]
        + " ⇒ ~"
        + String(exp(-val_final) * 100.0)[byte=:5]
        + "%)"
    )

    # ---------- Sampling ----------
    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))

    print("\n[sample] greedy (T=0.0):")
    var greedy = _generate_text_gpu(
        ctx, state, tok, prompt, 200, 0.0, top_k=0
    )
    print(prompt + greedy)

    # top_k=0 → no filter, sample from the full softmax (matches nanoGPT
    # `sample.py` default top_k=200 with vocab=65, i.e. effectively no filter).
    print("\n[sample] temperature (T=0.8, no top-k):")
    var temp = _generate_text_gpu(
        ctx, state, tok, prompt, 200, 0.8, top_k=0
    )
    print(prompt + temp)

    print("\n" + "=" * 70)
