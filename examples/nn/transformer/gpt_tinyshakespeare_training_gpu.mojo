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
from std.math import log, exp, sqrt
from std.time import perf_counter_ns

from mojo_rl.nn.constants import dtype, TPB
from std.gpu import block_dim, block_idx, thread_idx
from mojo_rl.nn.composites import GPTDrop
from mojo_rl.nn.training import (
    NetworkState,
    GPUNetworkState,
    CosineWarmupSchedule,
)
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.initializer import Normal
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
comptime EMBED = 384         # transformer width (matches nanoGPT char-Shakespeare)
comptime HEADS = 6           # head_dim = 32
comptime LAYERS = 6          # transformer blocks
comptime FF_MULT = 4         # FFN inner dim = 4 * EMBED = 768

comptime BATCH = 64          # nanoGPT char-Shakespeare batch size

comptime BASE_LR = 1e-3      # nanoGPT char-Shakespeare LR (3.3× our prior 3e-4)
comptime BETA1 = 0.9
comptime BETA2 = 0.99        # nanoGPT char-Shakespeare beta2 (small batch → bigger)
comptime WD = 0.1
comptime GRAD_CLIP = 1.0     # max-abs clip on params grads each step

# Per-step random window sampling (matches nanoGPT's `get_batch`):
#   each iter samples a fresh BATCH of random windows from `split.train`,
#   instead of pre-sampling 32k windows up-front. Avoids the overfit-to-
#   fixed-corpus failure mode where the same windows are revisited 10× and
#   the model memorises local statistics rather than learning long-range
#   structure.
# Total examples seen ≈ TOTAL_ITERS × BATCH = 5000 × 64 = 320 000, exact
# match with nanoGPT's `max_iters=5000, batch_size=64`.
comptime TOTAL_ITERS = 5000
comptime WARMUP_ITERS = 100            # nanoGPT default
comptime EVAL_INTERVAL = 250           # nanoGPT eval_interval
comptime MIN_LR_SCALE = 0.1            # min_lr / lr = 1e-4 / 1e-3

# Validation kept pre-sampled — cheap and gives a stable per-eval signal.
comptime N_VAL_WINDOWS = 256           # 4 batches × BATCH=64
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
# CosineWarmupSchedule is unit-agnostic — we feed it (iter, TOTAL_ITERS)
# so warmup over WARMUP_ITERS, cosine decay to MIN_LR_SCALE thereafter.
comptime GPT_SCHEDULER = CosineWarmupSchedule[WARMUP_ITERS, MIN_LR_SCALE]


# =============================================================================
# Weight tying — LM head shares params with the input embedding (nanoGPT).
#
# Embedding W layout : (VOCAB, EMBED) row-major → flat index v*EMBED + e
# LM head Linear W   : (EMBED, VOCAB) row-major → flat index e*VOCAB + v
# Tying enforces  lm_head_W[e, v] == embedding_W[v, e]  (transpose).
#
# In the GPTDrop chain, Embedding is the first param-bearing layer and the
# LM head Linear is the last; Linear[EMBED, VOCAB] stores [W (EMBED*VOCAB) | b (VOCAB)],
# so:
#   EMB_W_OFF  = 0
#   LM_W_OFF   = PARAM_SIZE - VOCAB*EMBED - VOCAB
#
# Each iter we (a) accumulate the LM head W gradient into the embedding W
# gradient (transposed) and zero the LM head W grad before optimizer_step,
# then (b) copy the embedding W back into the LM head W slot afterwards so
# the two stay perfectly tied. The LM head's bias and optimizer moments
# evolve normally — only its weight is forced to mirror the embedding.
# =============================================================================
comptime EMB_W_OFF = 0
comptime LM_W_OFF = GPT_MODEL.PARAM_SIZE - VOCAB * EMBED - VOCAB
comptime TIE_NCELL = VOCAB * EMBED

# =============================================================================
# Per-block layout (GPTDrop / TransformerBlockDrop) — used by the c_proj
# scaled-init pass below. Each block is laid out in this order:
#
#   LN1 (Tokenwise[seq, LayerNorm[D]])  : 2*D
#   Linear[D, 3D]   (QKV proj)          : 3*D² + 3*D
#   ScaledDotProductAttention            : 0
#   Linear[D, D]    (attn out, c_proj)  : D² + D     ← W scaled by 1/√(2L)
#   Dropout                              : 0
#   LN2                                  : 2*D
#   Linear[D, F]    (FFN first)         : D*F + F
#   GELU                                 : 0
#   Linear[F, D]    (FFN out, c_proj)   : F*D + D    ← W scaled by 1/√(2L)
#   Dropout                              : 0
#
# The Linear layout inside each fused block is [W_flat | b], so the W of
# each c_proj layer is the first D² (attn-out) or F*D (FFN-out) entries.
# We scale only the W portion; biases (initialized via the Initializer)
# are left untouched by this pass.
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

# First block starts after Embedding (vocab*embed) + position BiasAdd (seq*embed).
comptime BLOCKS_BASE = VOCAB * EMBED + SEQ * EMBED


def _apply_c_proj_scaled_init(
    mut p: LayoutTensor[
        dtype, Layout.row_major(GPT_MODEL.PARAM_SIZE), MutAnyOrigin
    ],
) raises:
    """Scale attn-output-proj W and FFN-output-proj W per block by 1/√(2L).

    Matches nanoGPT's GPT-2-style scaled init for residual output projections.
    Run on CPU after `cpu.initialize[Normal[0, 0.02]]()` and before the
    `state.upload_from(cpu, ctx)` that uploads weights to the GPU.

    Takes the raw param `LayoutTensor` (not the wrapping NetworkState) so
    the function's mangled name doesn't carry the deep `GPTDrop[…]` chain
    — that pushed the Apple `ld` symbol-name-length limit over the cliff.
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


@always_inline
def tie_grads_kernel[
    PARAM_SIZE: Int,
    EMB_OFF: Int,
    LM_OFF: Int,
    VOCAB_: Int,
    EMBED_: Int,
    dtype: DType,
](
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= VOCAB_ * EMBED_:
        return
    var v = idx // EMBED_
    var e = idx % EMBED_
    var emb_idx = EMB_OFF + v * EMBED_ + e
    var lm_idx = LM_OFF + e * VOCAB_ + v
    var lm_g = rebind[Scalar[dtype]](grads[lm_idx])
    grads[emb_idx] = rebind[Scalar[dtype]](grads[emb_idx]) + lm_g
    grads[lm_idx] = Scalar[dtype](0.0)


@always_inline
def tie_params_kernel[
    PARAM_SIZE: Int,
    EMB_OFF: Int,
    LM_OFF: Int,
    VOCAB_: Int,
    EMBED_: Int,
    dtype: DType,
](
    params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= VOCAB_ * EMBED_:
        return
    var v = idx // EMBED_
    var e = idx % EMBED_
    var emb_idx = EMB_OFF + v * EMBED_ + e
    var lm_idx = LM_OFF + e * VOCAB_ + v
    params[lm_idx] = rebind[Scalar[dtype]](params[emb_idx])


def _tie_grads(ctx: DeviceContext, state: GPUNetworkState[GPT_MODEL, GPT_OPT]) raises:
    var g = state.grads_view()
    comptime BLOCKS = (TIE_NCELL + TPB - 1) // TPB
    ctx.enqueue_function[
        tie_grads_kernel[
            GPT_MODEL.PARAM_SIZE, EMB_W_OFF, LM_W_OFF, VOCAB, EMBED, dtype
        ],
        tie_grads_kernel[
            GPT_MODEL.PARAM_SIZE, EMB_W_OFF, LM_W_OFF, VOCAB, EMBED, dtype
        ],
    ](g, grid_dim=(BLOCKS,), block_dim=(TPB,))


def _tie_params(ctx: DeviceContext, state: GPUNetworkState[GPT_MODEL, GPT_OPT]) raises:
    var p = state.params_view()
    comptime BLOCKS = (TIE_NCELL + TPB - 1) // TPB
    ctx.enqueue_function[
        tie_params_kernel[
            GPT_MODEL.PARAM_SIZE, EMB_W_OFF, LM_W_OFF, VOCAB, EMBED, dtype
        ],
        tie_params_kernel[
            GPT_MODEL.PARAM_SIZE, EMB_W_OFF, LM_W_OFF, VOCAB, EMBED, dtype
        ],
    ](p, grid_dim=(BLOCKS,), block_dim=(TPB,))


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
        "  total_iters=" + String(TOTAL_ITERS)
        + " warmup_iters=" + String(WARMUP_ITERS)
        + " eval_interval=" + String(EVAL_INTERVAL)
        + " n_val_windows=" + String(N_VAL_WINDOWS)
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

    # ---------- Pre-sample val windows; train sampled per-iter ----------
    print(
        "\n[data] pre-sampling " + String(N_VAL_WINDOWS) + " val windows"
        + " (train resampled per iter)..."
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
    # nanoGPT's char-Shakespeare init: N(0, 0.02) for every Linear / Embedding
    # weight, plus an additional 1/√(2L) scaling on every attention-output and
    # FFN-output projection (the GPT-2 "scaled init"). Keeps the residual
    # stream variance bounded as depth grows.
    cpu.initialize[Normal[0.0, 0.02]]()
    var cpu_params = cpu.params_view()
    _apply_c_proj_scaled_init(cpu_params)
    state.upload_from(cpu, ctx)
    # Tie LM head weight to embedding weight at init (overwrites the LM
    # head's freshly-sampled weights with embedding.T). After this every
    # forward sees a coherent tied pair; subsequent ties happen post-step.
    _tie_params(ctx, state)

    # ---------- Upload val once; train uses a per-iter staging batch ----------
    print("[data] uploading val dataset + allocating per-iter train staging...")
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

    # Per-iter train staging — one BATCH worth of one-hot, reused every step.
    var train_inp_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * GPT_MODEL.IN_DIM
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * GPT_MODEL.OUT_DIM
    )
    var train_inp_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * GPT_MODEL.IN_DIM
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * GPT_MODEL.OUT_DIM
    )
    var batch_input = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT_MODEL.IN_DIM), MutAnyOrigin
    ](train_inp_buf.unsafe_ptr())
    var batch_target_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](train_tgt_buf.unsafe_ptr())

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
        "\n[iter 0] initial val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    # ---------- Per-iter loop: sample → forward → backward → step ----------
    print("\n── Training ──")
    var t_start = perf_counter_ns()
    var final_loss: Float64 = 0.0
    for iter in range(TOTAL_ITERS):
        var lr_s = GPT_SCHEDULER.lr_scale_at(iter, TOTAL_ITERS)
        state.set_lr_scale(lr_s, ctx)

        var p_view = state.params_view()
        var s_view = state.model_state_view()

        # Sample a fresh BATCH of windows from train, build one-hot, upload.
        var mb = make_batch(split.train, BATCH, SEQ)
        var inp_oh = to_one_hot(mb.inputs, VOCAB, BATCH, SEQ)
        var tgt_oh = to_one_hot(mb.targets, VOCAB, BATCH, SEQ)
        for i in range(BATCH * GPT_MODEL.IN_DIM):
            train_inp_host.unsafe_ptr()[i] = inp_oh[i]
            train_tgt_host.unsafe_ptr()[i] = tgt_oh[i]
        ctx.enqueue_copy(train_inp_buf, train_inp_host)
        ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

        GPT_MODEL.forward_gpu[BATCH, dtype](
            ctx, output_t, batch_input, p_view, s_view, cache_t, ws_buf
        )
        CrossEntropyLoss.forward_gpu[BATCH * SEQ, VOCAB, dtype](
            ctx, loss_t, output_v, batch_target_v
        )
        CrossEntropyLoss.backward_gpu[BATCH * SEQ, VOCAB, dtype](
            ctx, grad_out_v, output_v, batch_target_v
        )

        state.zero_grads(ctx)
        var grads_view = state.grads_view()
        GPT_MODEL.backward_gpu[BATCH, dtype](
            ctx, grad_in_t, grad_out_t, p_view, s_view, cache_t,
            grads_view, ws_buf,
        )
        # Weight tying — fold lm_head W grad (transposed) into embedding W
        # grad and zero the lm_head W grad slot. Must run before clip+step.
        _tie_grads(ctx, state)
        # Per-element max-abs grad clip — cheap insurance against the
        # rare-spike instability common at this depth, standard in nanoGPT.
        state.clip_grads(ctx, Scalar[dtype](GRAD_CLIP))
        state.optimizer_step(ctx)
        # Re-tie lm_head W to embedding W after step (in case Adam moments
        # in the lm_head slot ever produce a non-trivial update — shouldn't
        # since grad was zeroed, but keeps the tie strict against numerical
        # drift / weight decay).
        _tie_params(ctx, state)

        # Eval + log every EVAL_INTERVAL iters (and on the last iter).
        if (iter + 1) % EVAL_INTERVAL == 0 or (iter + 1) == TOTAL_ITERS:
            ctx.enqueue_copy(loss_host, loss_buf)
            ctx.synchronize()
            final_loss = Float64(loss_host[0])
            var v = _eval_loss_seq_gpu(
                ctx, state, val_inp_lt, val_tgt_lt,
                output_buf, cache_buf, ws_buf, loss_buf, loss_host,
            )
            print(
                "  iter " + String(iter + 1) + "/" + String(TOTAL_ITERS)
                + "  train_loss=" + String(Float32(final_loss))
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

    # Diagnostic: long-prompt sampling.
    #
    # The val_loss=1.6 + top-1=52 % numbers were measured with full 256-char
    # context. The "ROMEO:" prompt above only feeds 6 chars, so position 5
    # has very little context to work with. To distinguish (a) a real bug in
    # the front-anchor inference path from (b) a low-context capacity issue,
    # generate from a 250-char real Shakespeare prefix. With ~250 chars of
    # context the model is in the regime the val diagnostic actually measured.
    #   - If output is coherent → low-context generation just needs more
    #     prompt / bigger model; the inference pipeline is fine.
    #   - If output is still degenerate (newlines, repeated bigrams) →
    #     there's a real bug somewhere in the front-anchor / generation path.
    var long_prompt = String(text[byte=0:250])
    print(
        "\n[sample] long prompt diagnostic (250 real Shakespeare chars):\n"
        + "---- prompt ----\n" + long_prompt + "\n---- continuation (greedy) ----"
    )
    var long_cont = _generate_text_gpu(
        ctx, state, tok, long_prompt, 200, 0.0, top_k=0
    )
    print(long_cont)
    print("---- continuation (T=0.8, no top-k) ----")
    var long_temp = _generate_text_gpu(
        ctx, state, tok, long_prompt, 200, 0.8, top_k=0
    )
    print(long_temp)

    print("\n" + "=" * 70)
