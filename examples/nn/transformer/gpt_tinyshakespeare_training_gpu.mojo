"""TinyShakespeare char-GPT training — GPU.

GPU training script targeting the docs/TRANSFORMER_VIT.md Phase A goal:
val loss ≤ 1.5 nats/char on the default nanoGPT-class config.

Default config below (S=256, D=192, H=6, N=6, ~5M params, 20k steps) is sized
for NVIDIA GPUs (~RTX 4090 / 5090). On the M1 Pro development hardware the
config OOMs — for local iteration shrink SEQ, EMBED, LAYERS, and N_STEPS to
the values commented in the constants block below.

All compute lives on device:
- Native GPU kernels for ScaledDotProductAttention (forward + 4-stage backward),
  MatMul/BiasAdd (AutoFused), LayerNorm, Embedding, Tokenwise.
- `CrossEntropyLoss.forward_gpu` / `.backward_gpu` operate on the (BATCH*S, V)
  reinterpretation of the GPT's flat (BATCH, S*V) output — averaging over all
  (sample, position) pairs as standard per-token CE.
- AdamW step + on-device step counter, gradient clipping.
- Sampling at the end runs `GPUNetworkState.forward_gpu` per generated token.

Run on NVIDIA (production target):
    pixi run -e nvidia mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo
Run on Apple Metal (dev iteration only — shrink config first):
    pixi run -e apple mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed, random_float64
from std.math import cos, log, exp

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import GPT
from mojo_rl.nn.training import NetworkState, GPUNetworkState
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
# Matches docs/TRANSFORMER_VIT.md Phase A defaults. Designed for NVIDIA GPUs;
# on Apple M1 Pro (development hardware) reduce SEQ→64, EMBED→64, LAYERS→4
# and N_STEPS→1000 to fit memory and iterate quickly.
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
comptime BETA2 = 0.95
comptime WD = 0.1
comptime GRAD_CLIP = 1.0     # max-abs clip on params grads each step

# Schedule: linear warmup over WARMUP_STEPS, then cosine decay to 10% over
# the remaining (N_STEPS - WARMUP_STEPS). 20k steps × BATCH=16 × SEQ=256 =
# ~82M tokens seen, comparable to nanoGPT's 5k iters at BATCH=64 SEQ=256.
comptime N_STEPS = 20000
comptime WARMUP_STEPS = 500
comptime PRINT_EVERY = 100
comptime EVAL_EVERY = 1000
comptime EVAL_BATCHES = 8


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
# Per-token cross-entropy on device (BATCH, S*V) viewed as (BATCH*S, V).
# =============================================================================
def _ce_loss_and_grad_gpu(
    ctx: DeviceContext,
    output_dev: DeviceBuffer[dtype],
    target_dev: DeviceBuffer[dtype],
    grad_dev: DeviceBuffer[dtype],
    loss_dev: DeviceBuffer[dtype],
) raises:
    var output_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](output_dev.unsafe_ptr())
    var target_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](target_dev.unsafe_ptr())
    var grad_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](grad_dev.unsafe_ptr())
    var loss_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](loss_dev.unsafe_ptr())

    CrossEntropyLoss.forward_gpu[BATCH * SEQ, VOCAB, dtype](
        ctx, loss_t, output_v, target_v
    )
    CrossEntropyLoss.backward_gpu[BATCH * SEQ, VOCAB, dtype](
        ctx, grad_v, output_v, target_v
    )


# =============================================================================
# One training step on device — returns scalar loss in nats.
# =============================================================================
def _train_step_gpu(
    ctx: DeviceContext,
    mut state: GPUNetworkState[
        GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True],
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
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True]
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
    # Gradient clipping (per-element max-abs). Cheap insurance against the
    # rare-spike instability common at this depth; standard in nanoGPT.
    state.clip_grads(ctx, Scalar[dtype](GRAD_CLIP))
    state.optimizer_step(ctx)

    # Read scalar loss back to host.
    ctx.enqueue_copy(loss_host, loss_dev)
    ctx.synchronize()
    return Float64(loss_host[0])


# =============================================================================
# Eval loss (no parameter update). Reuses the train buffers.
# =============================================================================
def _eval_loss_gpu(
    ctx: DeviceContext,
    state: GPUNetworkState[
        GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True],
        AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD],
    ],
    val_ids: List[Int],
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
) raises -> Float64:
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True]
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var total: Float64 = 0.0
    for _ in range(n_batches):
        var batch = make_batch(val_ids, BATCH, SEQ)
        var inp_data = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)
        var tgt_data = to_one_hot(batch.targets, VOCAB, BATCH, SEQ)
        for i in range(BATCH * Model.IN_DIM):
            inp_host[i] = inp_data[i]
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
            dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
        ](out_dev.unsafe_ptr())
        var tgt_v = LayoutTensor[
            dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
        ](tgt_dev.unsafe_ptr())
        var loss_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](loss_dev.unsafe_ptr())
        CrossEntropyLoss.forward_gpu[BATCH * SEQ, VOCAB, dtype](
            ctx, loss_t, out_v, tgt_v
        )
        ctx.enqueue_copy(loss_host, loss_dev)
        ctx.synchronize()
        total += Float64(loss_host[0])
    return total / Float64(n_batches)


# =============================================================================
# Sampling on device. Greedy if T=0, categorical otherwise. Slow path runs
# one BATCH=1 forward per generated token (no KV cache). Pulls only the
# last-position logits row back to host for sampling.
# =============================================================================
def _sample_categorical_host(
    logits_row: List[Scalar[dtype]], vocab: Int, temperature: Float64
) -> Int:
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
    var max_l = Float64(logits_row[0]) * inv_t
    for v in range(1, vocab):
        var x = Float64(logits_row[v]) * inv_t
        if x > max_l:
            max_l = x
    var sum_exp = 0.0
    var exps = List[Float64](capacity=vocab)
    for v in range(vocab):
        var e = exp(Float64(logits_row[v]) * inv_t - max_l)
        exps.append(e)
        sum_exp += e
    var u = random_float64(0.0, 1.0) * sum_exp
    var acc = 0.0
    for v in range(vocab):
        acc += exps[v]
        if u < acc:
            return v
    return vocab - 1


def _generate_text_gpu(
    ctx: DeviceContext,
    state: GPUNetworkState[
        GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True],
        AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD],
    ],
    tok: CharTokenizer,
    prompt: String,
    n_tokens: Int,
    temperature: Float64 = 0.8,
    pad_id: Int = 0,
) raises -> String:
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True]
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var all_ids = tok.encode(prompt)
    var prompt_len = len(all_ids)
    if prompt_len == 0:
        raise Error("generate_text: prompt is empty after tokenization")

    # BATCH=1 forward buffers.
    var inp_host = ctx.enqueue_create_host_buffer[dtype](Model.IN_DIM)
    var inp_dev = ctx.enqueue_create_buffer[dtype](Model.IN_DIM)
    var out_dev = ctx.enqueue_create_buffer[dtype](Model.OUT_DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](Model.CACHE_SIZE)
    var ws_dev = ctx.enqueue_create_buffer[dtype](
        max(1, Model.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var out_host = ctx.enqueue_create_host_buffer[dtype](Model.OUT_DIM)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(1, Model.IN_DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(1, Model.OUT_DIM), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(1, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_dev.unsafe_ptr())

    for _ in range(n_tokens):
        # Build the SEQ-token context window. Last SEQ ids of the running
        # sequence; when shorter than SEQ, BACK-anchor — pad the front with
        # pad_id (newline) so the prompt sits at the end and logits are
        # read at position SEQ-1.
        #
        # Empirically this produces noticeably better generations than
        # front-anchoring at position n_have-1 with backwards padding. The
        # front-anchored variant in theory feeds the model a more in-
        # distribution short context, but in practice the model's
        # position-(n_have-1) embedding carries an "early-in-window" prior
        # toward common-character defaults (space, the), and the resulting
        # generations collapse to that. Position-(SEQ-1) embeddings have
        # been trained as "deep-in-window, look back" — the long run of
        # leading pad newlines is unusual but attention learns to discount
        # them; the prompt at the tail dominates the prediction.
        for i in range(Model.IN_DIM):
            inp_host[i] = 0
        var n_have = len(all_ids)
        var pad_n = SEQ - n_have if n_have < SEQ else 0
        var first_real = 0 if n_have <= SEQ else n_have - SEQ
        for t in range(SEQ):
            var tid: Int
            if t < pad_n:
                tid = pad_id
            else:
                tid = all_ids[first_real + (t - pad_n)]
            if tid < 0 or tid >= VOCAB:
                continue
            inp_host[t * VOCAB + tid] = Scalar[dtype](1.0)
        ctx.enqueue_copy(inp_dev, inp_host)

        Model.forward_gpu[1, dtype](
            ctx, out_t, inp_t, p_view, s_view, cache_t, ws_dev
        )
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        # Pull the SEQ-1-row logits to host, sample.
        var last_row = List[Scalar[dtype]](capacity=VOCAB)
        for v in range(VOCAB):
            last_row.append(out_host[(SEQ - 1) * VOCAB + v])
        var next_id = _sample_categorical_host(last_row, VOCAB, temperature)
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
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True]
    comptime Opt = AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD]

    print("=" * 70)
    print("TinyShakespeare GPT training (GPU)")
    print("=" * 70)
    print("  vocab=" + String(VOCAB) + " seq=" + String(SEQ) + " embed=" + String(EMBED) + " heads=" + String(HEADS) + " layers=" + String(LAYERS))
    print("  batch=" + String(BATCH) + " base_lr=" + String(BASE_LR) + " wd=" + String(WD))
    print("  steps=" + String(N_STEPS) + " warmup=" + String(WARMUP_STEPS))
    print("  PARAM_SIZE=" + String(Model.PARAM_SIZE) + " CACHE/sample=" + String(Model.CACHE_SIZE) + " WS/sample=" + String(Model.WORKSPACE_SIZE_PER_SAMPLE))

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
        "  total tokens="
        + String(len(ids))
        + " train="
        + String(len(split.train))
        + " val="
        + String(len(split.val))
    )

    # ---------- Device + state ----------
    var ctx = DeviceContext()
    var state = GPUNetworkState[Model, Opt](ctx)

    # Init via transient CPU NetworkState then upload (Trainer.init_state_gpu pattern).
    var cpu = NetworkState[Model, Opt]()
    cpu.initialize[Xavier[]]()
    state.upload_from(cpu, ctx)

    # ---------- Pre-allocate device + host buffers (reused every step) ----------
    var inp_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var tgt_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.CACHE_SIZE)
    var gin_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var gout_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var ws_dev = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * Model.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var loss_dev = ctx.enqueue_create_buffer[dtype](1)
    var inp_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.IN_DIM)
    var tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.OUT_DIM)
    var loss_host = ctx.enqueue_create_host_buffer[dtype](1)

    # ---------- Initial val loss ----------
    var val_init = _eval_loss_gpu(
        ctx, state, split.val, EVAL_BATCHES,
        inp_dev, tgt_dev, out_dev, cache_dev, ws_dev, loss_dev,
        inp_host, tgt_host, loss_host,
    )
    print(
        "\n[step 0] initial val loss="
        + String(val_init)
        + "  (random ≈ ln(V)="
        + String(log(Float64(VOCAB)))
        + ")"
    )

    # ---------- Training loop ----------
    var loss_running: Float64 = 0.0
    var loss_count: Int = 0
    for step in range(N_STEPS):
        var s = lr_scale(step, WARMUP_STEPS, N_STEPS)
        state.set_lr_scale(s, ctx)

        # Sample minibatch on host, upload one-hots.
        var batch = make_batch(split.train, BATCH, SEQ)
        var inp_data = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)
        var tgt_data = to_one_hot(batch.targets, VOCAB, BATCH, SEQ)
        for i in range(BATCH * Model.IN_DIM):
            inp_host[i] = inp_data[i]
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
            loss_running = 0.0
            loss_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            var v = _eval_loss_gpu(
                ctx, state, split.val, EVAL_BATCHES,
                inp_dev, tgt_dev, out_dev, cache_dev, ws_dev, loss_dev,
                inp_host, tgt_host, loss_host,
            )
            print("           val_loss=" + String(v))

    # ---------- Final eval ----------
    var val_final = _eval_loss_gpu(
        ctx, state, split.val, EVAL_BATCHES,
        inp_dev, tgt_dev, out_dev, cache_dev, ws_dev, loss_dev,
        inp_host, tgt_host, loss_host,
    )
    print(
        "\n[final] val_loss="
        + String(val_final)
        + " (start "
        + String(val_init)
        + ")"
    )
    if val_final < val_init - 0.1:
        print("  PASS: validation loss decreased by > 0.1 nats")
    else:
        print(
            "  WARN: validation loss did not improve substantially — increase N_STEPS or check"
        )

    # ---------- Sampling ----------
    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))

    print("\n[sample] greedy (T=0.0):")
    var greedy = _generate_text_gpu(ctx, state, tok, prompt, 200, 0.0)
    print(prompt + greedy)

    print("\n[sample] temperature (T=0.8):")
    var temp = _generate_text_gpu(ctx, state, tok, prompt, 200, 0.8)
    print(prompt + temp)

    print("\n" + "=" * 70)
