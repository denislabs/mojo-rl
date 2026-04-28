"""TinyShakespeare char-GPT training (CPU smoke test).

Validates the GPT composite, attention causal mask, Tokenwise combinator,
TransformerBlock, embedding+pos-embed, and the cross-entropy training path
end-to-end on real TinyShakespeare data.

This is a CPU smoke harness — uses a tiny config so each step takes <1s. The
goal is to demonstrate the loss decreases monotonically over a few hundred
iterations, not to reach a target validation score (that is the GPU
training script's job).

Run:
    pixi run mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training.mojo
"""

from std.random import seed, random_float64
from std.math import cos, log, exp

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import GPT
from mojo_rl.nn.training import NetworkState
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
# Hyperparameters (tiny config for CPU smoke)
# =============================================================================
comptime VOCAB = 65       # TinyShakespeare standard vocab
comptime SEQ = 32         # sequence length
comptime EMBED = 32       # embedding dim
comptime HEADS = 2        # attention heads (head_dim = 16)
comptime LAYERS = 2       # transformer blocks
comptime FF_MULT = 4      # FFN hidden = 4 * EMBED

comptime BATCH = 4

comptime BASE_LR = 3e-4
comptime BETA1 = 0.9
comptime BETA2 = 0.95
comptime WD = 0.1

# Total optimizer steps for the smoke run. Tweak up for richer training.
comptime N_STEPS = 200
comptime WARMUP_STEPS = 20
comptime PRINT_EVERY = 10
comptime EVAL_EVERY = 50
comptime EVAL_BATCHES = 4


# =============================================================================
# LR schedule: linear warmup then cosine decay to 10 % of peak
# =============================================================================
def lr_scale(step: Int, warmup: Int, total: Int) -> Float64:
    if step < warmup:
        return Float64(step + 1) / Float64(warmup)
    var progress = Float64(step - warmup) / Float64(max(1, total - warmup))
    if progress > 1.0:
        progress = 1.0
    # Cosine from 1.0 down to 0.1
    var c = 0.5 * (1.0 + cos(progress * 3.141592653589793))
    return 0.1 + 0.9 * c


# =============================================================================
# Compute per-token cross-entropy loss + grad_output for a GPT batch
# =============================================================================
# `output` and `target_oh` are (BATCH, SEQ * VOCAB) row-major. The exact same
# memory, viewed as (BATCH * SEQ, VOCAB), is BATCH*SEQ independent classifiers
# (one per token position). CrossEntropyLoss averages over the inner BATCH
# dimension, so calling it at BATCH'=BATCH*SEQ correctly averages over all
# (sample, position) pairs.
def _compute_loss_and_grad(
    output: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
    target_oh: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
    mut grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
) -> Float64:
    var output_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](output.ptr)
    var target_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](target_oh.ptr)
    var grad_v = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
    ](grad_output.ptr)

    var loss = CrossEntropyLoss.forward[BATCH * SEQ, VOCAB, dtype](
        output_v, target_v
    )
    CrossEntropyLoss.backward[BATCH * SEQ, VOCAB, dtype](
        output_v, target_v, grad_v
    )
    return loss


# =============================================================================
# Forward + backward + optimizer step on one minibatch.
# Returns the per-token CE loss in nats.
# =============================================================================
def _train_step(
    mut state: NetworkState[GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS], AdamW[
        BASE_LR, BETA1, BETA2, 1e-8, WD
    ]],
    inp_oh: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
    tgt_oh: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
    mut output: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
    mut cache: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut grad_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
    mut grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ * VOCAB), MutAnyOrigin
    ],
) -> Float64:
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS]
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    Model.forward[BATCH, dtype](inp_oh, output, p_view, s_view, cache)
    var loss = _compute_loss_and_grad(output, tgt_oh, grad_output)

    state.zero_grads()
    var grads_v = state.grads_view()
    Model.backward[BATCH, dtype](
        grad_output, grad_input, p_view, s_view, cache, grads_v
    )
    state.optimizer_step()
    return loss


# =============================================================================
# Sampling: greedy (argmax) and temperature
# =============================================================================
# Sampling is a sliding-window forward pass: at every step we feed the last
# SEQ tokens (front-padded with `pad_id` if shorter than SEQ), run GPT.forward
# with cache, and read the next-token logits at position SEQ-1.
#
# At T=0 we take argmax; at T>0 we apply temperature scaling and categorical
# sample from the softmax. No KV cache for v1 — re-runs the full forward each
# step, which is fine at small SEQ.
def _sample_categorical(
    logits_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    vocab: Int,
    temperature: Float64,
) -> Int:
    # Greedy
    if temperature <= 0.0:
        var best_v = Float64(logits_row[0])
        var best_idx = 0
        for v in range(1, vocab):
            var x = Float64(logits_row[v])
            if x > best_v:
                best_v = x
                best_idx = v
        return best_idx

    # Temperature softmax with log-sum-exp stabilization.
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


def generate_text(
    state: NetworkState[GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS], AdamW[
        BASE_LR, BETA1, BETA2, 1e-8, WD
    ]],
    tok: CharTokenizer,
    prompt: String,
    n_tokens: Int,
    temperature: Float64 = 0.8,
    pad_id: Int = 0,
) raises -> String:
    """Generate `n_tokens` characters after `prompt` using the trained GPT.

    Args:
        state:        Trained NetworkState wrapping the GPT.
        tok:          Tokenizer used to encode the prompt.
        prompt:       Seed text to condition on.
        n_tokens:     Number of new tokens (chars) to generate.
        temperature:  Sampling temperature. T=0 → greedy/argmax,
                      higher T = more diverse samples.
        pad_id:       Token id used to front-pad the context window when the
                      generated sequence is shorter than SEQ. Newline (id=0)
                      is the typical choice.
    """
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS]
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    # Single mutable token-id buffer: prompt + generated tokens, in order.
    var all_ids = tok.encode(prompt)
    var prompt_len = len(all_ids)
    if prompt_len == 0:
        raise Error("generate_text: prompt is empty after tokenization")

    # Reusable forward buffers. Sampling uses BATCH=1.
    var inp_data = List[Scalar[dtype]](capacity=Model.IN_DIM)
    for _ in range(Model.IN_DIM):
        inp_data.append(0)
    var out_data = List[Scalar[dtype]](capacity=Model.OUT_DIM)
    for _ in range(Model.OUT_DIM):
        out_data.append(0)
    var cache_data = List[Scalar[dtype]](capacity=Model.CACHE_SIZE)
    for _ in range(Model.CACHE_SIZE):
        cache_data.append(0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(1, Model.IN_DIM), MutAnyOrigin
    ](inp_data.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(1, Model.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(1, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    for _ in range(n_tokens):
        # Build the SEQ-length context window: last SEQ ids of the running
        # sequence. When the sequence is shorter than SEQ we BACK-anchor —
        # pad the front with pad_id (newline by default) so the prompt
        # always sits at the END of the context and we read logits at
        # position SEQ-1.
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
            inp_data[i] = 0
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
            inp_data[t * VOCAB + tid] = Scalar[dtype](1.0)

        Model.forward[1, dtype](inp_t, out_t, p_view, s_view, cache_t)

        # Logits row at position SEQ-1: layout per sample is
        # [pos_0[0..V], pos_1[0..V], ..., pos_{S-1}[0..V]].
        var last_pos_ptr = out_data.unsafe_ptr() + (SEQ - 1) * VOCAB
        var next_id = _sample_categorical(last_pos_ptr, VOCAB, temperature)
        all_ids.append(next_id)

    # Return only the generated tail, not the prompt.
    var gen_only = List[Int](capacity=n_tokens)
    for i in range(prompt_len, len(all_ids)):
        gen_only.append(all_ids[i])
    return tok.decode(gen_only)


# =============================================================================
# Validation loss (no parameter update)
# =============================================================================
def _eval_loss(
    state: NetworkState[GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS], AdamW[
        BASE_LR, BETA1, BETA2, 1e-8, WD
    ]],
    val_ids: List[Int],
    n_batches: Int,
) raises -> Float64:
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS]
    var p_view = state.params_view()
    var s_view = state.model_state_view()

    var total: Float64 = 0.0
    for _ in range(n_batches):
        var batch = make_batch(val_ids, BATCH, SEQ)
        var inp_data = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)
        var tgt_data = to_one_hot(batch.targets, VOCAB, BATCH, SEQ)
        var out_data = List[Scalar[dtype]](capacity=BATCH * Model.OUT_DIM)
        for _ in range(BATCH * Model.OUT_DIM):
            out_data.append(0)
        var cache_data = List[Scalar[dtype]](capacity=BATCH * Model.CACHE_SIZE)
        for _ in range(BATCH * Model.CACHE_SIZE):
            cache_data.append(0)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
        ](inp_data.unsafe_ptr())
        var tgt_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](tgt_data.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](out_data.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](cache_data.unsafe_ptr())

        Model.forward[BATCH, dtype](inp_t, out_t, p_view, s_view, cache_t)
        var out_v = LayoutTensor[
            dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
        ](out_t.ptr)
        var tgt_v = LayoutTensor[
            dtype, Layout.row_major(BATCH * SEQ, VOCAB), MutAnyOrigin
        ](tgt_t.ptr)
        total += CrossEntropyLoss.forward[BATCH * SEQ, VOCAB, dtype](out_v, tgt_v)
    return total / Float64(n_batches)


# =============================================================================
# Driver
# =============================================================================
def main() raises:
    seed(42)
    comptime Model = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS]
    comptime Opt = AdamW[BASE_LR, BETA1, BETA2, 1e-8, WD]

    print("=" * 70)
    print("TinyShakespeare GPT training (CPU smoke)")
    print("=" * 70)
    print("  vocab=" + String(VOCAB) + " seq=" + String(SEQ) + " embed=" + String(EMBED) + " heads=" + String(HEADS) + " layers=" + String(LAYERS))
    print("  batch=" + String(BATCH) + " base_lr=" + String(BASE_LR) + " wd=" + String(WD))
    print("  steps=" + String(N_STEPS) + " warmup=" + String(WARMUP_STEPS))
    print("  PARAM_SIZE=" + String(Model.PARAM_SIZE) + " CACHE_SIZE/sample=" + String(Model.CACHE_SIZE))

    # ---------- Load data ----------
    print("\n[data] loading TinyShakespeare...")
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error(
            "vocab mismatch: tokenizer found "
            + String(tok.vocab_size)
            + " unique chars, expected VOCAB="
            + String(VOCAB)
            + ". Update VOCAB constant."
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

    # ---------- Initialize network ----------
    var state = NetworkState[Model, Opt]()
    state.initialize[Xavier[]]()

    # ---------- Pre-allocate buffers (reused every step) ----------
    var out_data = List[Scalar[dtype]](capacity=BATCH * Model.OUT_DIM)
    for _ in range(BATCH * Model.OUT_DIM):
        out_data.append(0)
    var cache_data = List[Scalar[dtype]](capacity=BATCH * Model.CACHE_SIZE)
    for _ in range(BATCH * Model.CACHE_SIZE):
        cache_data.append(0)
    var grad_in_data = List[Scalar[dtype]](capacity=BATCH * Model.IN_DIM)
    for _ in range(BATCH * Model.IN_DIM):
        grad_in_data.append(0)
    var grad_out_data = List[Scalar[dtype]](capacity=BATCH * Model.OUT_DIM)
    for _ in range(BATCH * Model.OUT_DIM):
        grad_out_data.append(0)

    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](grad_in_data.unsafe_ptr())
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](grad_out_data.unsafe_ptr())

    # ---------- Initial val loss (sanity baseline) ----------
    # Random init logits should give CE ≈ ln(VOCAB) ≈ 4.17 for VOCAB=65.
    var val_init = _eval_loss(state, split.val, EVAL_BATCHES)
    print("\n[step 0] initial val loss=" + String(val_init) + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")")

    # ---------- Training loop ----------
    var loss_running: Float64 = 0.0
    var loss_count: Int = 0
    for step in range(N_STEPS):
        var s = lr_scale(step, WARMUP_STEPS, N_STEPS)
        state.set_lr_scale(s)

        var batch = make_batch(split.train, BATCH, SEQ)
        var inp_data = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)
        var tgt_data = to_one_hot(batch.targets, VOCAB, BATCH, SEQ)
        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
        ](inp_data.unsafe_ptr())
        var tgt_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](tgt_data.unsafe_ptr())

        var loss = _train_step(
            state, inp_t, tgt_t, out_t, cache_t, grad_in_t, grad_out_t
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
            var v = _eval_loss(state, split.val, EVAL_BATCHES)
            print("           val_loss=" + String(v))

    # ---------- Final eval ----------
    var val_final = _eval_loss(state, split.val, EVAL_BATCHES)
    print("\n[final] val_loss=" + String(val_final) + " (start " + String(val_init) + ")")
    if val_final < val_init - 0.1:
        print("  PASS: validation loss decreased by > 0.1 nats")
    else:
        print(
            "  WARN: validation loss did not improve substantially — increase N_STEPS or check"
        )

    # ---------- Sampling ----------
    # Qualitative check: at this scale (200 steps, tiny model) the output won't
    # look like Shakespeare, but we should still see (a) char distribution
    # has tilted away from uniform, (b) sampling code path runs without
    # crashing, and (c) greedy and temperature samples differ.
    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))

    print("\n[sample] greedy (T=0.0):")
    var greedy = generate_text(state, tok, prompt, 200, 0.0)
    print(prompt + greedy)

    print("\n[sample] temperature (T=0.8):")
    var temp = generate_text(state, tok, prompt, 200, 0.8)
    print(prompt + temp)

    print("\n" + "=" * 70)
