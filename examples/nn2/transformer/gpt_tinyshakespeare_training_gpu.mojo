"""TinyShakespeare char-GPT training — nn2 GPU (full budget + eval + sampling).

nn2 port of `examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo`,
matching its training budget (nanoGPT-class config, 5000 iters, per-iter random
window sampling) and porting the full eval suite so we can compare nn2's
generation/eval behaviour against the legacy run:

  - per-eval mean per-token val loss (nats),
  - per-token top-1 argmax diagnostic on the val windows (is the loss
    *consistent* with good next-token prediction, or artifactually low?),
  - greedy + temperature text generation (ROMEO prompt) and a 250-char
    long-prompt diagnostic — the legacy generation looked degenerate; this
    lets us check whether nn2 reproduces that.

nn2 differences vs the legacy script (simpler, not awkward):
  - The model is *stateful* (owns params); training goes through
    `Trainer.train_step`, eval calls `trainer.net.forward` / `trainer.loss_fn`
    directly. No stateless param/cache/ws view juggling.
  - Eval/generation flip dropout off via `net.set_attr["training"](0.0)`
    (propagates to every Dropout leaf), restoring 1.0 after — no separate
    eval-mode model needed.
  - Per-token CE is the `SequenceCrossEntropyLoss[SEQ, VOCAB]` op (used for
    both training and val), instead of reinterpreting + a static CE kernel.

Generation-quality features ported from gen-1 (the full nanoGPT recipe now):
dropout (`GPTDrop`, p=0.2), `Normal(0,0.02)` init, **LM-head↔embedding weight
tying** (`TIE_WEIGHTS`), **1/√(2L) c_proj scaled init** (`SCALED_INIT`, applied
to each block's attention-out + FFN-out projection), and a **bias-less LM head**
(`HEAD_NO_BIAS`, matching nanoGPT's `lm_head bias=False` — frozen at 0 rather
than a separate Linear variant). Tying drives the step manually so the grad-fold
lands between net.vjp and optim.step. Still deferred (smallest effect): per-step
grad clip (nn2 AdamW has none). See docs/NN2_TRANSFORMER_PORT.md.

Default config is sized for NVIDIA. On Apple it OOMs — shrink SEQ→64,
EMBED→64, LAYERS→2, BATCH→16, TOTAL_ITERS→400 (the prior dev config).

Run on NVIDIA:
    pixi run -e nvidia mojo run -I . examples/nn2/transformer/gpt_tinyshakespeare_training_gpu.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.math import log, exp, cos, sqrt
from std.time import perf_counter_ns
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import row_major, TileTensor, Layout, LayoutTensor

from mojo_rl.nn.datasets import (
    CharTokenizer, load_text, train_val_split, make_batch, to_one_hot,
)
from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.composites import GPTDrop
from mojo_rl.nn2.loss import SequenceCrossEntropyLoss
from mojo_rl.nn2.optimizer import AdamW
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Normal


# ── Full nanoGPT-class config (NVIDIA) — same budget as the gen-1 script ──
comptime VOCAB = 65
comptime SEQ = 256
comptime EMBED = 384
comptime HEADS = 6
comptime LAYERS = 6
comptime FF_MULT = 4
comptime BATCH = 64

comptime BASE_LR: Scalar[DT] = 1e-3
comptime BETA2: Scalar[DT] = 0.99
comptime WD: Scalar[DT] = 0.1

comptime TOTAL_ITERS = 5000
comptime WARMUP_ITERS = 100
comptime EVAL_INTERVAL = 250
comptime MIN_LR_SCALE: Float64 = 0.1

comptime N_VAL_WINDOWS = 256
comptime N_VAL_BATCHES = N_VAL_WINDOWS // BATCH
comptime IN_DIM = SEQ * VOCAB
comptime OUT_DIM = SEQ * VOCAB

# Attention kernel path: True → batched-GEMM (USE_MAX_KERNELS, fast on NVIDIA);
# False → portable serial per-(b,h) custom kernels. Flip to compare timings;
# the two are bit-identical on Metal (TF32 may widen the gap on NVIDIA).
comptime USE_MAX_ATTN = True
# Dropout (nanoGPT char-Shakespeare uses 0.2). Regularizes against overfitting
# the small corpus — without it val loss collapses (memorization) and greedy
# generation degenerates. Toggled off for eval/generation via set_attr.
comptime DROPOUT_P: Float64 = 0.2
comptime GPT_MODEL = GPTDrop[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
    UInt64(0xC0FFEE), USE_MAX_ATTN,
]
comptime GPT_LOSS = SequenceCrossEntropyLoss[SEQ, VOCAB]
comptime GPT_TRAINER = Trainer[GPT_MODEL, AdamW, GPT_LOSS, BATCH, target="gpu"]

# Weight tying: LM-head W shares the embedding W (transposed) — nanoGPT's
# `lm_head.weight = wte.weight`. In your nn experience this was the key fix
# for degenerate generation (val collapsing low while greedy decodes "the the
# the"). Each step we fold the LM-head W grad (transposed) into the embedding
# W grad + zero the LM-head's, step, then copy emb→lm so the two stay tied.
# Requires the manual training pipeline (the grad-fold lands between net.vjp
# and optim.step, which the packaged train_step does atomically).
#   Embedding W : (VOCAB, EMBED) row-major → v*EMBED + e
#   LM-head  W  : (EMBED, VOCAB) row-major → e*VOCAB + v   ⇒ tie lm[e,v]=emb[v,e]
# GPTDrop children: [Tok[Embed], BiasAdd, Dropout, Repeat, Tok[LN], Tok[LMHead]]
# → embedding is child 0, LM head is the last child.
comptime TIE_WEIGHTS = True
comptime LM_IDX = GPT_MODEL.N - 1

# nanoGPT's lm_head is `nn.Linear(..., bias=False)`. nn2 Linear always carries
# a bias, so instead of a bias-less Linear variant (would touch the core
# primitive used everywhere) we make the head bias-less by freezing it at 0:
# Normal init sets bias=0, and we zero its gradient every step so the optimizer
# never moves it (decay is already off for biases). forward = x@W + 0 ≡ no bias.
comptime HEAD_NO_BIAS = True


def _tie_grads_kernel[
    VOCAB_: Int, EMBED_: Int
](
    emb_g: LayoutTensor[DT, Layout.row_major(VOCAB_ * EMBED_), MutAnyOrigin],
    lm_g: LayoutTensor[DT, Layout.row_major(EMBED_ * VOCAB_), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= VOCAB_ * EMBED_:
        return
    var v = idx // EMBED_
    var e = idx % EMBED_
    var lm_i = e * VOCAB_ + v
    emb_g[idx] = rebind[Scalar[DT]](emb_g[idx]) + rebind[Scalar[DT]](lm_g[lm_i])
    lm_g[lm_i] = Scalar[DT](0)


def _tie_params_kernel[
    VOCAB_: Int, EMBED_: Int
](
    emb_v: LayoutTensor[DT, Layout.row_major(VOCAB_ * EMBED_), MutAnyOrigin],
    lm_v: LayoutTensor[DT, Layout.row_major(EMBED_ * VOCAB_), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= VOCAB_ * EMBED_:
        return
    var v = idx // EMBED_
    var e = idx % EMBED_
    lm_v[e * VOCAB_ + v] = rebind[Scalar[DT]](emb_v[idx])


def _tie_grads(mut trainer: GPT_TRAINER, ctx: DeviceContext) raises:
    var eg = LayoutTensor[DT, Layout.row_major(VOCAB * EMBED), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            trainer.net.children[0].inner.weight.grad_dev.value().unsafe_ptr()
        )
    )
    var lg = LayoutTensor[DT, Layout.row_major(EMBED * VOCAB), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            trainer.net.children[LM_IDX].inner.weight.grad_dev.value().unsafe_ptr()
        )
    )
    comptime nb = (VOCAB * EMBED + TPB - 1) // TPB
    ctx.enqueue_function[_tie_grads_kernel[VOCAB, EMBED]](
        eg, lg, grid_dim=nb, block_dim=TPB
    )


def _tie_params(mut trainer: GPT_TRAINER, ctx: DeviceContext) raises:
    var ev = LayoutTensor[DT, Layout.row_major(VOCAB * EMBED), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            trainer.net.children[0].inner.weight.value_dev.value().unsafe_ptr()
        )
    )
    var lv = LayoutTensor[DT, Layout.row_major(EMBED * VOCAB), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            trainer.net.children[LM_IDX].inner.weight.value_dev.value().unsafe_ptr()
        )
    )
    comptime nb = (VOCAB * EMBED + TPB - 1) // TPB
    ctx.enqueue_function[_tie_params_kernel[VOCAB, EMBED]](
        ev, lv, grid_dim=nb, block_dim=TPB
    )


# nanoGPT/GPT-2 scaled init: divide each residual output projection
# (attention-out and FFN-out) weight by 1/√(2L) after Normal init, keeping the
# residual-stream variance bounded as depth grows. Reached through the GPTDrop
# child tree (per-block, deeply nested — see the chain below).
comptime SCALED_INIT = True


def _scale_kernel[
    N: Int
](buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin], s: Scalar[DT]):
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = rebind[Scalar[DT]](buf[idx]) * s


def _scale_c_proj(mut trainer: GPT_TRAINER, ctx: DeviceContext) raises:
    var s = Scalar[DT](1.0 / sqrt(Float64(2 * LAYERS)))
    comptime DD = EMBED * EMBED                 # attn-out Linear[D, D]
    comptime FD = (FF_MULT * EMBED) * EMBED      # FFN-out  Linear[F, D]
    comptime db = (DD + TPB - 1) // TPB
    comptime fb = (FD + TPB - 1) // TPB
    # GPTDrop.children[3] = Repeat; .children[L] = TransformerBlockDrop:
    #   [0]=Residual(LN+MHADrop), [1]=Residual(LN+FFNDrop).
    # MHADrop  = Seq[Tok[Lin d,3d], QKVToMajor, Attn, Tok[Lin d,d] (c_proj@3), Dropout]
    # FFNDrop  = Seq[Tok[Lin d,ff], GELU, Tok[Lin ff,d] (c_proj@2), Dropout]
    for L in range(LAYERS):
        var a = LayoutTensor[DT, Layout.row_major(DD), MutAnyOrigin](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                trainer.net.children[3].children[L].children[0].inner
                .children[1].children[3].inner.weight.value_dev.value()
                .unsafe_ptr()
            )
        )
        ctx.enqueue_function[_scale_kernel[DD]](a, s, grid_dim=db, block_dim=TPB)
        var f = LayoutTensor[DT, Layout.row_major(FD), MutAnyOrigin](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                trainer.net.children[3].children[L].children[1].inner
                .children[1].children[2].inner.weight.value_dev.value()
                .unsafe_ptr()
            )
        )
        ctx.enqueue_function[_scale_kernel[FD]](f, s, grid_dim=fb, block_dim=TPB)


def _lr_scale(it: Int) -> Scalar[DT]:
    """Linear warmup then cosine decay to MIN_LR_SCALE (per-iter)."""
    if it < WARMUP_ITERS:
        return Scalar[DT](Float64(it + 1) / Float64(WARMUP_ITERS))
    var denom = TOTAL_ITERS - WARMUP_ITERS
    if denom < 1:
        denom = 1
    var prog = Float64(it - WARMUP_ITERS) / Float64(denom)
    if prog > 1.0:
        prog = 1.0
    var c = 0.5 * (1.0 + cos(3.14159265358979 * prog))
    return Scalar[DT](MIN_LR_SCALE + (1.0 - MIN_LR_SCALE) * c)


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _host_one_hot_into(
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin], ids: List[Int], n_rows: Int,
):
    for i in range(n_rows * IN_DIM):
        dst[i] = 0.0
    for r in range(n_rows):
        for t in range(SEQ):
            var tid = ids[r * SEQ + t]
            if tid >= 0 and tid < VOCAB:
                dst[r * IN_DIM + t * VOCAB + tid] = 1.0


# ── Eval: device per-token val loss (nats) over the pre-uploaded windows ──
def _eval_val_loss(
    mut trainer: GPT_TRAINER,
    val_in: UnsafePointer[Scalar[DT], MutAnyOrigin],
    val_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_dev: DeviceBuffer[DT],
) raises -> Float64:
    # Eval mode: dropout off (set_attr propagates to every Dropout leaf).
    trainer.net.set_attr["training"](Scalar[DT](0.0))
    var total: Float64 = 0.0
    for vb in range(N_VAL_BATCHES):
        var in_tt = TileTensor(val_in + vb * BATCH * IN_DIM, row_major[BATCH, IN_DIM]())
        var out_tt = TileTensor(_mao(out_dev), row_major[BATCH, OUT_DIM]())
        trainer.net.forward["gpu", BATCH](in_tt, output=out_tt)
        var tgt_tt = TileTensor(val_tgt + vb * BATCH * OUT_DIM, row_major[BATCH, OUT_DIM]())
        total += Float64(trainer.loss_fn.forward["gpu", BATCH](out_tt, tgt_tt))
    trainer.net.set_attr["training"](Scalar[DT](1.0))
    return total / Float64(N_VAL_BATCHES)


# ── Diagnostic: per-token top-1 argmax accuracy on val windows ──
def _eval_top1(
    mut trainer: GPT_TRAINER,
    val_in: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_ids: List[Int],
    out_dev: DeviceBuffer[DT],
    out_host: HostBuffer[DT],
    ctx: DeviceContext,
) raises -> Float64:
    trainer.net.set_attr["training"](Scalar[DT](0.0))  # eval mode
    var correct: Int = 0
    var count: Int = 0
    var oh = out_host.unsafe_ptr()
    for vb in range(N_VAL_BATCHES):
        var in_tt = TileTensor(val_in + vb * BATCH * IN_DIM, row_major[BATCH, IN_DIM]())
        var out_tt = TileTensor(_mao(out_dev), row_major[BATCH, OUT_DIM]())
        trainer.net.forward["gpu", BATCH](in_tt, output=out_tt)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for b in range(BATCH):
            for t in range(SEQ):
                var row = b * OUT_DIM + t * VOCAB
                var best_v = Float64(oh[row])
                var best_i = 0
                for v in range(1, VOCAB):
                    var x = Float64(oh[row + v])
                    if x > best_v:
                        best_v = x
                        best_i = v
                var tgt = target_ids[vb * BATCH * SEQ + b * SEQ + t]
                if best_i == tgt:
                    correct += 1
                count += 1
    trainer.net.set_attr["training"](Scalar[DT](1.0))
    return Float64(correct) / Float64(count)


# ── Sampling (nanoGPT-style; greedy if T<=0, else top-k softmax) ──
def _sample_token(
    row: UnsafePointer[Scalar[DT], MutAnyOrigin], temperature: Float64, top_k: Int,
) -> Int:
    if temperature <= 0.0:
        var bv = Float64(row[0])
        var bi = 0
        for v in range(1, VOCAB):
            if Float64(row[v]) > bv:
                bv = Float64(row[v])
                bi = v
        return bi
    var inv_t = 1.0 / temperature
    var scaled = List[Float64](capacity=VOCAB)
    for v in range(VOCAB):
        scaled.append(Float64(row[v]) * inv_t)
    var keep = List[Bool](capacity=VOCAB)
    if top_k > 0 and top_k < VOCAB:
        for _ in range(VOCAB):
            keep.append(False)
        var work = List[Float64](capacity=VOCAB)
        for v in range(VOCAB):
            work.append(scaled[v])
        for _ in range(top_k):
            var bv: Float64 = -1e30
            var bi = 0
            for v in range(VOCAB):
                if work[v] > bv:
                    bv = work[v]
                    bi = v
            keep[bi] = True
            work[bi] = -1e30
    else:
        for _ in range(VOCAB):
            keep.append(True)
    var m: Float64 = -1e30
    for v in range(VOCAB):
        if keep[v] and scaled[v] > m:
            m = scaled[v]
    var se: Float64 = 0.0
    var exps = List[Float64](capacity=VOCAB)
    for v in range(VOCAB):
        if keep[v]:
            var e = exp(scaled[v] - m)
            exps.append(e)
            se += e
        else:
            exps.append(0.0)
    var u = random_float64(0.0, 1.0) * se
    var acc = 0.0
    for v in range(VOCAB):
        acc += exps[v]
        if u < acc:
            return v
    return VOCAB - 1


def _generate(
    mut trainer: GPT_TRAINER,
    tok: CharTokenizer,
    prompt: String,
    n_tokens: Int,
    temperature: Float64,
    top_k: Int,
    ctx: DeviceContext,
    pad_id: Int = 0,
) raises -> String:
    trainer.net.set_attr["training"](Scalar[DT](0.0))  # eval mode
    var all_ids = tok.encode(prompt)
    var prompt_len = len(all_ids)
    if prompt_len == 0:
        raise Error("generate: empty prompt")

    var inp_h = ctx.enqueue_create_host_buffer[DT](IN_DIM)
    var inp_d = ctx.enqueue_create_buffer[DT](IN_DIM)
    var out_d = ctx.enqueue_create_buffer[DT](OUT_DIM)
    var out_h = ctx.enqueue_create_host_buffer[DT](OUT_DIM)
    ctx.synchronize()

    for _gen in range(n_tokens):
        # Front-anchored window: last min(n_have, SEQ) ids at positions 0.. ,
        # read logits at read_pos = n_eff - 1 (causal → tail pad is invisible).
        for i in range(IN_DIM):
            inp_h.unsafe_ptr()[i] = 0.0
        var n_have = len(all_ids)
        var n_eff = n_have if n_have <= SEQ else SEQ
        var first = 0 if n_have <= SEQ else n_have - SEQ
        for t in range(SEQ):
            var tid = all_ids[first + t] if t < n_eff else pad_id
            if tid >= 0 and tid < VOCAB:
                inp_h.unsafe_ptr()[t * VOCAB + tid] = 1.0
        ctx.enqueue_copy(inp_d, inp_h)

        var inp_t = TileTensor(_mao(inp_d), row_major[1, IN_DIM]())
        var out_t = TileTensor(_mao(out_d), row_major[1, OUT_DIM]())
        trainer.net.forward["gpu", 1](inp_t, output=out_t)
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()

        var read_pos = n_eff - 1
        var row_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            out_h.unsafe_ptr()
        ) + read_pos * VOCAB
        all_ids.append(_sample_token(row_ptr, temperature, top_k))

    var gen = List[Int](capacity=n_tokens)
    for i in range(prompt_len, len(all_ids)):
        gen.append(all_ids[i])
    trainer.net.set_attr["training"](Scalar[DT](1.0))
    return tok.decode(gen)


def main() raises:
    seed(42)
    print("=" * 70)
    print("TinyShakespeare GPT training — nn2 GPU (full budget + eval)")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS)
    )
    print(
        "  batch=" + String(BATCH) + " base_lr=" + String(BASE_LR)
        + " wd=" + String(WD)
        + " | dropout_p=" + String(DROPOUT_P)
        + " tie_weights=" + String(TIE_WEIGHTS)
        + " scaled_init=" + String(SCALED_INIT)
        + " head_no_bias=" + String(HEAD_NO_BIAS)
        + " use_max_attn=" + String(USE_MAX_ATTN)
    )
    print(
        "  total_iters=" + String(TOTAL_ITERS)
        + " warmup=" + String(WARMUP_ITERS)
        + " eval_interval=" + String(EVAL_INTERVAL)
        + " n_val_windows=" + String(N_VAL_WINDOWS)
    )

    print("\n[data] loading TinyShakespeare...")
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error(
            "vocab mismatch: " + String(tok.vocab_size) + " vs " + String(VOCAB)
        )
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  tokens=" + String(len(ids)) + " train=" + String(len(split.train))
        + " val=" + String(len(split.val))
    )

    var ctx = DeviceContext()
    print("[init] building nn2 GPT on GPU...")
    var net = GPT_MODEL.make["gpu", INIT = Normal[0.0, 0.02]](ctx)
    var loss_fn = GPT_LOSS.make["gpu"](ctx)
    var optim = AdamW.make["gpu", M = type_of(net)](net, ctx)
    optim.lr = BASE_LR
    optim.weight_decay = WD
    optim.beta2 = BETA2
    var trainer = GPT_TRAINER.make_from(net^, optim^, loss_fn^, ctx)
    print("  in_dim=" + String(GPT_MODEL.IN_DIMS[0]) + " out_dim=" + String(OUT_DIM))

    # ── Pre-sample val windows; upload one-hot input + target once ──
    print("\n[data] pre-sampling " + String(N_VAL_WINDOWS) + " val windows...")
    var val_batch = make_batch(split.val, N_VAL_WINDOWS, SEQ)
    var val_in_h = ctx.enqueue_create_host_buffer[DT](N_VAL_WINDOWS * IN_DIM)
    var val_tgt_h = ctx.enqueue_create_host_buffer[DT](N_VAL_WINDOWS * OUT_DIM)
    ctx.synchronize()
    _host_one_hot_into(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](val_in_h.unsafe_ptr()),
        val_batch.inputs, N_VAL_WINDOWS,
    )
    _host_one_hot_into(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](val_tgt_h.unsafe_ptr()),
        val_batch.targets, N_VAL_WINDOWS,
    )
    var val_in_d = ctx.enqueue_create_buffer[DT](N_VAL_WINDOWS * IN_DIM)
    var val_tgt_d = ctx.enqueue_create_buffer[DT](N_VAL_WINDOWS * OUT_DIM)
    ctx.enqueue_copy(val_in_d, val_in_h)
    ctx.enqueue_copy(val_tgt_d, val_tgt_h)
    var val_in_p = _mao(val_in_d)
    var val_tgt_p = _mao(val_tgt_d)

    # Eval scratch: a (BATCH, OUT_DIM) device output + host mirror.
    var eval_out_d = ctx.enqueue_create_buffer[DT](BATCH * OUT_DIM)
    var eval_out_h = ctx.enqueue_create_host_buffer[DT](BATCH * OUT_DIM)
    # Per-iter train staging + the manual-pipeline device buffers (weight tying
    # needs the grad-fold between net.vjp and optim.step, so we drive the step
    # ourselves instead of trainer.train_step).
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN_DIM)
    var tgt_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT_DIM)
    var in_d = ctx.enqueue_create_buffer[DT](BATCH * IN_DIM)
    var tgt_d = ctx.enqueue_create_buffer[DT](BATCH * OUT_DIM)
    var out_d = ctx.enqueue_create_buffer[DT](BATCH * OUT_DIM)
    var go_d = ctx.enqueue_create_buffer[DT](BATCH * OUT_DIM)
    var gi_d = ctx.enqueue_create_buffer[DT](BATCH * IN_DIM)
    var in_hp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        in_host.unsafe_ptr()
    )
    var tgt_hp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        tgt_host.unsafe_ptr()
    )
    ctx.synchronize()

    # nanoGPT scaled init on residual c_proj weights (after Normal init).
    comptime if SCALED_INIT:
        _scale_c_proj(trainer, ctx)
        ctx.synchronize()

    # Tie LM-head W to the embedding W at init so step 0 sees a coherent pair.
    comptime if TIE_WEIGHTS:
        _tie_params(trainer, ctx)
        ctx.synchronize()

    var val_init = _eval_val_loss(trainer, val_in_p, val_tgt_p, eval_out_d)
    print(
        "\n[iter 0] val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var final_train: Float64 = 0.0
    for it in range(TOTAL_ITERS):
        trainer.optim.lr = BASE_LR * _lr_scale(it)
        var mb = make_batch(split.train, BATCH, SEQ)
        _host_one_hot_into(in_hp, mb.inputs, BATCH)
        _host_one_hot_into(tgt_hp, mb.targets, BATCH)
        ctx.enqueue_copy(in_d, in_host)
        ctx.enqueue_copy(tgt_d, tgt_host)

        # Manual train step (= trainer.train_step) with the tying grad-fold
        # injected between net.vjp and optim.step.
        var in_tt = TileTensor(_mao(in_d), row_major[BATCH, IN_DIM]())
        var tgt_tt = TileTensor(_mao(tgt_d), row_major[BATCH, OUT_DIM]())
        var out_tt = TileTensor(_mao(out_d), row_major[BATCH, OUT_DIM]())
        var go_tt = TileTensor(_mao(go_d), row_major[BATCH, OUT_DIM]())
        var gi_tt = TileTensor(_mao(gi_d), row_major[BATCH, IN_DIM]())
        trainer.optim.zero_grad["gpu"](trainer.net)
        trainer.net.forward["gpu", BATCH](in_tt, output=out_tt)
        final_train = Float64(trainer.loss_fn.forward["gpu", BATCH](out_tt, tgt_tt))
        trainer.loss_fn.vjp["gpu", BATCH](tgt_tt, go_tt)
        trainer.net.vjp["gpu", BATCH](go_tt, gi_tt)
        comptime if TIE_WEIGHTS:
            _tie_grads(trainer, ctx)
        comptime if HEAD_NO_BIAS:
            # Freeze the LM-head bias at 0 (≡ bias=False): zero its grad so
            # the optimizer never moves it from the 0 it was initialized to.
            trainer.net.children[LM_IDX].inner.bias.grad_dev.value().enqueue_fill(
                Scalar[DT](0.0)
            )
        trainer.optim.step["gpu"](trainer.net)
        comptime if TIE_WEIGHTS:
            _tie_params(trainer, ctx)

        if (it + 1) % EVAL_INTERVAL == 0 or (it + 1) == TOTAL_ITERS:
            var v = _eval_val_loss(trainer, val_in_p, val_tgt_p, eval_out_d)
            print(
                "  iter " + String(it + 1) + "/" + String(TOTAL_ITERS)
                + "  train=" + String(Float32(final_train))
                + "  val=" + String(Float32(v))
                + "  lr_scale=" + String(_lr_scale(it))
            )

    var t1 = perf_counter_ns()
    print("\n  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s")

    var val_final = _eval_val_loss(trainer, val_in_p, val_tgt_p, eval_out_d)
    print("\n[final] val_loss=" + String(val_final) + " (start " + String(val_init) + ")")
    if val_final < val_init - 0.1:
        print("  PASS: val loss decreased by > 0.1 nats")
    else:
        print("  WARN: val loss did not improve substantially")

    # ── Diagnostic: per-token top-1 (is the loss consistent with prediction?) ──
    var acc = _eval_top1(
        trainer, val_in_p, val_batch.targets, eval_out_d, eval_out_h, ctx
    )
    print(
        "[diagnostic] val per-token top-1=" + String(acc * 100.0)[byte=:5]
        + "%  (random ≈ " + String(100.0 / Float64(VOCAB))[byte=:4]
        + "%, from loss≈" + String(val_final)[byte=:5]
        + " expect ~" + String(exp(-val_final) * 100.0)[byte=:5] + "%)"
    )

    # ── Sampling — compare nn2 generation quality vs the legacy run ──
    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))
    print("\n[sample] greedy (T=0.0):")
    print(prompt + _generate(trainer, tok, prompt, 200, 0.0, 0, ctx))
    print("\n[sample] temperature (T=0.8, no top-k):")
    print(prompt + _generate(trainer, tok, prompt, 200, 0.8, 0, ctx))

    var long_prompt = String(text[byte=0:250])
    print(
        "\n[sample] long-prompt diagnostic (250 real chars):\n"
        + "---- prompt ----\n" + long_prompt + "\n---- continuation (greedy) ----"
    )
    print(_generate(trainer, tok, long_prompt, 200, 0.0, 0, ctx))
    print("---- continuation (T=0.8) ----")
    print(_generate(trainer, tok, long_prompt, 200, 0.8, 0, ctx))
    print("\n" + "=" * 70)
