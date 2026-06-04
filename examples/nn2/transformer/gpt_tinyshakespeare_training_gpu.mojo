"""TinyShakespeare char-GPT training — nn2 GPU (real-dataset parity run).

nn2 port of `examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo`.
Trains the nn2 `GPT` composite with the new per-token `SequenceCrossEntropyLoss`
through the stateful nn2 `Trainer`. Per-iter random-window sampling (matches
nanoGPT's `get_batch`); periodic mean-per-token val loss in nats.

Config note: the DEFAULT below is a *dev/Apple-smoke* config (small model,
few iters) so it runs on Apple Metal / CI. For the gen-1 Phase-A parity
target (val loss ≤ 1.5 nats) flip to the PRODUCTION config in the comment
block and run on NVIDIA — that config OOMs on an M1.

Deferred vs gen-1 (convergence refinements, not architecture): weight tying
(LM head ↔ embedding), `Normal(0,0.02)` + 1/√(2L) c_proj scaled init, and
dropout (GPTDrop) are not applied here; nn2 uses Kaiming + plain GPT. See
docs/NN2_TRANSFORMER_PORT.md.

Run:
    pixi run -e apple  mojo run -I . examples/nn2/transformer/gpt_tinyshakespeare_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/nn2/transformer/gpt_tinyshakespeare_training_gpu.mojo
"""

from std.memory import alloc
from std.random import seed
from std.math import log, exp, cos
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import (
    CharTokenizer, load_text, train_val_split, make_batch, to_one_hot,
)
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.composites import GPT
from mojo_rl.nn2.loss import SequenceCrossEntropyLoss
from mojo_rl.nn2.optimizer import AdamW
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


# ── DEV / Apple-smoke config (runs on M1) ──────────────────────────────
# PRODUCTION (NVIDIA, ≤1.5 nats target): SEQ=256, EMBED=384, HEADS=6,
#   LAYERS=6, FF_MULT=4, BATCH=64, TOTAL_ITERS=5000, WARMUP_ITERS=100,
#   BASE_LR=1e-3, WD=0.1 (+ weight tying + scaled init + dropout — deferred).
comptime VOCAB = 65
comptime SEQ = 64
comptime EMBED = 64
comptime HEADS = 4
comptime LAYERS = 2
comptime FF_MULT = 4
comptime BATCH = 16

comptime TOTAL_ITERS = 400
comptime WARMUP_ITERS = 40
comptime EVAL_INTERVAL = 50
comptime BASE_LR: Scalar[DT] = 1e-3
comptime MIN_LR_SCALE: Float64 = 0.1
comptime WD: Scalar[DT] = 0.1

comptime N_VAL_WINDOWS = 4 * BATCH
comptime N_VAL_BATCHES = N_VAL_WINDOWS // BATCH
comptime IN_DIM = SEQ * VOCAB
comptime OUT_DIM = SEQ * VOCAB

comptime GPT_MODEL = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True]


def _lr_scale(it: Int) -> Scalar[DT]:
    """Linear warmup then cosine decay to MIN_LR_SCALE."""
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


def _host_one_hot_into(
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ids: List[Int],
    n_rows: Int,
):
    """ids is (n_rows*SEQ) token ids → one-hot (n_rows, SEQ*VOCAB) into dst."""
    for i in range(n_rows * IN_DIM):
        dst[i] = 0.0
    for r in range(n_rows):
        for t in range(SEQ):
            var tid = ids[r * SEQ + t]
            if tid >= 0 and tid < VOCAB:
                dst[r * IN_DIM + t * VOCAB + tid] = 1.0


def _seq_ce_host(
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_ids: List[Int],
    base_off: Int,
) -> Float64:
    """Mean per-token CE (nats) over one BATCH of (SEQ*VOCAB) logit rows."""
    var total: Float64 = 0.0
    for b in range(BATCH):
        for t in range(SEQ):
            var row = b * OUT_DIM + t * VOCAB
            var m = Float64(logits[row])
            for v in range(1, VOCAB):
                if Float64(logits[row + v]) > m:
                    m = Float64(logits[row + v])
            var se: Float64 = 0.0
            for v in range(VOCAB):
                se += exp(Float64(logits[row + v]) - m)
            var lse = m + log(se)
            var tid = target_ids[base_off + b * SEQ + t]
            total += -(Float64(logits[row + tid]) - lse)
    return total / Float64(BATCH * SEQ)


def main() raises:
    seed(42)
    print("=" * 70)
    print("TinyShakespeare GPT training — nn2 GPU")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS)
    )
    print(
        "  batch=" + String(BATCH) + " iters=" + String(TOTAL_ITERS)
        + " base_lr=" + String(BASE_LR) + " wd=" + String(WD)
    )

    print("\n[data] loading TinyShakespeare...")
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error(
            "vocab mismatch: tokenizer " + String(tok.vocab_size)
            + " vs VOCAB=" + String(VOCAB)
        )
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  tokens=" + String(len(ids)) + " train=" + String(len(split.train))
        + " val=" + String(len(split.val))
    )

    var ctx = DeviceContext()
    print("[init] building nn2 GPT on GPU...")
    var net = GPT_MODEL.make["gpu", INIT=Kaiming](ctx)
    var loss_fn = SequenceCrossEntropyLoss[SEQ, VOCAB].make["gpu"](ctx)
    var optim = AdamW.make["gpu", M = type_of(net)](net, ctx)
    optim.lr = BASE_LR
    optim.weight_decay = WD
    optim.beta2 = 0.99

    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH, target="gpu",
    ].make_from(net^, optim^, loss_fn^, ctx)

    # Pre-sample val windows.
    var val_batch = make_batch(split.val, N_VAL_WINDOWS, SEQ)
    var val_in_host = alloc[Scalar[DT]](N_VAL_WINDOWS * IN_DIM)
    var val_in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](val_in_host)
    _host_one_hot_into(val_in_p, val_batch.inputs, N_VAL_WINDOWS)

    var in_host = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * IN_DIM)
    )
    var tgt_host = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * OUT_DIM)
    )
    var out_host = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * OUT_DIM)
    )

    print(
        "\n[iter 0] random baseline ≈ ln(V)=" + String(log(Float64(VOCAB)))
    )
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var first_val: Float64 = 0.0
    var last_val: Float64 = 0.0
    for it in range(TOTAL_ITERS):
        trainer.optim.lr = BASE_LR * _lr_scale(it)

        var mb = make_batch(split.train, BATCH, SEQ)
        _host_one_hot_into(in_host, mb.inputs, BATCH)
        _host_one_hot_into(tgt_host, mb.targets, BATCH)
        var l = Float64(trainer.train_step(in_host, tgt_host))

        if (it + 1) % EVAL_INTERVAL == 0 or it == 0 or (it + 1) == TOTAL_ITERS:
            var vtot: Float64 = 0.0
            for vb in range(N_VAL_BATCHES):
                var vptr = val_in_p + vb * BATCH * IN_DIM
                trainer.predict(vptr, out_host)
                vtot += _seq_ce_host(
                    out_host, val_batch.targets, vb * BATCH * SEQ
                )
            var v = vtot / Float64(N_VAL_BATCHES)
            if it == 0:
                first_val = v
            last_val = v
            print(
                "  iter " + String(it + 1) + "/" + String(TOTAL_ITERS)
                + "  train=" + String(Float32(l))
                + "  val=" + String(Float32(v))
            )

    var t1 = perf_counter_ns()
    print("  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s")

    print("\n── Final ──")
    print("  first_val=" + String(Float32(first_val))
          + "  last_val=" + String(Float32(last_val)))
    in_host.free()
    tgt_host.free()
    out_host.free()
    val_in_host.free()

    print("=" * 70)
    if last_val < first_val - 0.1:
        print("PASS — nn2 GPT val loss dropped > 0.1 nats on real data")
    else:
        raise Error(
            "GPT did not improve: first=" + String(first_val)
            + " last=" + String(last_val)
        )
