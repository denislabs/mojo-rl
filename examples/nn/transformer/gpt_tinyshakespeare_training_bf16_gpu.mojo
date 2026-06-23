"""TinyShakespeare char-GPT training — bf16 AMP (bf16-FLOW), via
`AutoregressiveTrainer`.

The bf16 twin of gpt_tinyshakespeare_training_gpu.mojo: identical nanoGPT recipe
and config, but the model is `GPTDropTied[..., DType.bfloat16]` — every leaf
(Embedding/BiasAdd/Dropout/LayerNorm/Linear/Attention/QKVToMajor/GELU/TiedLinear)
is bf16, so `GPT_MODEL.ACT_DT == bfloat16` and the AutoregressiveTrainer runs its
bf16-flow path: ACTIVATIONS are STORED + flow at bf16 (≈ half the activation
memory), while the SeqCE loss, softmax, LayerNorm stats, and the master
weights/grads/optimizer state stay fp32. The tie/scale helpers take the same
`BF16` ADT (they wire the fp32-MASTER embedding table — tying is dtype-agnostic).

This is where the bf16 throughput win actually shows: a GPT is dominated by WIDE
GEMMs (QKV/attention/FFN at EMBED=384, FF=1536), well past the size where bf16
tensor cores beat fp32 — unlike a small MLP.

⚠️ NVIDIA-only for real numerics: Apple Metal's `linalg.matmul` mis-computes bf16
GEMMs at realistic dims (a known toolchain bug), so on Apple this runs but the
GEMM-heavy parts (attention/FFN/head) are inaccurate. On NVIDIA (cutlass bf16) it
should reach ~the same val loss as fp32 at lower activation memory and (for this
wide model) competitive-or-better throughput.

Run on NVIDIA:
    pixi run -e nvidia mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training_bf16_gpu.mojo
"""

from std.random import seed
from std.math import log, exp
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CharTokenizer, load_text, train_val_split
from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.gpt import GPTDropTied
from mojo_rl.nn.models.gpt import gpt_scale_residual_proj, gpt_wire_tie
from mojo_rl.nn.optimizer.adam import AdamW
from mojo_rl.nn.training.autoregressive_trainer import (
    AutoregressiveTrainer,
)
from mojo_rl.nn.core.initializer import Normal


# ── Full nanoGPT-class config (NVIDIA) ──
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

comptime USE_MAX_ATTN = True
comptime DROPOUT_P: Float64 = 0.2
comptime SCALED_INIT = True
comptime GRAD_CLIP: Scalar[DT] = 1.0

# bf16-flow: the trailing ADT=BF16 threads bf16 to every leaf of the GPT.
comptime BF16 = DType.bfloat16
comptime GPT_MODEL = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
    UInt64(0xC0FFEE), USE_MAX_ATTN, BF16,
]
comptime GPT_AR = AutoregressiveTrainer[
    GPT_MODEL, AdamW, VOCAB, SEQ, BATCH, target="gpu"
]


def main() raises:
    comptime assert GPT_MODEL.ACT_DT == BF16, "GPT must flow at bf16"
    seed(42)
    print("=" * 70)
    print("TinyShakespeare GPT training — bf16 AMP (AutoregressiveTrainer)")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS) + " batch=" + String(BATCH)
        + " | ACT_DT=bf16 (activations stored bf16; loss+master fp32)"
    )
    print(
        "  base_lr=" + String(BASE_LR) + " wd=" + String(WD)
        + " dropout_p=" + String(DROPOUT_P)
        + " | tie=structural(TiedLinear) scaled_init=" + String(SCALED_INIT)
        + " grad_clip=" + String(GRAD_CLIP)
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

    # ── Build model + optimizer ──
    var ctx = DeviceContext()
    print("[init] building bf16 nn GPT on GPU...")
    var net = GPT_MODEL.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var optim = AdamW(lr=BASE_LR, beta2=BETA2, wd=WD)

    var artr = GPT_AR.make_from(
        net^, optim^, tok^, split^, ctx,
        BASE_LR, WARMUP_ITERS, TOTAL_ITERS, MIN_LR_SCALE, GRAD_CLIP,
    )

    # ── Model-construction surgery (same bf16 ADT; fp32-master tie) ──
    comptime if SCALED_INIT:
        gpt_scale_residual_proj[
            "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
            UInt64(0xC0FFEE), USE_MAX_ATTN, BF16,
        ](artr.net, Optional(ctx))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        UInt64(0xC0FFEE), USE_MAX_ATTN, BF16,
    ](artr.net)
    ctx.synchronize()

    var val_init = artr.val_loss(N_VAL_WINDOWS)
    print(
        "\n[iter 0] val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    print("\n── Training (bf16-flow) ──")
    var t0 = perf_counter_ns()
    var val_final = artr.fit(
        eval_every=EVAL_INTERVAL, n_val_windows=N_VAL_WINDOWS
    )
    var t1 = perf_counter_ns()
    print("\n  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s")

    print("\n[final] val_loss=" + String(val_final) + " (start " + String(val_init) + ")")
    if val_final < val_init - 0.1:
        print("  PASS: val loss decreased by > 0.1 nats")
    else:
        print("  WARN: val loss did not improve substantially")

    var acc = artr.val_top1(N_VAL_WINDOWS)
    print(
        "[diagnostic] val per-token top-1=" + String(acc * 100.0)[byte=:5]
        + "%  (random ≈ " + String(100.0 / Float64(VOCAB))[byte=:4]
        + "%, from loss≈" + String(val_final)[byte=:5]
        + " expect ~" + String(exp(-val_final) * 100.0)[byte=:5] + "%)"
    )

    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))
    print("\n[sample] greedy (T=0.0):")
    print(prompt + artr.generate(prompt, 200, temperature=0.0))
    print("\n[sample] temperature (T=0.8, no top-k):")
    print(prompt + artr.generate(prompt, 200, temperature=0.8))
    print("\n" + "=" * 70)
