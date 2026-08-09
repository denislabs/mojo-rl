"""AMP bf16 GPT compile + run smoke (GPU): a fully-bf16 weight-tied char-GPT
driven through the AutoregressiveTrainer end-to-end.

This is the capstone of the bf16 GPT work: `GPTDropTied[..., DType.bfloat16]`
expands to a Sequential whose EVERY leaf is bf16 (Embedding/BiasAdd/Dropout/
LayerNorm/Linear/Attention/QKVToMajor/GELU/TiedLinear), so `Net.ACT_DT ==
bfloat16` and the (AMP-wired) AutoregressiveTrainer runs its bf16-flow path:
activations stored bf16, fp32 SeqCE loss + fp32 master weights. The tie/scale
helpers take the same `ADT=bf16` so they wire the fp32-master tie correctly.

⚠️ Apple: `linalg.matmul` mis-computes bf16 GEMMs at realistic dims (the Metal
toolchain bug), so the bf16 attention/linear GEMMs here may be inaccurate — this
test asserts the bf16 GPT COMPILES + RUNS end-to-end (finite loss, generate
returns text), NOT accuracy. Real bf16 GPT accuracy (~matching fp32) is the
NVIDIA gate (cutlass bf16), via the bf16 tinyshakespeare example.

Run: pixi run -e apple mojo run -I . tests/nn/test_amp_gpt_bf16.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CharTokenizer, train_val_split
from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.gpt import (
    GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie,
)
from mojo_rl.nn.optimizer.adam import AdamW
from mojo_rl.nn.training.autoregressive_trainer import AutoregressiveTrainer
from mojo_rl.nn.core.initializer import Normal


comptime VOCAB = 4  # "abcd"
comptime SEQ = 8
comptime EMBED = 16
comptime HEADS = 2
comptime LAYERS = 1
comptime FF_MULT = 2
comptime BATCH = 8
comptime DROPOUT_P: Float64 = 0.0
comptime USE_MAX = True
comptime SEED = UInt64(0xC0FFEE)
comptime BF16 = DType.bfloat16

# Fully-bf16 GPT: the trailing ADT=BF16 threads bf16 to every leaf.
comptime GPT_MODEL = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P, SEED, USE_MAX,
    BF16,
]
comptime GPT_AR = AutoregressiveTrainer[
    GPT_MODEL, AdamW, VOCAB, SEQ, BATCH, target="gpu"
]


def main() raises:
    print("=" * 60)
    print("AMP bf16 GPT compile+run smoke (tiny weight-tied char-GPT)")
    print("=" * 60)
    comptime assert GPT_MODEL.ACT_DT == BF16, "GPT must flow at bf16"

    var text = String("")
    for _ in range(256):
        text += "abcd"
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error("vocab mismatch: " + String(tok.vocab_size))
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)

    var ctx = DeviceContext()
    var net = GPT_MODEL.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var optim = AdamW(
        lr=Scalar[DT](3e-3), beta2=Scalar[DT](0.99), wd=Scalar[DT](0.0)
    )
    var artr = GPT_AR.make_from(
        net^, optim^, tok^, split^, ctx,
        Scalar[DT](3e-3), 5, 40, 0.1, Scalar[DT](1.0),
    )
    # Scaled-init + tie wiring with the SAME bf16 ADT (fp32-master tie).
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED, USE_MAX, BF16,
    ](artr.net, Optional(ctx))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED, USE_MAX, BF16,
    ](artr.net)
    ctx.synchronize()

    var v0 = artr.val_loss(BATCH)
    print("  val_loss start =", v0)
    assert_true(v0 == v0, "start val finite (bf16 GPT eval ran)")

    var vf = artr.fit(eval_every=20, n_val_windows=BATCH)
    print("  val_loss final =", vf)
    assert_true(vf == vf, "final val finite (bf16 GPT trained end-to-end)")

    var acc = artr.val_top1(BATCH)
    print("  val top-1 =", acc, "(accuracy is NVIDIA-gated; Metal bf16 may skew)")

    var gen = artr.generate("a", 12, temperature=0.0)
    print("  generate('a', 12) =", repr(gen))
    assert_true(gen.byte_length() == 12, "bf16 GPT generate returned length")

    print("ALL PASSED (bf16 GPT ran end-to-end through AutoregressiveTrainer)")
