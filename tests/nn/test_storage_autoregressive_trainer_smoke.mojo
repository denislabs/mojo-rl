"""Storage `AutoregressiveTrainer` smoke (GPU): a tiny weight-tied char-GPT on a
trivially-learnable periodic corpus (`abcd`…). Proves the self-contained storage
driver end-to-end.

  build net + AdamW → make_from (engages opt.adopt arena) → scaled-init + tie
  surgery on artr.net → val_loss (start) → fit (per-step forward/SeqCE/vjp/clip/
  step + periodic eval) → val_loss drops → val_top1 → generate returns text.

Periodic data with SEQ ≥ 2 periods is learnable in a handful of iters, so the
val CE must fall well below the random ≈ ln(VOCAB) baseline.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_storage_autoregressive_trainer_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CharTokenizer, train_val_split
from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.gpt import (
    GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie,
)
from mojo_rl.nn.optimizer.adam import AdamW
from mojo_rl.nn.training.autoregressive_trainer import (
    AutoregressiveTrainer,
)
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

comptime GPT_MODEL = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P, SEED, USE_MAX
]
comptime GPT_AR = AutoregressiveTrainer[
    GPT_MODEL, AdamW, VOCAB, SEQ, BATCH, target="gpu"
]


def main() raises:
    print("=" * 60)
    print("nn.storage AutoregressiveTrainer smoke (tiny periodic GPT)")
    print("=" * 60)

    # Periodic corpus: vocab is exactly {a,b,c,d}.
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
    var optim = AdamW(lr=Scalar[DT](3e-3), beta2=Scalar[DT](0.99), wd=Scalar[DT](0.0))
    var artr = GPT_AR.make_from(
        net^, optim^, tok^, split^, ctx,
        Scalar[DT](3e-3), 5, 120, 0.1, Scalar[DT](1.0),
    )
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED, USE_MAX,
    ](artr.net, Optional(ctx))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED, USE_MAX,
    ](artr.net)
    ctx.synchronize()

    var v0 = artr.val_loss(BATCH)
    print("  val_loss start =", v0)
    assert_true(v0 == v0, "start val finite")

    var vf = artr.fit(eval_every=40, n_val_windows=BATCH)
    print("  val_loss final =", vf)
    assert_true(vf == vf, "final val finite")
    assert_true(vf < v0, "val loss decreased")
    # Periodic data → CE should fall well under random ≈ ln(4) ≈ 1.386.
    assert_true(Float64(vf) < 1.0, "val CE below random baseline")

    var acc = artr.val_top1(BATCH)
    print("  val top-1 =", acc)
    assert_true(acc > 0.5, "top-1 above chance")

    var gen = artr.generate("a", 12, temperature=0.0)
    print("  generate('a', 12) =", repr(gen))
    assert_true(len(gen) == 12, "generated requested length")

    print("ALL PASSED")
