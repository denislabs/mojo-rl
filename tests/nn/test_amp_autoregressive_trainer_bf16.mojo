"""AMP AutoregressiveTrainer bf16 compile + run smoke (GPU).

Proves the autoregressive trainer's bf16-flow AMP path (MADT = NET.ACT_DT)
end-to-end: a bf16 net (every leaf bf16 → Sequential.ACT_DT == bfloat16) is
driven through make_from → val_loss → fit → val_top1 → generate. The trainer
casts only at the boundaries — host one-hot built directly at bf16 for the net
input, bf16 logits → fp32 for the SeqCE loss, fp32 grad → bf16 for net.vjp.

A degenerate `Sequence[Linear]` "LM" is enough to exercise the plumbing (it is
NOT a real language model — no causal structure). The point is the bf16 path
COMPILES + RUNS through the trainer. Numerics are garbage on Apple (Metal bf16
linalg is broken — the same toolchain bug the leaf tests document); the real
accuracy validation is a bf16 GPT on NVIDIA (which needs the GPT model
parametrized by ADT — a separate task).

Run: pixi run -e apple mojo run -I . tests/nn/test_amp_autoregressive_trainer_bf16.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CharTokenizer, train_val_split
from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.autoregressive_trainer import AutoregressiveTrainer
from mojo_rl.nn.core.initializer import Normal


comptime VOCAB = 4  # "abcd"
comptime SEQ = 8
comptime BATCH = 8
comptime IO = SEQ * VOCAB  # one-hot in = logits out = SEQ·VOCAB
comptime BF16 = DType.bfloat16

# bf16-flow net: the single leaf is bf16, so Net.ACT_DT == bfloat16 and the
# trainer runs its MADT (bf16) path.
comptime Net = Sequential[Linear[IO, IO, BF16], Linear[IO, IO, BF16]]
comptime AR = AutoregressiveTrainer[Net, Adam, VOCAB, SEQ, BATCH, target="gpu"]


def main() raises:
    print("=" * 60)
    print("AMP AutoregressiveTrainer bf16 compile+run smoke")
    print("=" * 60)
    comptime assert Net.ACT_DT == BF16, "net must flow at bf16"

    var text = String("")
    for _ in range(256):
        text += "abcd"
    var tok = CharTokenizer(text)
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)

    var ctx = DeviceContext()
    var net = Net.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var optim = Adam(lr=Scalar[DT](3e-3))
    var artr = AR.make_from(
        net^, optim^, tok^, split^, ctx,
        Scalar[DT](3e-3), 2, 10, 0.1, Scalar[DT](1.0),
    )
    ctx.synchronize()

    # Each step exercises: one-hot@bf16 input, net.forward (bf16), logits→fp32,
    # SeqCE (fp32), grad fp32→bf16, net.vjp (bf16), clip, step. Just must RUN.
    var v0 = artr.val_loss(BATCH)
    print("  val_loss start =", v0)
    assert_true(v0 == v0, "start val finite (eval bf16 path ran)")

    var vf = artr.fit(eval_every=5, n_val_windows=BATCH)
    print("  val_loss final =", vf)
    assert_true(vf == vf, "final val finite (train bf16 path ran end-to-end)")

    var acc = artr.val_top1(BATCH)
    print("  val top-1 =", acc, "(numerics garbage on Apple Metal bf16)")

    var gen = artr.generate("a", 12, temperature=0.0)
    print("  generate('a', 12) =", repr(gen))
    assert_true(gen.byte_length() == 12, "generate bf16 path returned length")

    print("ALL PASSED (bf16 AMP autoregressive trainer ran end-to-end)")
