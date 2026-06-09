"""Char-LSTM on TinyShakespeare — nn2 GPU, via `AutoregressiveTrainer`.

The LSTM analog of the GPT example (Karpathy's char-rnn). The recurrence
is wrapped behind the standard `Module` interface by `LSTMSeq` (nn2's
`nn.LSTM`-over-a-window), so the whole model is a plain `Module`:

    Sequential[LSTMSeq[VOCAB, HIDDEN, SEQ], Tokenwise[SEQ, Linear[HIDDEN, VOCAB]]]

Because that conforms to `Module.forward`/`vjp`, training goes through the
SAME `AutoregressiveTrainer` as the GPT — `fit()` packages the run
(streaming random-window sampling, cosine LR, the packaged `train_step`,
periodic per-token val CE), and `val_loss`/`val_top1`/`generate` cover
eval + sampling. There is no hand-written BPTT loop, no second optimizer,
and no bespoke per-token CE plumbing here anymore — `LSTMSeq.vjp` runs the
BPTT internally and one `Adam` trains the whole net (`max_grad_norm` gives
a per-net global grad-norm clip).

`LSTMSeq`'s GPU path is an unfused RNN (SEQ sequential per-step kernel
launches), so keep SEQ modest. Run:
    pixi run -e nvidia mojo run -I . examples/nn2/lstm/lstm_tinyshakespeare_training.mojo
"""

from std.random import seed
from std.math import log, exp
from std.gpu.host import DeviceContext

from mojo_rl.nn2.datasets import CharTokenizer, load_text, train_val_split
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.lstm_seq import LSTMSeq
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators import Sequential, Tokenwise
from mojo_rl.nn2.loss import SequenceCrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer, AutoregressiveTrainer
from mojo_rl.nn2.initializer import Xavier


comptime VOCAB = 65
comptime HIDDEN = 128
comptime SEQ = 32
comptime BATCH = 16

comptime LR: Scalar[DT] = 0.003
comptime CLIP_NORM: Scalar[DT] = 5.0
comptime TOTAL_ITERS = 1000
comptime WARMUP_ITERS = 50
comptime EVAL_EVERY = 200
comptime MIN_LR_SCALE: Float64 = 0.1
comptime N_VAL_WINDOWS = 64

comptime NET = Sequential[
    LSTMSeq[VOCAB, HIDDEN, SEQ],
    Tokenwise[SEQ, Linear[HIDDEN, VOCAB]],
]
comptime LOSS = SequenceCrossEntropyLoss[SEQ, VOCAB]
comptime TRAINER = Trainer[NET, Adam, LOSS, BATCH, target="gpu"]
comptime AR = AutoregressiveTrainer[NET, Adam, VOCAB, SEQ, BATCH, target="gpu"]


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 TinyShakespeare char-LSTM — GPU (AutoregressiveTrainer)")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " hidden=" + String(HIDDEN)
        + " seq=" + String(SEQ) + " batch=" + String(BATCH)
        + " lr=" + String(LR) + " clip=" + String(CLIP_NORM)
        + " total_iters=" + String(TOTAL_ITERS)
    )

    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error("vocab mismatch: got " + String(tok.vocab_size))
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  tokens=" + String(len(ids)) + " train=" + String(len(split.train))
        + " val=" + String(len(split.val))
    )

    # ── Build model + optimizer + the generic Trainer ──
    var ctx = DeviceContext()
    print("[init] building LSTM on GPU...")
    var net = NET.make["gpu", INIT=Xavier](ctx)
    var loss_fn = LOSS.make["gpu"](ctx)
    var optim = Adam.make["gpu", M = type_of(net)](net, ctx)
    optim.lr = LR
    optim.max_grad_norm = CLIP_NORM  # per-net global grad-norm clip
    var trainer = TRAINER.make_from(net^, optim^, loss_fn^, ctx)

    # ── Wrap in the autoregressive driver (owns corpus + sampling + LR) ──
    var artr = AR.make_from(
        trainer^, tok^, split^, LR, WARMUP_ITERS, TOTAL_ITERS, MIN_LR_SCALE
    )

    var val_init = artr.val_loss(N_VAL_WINDOWS)
    print(
        "\n[iter 0] val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    print("\n── Training ──")
    var val_final = artr.fit(
        eval_every=EVAL_EVERY, n_val_windows=N_VAL_WINDOWS
    )

    print(
        "\n[final] val_loss=" + String(val_final)
        + " (start " + String(val_init) + ")"
    )
    if val_final < val_init - 0.5:
        print("  PASS: validation loss dropped > 0.5 nats")
    else:
        print("  WARN: little improvement — raise TOTAL_ITERS or tune LR")

    var acc = artr.val_top1(N_VAL_WINDOWS)
    print(
        "[diagnostic] val per-token top-1=" + String(acc * 100.0)[byte=:5]
        + "%  (random ≈ " + String(100.0 / Float64(VOCAB))[byte=:4]
        + "%, from loss≈" + String(val_final)[byte=:5]
        + " expect ~" + String(exp(-val_final) * 100.0)[byte=:5] + "%)"
    )

    var prompt = String("ROMEO:")
    print("\n[sample] greedy (T=0.0):")
    print(prompt + artr.generate(prompt, 200, temperature=0.0))
    print("\n[sample] temperature (T=0.8):")
    print(prompt + artr.generate(prompt, 200, temperature=0.8))
    print("=" * 70)
