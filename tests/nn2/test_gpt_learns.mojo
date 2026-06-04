"""GPT learning lighthouse (Wave D) — synthetic next-token task.

Proves the full GPT stack (token Embedding → pos BiasAdd → causal
TransformerBlocks → LayerNorm → LM head) trains end-to-end through the
nn2 Trainer + Adam + SequenceCrossEntropyLoss (per-token CE).

Task: each sample is an arithmetic sequence mod VOCAB with a per-sample
phase — token at position t is (phase+t)%VOCAB, and the target (next
token) is (phase+t+1)%VOCAB. A correct causal GPT learns "predict
current+1" and drives per-token loss → 0 / next-token accuracy → 100%.

Run: pixi run mojo run -I . tests/nn2/test_gpt_learns.mojo
Docs: docs/NN2_TRANSFORMER_PORT.md Phase 1 Wave D.
"""

from std.memory import alloc
from std.random import seed
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.composites import GPT
from mojo_rl.nn2.loss import SequenceCrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


comptime VOCAB = 8
comptime SEQ = 4
comptime EMBED = 16
comptime HEADS = 2
comptime LAYERS = 2
comptime BATCH = 16
comptime STEPS = 120
comptime LR: Scalar[DT] = 0.01
comptime IN_DIM = SEQ * VOCAB
comptime OUT_DIM = SEQ * VOCAB


def _make_dataset(
    in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    for s in range(BATCH):
        var p = s % VOCAB
        for t in range(SEQ):
            var cur = (p + t) % VOCAB
            var nxt = (p + t + 1) % VOCAB
            for k in range(VOCAB):
                in_buf[s * IN_DIM + t * VOCAB + k] = 0.0
                tgt_buf[s * OUT_DIM + t * VOCAB + k] = 0.0
            in_buf[s * IN_DIM + t * VOCAB + cur] = 1.0
            tgt_buf[s * OUT_DIM + t * VOCAB + nxt] = 1.0


def main() raises:
    seed(11)
    print("=" * 70)
    print("GPT learning lighthouse (Wave D) — synthetic next-token task")
    print("=" * 70)

    var net = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS].make[
        target="cpu", INIT=Kaiming
    ]()
    var loss_fn = SequenceCrossEntropyLoss[SEQ, VOCAB].make["cpu"]()
    var optim = Adam.make["cpu", M = type_of(net)](net)
    optim.lr = LR
    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH, target="cpu",
    ].make_from(net^, optim^, loss_fn^)

    var in_buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * IN_DIM)
    )
    var tgt_buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * OUT_DIM)
    )
    var out_buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * OUT_DIM)
    )
    _make_dataset(in_buf, tgt_buf)

    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0
    for step in range(STEPS):
        var l = Float64(trainer.train_step(in_buf, tgt_buf))
        if step == 0:
            first_loss = l
        last_loss = l
        if step % 20 == 0 or step == STEPS - 1:
            print("  step " + String(step) + "  loss=" + String(l)[byte=:7])

    # Per-token next-token accuracy.
    trainer.predict(in_buf, out_buf)
    var correct = 0
    var total = 0
    for s in range(BATCH):
        var p = s % VOCAB
        for t in range(SEQ):
            var nxt = (p + t + 1) % VOCAB
            var base = s * OUT_DIM + t * VOCAB
            var best_k = 0
            var best_v = out_buf[base]
            for k in range(1, VOCAB):
                if out_buf[base + k] > best_v:
                    best_v = out_buf[base + k]
                    best_k = k
            if best_k == nxt:
                correct += 1
            total += 1
    var acc = Float64(correct) / Float64(total)

    print("-" * 70)
    print(
        "  first_loss=" + String(first_loss)[byte=:7]
        + "  last_loss=" + String(last_loss)[byte=:7]
        + "  next_token_acc=" + String(acc * 100.0)[byte=:6] + "%"
    )
    in_buf.free()
    tgt_buf.free()
    out_buf.free()

    print("=" * 70)
    if last_loss < first_loss * 0.3 and acc >= 0.95:
        print("PASS — GPT learns the synthetic next-token task")
    else:
        raise Error(
            "GPT failed to learn: first=" + String(first_loss)
            + " last=" + String(last_loss) + " acc=" + String(acc)
        )
