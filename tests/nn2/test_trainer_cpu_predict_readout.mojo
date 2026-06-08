"""Regression: Trainer CPU `predict` / `train_cpu` flat-readout.

Pins the bug where `predict`'s CPU readout indexed the 2-D `output` view
with a single int (`output[k]`) — which selects row `k`, column 0, NOT the
flat element `k`. The effect: `result` got column-0 of consecutive samples
instead of the OUT_DIM class logits of one sample, so argmax was garbage and
test accuracy collapsed to chance (the MNIST CPU example dropped to ~10%
while still training fine). The fix reads `self.output_buf[k]` flat.

Two checks, both dataset-free on synthetic separable data:

  * `test_predict_readout`  — train via per-step `train_step`, then `predict`
    and assert per-sample argmax accuracy is ~100%. With the column-0 bug
    this collapses to ~chance.
  * `test_train_cpu`        — the whole-dataset `train_cpu[N_TRAIN, N_TEST]`
    loop reaches high `epoch_test_top1` (its eval shares the flat-read path).

Run:
    pixi run mojo run -I . tests/nn2/test_trainer_cpu_predict_readout.mojo
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


comptime IN_DIM = 8
comptime H = 32
comptime N_CLASSES = 4
comptime BATCH = 16

comptime MLP = Sequential[
    Linear[IN_DIM, H], ReLU[H], Linear[H, N_CLASSES],
]
comptime TRAINER = Trainer[
    MLP, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="cpu"
]


def _fill_batch(
    mut inp: List[Scalar[DT]],
    mut tgt: List[Scalar[DT]],
    base: Int = 0,
):
    """Sample `b` belongs to class `(base + b) % N_CLASSES`, encoded as a
    one-hot over the first N_CLASSES input features (trivially separable, so
    a correct readout reaches ~100%)."""
    for b in range(BATCH):
        var cls = (base + b) % N_CLASSES
        for d in range(IN_DIM):
            inp[b * IN_DIM + d] = 0.0
        inp[b * IN_DIM + cls] = 1.0
        for c in range(N_CLASSES):
            tgt[b * N_CLASSES + c] = 0.0
        tgt[b * N_CLASSES + cls] = 1.0


def _argmax_accuracy(out_flat: List[Scalar[DT]], base: Int) -> Float64:
    var correct = 0
    for b in range(BATCH):
        var best_c = 0
        var best_v = out_flat[b * N_CLASSES + 0]
        for c in range(1, N_CLASSES):
            var v = out_flat[b * N_CLASSES + c]
            if v > best_v:
                best_v = v
                best_c = c
        if best_c == (base + b) % N_CLASSES:
            correct += 1
    return Float64(correct) / Float64(BATCH)


def test_predict_readout() raises:
    print("--- predict() flat readout ---")
    seed(7)
    var trainer = TRAINER.make[Kaiming]()

    var inp = List[Scalar[DT]](length=BATCH * IN_DIM, fill=0.0)
    var tgt = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=0.0)
    var out = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=0.0)
    _fill_batch(inp, tgt)

    var first = trainer.train_step(inp, tgt)
    var last: Scalar[DT] = first
    for _ in range(300):
        last = trainer.train_step(inp, tgt)
    print("  loss", first, "->", last)
    assert_true(last < first, "trainer should reduce the loss")

    trainer.predict(inp, out)
    var acc = _argmax_accuracy(out, 0)
    print("  predict argmax accuracy:", acc)
    # The trivially separable task is memorized; a correct flat readout hits
    # 100%. The column-0 bug returns col-0 of consecutive samples → ~chance.
    assert_true(
        acc >= 0.95,
        "predict readout should reach ~100% (col-0 bug → chance ~0.25); got "
        + String(acc),
    )

    # Input-sensitivity: a different batch (shifted classes) must produce a
    # different output vector for sample 0 (the col-0 bug leaves rows 1..n-1
    # of a partially-written batch stale, masking input changes).
    var inp2 = List[Scalar[DT]](length=BATCH * IN_DIM, fill=0.0)
    var tgt2 = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=0.0)
    var out2 = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=0.0)
    _fill_batch(inp2, tgt2, base=1)
    trainer.predict(inp2, out2)
    var acc2 = _argmax_accuracy(out2, 1)
    print("  predict argmax accuracy (shifted):", acc2)
    assert_true(acc2 >= 0.95, "shifted-batch readout should also be correct")
    print("  ok")


def test_train_cpu() raises:
    print("--- train_cpu[N_TRAIN, N_TEST] whole-dataset loop ---")
    seed(7)
    comptime N_TRAIN = BATCH * 4
    comptime N_TEST = BATCH * 2
    var trainer = TRAINER.make[Kaiming]()

    var train_x = List[Scalar[DT]](length=N_TRAIN * IN_DIM, fill=0.0)
    var train_y = List[Scalar[DT]](length=N_TRAIN * N_CLASSES, fill=0.0)
    var test_x = List[Scalar[DT]](length=N_TEST * IN_DIM, fill=0.0)
    var test_labels = List[Int32](length=N_TEST, fill=0)
    for i in range(N_TRAIN):
        var cls = i % N_CLASSES
        train_x[i * IN_DIM + cls] = 1.0
        train_y[i * N_CLASSES + cls] = 1.0
    for i in range(N_TEST):
        var cls = i % N_CLASSES
        test_x[i * IN_DIM + cls] = 1.0
        test_labels[i] = Int32(cls)

    var result = trainer.train_cpu[N_TRAIN, N_TEST](
        train_x, train_y, test_x, test_labels, epochs=80, print_progress=False
    )
    var best: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best:
            best = a
    print("  best test_top1:", best)
    assert_true(
        best >= 0.95,
        "train_cpu should reach ~100% on a separable set; got " + String(best),
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Trainer CPU predict / train_cpu flat-readout regression")
    print("=" * 70)
    test_predict_readout()
    test_train_cpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
