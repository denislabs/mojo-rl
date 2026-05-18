"""End-to-end MNIST MLP — Phase 1 exit criterion.

Architecture: Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10).
Loss:         CrossEntropyLoss[10] on one-hot targets.
Optimizer:    Adam (lr=0.001, default betas/eps).
Data:         MNIST via existing `mojo_rl.nn.datasets.MNIST` (pre-normalized to [0, 1]).

The design doc's exit criterion is >=97% test accuracy after 5 epochs on
the GPU example. CPU is slower per step, so we set EPOCHS=3 and a
relaxed bar of >=95% — enough to demonstrate the framework trains on
real data. If iteration is fast enough, we can crank EPOCHS up later.

Network composition uses nested `Sequential2` because variadic
`Sequential[*L]` is a Phase 1.x follow-up. The nesting reads as a
right-fold:
  Sequential2(
    Sequential2(
      Sequential2(
        Sequential2(lin0, ReLU),
        lin1
      ),
      ReLU
    ),
    lin2
  )
"""

from std.math import sqrt as fsqrt
from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from std.time import perf_counter_ns
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn.datasets import MNIST   # reuse the existing loader
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential2
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam


# ──────────────────────────────────────────────────────────────────────────
# Initializer helpers (Phase 1 has no Initializer abstraction yet).
# ──────────────────────────────────────────────────────────────────────────

def he_init_uniform(mut data: List[Scalar[DT]], n_elems: Int, fan_in: Int):
    """He-style uniform init: U[-bound, bound], bound = sqrt(6/fan_in).
    Right shape for ReLU networks; equivalent to PyTorch's
    `kaiming_uniform_` with default args."""
    var bound = fsqrt(6.0 / Float64(fan_in))
    for i in range(n_elems):
        var r = random_float64()     # [0, 1)
        data[i] = Scalar[DT]((r * 2.0 - 1.0) * bound)


def fill_const(mut data: List[Scalar[DT]], n_elems: Int, val: Scalar[DT]):
    for i in range(n_elems):
        data[i] = val


# ──────────────────────────────────────────────────────────────────────────
# Training run
# ──────────────────────────────────────────────────────────────────────────

def test_mnist_mlp() raises:
    comptime IN_DIM = 784
    comptime H1 = 256
    comptime H2 = 128
    comptime N_CLASSES = 10
    comptime BATCH = 100      # divides 60000 and 10000 cleanly
    comptime N_EPOCHS = 5
    comptime LR: Scalar[DT] = 0.001
    comptime TARGET_ACC: Float64 = 0.97   # Design doc's exit criterion.

    seed(42)

    print("loading MNIST...")
    var ds = MNIST()
    var N_TRAIN = MNIST.N_TRAIN
    var N_TEST = MNIST.N_TEST
    comptime N_BATCHES_TRAIN = 60000 // BATCH    # 600
    comptime N_BATCHES_TEST  = 10000 // BATCH    # 100

    # ── Build + init the network ──────────────────────────────────────
    print("initializing network...")
    var lin0 = Linear[IN_DIM, H1]()
    he_init_uniform(lin0.weight, IN_DIM * H1, IN_DIM)
    fill_const(lin0.bias, H1, 0.01)

    var lin1 = Linear[H1, H2]()
    he_init_uniform(lin1.weight, H1 * H2, H1)
    fill_const(lin1.bias, H2, 0.01)

    var lin2 = Linear[H2, N_CLASSES]()
    he_init_uniform(lin2.weight, H2 * N_CLASSES, H2)
    # final-layer bias defaults to 0

    var net = Sequential2(
        Sequential2(
            Sequential2(
                Sequential2(lin0^, ReLU[H1]()),
                lin1^
            ),
            ReLU[H2]()
        ),
        lin2^
    )

    var loss_fn = CrossEntropyLoss[N_CLASSES]()
    var optim = Adam.make(net, lr=LR)

    # ── Per-batch I/O buffers (reused across steps) ───────────────────
    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N_CLASSES)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N_CLASSES)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N_CLASSES)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var input    = TileTensor(in_buf,  row_major[BATCH, IN_DIM]())
    var targets  = TileTensor(tgt_buf, row_major[BATCH, N_CLASSES]())
    var output   = TileTensor(out_buf, row_major[BATCH, N_CLASSES]())
    var grad_out = TileTensor(go_buf,  row_major[BATCH, N_CLASSES]())
    var grad_in  = TileTensor(gi_buf,  row_major[BATCH, IN_DIM]())

    # ── Train ─────────────────────────────────────────────────────────
    var final_acc: Float64 = 0.0
    for epoch in range(N_EPOCHS):
        var t0 = perf_counter_ns()
        var epoch_loss: Scalar[DT] = 0.0

        for batch_idx in range(N_BATCHES_TRAIN):
            # Load batch into I/O buffers.
            for b in range(BATCH):
                var sample_idx = batch_idx * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.train_images[sample_idx * IN_DIM + px]
                for c in range(N_CLASSES):
                    tgt_buf[b * N_CLASSES + c] = 0.0
                tgt_buf[b * N_CLASSES + Int(ds.train_labels[sample_idx])] = 1.0

            optim.zero_grad(net)
            net.forward[BATCH](input, output)
            var L = loss_fn.forward[BATCH](output, targets)
            epoch_loss += L
            loss_fn.backward[BATCH](targets, grad_out)
            net.backward[BATCH](grad_out, grad_in)
            optim.step(net)

        var t_train = perf_counter_ns()
        var train_s = Float64(t_train - t0) / 1e9

        # ── Eval on full test set ─────────────────────────────────────
        var n_correct: Int = 0
        for batch_idx in range(N_BATCHES_TEST):
            for b in range(BATCH):
                var sample_idx = batch_idx * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.test_images[sample_idx * IN_DIM + px]
            net.forward[BATCH](input, output)
            for b in range(BATCH):
                var best_c: Int = 0
                var best_v: Scalar[DT] = output[b, 0]
                for c in range(1, N_CLASSES):
                    if output[b, c] > best_v:
                        best_v = output[b, c]
                        best_c = c
                if best_c == Int(ds.test_labels[batch_idx * BATCH + b]):
                    n_correct += 1
        var t_eval = perf_counter_ns()
        var eval_s = Float64(t_eval - t_train) / 1e9

        var acc = Float64(n_correct) / Float64(N_TEST)
        final_acc = acc
        var avg_loss = epoch_loss / Scalar[DT](N_BATCHES_TRAIN)
        print("epoch " + String(epoch)
            + " | train_loss=" + String(avg_loss)
            + " | test_acc=" + String(acc * 100.0) + "%"
            + " | train=" + String(train_s) + "s"
            + " | eval=" + String(eval_s) + "s")

    print("")
    print("final test accuracy: " + String(final_acc * 100.0) + "%")
    assert_true(final_acc >= TARGET_ACC,
        "Expected test acc >= " + String(TARGET_ACC * 100.0)
        + "%, got " + String(final_acc * 100.0) + "%")

    in_buf.free()
    tgt_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_mnist_mlp PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 MNIST MLP end-to-end (CPU, Phase 1 exit criterion)")
    print("=" * 60)
    test_mnist_mlp()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
