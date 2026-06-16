"""CPU MNIST MLP via the Phase 2.4 architecture-as-type-alias form.

Mirror of `test_mnist_mlp_gpu_alias.mojo` but on CPU. See that file for
explanation of the alias-form construction.
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer import Kaiming


comptime IN_DIM    = 784
comptime H1        = 256
comptime H2        = 128
comptime N_CLASSES = 10
comptime BATCH     = 100
comptime N_EPOCHS  = 5
comptime TARGET_ACC: Float64 = 0.97

comptime MLP = Sequential[
    Linear[IN_DIM, H1],
    ReLU[H1],
    Linear[H1, H2],
    ReLU[H2],
    Linear[H2, N_CLASSES],
]

comptime TRAINER = Trainer[
    MLP,
    Adam,
    CrossEntropyLoss[N_CLASSES],
    BATCH,
    target="cpu",
]


def test_mnist_mlp_cpu_alias() raises:
    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    comptime N_BATCHES_TRAIN = 60000 // BATCH
    comptime N_BATCHES_TEST  = 10000 // BATCH

    print("building trainer via TRAINER.make[Kaiming]()...")
    var trainer = TRAINER.make[Kaiming]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N_CLASSES)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N_CLASSES)

    var final_acc: Float64 = 0.0
    for epoch in range(N_EPOCHS):
        var t0 = perf_counter_ns()
        var epoch_loss: Scalar[DT] = 0.0
        for batch_idx in range(N_BATCHES_TRAIN):
            for b in range(BATCH):
                var sample_idx = batch_idx * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.train_images[sample_idx * IN_DIM + px]
                for c in range(N_CLASSES):
                    tgt_buf[b * N_CLASSES + c] = 0.0
                tgt_buf[b * N_CLASSES + Int(ds.train_labels[sample_idx])] = 1.0
            var L = trainer.train_step(in_buf, tgt_buf)
            epoch_loss += L
        var t_train = perf_counter_ns()
        var train_s = Float64(t_train - t0) / 1e9

        var n_correct: Int = 0
        for batch_idx in range(N_BATCHES_TEST):
            for b in range(BATCH):
                var sample_idx = batch_idx * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.test_images[sample_idx * IN_DIM + px]
            trainer.predict(in_buf, out_buf)
            for b in range(BATCH):
                var best_c: Int = 0
                var best_v: Scalar[DT] = out_buf[b * N_CLASSES + 0]
                for c in range(1, N_CLASSES):
                    var v = out_buf[b * N_CLASSES + c]
                    if v > best_v:
                        best_v = v
                        best_c = c
                if best_c == Int(ds.test_labels[batch_idx * BATCH + b]):
                    n_correct += 1
        var t_eval = perf_counter_ns()
        var eval_s = Float64(t_eval - t_train) / 1e9

        var acc = Float64(n_correct) / 10000.0
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

    in_buf.free(); tgt_buf.free(); out_buf.free()
    print("  test_mnist_mlp_cpu_alias PASSED")


def main() raises:
    print("=" * 60)
    print("nn CPU MNIST MLP — Phase 2.4 alias form")
    print("=" * 60)
    test_mnist_mlp_cpu_alias()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
