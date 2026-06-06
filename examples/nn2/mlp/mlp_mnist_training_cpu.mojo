"""nn2 MLP on MNIST (CPU).

Port of `examples/nn/mlp/mlp_mnist_training_cpu.mojo` to the nn2 framework.

Architecture: Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)
Loss:         CrossEntropyLoss[10] on one-hot targets
Optimizer:    Adam (lr=1e-3)
Data:         MNIST via `mojo_rl.nn2.datasets.MNIST` (framework-agnostic loader)

Run:
    pixi run mojo run -I . examples/nn2/mlp/mlp_mnist_training_cpu.mojo
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


def main() raises:
    comptime IN_DIM = 784
    comptime H1 = 256
    comptime H2 = 128
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 5
    comptime LR: Scalar[DT] = 0.001
    comptime TARGET_ACC: Float64 = 0.97

    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    comptime N_BATCHES_TRAIN = 60000 // BATCH
    comptime N_BATCHES_TEST = 10000 // BATCH

    print("initializing network...")
    var net = Sequential(
        Linear[IN_DIM, H1].make["cpu", INIT=Kaiming](),
        ReLU[H1].make["cpu", INIT=Kaiming](),
        Linear[H1, H2].make["cpu", INIT=Kaiming](),
        ReLU[H2].make["cpu", INIT=Kaiming](),
        Linear[H2, N_CLASSES].make["cpu", INIT=Kaiming](),
    )
    var loss_fn = CrossEntropyLoss[N_CLASSES].make["cpu"]()
    var optim = Adam.make["cpu", M=type_of(net)](net)
    optim.lr = LR

    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH, target="cpu",
    ].make_from(net^, optim^, loss_fn^)

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_DIM
    )
    var tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )

    var final_acc: Float64 = 0.0
    for epoch in range(N_EPOCHS):
        var t0 = perf_counter_ns()
        var epoch_loss: Scalar[DT] = 0.0
        for batch_idx in range(N_BATCHES_TRAIN):
            for b in range(BATCH):
                var sample_idx = batch_idx * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.train_images[
                        sample_idx * IN_DIM + px
                    ]
                for c in range(N_CLASSES):
                    tgt_buf[b * N_CLASSES + c] = 0.0
                tgt_buf[b * N_CLASSES + Int(ds.train_labels[sample_idx])] = 1.0
            epoch_loss += trainer.train_step(in_buf, tgt_buf)
        var t_train = perf_counter_ns()

        var n_correct: Int = 0
        for batch_idx in range(N_BATCHES_TEST):
            for b in range(BATCH):
                var sample_idx = batch_idx * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.test_images[
                        sample_idx * IN_DIM + px
                    ]
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

        var acc = Float64(n_correct) / 10000.0
        final_acc = acc
        print(
            "epoch " + String(epoch)
            + " | train_loss=" + String(epoch_loss / Scalar[DT](N_BATCHES_TRAIN))
            + " | test_acc=" + String(acc * 100.0) + "%"
            + " | train=" + String(Float64(t_train - t0) / 1e9) + "s"
        )

    print("\nfinal test accuracy: " + String(final_acc * 100.0) + "%")
    assert_true(
        final_acc >= TARGET_ACC,
        "Expected >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(final_acc * 100.0) + "%",
    )
    in_buf.free(); tgt_buf.free(); out_buf.free()
    print("DONE")
