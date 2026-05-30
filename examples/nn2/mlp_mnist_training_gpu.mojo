"""End-to-end MLP training on MNIST — nn2 GPU perf benchmark.

Aligned with `examples/nn/mlp/mlp_mnist_training_gpu.mojo` (nn1 baseline)
and with `examples/nn2/mlp_mnist_training_cpu.mojo` (nn2 CPU) for
apples-to-apples perf comparison. Same architecture, batch size, epochs,
shuffle, and output format.

Architecture: Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10).
Loss:         CrossEntropyLoss[10] on one-hot targets.
Optimizer:    Adam (lr=0.001, default betas/eps).
Init:         Kaiming uniform on weights, zero bias.
Shuffle:      Fisher-Yates host permutation per epoch; per-batch host gather
              into the trainer's pinned input buffer, which trainer.train_step
              H2D-copies to the device. This is the idiomatic nn2 GPU per-batch
              path (vs nn1's whole-dataset upload + device-side shuffle).

Run:
    pixi run -e apple  mojo run -I . examples/nn2/mlp_mnist_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/nn2/mlp_mnist_training_gpu.mojo
"""

from std.memory import alloc
from std.random import seed, random_ui64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


comptime BATCH = 128
comptime EPOCHS = 5
comptime IN_DIM = 784
comptime H1 = 256
comptime H2 = 128
comptime N_CLASSES = 10
comptime LR: Scalar[DT] = 0.001


@always_inline
def _fisher_yates(mut idx: List[Int], n: Int):
    for i in range(n - 1, 0, -1):
        var j = Int(random_ui64(0, UInt64(i)))
        var tmp = idx[i]
        idx[i] = idx[j]
        idx[j] = tmp


def main() raises:
    seed(42)

    print("=" * 65)
    print("MNIST MLP training — nn2 GPU (perf benchmark)")
    print("=" * 65)
    print(
        "  architecture: Linear(784→256) → ReLU → Linear(256→128) →"
        " ReLU → Linear(128→10)"
    )
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    print("initializing network...")
    var net = Sequential(
        Linear[IN_DIM, H1].make["gpu", INIT=Kaiming](ctx),
        ReLU[H1].make["gpu", INIT=Kaiming](ctx),
        Linear[H1, H2].make["gpu", INIT=Kaiming](ctx),
        ReLU[H2].make["gpu", INIT=Kaiming](ctx),
        Linear[H2, N_CLASSES].make["gpu", INIT=Kaiming](ctx),
    )

    var loss_fn = CrossEntropyLoss[N_CLASSES].make["gpu"](ctx)
    var eval_loss_fn = CrossEntropyLoss[N_CLASSES].make["cpu"]()
    var optim = Adam.make["gpu", M=type_of(net)](net, ctx)
    optim.lr = LR

    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH,
        target="gpu",
    ].make_from(net^, optim^, loss_fn^, ctx)

    # Per-minibatch reusable host buffers (trainer H2D-copies these to device).
    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_DIM
    )
    var tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )

    # Shuffle indices, re-permuted each epoch.
    var perm = List[Int](capacity=MNIST.N_TRAIN)
    for i in range(MNIST.N_TRAIN):
        perm.append(i)

    comptime BATCHES_PER_EPOCH = 60000 // BATCH
    comptime N_BATCHES_TEST = 10000 // BATCH

    var last_acc: Float64 = 0.0
    var last_test_loss: Float64 = 0.0
    var last_epoch_loss: Scalar[DT] = 0.0

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    for epoch in range(EPOCHS):
        _fisher_yates(perm, MNIST.N_TRAIN)

        var epoch_loss: Scalar[DT] = 0.0
        for batch_idx in range(BATCHES_PER_EPOCH):
            var base = batch_idx * BATCH
            for b in range(BATCH):
                var sample = perm[base + b]
                var src = sample * IN_DIM
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.train_images[src + px]
                for c in range(N_CLASSES):
                    tgt_buf[b * N_CLASSES + c] = 0.0
                tgt_buf[b * N_CLASSES + Int(ds.train_labels[sample])] = 1.0

            epoch_loss = trainer.train_step(in_buf, tgt_buf)

        last_epoch_loss = epoch_loss

        # Per-epoch eval on the test set (forward-only, drop trailing partial batch).
        var n_correct: Int = 0
        var seen: Int = 0
        for batch_idx in range(N_BATCHES_TEST):
            var base = batch_idx * BATCH
            for b in range(BATCH):
                var sample_idx = base + b
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
                if best_c == Int(ds.test_labels[base + b]):
                    n_correct += 1
            seen += BATCH

        last_acc = Float64(n_correct) / Float64(seen)

        # Test loss on first batch (host-side CE on predict output; same as nn1 example).
        for b in range(BATCH):
            var src = b * IN_DIM
            for px in range(IN_DIM):
                in_buf[b * IN_DIM + px] = ds.test_images[src + px]
            for c in range(N_CLASSES):
                tgt_buf[b * N_CLASSES + c] = 0.0
            tgt_buf[b * N_CLASSES + Int(ds.test_labels[b])] = 1.0
        trainer.predict(in_buf, out_buf)
        var logits_tt = TileTensor(out_buf, row_major[BATCH, N_CLASSES]())
        var targets_tt = TileTensor(tgt_buf, row_major[BATCH, N_CLASSES]())
        last_test_loss = Float64(
            eval_loss_fn.forward["cpu", BATCH](logits_tt, targets_tt)
        )

        print(
            "  epoch "
            + String(epoch + 1)
            + "/"
            + String(EPOCHS)
            + "  loss="
            + String(epoch_loss)[byte=:7]
            + "  test_loss="
            + String(last_test_loss)[byte=:7]
            + "  top1="
            + String(last_acc * 100.0)[byte=:6]
            + "%"
        )

    ctx.synchronize()
    var t1 = perf_counter_ns()
    print(
        "  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s"
    )
    print("  final batch loss: " + String(last_epoch_loss)[byte=:8])

    # Final report.
    print("\n── Final evaluation (full test set) ──")
    print(
        "  test_loss="
        + String(last_test_loss)
        + "  top1="
        + String(last_acc * 100.0)[byte=:6]
        + "%"
    )

    in_buf.free()
    tgt_buf.free()
    out_buf.free()

    print("=" * 65)
    if last_acc >= 0.97:
        print("PASS — nn2 MLP converges on MNIST (>=97%)")
    else:
        print(
            "FAIL — expected >=97% test accuracy, got " + String(last_acc)
        )
        raise Error("accuracy below threshold")
