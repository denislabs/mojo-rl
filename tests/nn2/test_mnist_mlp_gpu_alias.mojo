"""GPU MNIST MLP via the Phase 2.4 architecture-as-type-alias form.

Demonstrates the canonical end-to-end shape that Phase 2.4 unlocks:

  comptime MLP     = Sequential[Linear[...], ReLU[...], ...]
  comptime TRAINER = Trainer[MLP, Adam, CrossEntropyLoss[10], BATCH, target="gpu"]
  var trainer = TRAINER.make[Kaiming](ctx)

No `^` on layer values. No `type_of(...)`. No separately-built
net/optim/loss. The Trainer's one-call `make[INIT](ctx)` factory builds
the entire net/optim/loss tree internally via:
  - `Sequential.make["gpu", Kaiming](ctx)`   ← recursive over children
  - `CrossEntropyLoss.make["gpu"](ctx)`
  - `Adam.make["gpu"](net, ctx)`

Numerics: identical to the manual-form test (`test_mnist_mlp_gpu.mojo`)
since both use the same architecture + Kaiming init + lr defaults.
"""

from std.random import seed
from std.testing import assert_true
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


# ── Architecture, declared once as a comptime type alias. ─────────────
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
    target="gpu",
]


def test_mnist_mlp_gpu_alias() raises:
    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    # ── ONE-CALL TRAINER CONSTRUCTION ──────────────────────────────────
    # `TRAINER.make[Kaiming](ctx)` recursively builds the whole network
    # tree (5 layers), CrossEntropyLoss, and Adam — internally.
    print("building trainer via TRAINER.make[Kaiming](ctx)...")
    var trainer = TRAINER.make[Kaiming](ctx)

    # ── Upload dataset to GPU once. ───────────────────────────────────
    print("uploading dataset to GPU...")
    var train_x_host = ctx.enqueue_create_host_buffer[DT](MNIST.N_TRAIN * IN_DIM)
    var train_y_host = ctx.enqueue_create_host_buffer[DT](MNIST.N_TRAIN * N_CLASSES)
    var test_x_host  = ctx.enqueue_create_host_buffer[DT](MNIST.N_TEST * IN_DIM)
    ctx.synchronize()
    for i in range(MNIST.N_TRAIN * IN_DIM):
        train_x_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(MNIST.N_TRAIN * N_CLASSES):
        train_y_host.unsafe_ptr()[i] = 0.0
    for i in range(MNIST.N_TRAIN):
        train_y_host.unsafe_ptr()[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0
    for i in range(MNIST.N_TEST * IN_DIM):
        test_x_host.unsafe_ptr()[i] = ds.test_images[i]

    var test_labels_host: UnsafePointer[Int32, MutAnyOrigin] = (
        ctx.enqueue_create_host_buffer[DType.int32](MNIST.N_TEST).unsafe_ptr()
    )
    for i in range(MNIST.N_TEST):
        test_labels_host[i] = Int32(ds.test_labels[i])

    var train_x_dev = ctx.enqueue_create_buffer[DT](MNIST.N_TRAIN * IN_DIM)
    var train_y_dev = ctx.enqueue_create_buffer[DT](MNIST.N_TRAIN * N_CLASSES)
    var test_x_dev  = ctx.enqueue_create_buffer[DT](MNIST.N_TEST * IN_DIM)
    ctx.enqueue_copy(train_x_dev, train_x_host)
    ctx.enqueue_copy(train_y_dev, train_y_host)
    ctx.enqueue_copy(test_x_dev,  test_x_host)
    ctx.synchronize()

    var train_x = TileTensor(train_x_dev, row_major[MNIST.N_TRAIN, IN_DIM]())
    var train_y = TileTensor(train_y_dev, row_major[MNIST.N_TRAIN, N_CLASSES]())
    var test_x  = TileTensor(test_x_dev,  row_major[MNIST.N_TEST, IN_DIM]())

    # ── Train. ────────────────────────────────────────────────────────
    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[MNIST.N_TRAIN, MNIST.N_TEST](
        train_x, train_y, test_x, test_labels_host,
        epochs=N_EPOCHS,
    )
    var t1 = perf_counter_ns()
    var total_s = Float64(t1 - t0) / 1e9

    var best_acc: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best_acc:
            best_acc = a
    var final_acc = result.epoch_test_top1[N_EPOCHS - 1]

    print("")
    print("final test accuracy: " + String(final_acc * 100.0) + "%")
    print("best  test accuracy: " + String(best_acc * 100.0) + "%")
    print("total wall time:     " + String(total_s) + "s")
    assert_true(best_acc >= TARGET_ACC,
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%")
    print("  test_mnist_mlp_gpu_alias PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 GPU MNIST MLP — Phase 2.4 alias form")
    print("=" * 60)
    test_mnist_mlp_gpu_alias()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
