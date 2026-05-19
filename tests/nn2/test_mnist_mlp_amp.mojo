"""Phase 3 exit criterion: MNIST MLP with `POLICY=Bf16Compute`.

Same architecture + hyperparameters as `test_mnist_mlp_gpu_alias.mojo`,
but the Trainer carries `POLICY=Bf16Compute` so Linear runs its matmuls
in bf16. Weights, gradients, and Adam moments stay fp32; cast-around the
linalg.matmul call.

Exit criterion (from docs/NN2_DESIGN.md Phase 3): test top-1 ≥ 96.7%
(within 0.3% of the fp32 baseline 97.19%).
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
from mojo_rl.nn2.core import Bf16Compute


# ── Same architecture as the fp32 baseline. ───────────────────────────
comptime IN_DIM    = 784
comptime H1        = 256
comptime H2        = 128
comptime N_CLASSES = 10
comptime BATCH     = 100
comptime N_EPOCHS  = 5
# 0.3% below fp32 baseline of 97.19%.
comptime TARGET_ACC: Float64 = 0.967

comptime MLP = Sequential[
    Linear[IN_DIM, H1],
    ReLU[H1],
    Linear[H1, H2],
    ReLU[H2],
    Linear[H2, N_CLASSES],
]

# ── POLICY=Bf16Compute is the only difference from the fp32 trainer. ──
comptime TRAINER = Trainer[
    MLP,
    Adam,
    CrossEntropyLoss[N_CLASSES],
    BATCH,
    target="gpu",
    POLICY=Bf16Compute,
]


def test_mnist_mlp_amp() raises:
    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    print("building AMP trainer (POLICY=Bf16Compute) via TRAINER.make[Kaiming](ctx)...")
    var trainer = TRAINER.make[Kaiming](ctx)

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
    print("AMP MNIST results (Bf16Compute, fp32 master weights):")
    print("  final test accuracy: " + String(final_acc * 100.0) + "%")
    print("  best  test accuracy: " + String(best_acc * 100.0) + "%")
    print("  total wall time:     " + String(total_s) + "s")
    print("  fp32 baseline:       97.19% (test_mnist_mlp_gpu_alias)")
    print("  target floor:        " + String(TARGET_ACC * 100.0) + "%")
    assert_true(best_acc >= TARGET_ACC,
        "AMP exit criterion: expected best >= " + String(TARGET_ACC * 100.0)
        + "%, got " + String(best_acc * 100.0) + "%")
    print("  test_mnist_mlp_amp PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 Phase 3 AMP MNIST — POLICY=Bf16Compute")
    print("=" * 60)
    test_mnist_mlp_amp()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
