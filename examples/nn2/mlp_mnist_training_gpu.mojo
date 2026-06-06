"""End-to-end MLP training on MNIST — nn2 GPU perf benchmark.

Aligned with `examples/nn/mlp/mlp_mnist_training_gpu.mojo` (nn1 baseline)
for apples-to-apples perf comparison: full train+test sets uploaded to
device once, on-device Fisher-Yates shuffle of an Int32 permutation
buffer per epoch, parallel gather kernel produces shuffled (BATCH, DIM)
minibatches. No per-batch H2D in the inner loop.

Architecture: LinearReLU(784→256) → LinearReLU(256→128) → Linear(128→10).
Loss:         CrossEntropyLoss[10] on one-hot targets.
Optimizer:    Adam (lr=0.001, default betas/eps).
Init:         Kaiming uniform on weights, zero bias.
Shuffle:      shuffle=True, rng_seed=42 (matches nn1 baseline).

Run:
    pixi run -e apple  mojo run -I . examples/nn2/mlp_mnist_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/nn2/mlp_mnist_training_gpu.mojo
"""

from std.memory import alloc
from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
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
# Drop trailing partial batch to match nn1's BATCH-divisible asserts.
comptime N_TRAIN_FULL = 60000
comptime N_TEST_FULL = 10000
comptime N_TRAIN = (N_TRAIN_FULL // BATCH) * BATCH
comptime N_TEST = (N_TEST_FULL // BATCH) * BATCH


def main() raises:
    seed(42)

    print("=" * 65)
    print("MNIST MLP training — nn2 GPU (perf benchmark)")
    print("=" * 65)
    print(
        "  architecture: LinearReLU(784→256) → LinearReLU(256→128) →"
        " Linear(128→10)"
    )
    print(
        "  batch: "
        + String(BATCH)
        + " | epochs: "
        + String(EPOCHS)
        + " | shuffle: True"
    )

    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    print("initializing network...")
    var net = Sequential(
        LinearReLU[IN_DIM, H1].make["gpu", INIT=Kaiming](ctx),
        LinearReLU[H1, H2].make["gpu", INIT=Kaiming](ctx),
        Linear[H2, N_CLASSES].make["gpu", INIT=Kaiming](ctx),
        ctx=ctx,
    )

    var loss_fn = CrossEntropyLoss[N_CLASSES].make["gpu"](ctx)
    var optim = Adam.make["gpu", M=type_of(net)](net, ctx)
    optim.lr = LR

    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH,
        target="gpu",
    ].make_from(net^, optim^, loss_fn^, ctx)

    # ── Upload full training set (images + one-hot targets) to GPU once ──
    var train_img_host = ctx.enqueue_create_host_buffer[DT](
        N_TRAIN * IN_DIM
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[DT](
        N_TRAIN * N_CLASSES
    )
    for i in range(N_TRAIN * IN_DIM):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(N_TRAIN * N_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = 0.0
    for i in range(N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * N_CLASSES + Int(ds.train_labels[i])
        ] = 1.0

    var train_img_dev = ctx.enqueue_create_buffer[DT](N_TRAIN * IN_DIM)
    var train_tgt_dev = ctx.enqueue_create_buffer[DT](N_TRAIN * N_CLASSES)
    ctx.enqueue_copy(train_img_dev, train_img_host)
    ctx.enqueue_copy(train_tgt_dev, train_tgt_host)

    var train_img_ptr: UnsafePointer[
        Scalar[DT], MutAnyOrigin
    ] = train_img_dev.unsafe_ptr()
    var train_tgt_ptr: UnsafePointer[
        Scalar[DT], MutAnyOrigin
    ] = train_tgt_dev.unsafe_ptr()
    var train_x_tt = TileTensor(
        train_img_ptr, row_major[N_TRAIN, IN_DIM]()
    )
    var train_y_tt = TileTensor(
        train_tgt_ptr, row_major[N_TRAIN, N_CLASSES]()
    )

    # ── Upload test set (images on device, int32 labels on host) ──
    var test_img_host = ctx.enqueue_create_host_buffer[DT](
        N_TEST * IN_DIM
    )
    for i in range(N_TEST * IN_DIM):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_dev = ctx.enqueue_create_buffer[DT](N_TEST * IN_DIM)
    ctx.enqueue_copy(test_img_dev, test_img_host)
    var test_img_ptr: UnsafePointer[
        Scalar[DT], MutAnyOrigin
    ] = test_img_dev.unsafe_ptr()
    var test_x_tt = TileTensor(
        test_img_ptr, row_major[N_TEST, IN_DIM]()
    )

    # Trainer.eval_top1_gpu reads labels host-side per batch — Int32 host buffer.
    var test_lbl_host: UnsafePointer[Int32, MutAnyOrigin] = alloc[Int32](
        N_TEST
    )
    for i in range(N_TEST):
        test_lbl_host[i] = Int32(ds.test_labels[i])

    ctx.synchronize()

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[N_TRAIN, N_TEST](
        train_x_tt,
        train_y_tt,
        test_x_tt,
        test_lbl_host,
        epochs=EPOCHS,
        print_progress=True,
        shuffle=True,
        rng_seed=UInt64(42),
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print(
        "  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s"
    )
    var n_evals = len(result.epoch_train_loss)
    var final_train_loss = result.epoch_train_loss[n_evals - 1]
    var final_top1 = result.epoch_test_top1[n_evals - 1]
    print("  final train loss: " + String(final_train_loss)[byte=:8])

    print("\n── Final evaluation (full test set) ──")
    print("  top1=" + String(final_top1 * 100.0)[byte=:6] + "%")

    test_lbl_host.free()

    print("=" * 65)
    if final_top1 >= 0.97:
        print("PASS — nn2 MLP converges on MNIST (>=97%)")
    else:
        print(
            "FAIL — expected >=97% test accuracy, got "
            + String(final_top1)
        )
        raise Error("accuracy below threshold")
