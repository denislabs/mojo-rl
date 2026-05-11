"""End-to-end MLP training on MNIST — simple fully-connected baseline.

Trains a 2-hidden-layer MLP (784 → 256 → 128 → 10) on real MNIST and
checks that test accuracy exceeds 97%. Uses the auto-fused `LinearReLU`
(MatMul+BiasAdd+ReLU as a single GPU kernel).

Uses `Trainer.train_gpu_minibatch_full` with `IdentityAugmenter` (default,
no augmentation needed for MNIST) and on-device per-epoch eval.

Run:
    pixi run -e apple  mojo run -I . examples/nn/mlp/mlp_mnist_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/nn/mlp/mlp_mnist_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.linear_act import LinearReLU
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets import MNIST


comptime BATCH = 128
comptime EPOCHS = 5

comptime MLP = Sequential[
    LinearReLU[28 * 28, 256],
    LinearReLU[256, 128],
    Linear[128, 10],
]


def main() raises:
    seed(42)

    print("=" * 65)
    print("MNIST MLP training — fully-connected baseline")
    print("=" * 65)
    print("  architecture: LinearReLU(784→256) → LinearReLU(256→128) → Linear(128→10)")
    print("  params: " + String(MLP.PARAM_SIZE))
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    var ds = MNIST()
    var ctx = DeviceContext()

    comptime TRAINER = Trainer[MLP, Adam[LR=0.001], CrossEntropyLoss]
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # ── Upload full training set (images + one-hot labels) to GPU once ──
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * MNIST.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * MNIST.NUM_CLASSES
    )
    for i in range(MNIST.N_TRAIN * MNIST.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(MNIST.N_TRAIN * MNIST.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = 0.0
    for i in range(MNIST.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * MNIST.NUM_CLASSES + Int(ds.train_labels[i])
        ] = 1.0

    var train_img_buf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TRAIN * MNIST.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TRAIN * MNIST.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var train_img_lt = LayoutTensor[
        dtype, Layout.row_major(MNIST.N_TRAIN, MNIST.IMG_SIZE), MutAnyOrigin
    ](train_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype, Layout.row_major(MNIST.N_TRAIN, MNIST.NUM_CLASSES), MutAnyOrigin
    ](train_tgt_buf)

    # ── Upload test set (images + int32 labels) to GPU once ──
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    var test_lbl_host = ctx.enqueue_create_host_buffer[DType.int32](
        MNIST.N_TEST
    )
    for i in range(MNIST.N_TEST * MNIST.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    for i in range(MNIST.N_TEST):
        test_lbl_host.unsafe_ptr()[i] = ds.test_labels[i]

    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    var test_lbl_buf = ctx.enqueue_create_buffer[DType.int32](MNIST.N_TEST)
    ctx.enqueue_copy(test_img_buf, test_img_host)
    ctx.enqueue_copy(test_lbl_buf, test_lbl_host)

    var test_img_lt = LayoutTensor[
        dtype, Layout.row_major(MNIST.N_TEST, MNIST.IMG_SIZE), MutAnyOrigin
    ](test_img_buf)
    var test_lbl_lt = LayoutTensor[
        DType.int32, Layout.row_major(MNIST.N_TEST), MutAnyOrigin
    ](test_lbl_buf)

    # ── Train + per-epoch eval ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = TRAINER.train_gpu_minibatch_full[
        BATCH, MNIST.N_TRAIN, MNIST.N_TEST,
    ](
        state,
        ctx,
        train_img_lt, train_tgt_lt,
        test_img_lt, test_lbl_lt,
        epochs=EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        show_progress=True,
        eval_every_epochs=1,
        progress_label="MNIST-MLP",
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print(
        "  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s"
    )
    print("  final batch loss: " + String(result.final_loss)[byte=:8])

    # ── Final report ──
    var n_evals = len(result.val_top1_history)
    var acc = result.val_top1_history[n_evals - 1]
    var test_loss = result.val_loss_history[n_evals - 1]
    print("\n── Final evaluation (full test set) ──")
    print(
        "  test_loss=" + String(test_loss) + "  top1=" + String(acc * 100.0)[byte=:6] + "%"
    )

    print("=" * 65)
    if acc >= 0.97:
        print("PASS — MLP converges on MNIST (>=97%)")
    else:
        print("FAIL — expected >=97% test accuracy, got " + String(acc))
        raise Error("accuracy below threshold")
