"""End-to-end CNN training on CIFAR-10 — validates Conv2D RGB + BatchNorm.

Trains a deeper VGG-style CNN on real CIFAR-10 with per-channel input
normalization and standard crop+flip augmentation. Targets ≥65% test
accuracy, which is well above:
  - 10% random baseline
  - ~35% linear-on-pixels baseline
  - ~43% shallow-conv-without-BN baseline

A pass validates the fused Conv2D + BatchNorm + ReLU pipeline end-to-end,
including gradient flow through 6 conv layers, running-stat updates, and
Dropout train-vs-eval routing (forward_gpu applies mask, forward_gpu_no_cache
is identity).

Uses `Trainer.train_gpu_minibatch_full` with `CIFAR10CropFlipAugmenter`
(centralized in `mojo_rl.nn.datasets`) for the full training loop and
on-device per-epoch eval.

Architecture (mirrors a Kaggle CIFAR-10 recipe minus Dropout + aug):
    Conv2DBNReLU[3,  32, 3, 1, 1, 32, 32]   preserves 32x32
    Conv2DBNReLU[32, 32, 3, 1, 1, 32, 32]   preserves 32x32
    MaxPoolLayer[32, 32, 32, 2]             -> 16x16x32
    Conv2DBNReLU[32, 64, 3, 1, 1, 16, 16]   preserves 16x16
    Conv2DBNReLU[64, 64, 3, 1, 1, 16, 16]   preserves 16x16
    MaxPoolLayer[64, 16, 16, 2]             -> 8x8x64
    Conv2DBNReLU[64,  128, 3, 1, 1, 8, 8]   preserves 8x8
    Conv2DBNReLU[128, 128, 3, 1, 1, 8, 8]   preserves 8x8
    MaxPoolLayer[128, 8, 8, 2]              -> 4x4x128 = 2048
    Flatten
    LinearReLU[2048, 128]
    Linear[128, 10]

Run:
    pixi run -e apple  mojo run -I . examples/nn/conv2d/conv2d_cifar10_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/nn/conv2d/conv2d_cifar10_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.pool_layer import MaxPoolLayer
from mojo_rl.nn.model.flatten_layer import FlattenLayer
from mojo_rl.nn.model.linear_act import LinearReLU
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.dropout import Dropout
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets import CIFAR10, CIFAR10CropFlipAugmenter


comptime BATCH = 128
comptime EPOCHS = 50


comptime CNN = Sequential[
    # Block 1: 32×32 → 16×16, 32 channels
    Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
    Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
    MaxPoolLayer[32, 32, 32, 2],
    Dropout[32 * 16 * 16, 0.25, 101, True],
    # Block 2: 16×16 → 8×8, 64 channels
    Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16],
    Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16],
    MaxPoolLayer[64, 16, 16, 2],
    Dropout[64 * 8 * 8, 0.25, 202, True],
    # Block 3: 8×8 → 4×4, 128 channels
    Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8],
    Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8],
    MaxPoolLayer[128, 8, 8, 2],
    Dropout[128 * 4 * 4, 0.25, 303, True],
    # Classifier head
    FlattenLayer[128 * 4 * 4],
    LinearReLU[128 * 4 * 4, 128],
    Dropout[128, 0.25, 404, True],
    Linear[128, 10],
]


def main() raises:
    seed(42)

    print("=" * 65)
    print(
        "CIFAR-10 deep CNN — validates Conv2D + BatchNorm + deep gradient flow"
    )
    print("=" * 65)
    print(
        "  architecture: 6× Conv2DBatchNormReLU + 3 MaxPool + 4× Dropout + FC(2048→128→10)"
    )
    print("  params: " + String(CNN.PARAM_SIZE))
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime TRAINER = Trainer[CNN, Adam[LR=0.001], CrossEntropyLoss]
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # ── Upload full training set to GPU once (images + one-hot labels) ──
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    for i in range(CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = 0.0
    for i in range(CIFAR10.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.train_labels[i])
        ] = 1.0

    var train_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var train_img_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE),
        MutAnyOrigin,
    ](train_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.NUM_CLASSES),
        MutAnyOrigin,
    ](train_tgt_buf)

    # ── Upload test set (images + int32 labels) to GPU once ──
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    var test_lbl_host = ctx.enqueue_create_host_buffer[DType.int32](
        CIFAR10.N_TEST
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    for i in range(CIFAR10.N_TEST):
        test_lbl_host.unsafe_ptr()[i] = ds.test_labels[i]

    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    var test_lbl_buf = ctx.enqueue_create_buffer[DType.int32](CIFAR10.N_TEST)
    ctx.enqueue_copy(test_img_buf, test_img_host)
    ctx.enqueue_copy(test_lbl_buf, test_lbl_host)

    var test_img_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TEST, CIFAR10.IMG_SIZE),
        MutAnyOrigin,
    ](test_img_buf)
    var test_lbl_lt = LayoutTensor[
        DType.int32, Layout.row_major(CIFAR10.N_TEST), MutAnyOrigin
    ](test_lbl_buf)

    # ── Train + per-epoch eval ──
    # CIFAR10CropFlipAugmenter re-augments train_input each epoch into a
    # Trainer-owned aug buffer (random pad-4 crop + horizontal flip).
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = TRAINER.train_gpu_minibatch_full[
        BATCH, CIFAR10.N_TRAIN, CIFAR10.N_TEST,
        AUGMENTER=CIFAR10CropFlipAugmenter,
    ](
        state,
        ctx,
        train_img_lt, train_tgt_lt,
        test_img_lt, test_lbl_lt,
        epochs=EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
        show_progress=True,
        eval_every_epochs=1,
        progress_label="CIFAR10-CNN",
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
    if acc >= 0.65:
        print("PASS — deep CNN + BatchNorm converges on CIFAR-10 (>=65%)")
    else:
        print("FAIL — expected >=65% test accuracy, got " + String(acc))
        raise Error("accuracy below threshold")
