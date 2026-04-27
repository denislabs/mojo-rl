"""End-to-end CNN training on CIFAR-10 — validates Conv2D RGB + BatchNorm.

Trains a deeper VGG-style CNN on real CIFAR-10 with per-channel input
normalization and standard crop+flip augmentation. The test targets ≥65%
test accuracy, which is well above:
  - 10% random baseline
  - ~35% linear-on-pixels baseline
  - ~43% shallow-conv-without-BN baseline

A pass validates the fused Conv2D + BatchNorm + ReLU pipeline end-to-end,
including gradient flow through 6 conv layers, running-stat updates, and
Dropout train-vs-eval routing (forward_gpu applies mask, forward_gpu_no_cache
is identity).

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
    pixi run -e apple  mojo run -I . tests/nn/test_conv2d_cifar10.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_cifar10.mojo
"""

from std.random import seed
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.pool_layer import MaxPoolLayer
from mojo_rl.nn.model.flatten_layer import FlattenLayer
from mojo_rl.nn.model.linear_act import LinearReLU
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.dropout import Dropout
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets.cifar10 import CIFAR10


comptime BATCH = 128
comptime EPOCHS = 50


def _cifar_augment_kernel[
    N: Int,
    dtype: DType = DType.float32,
](
    aug: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    raw: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    epoch_seed: Scalar[DType.uint64],
):
    """Random crop (pad 4) + random horizontal flip, per sample.

    Grid: (N,), Block: (TPB,). One block per sample; threads parallelize
    the 3072 output pixels. All threads in a block derive dx/dy/flip from
    PhiloxRandom(epoch_seed, b) identically — out-of-bounds pixels get 0.
    """
    var b = Int(block_idx.x)
    if b >= N:
        return
    var tid = Int(thread_idx.x)

    comptime C = 3
    comptime H = 32
    comptime W = 32
    comptime CHAN = H * W
    comptime IMG_SIZE = C * CHAN

    var rng = PhiloxRandom(seed=UInt64(epoch_seed), offset=UInt64(b))
    var r = rng.step_uniform()
    var dx = Int(Scalar[DType.float32](r[0]) * 9.0) - 4  # [-4, 4]
    var dy = Int(Scalar[DType.float32](r[1]) * 9.0) - 4  # [-4, 4]
    var flip = Scalar[DType.float32](r[2]) > 0.5

    var idx = tid
    while idx < IMG_SIZE:
        var c = idx // CHAN
        var yx = idx % CHAN
        var oy = yx // W
        var ox = yx % W
        var src_y = oy + dy
        var vx = ox + dx
        var val = Scalar[dtype](0.0)
        if src_y >= 0 and src_y < H and vx >= 0 and vx < W:
            var src_x = (W - 1 - vx) if flip else vx
            val = rebind[Scalar[dtype]](raw[b, c * CHAN + src_y * W + src_x])
        aug[b, idx] = val
        idx += TPB


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

    var raw_train_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var aug_train_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(raw_train_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var raw_train_img_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE),
        MutAnyOrigin,
    ](raw_train_img_buf)
    var aug_train_img_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE),
        MutAnyOrigin,
    ](aug_train_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.NUM_CLASSES),
        MutAnyOrigin,
    ](train_tgt_buf)

    # ── Train ──
    # One augmented copy per epoch (random 32×32 crop from pad-4, + h-flip).
    # Trainer sees the augmented buffer; raw copy is preserved for next epoch.
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var final_loss: Float32 = 0.0
    comptime aug_k = _cifar_augment_kernel[CIFAR10.N_TRAIN, dtype]
    for epoch in range(EPOCHS):
        var aug_seed = Scalar[DType.uint64](UInt64(1000) + UInt64(epoch))
        ctx.enqueue_function[aug_k, aug_k](
            aug_train_img_lt,
            raw_train_img_lt,
            aug_seed,
            grid_dim=(CIFAR10.N_TRAIN,),
            block_dim=(TPB,),
        )

        var result = TRAINER.train_gpu_minibatch[
            BATCH, CIFAR10.N_TRAIN, USE_CUDA_GRAPH=False
        ](
            state,
            ctx,
            aug_train_img_lt,
            train_tgt_lt,
            epochs=1,
            print_every_batches=0,
            shuffle=True,
            rng_seed=UInt64(42 + epoch),
        )
        ctx.synchronize()
        final_loss = Float32(result.final_loss)
        print(
            "  epoch "
            + String(epoch + 1)
            + "/"
            + String(EPOCHS)
            + "  last-batch loss="
            + String(final_loss)
        )
    var t1 = perf_counter_ns()
    print("  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s")
    print("  final batch loss: " + String(final_loss)[byte=:8])

    # ── Evaluate test set ──
    # Uses forward_gpu_no_cache: BN layers normalize with their EMA-tracked
    # running_mean/running_var (populated during training), so test accuracy
    # reflects true generalization and isn't contaminated by eval-batch stats.
    print("\n── Evaluating ──")

    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_buf, test_img_host)

    comptime num_test_batches = CIFAR10.N_TEST // BATCH
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * CNN.OUT_DIM)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * CNN.WORKSPACE_SIZE_PER_SAMPLE
    )
    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * CNN.OUT_DIM)

    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CNN.OUT_DIM), MutAnyOrigin
    ](output_buf)

    var correct: Int = 0
    var total: Int = 0

    for batch_idx in range(num_test_batches):
        var batch_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, CNN.IN_DIM), MutAnyOrigin
        ](test_img_buf.unsafe_ptr() + batch_idx * BATCH * CNN.IN_DIM)

        var params_eval = state.params_view()
        CNN.forward_gpu_no_cache[BATCH](
            ctx, output_lt, batch_input, params_eval, state.model_state_view(), workspace_buf
        )
        ctx.enqueue_copy(output_host, output_buf)
        ctx.synchronize()

        for b in range(BATCH):
            var best_idx = 0
            var best_val = output_host.unsafe_ptr()[b * 10 + 0]
            for c in range(1, 10):
                var v = output_host.unsafe_ptr()[b * 10 + c]
                if v > best_val:
                    best_val = v
                    best_idx = c
            var true_label = Int(ds.test_labels[batch_idx * BATCH + b])
            if best_idx == true_label:
                correct += 1
            total += 1

    var acc = Float64(correct) / Float64(total)
    print(
        "  test accuracy: "
        + String(correct)
        + " / "
        + String(total)
        + " = "
        + String(acc * 100.0)[byte=:6]
        + "%"
    )

    print("=" * 65)
    if acc >= 0.65:
        print("PASS — deep CNN + BatchNorm converges on CIFAR-10 (>=65%)")
    else:
        print("FAIL — expected >=65% test accuracy, got " + String(acc))
        raise Error("accuracy below threshold")
