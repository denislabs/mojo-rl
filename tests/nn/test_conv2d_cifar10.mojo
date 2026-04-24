"""End-to-end CNN training on CIFAR-10 — validates Conv2D with RGB channels.

Trains a small CNN on real CIFAR-10 (32×32 RGB, 10 classes). The point is to
stress Conv2D with 3 input channels, which MNIST (1 channel) does not
exercise — a bug in multi-channel indexing would cap accuracy around the
linear-on-pixels baseline (~35%).

Target: ≥40% test accuracy after 1 epoch. Random baseline is 10%.

Architecture (fused Conv+ReLU via Conv2DReLU):
    Conv2DReLU[3,  32, 5×5, s=2]  32×32 -> 14×14×32
    Conv2DReLU[32, 64, 5×5, s=2]  14×14 -> 5×5×64
    Flatten -> 1600
    Linear[1600, 10]

Run:
    pixi run -e apple  mojo run -I . tests/nn/test_conv2d_cifar10.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_cifar10.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.conv2d_layer import Conv2DReLU
from mojo_rl.nn.model.flatten_layer import FlattenLayer
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets.cifar10 import CIFAR10


comptime BATCH = 128
comptime EPOCHS = 1

comptime CNN = Sequential[
    Conv2DReLU[3, 32, 5, 2, 0, 32, 32],    # 32 -> 14 (×32 ch)
    Conv2DReLU[32, 64, 5, 2, 0, 14, 14],   # 14 -> 5  (×64 ch) = 1600
    FlattenLayer[64 * 5 * 5],
    Linear[64 * 5 * 5, 10],
]


def main() raises:
    seed(42)

    print("=" * 65)
    print("CIFAR-10 CNN training — validates Conv2D RGB (3 input channels)")
    print("=" * 65)
    print(
        "  architecture: Conv2DReLU(3→32,5,s=2) → Conv2DReLU(32→64,5,s=2)"
        " → Flatten → FC(1600→10)"
    )
    print("  params: " + String(CNN.PARAM_SIZE))
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime TRAINER = Trainer[CNN, Adam[LR=0.001], CrossEntropyLoss]
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # ── Upload full training set (images + one-hot labels) to GPU once ──
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

    # ── Train ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = TRAINER.train_gpu_minibatch[BATCH, CIFAR10.N_TRAIN](
        state,
        ctx,
        train_img_lt,
        train_tgt_lt,
        epochs=EPOCHS,
        print_every_batches=1,  # per-epoch loss on end of each epoch
        shuffle=True,
        rng_seed=42,
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print(
        "  training time: "
        + String(Float64(t1 - t0) / 1e9)[byte=:6]
        + " s"
    )
    print("  final batch loss: " + String(result.final_loss)[byte=:8])

    # ── Evaluate test set ──
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
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CNN.CACHE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * CNN.WORKSPACE_SIZE_PER_SAMPLE
    )
    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * CNN.OUT_DIM)

    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CNN.OUT_DIM), MutAnyOrigin
    ](output_buf)
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CNN.CACHE_SIZE), MutAnyOrigin
    ](cache_buf)

    var correct: Int = 0
    var total: Int = 0

    for batch_idx in range(num_test_batches):
        var batch_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, CNN.IN_DIM), MutAnyOrigin
        ](test_img_buf.unsafe_ptr() + batch_idx * BATCH * CNN.IN_DIM)

        var params_eval = state.params_view()
        CNN.forward_gpu[BATCH](
            ctx, output_lt, batch_input, params_eval, cache_lt, workspace_buf
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
    if acc >= 0.40:
        print("PASS — Conv2D RGB converges on CIFAR-10 (>=40% after 1 epoch)")
    else:
        print("FAIL — expected >=40% test accuracy, got " + String(acc))
        raise Error("accuracy below threshold")
