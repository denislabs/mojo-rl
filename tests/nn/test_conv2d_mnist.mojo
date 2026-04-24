"""End-to-end CNN training on MNIST — validates Conv2D forward + backward.

Trains a small LeNet-style CNN on real MNIST and checks that test accuracy
exceeds 95%. If Conv2D forward or backward is broken this test fails:
broken forward → train loss plateaus; broken backward dx → conv1 doesn't
learn features; broken backward dW → filters don't update.

Uses Trainer.train_gpu_minibatch to iterate the full 60k dataset in
BATCH-sized slices — dataset is uploaded to GPU once, no per-batch H2D copy.

Architecture (unfused primitive Conv2D, NOT Conv2DReLU):
    Conv2DLayer[1,  16, 5×5, s=2]  28×28 -> 12×12×16
    ReLU
    Conv2DLayer[16, 32, 5×5, s=2]  12×12 -> 4×4×32
    ReLU
    Flatten -> 512
    Linear[512, 10]

Run:
    pixi run -e apple mojo run -I . tests/nn/test_conv2d_mnist.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.conv2d_layer import Conv2DLayer
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.model.flatten_layer import FlattenLayer
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets.mnist import MNIST


comptime BATCH = 64
comptime EPOCHS = 1

comptime CNN = Sequential[
    Conv2DLayer[1, 16, 5, 2, 0, 28, 28],   # 28 -> 12 (×16 ch) = 2304
    ReLU[16 * 12 * 12],
    Conv2DLayer[16, 32, 5, 2, 0, 12, 12],  # 12 -> 4  (×32 ch) = 512
    ReLU[32 * 4 * 4],
    FlattenLayer[32 * 4 * 4],
    Linear[32 * 4 * 4, 10],
]


def main() raises:
    seed(42)

    print("=" * 65)
    print("MNIST CNN training — validates Conv2D end-to-end")
    print("=" * 65)
    print(
        "  architecture: Conv(1→16,5,s=2) → ReLU → Conv(16→32,5,s=2) → ReLU"
        " → Flatten → FC(512→10)  [unfused Conv2D primitive]"
    )
    print("  params: " + String(CNN.PARAM_SIZE))
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    var ds = MNIST()
    var ctx = DeviceContext()

    comptime TRAINER = Trainer[CNN, Adam[LR=0.001], CrossEntropyLoss]
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

    # ── Train ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = TRAINER.train_gpu_minibatch[BATCH, MNIST.N_TRAIN](
        state,
        ctx,
        train_img_lt,
        train_tgt_lt,
        epochs=EPOCHS,
        print_every_batches=100,
        shuffle=True,
        rng_seed=42,
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print(
        "  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s"
    )
    print("  final batch loss: " + String(result.final_loss)[byte=:8])

    # ── Evaluate test set ──
    print("\n── Evaluating ──")

    # Upload test images (labels stay on host for argmax comparison)
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    for i in range(MNIST.N_TEST * MNIST.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_buf, test_img_host)
    var test_img_lt = LayoutTensor[
        dtype, Layout.row_major(MNIST.N_TEST, MNIST.IMG_SIZE), MutAnyOrigin
    ](test_img_buf)

    # Inference buffers for one BATCH at a time
    comptime num_test_batches = MNIST.N_TEST // BATCH
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
    if acc >= 0.95:
        print("PASS — Conv2D forward + backward converge on real MNIST")
    else:
        print("FAIL — expected >=95% test accuracy, got " + String(acc))
        raise Error("accuracy below threshold")
