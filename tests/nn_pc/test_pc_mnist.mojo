"""End-to-end MNIST PCN test (GPU) — validates real-data convergence.

Smaller-than-paper architecture so it completes in reasonable time on naive
GPU kernels (~minutes per epoch). The point is *convergence on real data*,
not chasing the paper's CIFAR accuracy.

Architecture:
    PCLinear[784, 128]              # input → hidden 1
    PCLinear[128, 64]               # hidden 1 → hidden 2
    PCLinear[10, 64, PCIdentity]    # readout (NUM_CLASSES=10, TOP_HIDDEN=64)

Pass criterion:
  - Test accuracy > 40% (4× random baseline) after 1 epoch.

  This is a POC budget — naive matmul kernels + 1 epoch + small architecture.
  Empirically: 200 batches → 37%, 600 batches (full epoch) → 47%, train acc
  ~59%. Scaling up compute or matmul perf would close the gap to backprop-MLP.

Run:
    pixi run -e apple  mojo run -I . tests/nn_pc/test_pc_mnist.mojo
    pixi run -e nvidia mojo run -I . tests/nn_pc/test_pc_mnist.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets.mnist import MNIST
from mojo_rl.experimental.nn_pc import PCLinear, PCIdentity, PCTrainer


comptime BATCH = 100
comptime EPOCHS = 1
comptime T_INFER = 50          # match paper
comptime T_LEARN = 200         # 6.7x my previous 30, still 2.5x less than paper's 500
comptime ETA_INFER: Float64 = 0.05
comptime ETA_LEARN: Float64 = 0.005

# For initial debugging: run only a SUBSET of training to iterate faster.
# Set N_TRAIN_BATCHES = MNIST.N_TRAIN // BATCH to use the full epoch.
comptime N_TRAIN_BATCHES = 600   # full epoch
comptime N_TEST_BATCHES = 30     # 3000 test samples, speeds up debugging
comptime N_TRAINEVAL_BATCHES = 5  # 500 training samples re-evaluated for train acc

comptime TRAINER = PCTrainer[
    PCLinear[784, 128],
    PCLinear[128, 64],
    PCLinear[10, 64, PCIdentity],
    dtype=dtype,
]


def main() raises:
    seed(42)
    print("=" * 65)
    print("MNIST PCN — validates convergence on real data (GPU)")
    print("=" * 65)
    print("  arch       : 784 → 128 → 64 → 10 (PCN, identity readout)")
    print("  params     :", TRAINER.MODEL.PARAM_SIZE)
    print(
        "  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER,
        " T_LEARN=", T_LEARN, " EPOCHS=", EPOCHS,
    )
    print("  rates      : eta_infer=", ETA_INFER, " eta_learn=", ETA_LEARN)

    var ds = MNIST()
    var ctx = DeviceContext()

    # ── Allocate + initialize params on GPU ──
    var params_host = ctx.enqueue_create_host_buffer[dtype](TRAINER.MODEL.PARAM_SIZE)
    for i in range(TRAINER.MODEL.PARAM_SIZE):
        params_host.unsafe_ptr()[i] = Scalar[dtype](0)
    var params_init_t = LayoutTensor[
        dtype, Layout.row_major(TRAINER.MODEL.PARAM_SIZE), MutAnyOrigin
    ](params_host.unsafe_ptr())
    TRAINER.MODEL.initialize_params[Xavier[], dtype](params_init_t)

    var params_dbuf = ctx.enqueue_create_buffer[dtype](TRAINER.MODEL.PARAM_SIZE)
    ctx.enqueue_copy(params_dbuf, params_host)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(TRAINER.MODEL.PARAM_SIZE), MutAnyOrigin
    ](params_dbuf)

    # ── Upload full training set + one-hot labels to GPU once ──
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * MNIST.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * MNIST.NUM_CLASSES
    )
    for i in range(MNIST.N_TRAIN * MNIST.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(MNIST.N_TRAIN * MNIST.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(MNIST.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * MNIST.NUM_CLASSES + Int(ds.train_labels[i])
        ] = Scalar[dtype](1.0)
    var train_img_dbuf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TRAIN * MNIST.IMG_SIZE
    )
    var train_tgt_dbuf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TRAIN * MNIST.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_dbuf, train_img_host)
    ctx.enqueue_copy(train_tgt_dbuf, train_tgt_host)

    # ── Per-batch latents buffer (reused across batches) ──
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    var lat_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_dbuf)

    # Latent randn init: do it on host, copy. Same Philox seed per batch slot
    # for reproducibility.
    var lat_host_init = ctx.enqueue_create_host_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    var lat_init_view = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_host_init.unsafe_ptr())

    comptime num_train_batches = N_TRAIN_BATCHES  # subset for debugging

    # ── Train ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    for epoch in range(EPOCHS):
        for b in range(num_train_batches):
            # Re-init latents N(0,1) per batch (PyTorch protocol)
            TRAINER.randn_init_latents[BATCH](
                lat_init_view,
                seed=UInt64(1000) + UInt64(epoch) * UInt64(num_train_batches) + UInt64(b),
                offset=UInt64(0),
            )
            ctx.enqueue_copy(lat_dbuf, lat_host_init)

            # View into the b-th batch slice of the uploaded training set.
            var batch_x = LayoutTensor[
                dtype, Layout.row_major(BATCH, MNIST.IMG_SIZE), MutAnyOrigin
            ](train_img_dbuf.unsafe_ptr() + b * BATCH * MNIST.IMG_SIZE)
            var batch_y = LayoutTensor[
                dtype, Layout.row_major(BATCH, MNIST.NUM_CLASSES), MutAnyOrigin
            ](train_tgt_dbuf.unsafe_ptr() + b * BATCH * MNIST.NUM_CLASSES)

            TRAINER.train_one_batch_gpu[BATCH](
                ctx, params_t, lat_t, batch_x, batch_y,
                T_infer=T_INFER, T_learn=T_LEARN,
                eta_infer=Scalar[dtype](ETA_INFER),
                eta_learn=Scalar[dtype](ETA_LEARN),
            )
            if b % 100 == 0:
                ctx.synchronize()
                var t_now = perf_counter_ns()
                print(
                    "  epoch", epoch + 1, " batch", b, "/", num_train_batches,
                    "  elapsed:",
                    String(Float64(t_now - t0) / 1e9)[byte=:6], "s",
                )
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print("  total training time:", String(Float64(t1 - t0) / 1e9)[byte=:6], "s")

    # ── Evaluate via free PCN inference on test set ──
    print("\n── Evaluating (free PCN inference) ──")
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    for i in range(MNIST.N_TEST * MNIST.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_dbuf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_dbuf, test_img_host)

    var y_hat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * MNIST.NUM_CLASSES)
    var y_hat_host = ctx.enqueue_create_host_buffer[dtype](BATCH * MNIST.NUM_CLASSES)
    var y_hat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MNIST.NUM_CLASSES), MutAnyOrigin
    ](y_hat_dbuf)

    comptime num_test_batches = N_TEST_BATCHES  # subset for debugging
    var correct: Int = 0
    var total: Int = 0
    for batch_idx in range(num_test_batches):
        # Fresh randn latents per inference batch
        TRAINER.randn_init_latents[BATCH](
            lat_init_view, seed=UInt64(9000) + UInt64(batch_idx), offset=UInt64(0)
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)

        var batch_x = LayoutTensor[
            dtype, Layout.row_major(BATCH, MNIST.IMG_SIZE), MutAnyOrigin
        ](test_img_dbuf.unsafe_ptr() + batch_idx * BATCH * MNIST.IMG_SIZE)

        TRAINER.inference_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, y_hat_t,
            T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
        )
        ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
        ctx.synchronize()

        for b in range(BATCH):
            var best_idx = 0
            var best_val = y_hat_host.unsafe_ptr()[b * MNIST.NUM_CLASSES + 0]
            for c in range(1, MNIST.NUM_CLASSES):
                var v = y_hat_host.unsafe_ptr()[b * MNIST.NUM_CLASSES + c]
                if v > best_val:
                    best_val = v
                    best_idx = c
            var true_label = Int(ds.test_labels[batch_idx * BATCH + b])
            if best_idx == true_label:
                correct += 1
            total += 1

    var acc = Float64(correct) / Float64(total)
    print(
        "  test accuracy:", correct, "/", total, "=",
        String(acc * 100.0)[byte=:6] + "%"
    )

    # ── Train-set accuracy (free inference on TRAINING samples) ──
    print("\n── Training-set accuracy (free inference) ──")
    var train_correct: Int = 0
    var train_total: Int = 0
    for batch_idx in range(N_TRAINEVAL_BATCHES):
        TRAINER.randn_init_latents[BATCH](
            lat_init_view, seed=UInt64(8000) + UInt64(batch_idx), offset=UInt64(0)
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)
        var batch_x = LayoutTensor[
            dtype, Layout.row_major(BATCH, MNIST.IMG_SIZE), MutAnyOrigin
        ](train_img_dbuf.unsafe_ptr() + batch_idx * BATCH * MNIST.IMG_SIZE)
        TRAINER.inference_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, y_hat_t,
            T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
        )
        ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
        ctx.synchronize()
        for b in range(BATCH):
            var best_idx = 0
            var best_val = y_hat_host.unsafe_ptr()[b * MNIST.NUM_CLASSES + 0]
            for c in range(1, MNIST.NUM_CLASSES):
                var v = y_hat_host.unsafe_ptr()[b * MNIST.NUM_CLASSES + c]
                if v > best_val:
                    best_val = v
                    best_idx = c
            var true_label = Int(ds.train_labels[batch_idx * BATCH + b])
            if best_idx == true_label:
                train_correct += 1
            train_total += 1
    var train_acc = Float64(train_correct) / Float64(train_total)
    print(
        "  train accuracy:", train_correct, "/", train_total, "=",
        String(train_acc * 100.0)[byte=:6] + "%"
    )

    # ── Diagnostic: print y_hat for first 3 test samples ──
    print("\n── Diagnostic (first 3 test samples) ──")
    # Re-run inference on the very first batch we evaluated (already in lat_init_view)
    TRAINER.randn_init_latents[BATCH](
        lat_init_view, seed=UInt64(9000), offset=UInt64(0)
    )
    ctx.enqueue_copy(lat_dbuf, lat_host_init)
    var first_batch = LayoutTensor[
        dtype, Layout.row_major(BATCH, MNIST.IMG_SIZE), MutAnyOrigin
    ](test_img_dbuf.unsafe_ptr())
    TRAINER.inference_gpu[BATCH](
        ctx, params_t, lat_t, first_batch, y_hat_t,
        T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
    )
    ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
    ctx.synchronize()
    for s in range(3):
        var label = Int(ds.test_labels[s])
        print("  sample", s, " label=", label, " y_hat=[", end="")
        for c in range(MNIST.NUM_CLASSES):
            var v = Float64(y_hat_host.unsafe_ptr()[s * MNIST.NUM_CLASSES + c])
            print(String(v)[byte=:7], "", end="")
        print("]")

    print("=" * 65)
    if acc >= 0.40:
        print(
            "PASS — PCN converges on real MNIST (>=40% test acc, 4x random baseline)"
        )
    else:
        print("FAIL — expected >=40% test accuracy, got", acc)
        raise Error("MNIST PCN accuracy below threshold")
