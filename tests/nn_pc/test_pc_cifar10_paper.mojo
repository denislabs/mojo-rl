"""Paper-faithful CIFAR-10 PCN training (arxiv 2506.06332).

Exact replication of the paper's setup:
    Architecture: 3072 → 1000 → 500 → 10 + readout 10×10  (3,577,100 params)
    BATCH       : 500   (100 train batches × 4 epochs = 400 batches total)
    T_INFER     : 50    (inference iterations per batch)
    T_LEARN     : 500   (weight-update iterations per batch with latents frozen)
    η_infer     : 0.05
    η_learn     : 0.005
    EPOCHS      : 4
    Init        : Xavier
    No biases, no momentum, no LR schedule.

Test-time inference is "free PCN" (no supervised signal): T_INFER settling
steps with eps_L = 0, then forward through readout for y_hat.

Paper's reported result: 99.92% top-1 / 99.99% top-3 (Table 1).
Wall time on an L4 GPU: 4 minutes.

Pass criterion: test accuracy >= 60% (a generous threshold; PCN on CIFAR is
sensitive to inference dynamics and we may not reproduce the exact paper
result, but anything well above the 10% random baseline confirms the
algorithm scales).

Run:
    pixi run -e nvidia mojo run -I . tests/nn_pc/test_pc_cifar10_paper.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets.cifar10 import CIFAR10
from mojo_rl.nn_pc import PCLinear, PCIdentity, PCTrainer


# ── Paper hyperparameters (exact) ──────────────────────────────────────────
comptime BATCH = 500
comptime EPOCHS = 4
comptime T_INFER = 50
comptime T_LEARN = 500
comptime ETA_INFER: Float64 = 0.05
comptime ETA_LEARN: Float64 = 0.005

comptime TRAINER = PCTrainer[
    PCLinear[3072, 1000],
    PCLinear[1000, 500],
    PCLinear[500, 10],
    PCLinear[10, 10, PCIdentity],
    dtype=dtype,
]


def main() raises:
    seed(42)
    print("=" * 65)
    print("CIFAR-10 PCN — paper-faithful run (arxiv 2506.06332)")
    print("=" * 65)
    print("  arch       : 3072 → 1000 → 500 → 10 (paper MLP)")
    print("  params     :", TRAINER.MODEL.PARAM_SIZE)
    print(
        "  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER,
        " T_LEARN=", T_LEARN, " EPOCHS=", EPOCHS,
    )
    print("  rates      : η_infer=", ETA_INFER, " η_learn=", ETA_LEARN)

    var ds = CIFAR10()
    var ctx = DeviceContext()

    # ── Allocate + initialize params on GPU ──
    var params_host = ctx.enqueue_create_host_buffer[dtype](
        TRAINER.MODEL.PARAM_SIZE
    )
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
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    for i in range(CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(CIFAR10.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.train_labels[i])
        ] = Scalar[dtype](1.0)
    var train_img_dbuf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_dbuf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_dbuf, train_img_host)
    ctx.enqueue_copy(train_tgt_dbuf, train_tgt_host)

    # ── Per-batch latents (re-init each batch from host PRNG) ──
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    var lat_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_dbuf)
    var lat_host_init = ctx.enqueue_create_host_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    var lat_init_view = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_host_init.unsafe_ptr())

    comptime num_train_batches = CIFAR10.N_TRAIN // BATCH

    # ── Train ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    for epoch in range(EPOCHS):
        var t_epoch_start = perf_counter_ns()
        for b in range(num_train_batches):
            TRAINER.randn_init_latents[BATCH](
                lat_init_view,
                seed=UInt64(1000)
                    + UInt64(epoch) * UInt64(num_train_batches)
                    + UInt64(b),
                offset=UInt64(0),
            )
            ctx.enqueue_copy(lat_dbuf, lat_host_init)

            var batch_x = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CIFAR10.IMG_SIZE),
                MutAnyOrigin,
            ](train_img_dbuf.unsafe_ptr() + b * BATCH * CIFAR10.IMG_SIZE)
            var batch_y = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CIFAR10.NUM_CLASSES),
                MutAnyOrigin,
            ](train_tgt_dbuf.unsafe_ptr() + b * BATCH * CIFAR10.NUM_CLASSES)

            TRAINER.train_one_batch_gpu[BATCH](
                ctx, params_t, lat_t, batch_x, batch_y,
                T_infer=T_INFER, T_learn=T_LEARN,
                eta_infer=Scalar[dtype](ETA_INFER),
                eta_learn=Scalar[dtype](ETA_LEARN),
            )
        ctx.synchronize()
        var t_epoch_end = perf_counter_ns()
        print(
            "  epoch", epoch + 1, "/", EPOCHS, " done in",
            String(Float64(t_epoch_end - t_epoch_start) / 1e9)[byte=:6], "s",
            " (cumulative",
            String(Float64(t_epoch_end - t0) / 1e9)[byte=:6], "s)",
        )
    var t1 = perf_counter_ns()
    print(
        "  total training time:",
        String(Float64(t1 - t0) / 1e9)[byte=:6], "s",
    )

    # ── Test-set evaluation (free PCN inference on full 10K test set) ──
    print("\n── Evaluating on test set (free PCN inference) ──")
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_dbuf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_dbuf, test_img_host)

    var y_hat_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * CIFAR10.NUM_CLASSES
    )
    var y_hat_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * CIFAR10.NUM_CLASSES
    )
    var y_hat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CIFAR10.NUM_CLASSES), MutAnyOrigin
    ](y_hat_dbuf)

    comptime num_test_batches = CIFAR10.N_TEST // BATCH
    var correct: Int = 0
    var top3: Int = 0
    var total: Int = 0
    var t_eval_start = perf_counter_ns()
    for batch_idx in range(num_test_batches):
        TRAINER.randn_init_latents[BATCH](
            lat_init_view,
            seed=UInt64(9000) + UInt64(batch_idx),
            offset=UInt64(0),
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)

        var batch_x = LayoutTensor[
            dtype, Layout.row_major(BATCH, CIFAR10.IMG_SIZE), MutAnyOrigin
        ](test_img_dbuf.unsafe_ptr() + batch_idx * BATCH * CIFAR10.IMG_SIZE)
        TRAINER.inference_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, y_hat_t,
            T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
        )
        ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
        ctx.synchronize()

        for b in range(BATCH):
            # top-1
            var best_idx = 0
            var best_val = y_hat_host.unsafe_ptr()[b * CIFAR10.NUM_CLASSES + 0]
            for c in range(1, CIFAR10.NUM_CLASSES):
                var v = y_hat_host.unsafe_ptr()[b * CIFAR10.NUM_CLASSES + c]
                if v > best_val:
                    best_val = v
                    best_idx = c
            var true_label = Int(ds.test_labels[batch_idx * BATCH + b])
            if best_idx == true_label:
                correct += 1
            # top-3
            var t1_idx = -1; var t2_idx = -1; var t3_idx = -1
            var t1v = Float64(-1e30); var t2v = Float64(-1e30); var t3v = Float64(-1e30)
            for c in range(CIFAR10.NUM_CLASSES):
                var v = Float64(y_hat_host.unsafe_ptr()[b * CIFAR10.NUM_CLASSES + c])
                if v > t1v:
                    t3v = t2v; t3_idx = t2_idx
                    t2v = t1v; t2_idx = t1_idx
                    t1v = v; t1_idx = c
                elif v > t2v:
                    t3v = t2v; t3_idx = t2_idx
                    t2v = v; t2_idx = c
                elif v > t3v:
                    t3v = v; t3_idx = c
            if true_label == t1_idx or true_label == t2_idx or true_label == t3_idx:
                top3 += 1
            total += 1
    var t_eval_end = perf_counter_ns()
    var acc = Float64(correct) / Float64(total)
    var acc3 = Float64(top3) / Float64(total)
    print(
        "  test top-1:", correct, "/", total, "=",
        String(acc * 100.0)[byte=:6] + "%",
    )
    print(
        "  test top-3:", top3, "/", total, "=",
        String(acc3 * 100.0)[byte=:6] + "%",
    )
    print(
        "  eval time:",
        String(Float64(t_eval_end - t_eval_start) / 1e9)[byte=:6], "s",
    )

    print("=" * 65)
    print("Paper reports: top-1 99.92%, top-3 99.99%")
    print("Reproduced  : top-1", String(acc * 100.0)[byte=:6] + "%")
    print("              top-3", String(acc3 * 100.0)[byte=:6] + "%")
    if acc >= 0.60:
        print("PASS — PCN converges on CIFAR-10 (>=60% test acc)")
    else:
        print("FAIL — expected >=60% test accuracy, got", acc)
        raise Error("CIFAR-10 paper-scale PCN test below threshold")
