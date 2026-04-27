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
    Per-epoch shuffle (Philox Fisher-Yates over batch indices).

Reports BOTH test-time protocols side-by-side:
  - SUPERVISED inference (paper's protocol — y_target drives top latent
    during settling; comparable to paper's 99.92% / 99.99% claim).
    NOT a generalization metric — labels leak into inference.
  - FREE inference (eps_L = 0 during settling — honest generalization).
    This is what fairly compares to a backprop baseline / ViT.

Run:
    pixi run -e nvidia mojo run -I . tests/nn_pc/test_pc_cifar10_paper.mojo
"""

from std.math import sqrt
from std.random import seed
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets.cifar10 import CIFAR10
from mojo_rl.experimental.nn_pc import PCLinear, PCIdentity, PCTrainer


def _shuffle_indices(mut idx: List[Int], seed_val: UInt64) raises:
    """In-place Fisher-Yates shuffle using Philox uniform draws."""
    var n = len(idx)
    var rng = PhiloxRandom(seed=seed_val, offset=UInt64(0))
    for i in range(n - 1, 0, -1):
        var r = rng.step_uniform()
        var j = Int(Float32(r[0]) * Float32(i + 1))
        if j > i:
            j = i
        var tmp = idx[i]
        idx[i] = idx[j]
        idx[j] = tmp


# ── Paper hyperparameters (exact) ──────────────────────────────────────────
comptime BATCH = 500
comptime EPOCHS = 4
comptime T_INFER = 50
comptime T_LEARN = 500
comptime ETA_INFER: Float64 = 0.05
comptime ETA_LEARN: Float64 = 0.005

# ── Diagnostics ───────────────────────────────────────────────────────────
comptime DIAG_INTERVAL = 25  # run a diagnostic every N batches
comptime N_DIAG_BATCHES = 5  # 5 train batches × 500 = 2500 samples / diag

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

    # ── Shuffled batch index list, regenerated each epoch ──
    var batch_idx = List[Int](capacity=num_train_batches)
    for i in range(num_train_batches):
        batch_idx.append(i)

    # ── Diagnostic buffers (re-used across all checkpoints) ──
    var params_diag_host = ctx.enqueue_create_host_buffer[dtype](
        TRAINER.MODEL.PARAM_SIZE
    )
    var y_hat_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * CIFAR10.NUM_CLASSES
    )
    var y_hat_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * CIFAR10.NUM_CLASSES
    )
    var y_hat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CIFAR10.NUM_CLASSES), MutAnyOrigin
    ](y_hat_dbuf)

    # ── Train ──
    print("\n── Training ──")
    print(
        "  diag every", DIAG_INTERVAL, "batches on",
        N_DIAG_BATCHES * BATCH, "train samples",
    )
    print(
        "  format: [diag epoch.batch] |W|2= ...  train_acc= ...  sup_loss= ..."
    )
    var t0 = perf_counter_ns()
    var total_b: Int = 0
    for epoch in range(EPOCHS):
        # Shuffle batch order for this epoch
        _shuffle_indices(batch_idx, seed_val=UInt64(2024) + UInt64(epoch))

        var t_epoch_start = perf_counter_ns()
        for b in range(num_train_batches):
            var bi = batch_idx[b]  # actual batch index after shuffle

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
            ](train_img_dbuf.unsafe_ptr() + bi * BATCH * CIFAR10.IMG_SIZE)
            var batch_y = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CIFAR10.NUM_CLASSES),
                MutAnyOrigin,
            ](train_tgt_dbuf.unsafe_ptr() + bi * BATCH * CIFAR10.NUM_CLASSES)

            TRAINER.train_one_batch_gpu[BATCH](
                ctx, params_t, lat_t, batch_x, batch_y,
                T_infer=T_INFER, T_learn=T_LEARN,
                eta_infer=Scalar[dtype](ETA_INFER),
                eta_learn=Scalar[dtype](ETA_LEARN),
            )
            total_b += 1

            # ── Diagnostic checkpoint ──
            if (total_b == 1) or (total_b % DIAG_INTERVAL == 0):
                ctx.synchronize()

                # Param L2 norm
                ctx.enqueue_copy(params_diag_host, params_dbuf)
                ctx.synchronize()
                var l2_sq: Float64 = 0.0
                for i in range(TRAINER.MODEL.PARAM_SIZE):
                    var v = Float64(params_diag_host.unsafe_ptr()[i])
                    l2_sq += v * v
                var l2 = sqrt(l2_sq)

                # Train-subset accuracy via BOTH protocols (free + supervised)
                # so we can see if training is producing useful weights at all.
                var dfree_correct: Int = 0
                var dsup_correct: Int = 0
                var d_sup_loss: Float64 = 0.0
                var d_total: Int = 0
                for di in range(N_DIAG_BATCHES):
                    var diag_x = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, CIFAR10.IMG_SIZE),
                        MutAnyOrigin,
                    ](
                        train_img_dbuf.unsafe_ptr()
                        + di * BATCH * CIFAR10.IMG_SIZE
                    )
                    var diag_y = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, CIFAR10.NUM_CLASSES),
                        MutAnyOrigin,
                    ](
                        train_tgt_dbuf.unsafe_ptr()
                        + di * BATCH * CIFAR10.NUM_CLASSES
                    )

                    # ── (a) FREE inference ──
                    TRAINER.randn_init_latents[BATCH](
                        lat_init_view,
                        seed=UInt64(80000) + UInt64(di),
                        offset=UInt64(0),
                    )
                    ctx.enqueue_copy(lat_dbuf, lat_host_init)
                    TRAINER.inference_gpu[BATCH](
                        ctx, params_t, lat_t, diag_x, y_hat_t,
                        T_infer=T_INFER,
                        eta_infer=Scalar[dtype](ETA_INFER),
                    )
                    ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
                    ctx.synchronize()
                    for sb in range(BATCH):
                        var sample_idx = di * BATCH + sb
                        var best_idx = 0
                        var best_val = y_hat_host.unsafe_ptr()[
                            sb * CIFAR10.NUM_CLASSES + 0
                        ]
                        for c in range(1, CIFAR10.NUM_CLASSES):
                            var v = y_hat_host.unsafe_ptr()[
                                sb * CIFAR10.NUM_CLASSES + c
                            ]
                            if v > best_val:
                                best_val = v
                                best_idx = c
                        if best_idx == Int(ds.train_labels[sample_idx]):
                            dfree_correct += 1
                        # sup_loss tracked from FREE inference (consistent with above)
                        for c in range(CIFAR10.NUM_CLASSES):
                            var yh = Float64(
                                y_hat_host.unsafe_ptr()[
                                    sb * CIFAR10.NUM_CLASSES + c
                                ]
                            )
                            var yt = Float64(
                                train_tgt_host.unsafe_ptr()[
                                    sample_idx * CIFAR10.NUM_CLASSES + c
                                ]
                            )
                            var d = yh - yt
                            d_sup_loss += d * d
                        d_total += 1

                    # ── (b) SUPERVISED inference (same latent init seed) ──
                    TRAINER.randn_init_latents[BATCH](
                        lat_init_view,
                        seed=UInt64(80000) + UInt64(di),
                        offset=UInt64(0),
                    )
                    ctx.enqueue_copy(lat_dbuf, lat_host_init)
                    TRAINER.supervised_inference_gpu[BATCH](
                        ctx, params_t, lat_t, diag_x, diag_y, y_hat_t,
                        T_infer=T_INFER,
                        eta_infer=Scalar[dtype](ETA_INFER),
                    )
                    ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
                    ctx.synchronize()
                    for sb in range(BATCH):
                        var sample_idx = di * BATCH + sb
                        var best_idx = 0
                        var best_val = y_hat_host.unsafe_ptr()[
                            sb * CIFAR10.NUM_CLASSES + 0
                        ]
                        for c in range(1, CIFAR10.NUM_CLASSES):
                            var v = y_hat_host.unsafe_ptr()[
                                sb * CIFAR10.NUM_CLASSES + c
                            ]
                            if v > best_val:
                                best_val = v
                                best_idx = c
                        if best_idx == Int(ds.train_labels[sample_idx]):
                            dsup_correct += 1

                var dfree_acc = Float64(dfree_correct) / Float64(d_total)
                var dsup_acc = Float64(dsup_correct) / Float64(d_total)
                var d_loss_per_sample = 0.5 * d_sup_loss / Float64(d_total)
                var diag_line = (
                    "  [diag "
                    + String(epoch + 1) + "." + String(b + 1) + "]"
                    + " |W|2=" + String(l2)[byte=:8]
                    + " sup_acc=" + String(dsup_acc * 100.0)[byte=:6] + "%"
                    + " free_acc=" + String(dfree_acc * 100.0)[byte=:6] + "%"
                    + " sup_loss=" + String(d_loss_per_sample)[byte=:8]
                )
                print(diag_line)
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

    # ── Test-set evaluation: BOTH supervised and free inference ──
    print("\n── Evaluating on test set (both protocols) ──")

    # Upload test images
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_dbuf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_dbuf, test_img_host)

    # Upload one-hot test labels (used by supervised inference)
    var test_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.NUM_CLASSES
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.NUM_CLASSES):
        test_tgt_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(CIFAR10.N_TEST):
        test_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.test_labels[i])
        ] = Scalar[dtype](1.0)
    var test_tgt_dbuf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(test_tgt_dbuf, test_tgt_host)

    comptime num_test_batches = CIFAR10.N_TEST // BATCH

    # Per-protocol counters: (top1_correct, top3_correct)
    var sup_top1: Int = 0
    var sup_top3: Int = 0
    var free_top1: Int = 0
    var free_top3: Int = 0
    var total_eval: Int = 0
    var t_eval_start = perf_counter_ns()

    for test_b in range(num_test_batches):
        var batch_x = LayoutTensor[
            dtype, Layout.row_major(BATCH, CIFAR10.IMG_SIZE), MutAnyOrigin
        ](test_img_dbuf.unsafe_ptr() + test_b * BATCH * CIFAR10.IMG_SIZE)
        var batch_y = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, CIFAR10.NUM_CLASSES),
            MutAnyOrigin,
        ](test_tgt_dbuf.unsafe_ptr() + test_b * BATCH * CIFAR10.NUM_CLASSES)

        # ── (1) SUPERVISED inference (paper's protocol) ──
        TRAINER.randn_init_latents[BATCH](
            lat_init_view,
            seed=UInt64(9000) + UInt64(test_b),
            offset=UInt64(0),
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)
        TRAINER.supervised_inference_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, batch_y, y_hat_t,
            T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
        )
        ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
        ctx.synchronize()
        for b in range(BATCH):
            var true_label = Int(ds.test_labels[test_b * BATCH + b])
            # top-1 / top-3 in one pass
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
            if true_label == t1_idx:
                sup_top1 += 1
            if true_label == t1_idx or true_label == t2_idx or true_label == t3_idx:
                sup_top3 += 1

        # ── (2) FREE inference (honest generalization) ──
        TRAINER.randn_init_latents[BATCH](
            lat_init_view,
            seed=UInt64(9000) + UInt64(test_b),  # SAME seed for fair compare
            offset=UInt64(0),
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)
        TRAINER.inference_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, y_hat_t,
            T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
        )
        ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
        ctx.synchronize()
        for b in range(BATCH):
            var true_label = Int(ds.test_labels[test_b * BATCH + b])
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
            if true_label == t1_idx:
                free_top1 += 1
            if true_label == t1_idx or true_label == t2_idx or true_label == t3_idx:
                free_top3 += 1

        total_eval += BATCH

    var t_eval_end = perf_counter_ns()
    var sup_acc1 = Float64(sup_top1) / Float64(total_eval)
    var sup_acc3 = Float64(sup_top3) / Float64(total_eval)
    var free_acc1 = Float64(free_top1) / Float64(total_eval)
    var free_acc3 = Float64(free_top3) / Float64(total_eval)
    var eval_time_s = Float64(t_eval_end - t_eval_start) / 1e9
    print("  eval time: " + String(eval_time_s)[byte=:6] + " s")

    print("=" * 65)
    print("Paper claim (supervised protocol): top-1 99.92%, top-3 99.99%")
    print("-" * 65)
    var sup_line = (
        "SUPERVISED inference (paper protocol)  top1="
        + String(sup_acc1 * 100.0)[byte=:6] + "%"
        + "  top3="
        + String(sup_acc3 * 100.0)[byte=:6] + "%"
    )
    var free_line = (
        "FREE       inference (honest generalize) top1="
        + String(free_acc1 * 100.0)[byte=:6] + "%"
        + "  top3="
        + String(free_acc3 * 100.0)[byte=:6] + "%"
    )
    print(sup_line)
    print(free_line)
    print("=" * 65)

    # Pass criterion: free-inference must clear random baseline by a wide margin
    # (we treat free inference as the honest signal; supervised is a sanity check).
    if free_acc1 >= 0.30:
        print(
            "PASS — free-inference top-1 (",
            String(free_acc1 * 100.0)[byte=:6] + "%",
            ") clears 30% (3x random baseline)",
        )
    else:
        print(
            "FAIL — free-inference top-1 (",
            String(free_acc1 * 100.0)[byte=:6] + "%",
            ") below 30%",
        )
        raise Error("CIFAR-10 paper-scale free-inference below threshold")
