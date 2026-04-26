"""Small-budget CIFAR-10 PCN test — runs the paper's exact MLP for a few
batches just to demonstrate the algorithm functions at the paper's scale.

This is NOT a convergence test. With naive matmul kernels, a full epoch of
the paper's setup (B=500, T_infer=50, T_learn=500, 100 batches/epoch) would
take many hours. We run a small subset to validate:
  1. The pipeline doesn't OOM/crash at 3M-param scale.
  2. Train accuracy on the trained subset rises above the 10% random baseline.

Architecture (paper, exact):
    PCLinear[3072, 1000]
    PCLinear[1000, 500]
    PCLinear[500, 10]
    PCLinear[10, 10, PCIdentity]   # readout

Run:
    pixi run -e apple  mojo run -I . tests/nn_pc/test_pc_cifar10.mojo
    pixi run -e nvidia mojo run -I . tests/nn_pc/test_pc_cifar10.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets.cifar10 import CIFAR10
from mojo_rl.nn_pc import PCLinear, PCIdentity, PCTrainer


comptime BATCH = 50               # smaller than paper's 500 to fit in time budget
comptime N_TRAIN_BATCHES = 20     # small budget — not a full epoch
comptime N_TRAINEVAL_BATCHES = 5  # 250 train samples re-evaluated for train acc
comptime T_INFER = 50             # match paper
comptime T_LEARN = 200            # less than paper's 500 to save time
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
    print("CIFAR-10 PCN — small-budget smoke test (paper architecture)")
    print("=" * 65)
    print("  arch       : 3072 → 1000 → 500 → 10 (paper MLP)")
    print("  params     :", TRAINER.MODEL.PARAM_SIZE)
    print(
        "  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER,
        " T_LEARN=", T_LEARN, " batches=", N_TRAIN_BATCHES,
    )

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

    # ── Upload only the slice of training data we'll touch ──
    comptime N_USED = N_TRAIN_BATCHES * BATCH
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        N_USED * CIFAR10.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        N_USED * CIFAR10.NUM_CLASSES
    )
    for i in range(N_USED * CIFAR10.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(N_USED * CIFAR10.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(N_USED):
        train_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.train_labels[i])
        ] = Scalar[dtype](1.0)
    var train_img_dbuf = ctx.enqueue_create_buffer[dtype](
        N_USED * CIFAR10.IMG_SIZE
    )
    var train_tgt_dbuf = ctx.enqueue_create_buffer[dtype](
        N_USED * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_dbuf, train_img_host)
    ctx.enqueue_copy(train_tgt_dbuf, train_tgt_host)

    # ── Per-batch latents ──
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

    # ── Train ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    for b in range(N_TRAIN_BATCHES):
        TRAINER.randn_init_latents[BATCH](
            lat_init_view, seed=UInt64(1000) + UInt64(b), offset=UInt64(0)
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)

        var batch_x = LayoutTensor[
            dtype, Layout.row_major(BATCH, CIFAR10.IMG_SIZE), MutAnyOrigin
        ](train_img_dbuf.unsafe_ptr() + b * BATCH * CIFAR10.IMG_SIZE)
        var batch_y = LayoutTensor[
            dtype, Layout.row_major(BATCH, CIFAR10.NUM_CLASSES), MutAnyOrigin
        ](train_tgt_dbuf.unsafe_ptr() + b * BATCH * CIFAR10.NUM_CLASSES)

        TRAINER.train_one_batch_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, batch_y,
            T_infer=T_INFER, T_learn=T_LEARN,
            eta_infer=Scalar[dtype](ETA_INFER),
            eta_learn=Scalar[dtype](ETA_LEARN),
        )
        ctx.synchronize()
        var t_now = perf_counter_ns()
        print(
            "  batch", b + 1, "/", N_TRAIN_BATCHES,
            "  elapsed:", String(Float64(t_now - t0) / 1e9)[byte=:6], "s",
        )
    var t1 = perf_counter_ns()
    print("  total training time:", String(Float64(t1 - t0) / 1e9)[byte=:6], "s")

    # ── Train-set accuracy on the SAME batches we trained on ──
    print("\n── Train-set accuracy (free inference) ──")
    var y_hat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * CIFAR10.NUM_CLASSES)
    var y_hat_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * CIFAR10.NUM_CLASSES
    )
    var y_hat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CIFAR10.NUM_CLASSES), MutAnyOrigin
    ](y_hat_dbuf)

    var correct: Int = 0
    var total: Int = 0
    for batch_idx in range(N_TRAINEVAL_BATCHES):
        TRAINER.randn_init_latents[BATCH](
            lat_init_view, seed=UInt64(7000) + UInt64(batch_idx), offset=UInt64(0)
        )
        ctx.enqueue_copy(lat_dbuf, lat_host_init)
        var batch_x = LayoutTensor[
            dtype, Layout.row_major(BATCH, CIFAR10.IMG_SIZE), MutAnyOrigin
        ](train_img_dbuf.unsafe_ptr() + batch_idx * BATCH * CIFAR10.IMG_SIZE)
        TRAINER.inference_gpu[BATCH](
            ctx, params_t, lat_t, batch_x, y_hat_t,
            T_infer=T_INFER, eta_infer=Scalar[dtype](ETA_INFER),
        )
        ctx.enqueue_copy(y_hat_host, y_hat_dbuf)
        ctx.synchronize()
        for b in range(BATCH):
            var best_idx = 0
            var best_val = y_hat_host.unsafe_ptr()[b * CIFAR10.NUM_CLASSES + 0]
            for c in range(1, CIFAR10.NUM_CLASSES):
                var v = y_hat_host.unsafe_ptr()[b * CIFAR10.NUM_CLASSES + c]
                if v > best_val:
                    best_val = v
                    best_idx = c
            var true_label = Int(ds.train_labels[batch_idx * BATCH + b])
            if best_idx == true_label:
                correct += 1
            total += 1

    var acc = Float64(correct) / Float64(total)
    print(
        "  train-subset accuracy:", correct, "/", total, "=",
        String(acc * 100.0)[byte=:6] + "%",
    )

    print("=" * 65)
    if acc > 0.10:
        print(
            "PASS — CIFAR-10 pipeline runs at paper scale and learns above"
            " random baseline (>10%) within budget."
        )
    else:
        print("FAIL — accuracy at random baseline (10%); check algorithm.")
        raise Error("CIFAR-10 small-budget test failed")
