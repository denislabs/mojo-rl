"""Smoke + behaviour test for Trainer.train_gpu_minibatch_full.

Trains a tiny MLP on a separable 2-class synthetic dataset and asserts:
  - LR schedule history matches the configured CosineWarmupSchedule
    (warmup ramp 1/W, 2/W, ... and the post-warmup cosine values are
    monotonically non-increasing).
  - Final top-1 accuracy on the held-out validation set exceeds 0.9.
  - Validation loss decreased meaningfully end-to-end.

Uses IdentityAugmenter — the augmenter trait is exercised but augment
is a no-op, so this also verifies the one-time raw→aug copy path.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_trainer_full.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_trainer_full.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.random.philox import Random as PhiloxRandom
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Sequential, LinearReLU, Linear
from mojo_rl.nn.training import (
    Trainer,
    GPUNetworkState,
    CosineWarmupSchedule,
    IdentityAugmenter,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.initializer import Xavier
from layout import Layout, LayoutTensor


comptime IN_DIM = 4
comptime HIDDEN = 16
comptime N_CLASSES = 2

comptime BATCH = 32
comptime N_TRAIN = 256
comptime N_VAL = 128
comptime EPOCHS = 20
comptime WARMUP_EPOCHS = 5

comptime MODEL = Sequential[LinearReLU[IN_DIM, HIDDEN], Linear[HIDDEN, N_CLASSES]]
comptime OPT = Adam[1e-2, 0.9, 0.999, 1e-8]
comptime SCHED = CosineWarmupSchedule[WARMUP_EPOCHS, 0.1]
comptime TRAINER_T = Trainer[MODEL, OPT, CrossEntropyLoss]


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def main() raises:
    seed(7)
    var fails = 0

    print("=" * 70)
    print("Trainer.train_gpu_minibatch_full — schedule + eval + identity aug")
    print("=" * 70)

    var ctx = DeviceContext()
    var state = TRAINER_T.init_state_gpu[Xavier[]](ctx)

    # ──────────────────────────────────────────────────────────────────────
    # Synthetic separable dataset:
    #   class 0  ⟺  x[0] < 0
    #   class 1  ⟺  x[0] >= 0
    # All other dims are noise. Trivially learnable, but exercises every
    # piece of the new method (shuffle, schedule, aug, eval).
    # ──────────────────────────────────────────────────────────────────────

    var train_in_host = ctx.enqueue_create_host_buffer[dtype](N_TRAIN * IN_DIM)
    var train_tg_host = ctx.enqueue_create_host_buffer[dtype](
        N_TRAIN * N_CLASSES
    )
    for i in range(N_TRAIN * IN_DIM):
        train_in_host[i] = 0
    for i in range(N_TRAIN * N_CLASSES):
        train_tg_host[i] = 0

    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    for i in range(N_TRAIN):
        for d in range(IN_DIM):
            var r = rng.step_uniform()
            train_in_host[i * IN_DIM + d] = Scalar[dtype](
                Float32(r[0]) * 2.0 - 1.0
            )
        var label = 1 if train_in_host[i * IN_DIM + 0] >= 0 else 0
        train_tg_host[i * N_CLASSES + label] = 1

    var val_in_host = ctx.enqueue_create_host_buffer[dtype](N_VAL * IN_DIM)
    var val_lb_host = ctx.enqueue_create_host_buffer[DType.int32](N_VAL)
    for i in range(N_VAL * IN_DIM):
        val_in_host[i] = 0
    for i in range(N_VAL):
        val_lb_host[i] = 0
    for i in range(N_VAL):
        for d in range(IN_DIM):
            var r = rng.step_uniform()
            val_in_host[i * IN_DIM + d] = Scalar[dtype](
                Float32(r[0]) * 2.0 - 1.0
            )
        val_lb_host[i] = Int32(
            1 if val_in_host[i * IN_DIM + 0] >= 0 else 0
        )

    var train_in_buf = ctx.enqueue_create_buffer[dtype](N_TRAIN * IN_DIM)
    var train_tg_buf = ctx.enqueue_create_buffer[dtype](N_TRAIN * N_CLASSES)
    var val_in_buf = ctx.enqueue_create_buffer[dtype](N_VAL * IN_DIM)
    var val_lb_buf = ctx.enqueue_create_buffer[DType.int32](N_VAL)
    ctx.enqueue_copy(train_in_buf, train_in_host)
    ctx.enqueue_copy(train_tg_buf, train_tg_host)
    ctx.enqueue_copy(val_in_buf, val_in_host)
    ctx.enqueue_copy(val_lb_buf, val_lb_host)

    var train_in_lt = LayoutTensor[
        dtype, Layout.row_major(N_TRAIN, IN_DIM), MutAnyOrigin
    ](train_in_buf.unsafe_ptr())
    var train_tg_lt = LayoutTensor[
        dtype, Layout.row_major(N_TRAIN, N_CLASSES), MutAnyOrigin
    ](train_tg_buf.unsafe_ptr())
    var val_in_lt = LayoutTensor[
        dtype, Layout.row_major(N_VAL, IN_DIM), MutAnyOrigin
    ](val_in_buf.unsafe_ptr())
    var val_lb_lt = LayoutTensor[
        DType.int32, Layout.row_major(N_VAL), MutAnyOrigin
    ](val_lb_buf.unsafe_ptr())

    # ──────────────────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────────────────
    var result = TRAINER_T.train_gpu_minibatch_full[
        BATCH, N_TRAIN, N_VAL, SCHED, IdentityAugmenter
    ](
        state,
        ctx,
        train_in_lt,
        train_tg_lt,
        val_in_lt,
        val_lb_lt,
        epochs=EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        show_progress=False,
        eval_every_epochs=1,
        progress_label="Test",
    )

    print("")
    print("  epochs trained = " + String(result.epochs_trained))
    print("  final train loss = " + String(Float32(result.final_loss)))
    print(
        "  final val loss = "
        + String(Float32(result.val_loss_history[len(result.val_loss_history) - 1]))
        + "  final top1 = "
        + String(Float32(result.val_top1_history[len(result.val_top1_history) - 1]))
    )

    # ──────────────────────────────────────────────────────────────────────
    # Asserts
    # ──────────────────────────────────────────────────────────────────────
    check(
        len(result.lr_scale_history) == EPOCHS,
        "lr_scale_history has one entry per epoch",
        fails,
    )
    check(
        len(result.val_loss_history) == EPOCHS,
        "val_loss_history populated each epoch (eval_every=1)",
        fails,
    )
    check(
        len(result.val_top1_history) == EPOCHS,
        "val_top1_history populated each epoch",
        fails,
    )

    # Warmup ramp: epoch 0 → 1/W, epoch W-1 → 1.0 (or close).
    var warmup_first = result.lr_scale_history[0]
    var warmup_peak = result.lr_scale_history[WARMUP_EPOCHS - 1]
    check(
        math_abs(warmup_first - 1.0 / Float64(WARMUP_EPOCHS)) < 1e-6,
        "lr_scale_history[0] = 1/WARMUP_EPOCHS",
        fails,
    )
    check(
        math_abs(warmup_peak - 1.0) < 1e-6,
        "lr_scale_history[WARMUP-1] = 1.0",
        fails,
    )

    # Cosine decay: lr should be monotonically non-increasing after warmup.
    var monotone = True
    for i in range(WARMUP_EPOCHS, EPOCHS - 1):
        if (
            result.lr_scale_history[i + 1] - result.lr_scale_history[i]
            > 1e-6
        ):
            monotone = False
    check(monotone, "lr_scale is non-increasing after warmup", fails)

    # Final lr should hit the cosine MIN_SCALE bound (within ~5%).
    var final_lr = result.lr_scale_history[EPOCHS - 1]
    check(
        final_lr >= 0.05 and final_lr <= 0.20,
        "final lr_scale is near MIN_SCALE=0.1 (got "
        + String(Float32(final_lr))
        + ")",
        fails,
    )

    # End-to-end learning: final top-1 > 0.9 on a separable problem.
    var final_top1 = result.val_top1_history[len(result.val_top1_history) - 1]
    check(
        final_top1 > 0.9,
        "final top-1 > 0.9 (got " + String(Float32(final_top1)) + ")",
        fails,
    )

    # Val loss went down.
    var first_vloss = result.val_loss_history[0]
    var last_vloss = result.val_loss_history[len(result.val_loss_history) - 1]
    check(
        last_vloss < first_vloss * 0.5,
        "val loss decreased to <50% of first eval",
        fails,
    )

    print("")
    print("=" * 70)
    if fails == 0:
        print("ALL TRAINER FULL TESTS PASSED")
    else:
        print("FAILED: " + String(fails) + " checks")
    print("=" * 70)
