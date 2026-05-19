"""Trainer[NET, OPT, LOSS, BATCH] tests — CPU + GPU.

CPU: XOR overfit. Same recipe as test_xor.mojo, but driven by Trainer
instead of the user wiring forward/backward/step by hand.

GPU: tiny dummy forward-only smoke (predict) to confirm host↔device
buffer plumbing works without needing the MNIST loader.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


def test_trainer_xor_cpu() raises:
    comptime IN = 2
    comptime HID = 4
    comptime OUT = 2
    comptime BATCH = 4
    comptime N_STEPS = 2000
    comptime LR: Scalar[DT] = 0.05

    seed(42)
    var net = Sequential(
        Linear[IN, HID].make["cpu", INIT=Kaiming](),
        ReLU[HID].make["cpu", INIT=Kaiming](),
        Linear[HID, OUT].make["cpu", INIT=Kaiming](),
    )
    var loss_fn = CrossEntropyLoss[OUT].make["cpu"]()
    var optim = Adam.make["cpu"](net, lr=LR)

    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH,
        target="cpu",
    ].make_from(net^, optim^, loss_fn^)

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var tg_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    in_buf[0] = 0.0; in_buf[1] = 0.0
    in_buf[2] = 0.0; in_buf[3] = 1.0
    in_buf[4] = 1.0; in_buf[5] = 0.0
    in_buf[6] = 1.0; in_buf[7] = 1.0
    for k in range(BATCH * OUT): tg_buf[k] = 0.0
    tg_buf[0 * OUT + 0] = 1.0
    tg_buf[1 * OUT + 1] = 1.0
    tg_buf[2 * OUT + 1] = 1.0
    tg_buf[3 * OUT + 0] = 1.0

    var initial_loss: Scalar[DT] = 0.0
    var final_loss:   Scalar[DT] = 0.0
    for step_i in range(N_STEPS):
        var L = trainer.train_step(in_buf, tg_buf)
        if step_i == 0:
            initial_loss = L
        final_loss = L

    print("  xor initial_loss=", initial_loss, " final_loss=", final_loss)
    assert_true(final_loss < Scalar[DT](0.01),
        "Expected final loss <0.01, got " + String(final_loss))

    # predict on the same 4 inputs
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    trainer.predict(in_buf, out_buf)
    var n_correct: Int = 0
    for b in range(BATCH):
        var pred: Int = 0
        if out_buf[b * OUT + 1] > out_buf[b * OUT + 0]:
            pred = 1
        var truth: Int = 0
        if tg_buf[b * OUT + 1] > tg_buf[b * OUT + 0]:
            truth = 1
        if pred == truth:
            n_correct += 1
    assert_true(n_correct == BATCH, "Expected 4/4 XOR after training")

    in_buf.free(); tg_buf.free(); out_buf.free()
    print("  test_trainer_xor_cpu PASSED")


def test_trainer_predict_gpu() raises:
    """GPU smoke: build a trainer on a tiny GPU net, run predict, ensure
    host buffers round-trip without errors."""
    comptime IN  = 3
    comptime HID = 4
    comptime OUT = 2
    comptime BATCH = 2

    var ctx = DeviceContext()
    var net = Sequential(
        Linear[IN, HID].make["gpu", INIT=Kaiming](ctx),
        ReLU[HID].make["gpu", INIT=Kaiming](ctx),
        Linear[HID, OUT].make["gpu", INIT=Kaiming](ctx),
        ctx=ctx,
    )
    var loss_fn = CrossEntropyLoss[OUT].make["gpu"](ctx)
    var optim = Adam.make["gpu"](net, ctx, lr=0.01)
    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH,
        target="gpu",
    ].make_from(net^, optim^, loss_fn^, ctx)

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var tg_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * IN):  in_buf[k] = Scalar[DT](0.1 * Float32(k))
    for k in range(BATCH * OUT): tg_buf[k] = 0.0
    tg_buf[0] = 1.0
    tg_buf[OUT + 1] = 1.0

    # One train step, then predict.
    var L = trainer.train_step(in_buf, tg_buf)
    trainer.predict(in_buf, out_buf)
    print("  gpu train_step loss=", L, " out[0,0]=", out_buf[0])

    in_buf.free(); tg_buf.free(); out_buf.free()
    print("  test_trainer_predict_gpu PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 Trainer tests (CPU + GPU)")
    print("=" * 60)
    test_trainer_xor_cpu()
    test_trainer_predict_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
