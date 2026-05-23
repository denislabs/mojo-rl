"""Test — Checkpoint v2 (Saveable-walked) round-trip.

Phase A.2 validation. Mirrors `tests/nn2/test_checkpoint.mojo` (v1)
but exercises the new `save_state_v2` / `load_state_v2` reflection
path, which depends on `Param` conforming to `Saveable`.

Two sub-tests:

  1. Single Linear round-trip: build Linear, save via v2, build a fresh
     Linear with a different init, load. Forward outputs must match
     bit-identically.
  2. Sequential MLP round-trip after training: train a few Adam steps,
     save, build a fresh MLP, load. Forward outputs must match within
     1e-6 (text round-trip slack for fp32).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.initializer import Kaiming, Zero
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.mse import MSELoss


def test_linear_roundtrip_v2() raises:
    print("test_linear_roundtrip_v2 ...")
    var path = String("/tmp/test_nn2_ckpt_v2_linear.txt")
    comptime BATCH = 2
    comptime IN = 3
    comptime OUT = 2

    var net1 = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    save_state_v2[Linear[IN, OUT]](net1, path)

    # Fresh net with Zero init — without load_state_v2 it would produce
    # all-zero output, very different from net1.
    var net2 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    load_state_v2[Linear[IN, OUT]](net2, path)

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var y1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var y2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for i in range(BATCH * IN):
        x[i] = Scalar[DT](0.1 * Float64(i + 1))

    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y1_t = TileTensor(y1, row_major[BATCH, OUT]())
    var y2_t = TileTensor(y2, row_major[BATCH, OUT]())
    net1.forward["cpu", BATCH](x_t, output=y1_t)
    net2.forward["cpu", BATCH](x_t, output=y2_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(BATCH * OUT):
        var d = y1[i] - y2[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y1 - y2| =", max_diff)
    # v2 round-trip should be bit-identical for clean fp32 values.
    assert_true(
        max_diff == Scalar[DT](0),
        "Linear v2 round-trip: forward outputs must match bit-identically"
    )
    print("  ok")


def test_sequential_roundtrip_v2_after_training() raises:
    """Mirror of the v1 test but exercising save/load_state_v2."""
    print("test_sequential_roundtrip_v2_after_training ...")
    var path = String("/tmp/test_nn2_ckpt_v2_sequential.txt")
    comptime BATCH = 4
    comptime IN = 4
    comptime HID = 8
    comptime OUT = 2

    comptime MLP = Sequential[
        Linear[IN, HID], ReLU[HID], Linear[HID, OUT],
    ]

    var net = MLP.make[target="cpu", INIT=Kaiming]()
    var opt = Adam.make[target="cpu", M=MLP](net)
    opt.lr = Scalar[DT](1e-2)
    var loss = MSELoss[OUT].make[target="cpu"]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var y_t_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var y_pred: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var grad_out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var grad_in: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for i in range(BATCH * IN):
        x[i] = Scalar[DT](0.1 * Float64(i + 1))
    for i in range(BATCH * OUT):
        y_t_buf[i] = Scalar[DT](0.05 * Float64(i + 1))

    var x_tt = TileTensor(x, row_major[BATCH, IN]())
    var yt_tt = TileTensor(y_t_buf, row_major[BATCH, OUT]())
    var yp_tt = TileTensor(y_pred, row_major[BATCH, OUT]())
    var go_tt = TileTensor(grad_out, row_major[BATCH, OUT]())
    var gi_tt = TileTensor(grad_in, row_major[BATCH, IN]())

    for _ in range(5):
        opt.zero_grad["cpu", M=MLP](net)
        net.forward["cpu", BATCH](x_tt, output=yp_tt)
        _ = loss.forward["cpu", BATCH](yp_tt, yt_tt)
        loss.vjp["cpu", BATCH](yt_tt, go_tt)
        net.vjp["cpu", BATCH](go_tt, gi_tt)
        opt.step["cpu", M=MLP](net)

    save_state_v2[MLP](net, path)

    var fresh = MLP.make[target="cpu", INIT=Zero]()
    load_state_v2[MLP](fresh, path)

    var x2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var y_orig: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var y_loaded: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for i in range(BATCH * IN):
        x2[i] = Scalar[DT](-0.05 + 0.03 * Float64(i))

    var x2_tt = TileTensor(x2, row_major[BATCH, IN]())
    var yo_tt = TileTensor(y_orig, row_major[BATCH, OUT]())
    var yl_tt = TileTensor(y_loaded, row_major[BATCH, OUT]())
    net.forward["cpu", BATCH](x2_tt, output=yo_tt)
    fresh.forward["cpu", BATCH](x2_tt, output=yl_tt)

    var max_diff: Scalar[DT] = 0.0
    for i in range(BATCH * OUT):
        var d = y_orig[i] - y_loaded[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y_orig - y_loaded| =", max_diff)
    assert_true(
        max_diff < Scalar[DT](1e-6),
        "Sequential v2 round-trip: outputs must match within 1e-6"
    )
    print("  ok (max_diff =", max_diff, ")")


def main() raises:
    print("=" * 70)
    print("Checkpoint v2 save/load round-trip (Phase A.2)")
    print("=" * 70)
    test_linear_roundtrip_v2()
    test_sequential_roundtrip_v2_after_training()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
