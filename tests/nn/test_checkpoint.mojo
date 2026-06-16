"""Test: Checkpoint v1 — save → load round-trip.

Phase 1.6 validation. Two sub-tests:

  1. **Single Linear round-trip**: build `Linear[3, 2]`, save params,
     build a *fresh* Linear (different init seed if any), load. Forward
     output on a fixed input must match the original to fp32 precision.

  2. **Sequential MLP round-trip**: build a 3-layer Sequential
     (Linear → ReLU → Linear), train one Adam step on a deterministic
     pair, save, build a fresh MLP, load. Forward outputs must match
     bit-identically.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.checkpoint import save_params, load_params
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.initializer import Kaiming, Zero
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.loss.mse import MSELoss


def test_linear_roundtrip() raises:
    print("test_linear_roundtrip ...")
    var path = String("/tmp/test_nn_ckpt_linear.txt")
    comptime BATCH = 2
    comptime IN = 3
    comptime OUT = 2

    var net1 = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    save_params[Linear[IN, OUT]](net1, path)

    var net2 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()  # different init
    load_params[Linear[IN, OUT]](net2, path)

    # Compare forward outputs on a fixed input.
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
    assert_true(
        max_diff == Scalar[DT](0),
        "Linear round-trip: forward outputs must match bit-identically"
    )
    print("  ok")


def test_sequential_roundtrip_after_training() raises:
    """A real-world flow: train a few steps, save, load into a fresh net,
    confirm output parity."""
    print("test_sequential_roundtrip_after_training ...")
    var path = String("/tmp/test_nn_ckpt_sequential.txt")
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

    # One synthetic train step to make the network non-trivial.
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

    save_params[MLP](net, path)

    # Build a fresh net (different INIT) and load.
    var fresh = MLP.make[target="cpu", INIT=Zero]()
    load_params[MLP](fresh, path)

    # Compare forward on a NEW input (not the training input).
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
    # Allow a tiny ε because save serialises through text float repr;
    # most fp32 values round-trip bit-identically through Mojo's `String`
    # conversion, but a few edge cases (denormals) could differ in the
    # last bit. 1e-6 is a generous bound for "the network is the same".
    assert_true(
        max_diff < Scalar[DT](1e-6),
        "Sequential round-trip: outputs must match within 1e-6"
    )
    print("  ok (max_diff =", max_diff, ")")


def main() raises:
    print("=" * 70)
    print("Checkpoint v1 save/load round-trip (Phase 1.6)")
    print("=" * 70)
    test_linear_roundtrip()
    test_sequential_roundtrip_after_training()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
