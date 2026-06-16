"""Test — Adam state save/load → training-equivalence round-trip.

Phase A.2 validation. The strongest check that Adam's Saveable
conformance captures every field needed for bit-exact resume.

Protocol:
  Build identical models + Adams (same seed for init). For each:

    BASELINE: train 200 steps straight.
    RESUMED:  train 100 steps → save model + Adam → reload into a fresh
              copy → train another 100 steps.

  At the end, weights of BASELINE and RESUMED must match within
  text-round-trip tolerance (1e-5). If Adam's `m_flat` / `v_flat` /
  `step_count` / `beta1_pow_t` / `beta2_pow_t` aren't all captured,
  the second 100 steps diverge wildly from the baseline.
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.random import seed
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.checkpoint import save_state_v2, load_state_v2
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.loss.mse import MSELoss


comptime BATCH = 4
comptime IN = 4
comptime HID = 8
comptime OUT = 2
comptime MLP = Sequential[
    Linear[IN, HID], ReLU[HID], Linear[HID, OUT],
]


def test_two_phase_training_bit_identical() raises:
    print("test_two_phase_training_bit_identical ...")
    var model_path = String("/tmp/test_nn_opt_state_model.txt")
    var opt_path = String("/tmp/test_nn_opt_state_adam.txt")

    # Shared inputs.
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var y_t_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for i in range(BATCH * IN):
        x[i] = Scalar[DT](0.1 * Float64(i + 1))
    for i in range(BATCH * OUT):
        y_t_buf[i] = Scalar[DT](0.05 * Float64(i + 1))
    var x_tt = TileTensor(x, row_major[BATCH, IN]())
    var yt_tt = TileTensor(y_t_buf, row_major[BATCH, OUT]())

    # ────────────────────────── BASELINE ──────────────────────────
    seed(42)
    var net_base = MLP.make[target="cpu", INIT=Kaiming]()
    var opt_base = Adam.make[target="cpu", M=MLP](net_base)
    opt_base.lr = Scalar[DT](1e-2)
    var loss_b = MSELoss[OUT].make[target="cpu"]()
    var yp_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var yp_b_tt = TileTensor(yp_b, row_major[BATCH, OUT]())
    var go_b_tt = TileTensor(go_b, row_major[BATCH, OUT]())
    var gi_b_tt = TileTensor(gi_b, row_major[BATCH, IN]())
    for _ in range(200):
        opt_base.zero_grad["cpu", M=MLP](net_base)
        net_base.forward["cpu", BATCH](x_tt, output=yp_b_tt)
        _ = loss_b.forward["cpu", BATCH](yp_b_tt, yt_tt)
        loss_b.vjp["cpu", BATCH](yt_tt, go_b_tt)
        net_base.vjp["cpu", BATCH](go_b_tt, gi_b_tt)
        opt_base.step["cpu", M=MLP](net_base)

    # ───────────────────────── RESUMED, phase 1 ─────────────────────────
    seed(42)  # same seed → same Kaiming init
    var net_a = MLP.make[target="cpu", INIT=Kaiming]()
    var opt_a = Adam.make[target="cpu", M=MLP](net_a)
    opt_a.lr = Scalar[DT](1e-2)
    var loss_a = MSELoss[OUT].make[target="cpu"]()
    var yp_a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var yp_a_tt = TileTensor(yp_a, row_major[BATCH, OUT]())
    var go_a_tt = TileTensor(go_a, row_major[BATCH, OUT]())
    var gi_a_tt = TileTensor(gi_a, row_major[BATCH, IN]())
    for _ in range(100):
        opt_a.zero_grad["cpu", M=MLP](net_a)
        net_a.forward["cpu", BATCH](x_tt, output=yp_a_tt)
        _ = loss_a.forward["cpu", BATCH](yp_a_tt, yt_tt)
        loss_a.vjp["cpu", BATCH](yt_tt, go_a_tt)
        net_a.vjp["cpu", BATCH](go_a_tt, gi_a_tt)
        opt_a.step["cpu", M=MLP](net_a)

    # Save model + Adam state.
    save_state_v2[MLP](net_a, model_path)
    var adam_dump = String("")
    opt_a.save(adam_dump, String("adam"))
    with open(opt_path, "w") as f:
        f.write(adam_dump)

    # ─────────────────── RESUMED, fresh copy, phase 2 ───────────────────
    var net_b = MLP.make[target="cpu", INIT=Kaiming]()
    var opt_b = Adam.make[target="cpu", M=MLP](net_b)
    opt_b.lr = Scalar[DT](999.0)  # nonsense — load must overwrite

    load_state_v2[MLP](net_b, model_path)

    # Load Adam from file directly via opt_b.load.
    var adam_content: String
    with open(opt_path, "r") as f:
        adam_content = String(f.read())
    var lines = List[String]()
    var current = String("")
    var bytes = adam_content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(c))
    var idx = 0
    opt_b.load(lines, idx, String("adam"))

    # Train 100 more steps from the resumed state.
    var loss_c = MSELoss[OUT].make[target="cpu"]()
    var yp_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var yp_c_tt = TileTensor(yp_c, row_major[BATCH, OUT]())
    var go_c_tt = TileTensor(go_c, row_major[BATCH, OUT]())
    var gi_c_tt = TileTensor(gi_c, row_major[BATCH, IN]())
    for _ in range(100):
        opt_b.zero_grad["cpu", M=MLP](net_b)
        net_b.forward["cpu", BATCH](x_tt, output=yp_c_tt)
        _ = loss_c.forward["cpu", BATCH](yp_c_tt, yt_tt)
        loss_c.vjp["cpu", BATCH](yt_tt, go_c_tt)
        net_b.vjp["cpu", BATCH](go_c_tt, gi_c_tt)
        opt_b.step["cpu", M=MLP](net_b)

    # ─── Compare baseline vs resumed final state via forward outputs ───
    var x2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for i in range(BATCH * IN):
        x2[i] = Scalar[DT](-0.05 + 0.03 * Float64(i))
    var y_base: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var y_res: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var x2_tt = TileTensor(x2, row_major[BATCH, IN]())
    var yb_tt = TileTensor(y_base, row_major[BATCH, OUT]())
    var yr_tt = TileTensor(y_res, row_major[BATCH, OUT]())
    net_base.forward["cpu", BATCH](x2_tt, output=yb_tt)
    net_b.forward["cpu", BATCH](x2_tt, output=yr_tt)

    var max_diff: Scalar[DT] = 0.0
    for i in range(BATCH * OUT):
        var d = y_base[i] - y_res[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y_baseline - y_resumed| =", max_diff)

    # Round-trip goes through text-stringified Adam scalars, which
    # introduces ~last-bit noise. 1e-5 is a generous bound for 100
    # additional Adam steps from near-identical state.
    assert_true(
        max_diff < Scalar[DT](1e-5),
        "Two-phase training: resumed weights must equal baseline within 1e-5"
    )
    print("  ok (max_diff =", max_diff, ")")

    # Cross-check: Adam counters round-tripped.
    assert_equal(opt_b.step_count, opt_a.step_count + 100)
    assert_true(
        opt_b.lr == Scalar[DT](1e-2),
        "Adam.lr must be overwritten by load (was 999.0 pre-load)"
    )
    print("  step_count + lr round-trip: ok")


def main() raises:
    print("=" * 70)
    print("Adam state save/load training-equivalence (Phase A.2)")
    print("=" * 70)
    test_two_phase_training_bit_identical()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
