"""The Euler schedule and the per-token concat — checked analytically.

Both pieces have failure modes that produce finite, plausibly-scaled garbage,
and both can be pinned exactly without a reference dump:

  1. **The schedule.** `time` must be exactly 1.0, 0.9, … 0.1 — it starts at 1
     and never reaches 0. Off-by-one variants (`1.0 + (step+1)*dt`, or a range
     that includes 0) shift every conditioning value the model sees.
  2. **The sign and the step count, together.** With a CONSTANT velocity field
     `v ≡ c`, the integral is `Σ dt·c = -c` exactly, so after `STEPS` advances
     `x == x0 - c`. That single identity catches a flipped `dt` (would give
     `x0 + c`), a wrong step count (`x0 - k·c/STEPS`), and a `dt` of the wrong
     magnitude. No model required.
  3. **The concat is PER TOKEN**, and the time half is the SAME vector in every
     token. `Concat2[CHUNK*DA, CHUNK*DB]` has the identical total width and is
     wrong; the test builds a case where the two layouts differ everywhere.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_flow.mojo
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.flow import EulerSchedule, token_concat

comptime STEPS = 10
comptime E = EulerSchedule[STEPS]
comptime N = 64


def main() raises:
    print("=" * 70)
    print("SmolVLA flow-matching sampler")
    print("=" * 70)

    # ── 1. the schedule ──────────────────────────────────────────────────
    print("  dt =", E.dt())
    assert_true(abs(E.dt() + 0.1) < 1e-12, "dt must be -1/STEPS")
    assert_true(E.dt() < 0.0, "dt must be NEGATIVE — t runs 1 -> 0")
    var bad = 0
    for s in range(STEPS):
        var want = 1.0 - 0.1 * Float64(s)
        if abs(E.time_at(s) - want) > 1e-9:
            bad += 1
    print("  [1] times: compared", STEPS, " wrong", bad, " first",
          E.time_at(0), " last", E.time_at(STEPS - 1))
    assert_true(bad == 0, "the time schedule is wrong")
    assert_true(abs(E.time_at(0) - 1.0) < 1e-12, "must start at exactly 1.0")
    assert_true(E.time_at(STEPS - 1) > 0.0, "time must never reach 0")
    var raised = False
    try:
        _ = E.time_at(STEPS)
    except:
        raised = True
    assert_true(raised, "an out-of-range step did not raise")

    # ── 2. constant velocity: x_final == x0 - c, exactly ─────────────────
    var x = Tensor.alloc(N)
    var x0 = Tensor.alloc(N)
    var v = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
        x0.data[i] = x.data[i]
        v.data[i] = Scalar[DT](((i * 53) % 23) - 11) * 0.07
    for _ in range(STEPS):
        E.advance["cpu", N](x, v, None)
    var worst = Scalar[DT](0)
    for i in range(N):
        var want = x0.data[i] - v.data[i]      # Σ dt·v = -v
        var d = abs(x.data[i] - want)
        if d > worst:
            worst = d
    print("  [2] constant v: x_final vs x0 - c over", N, "elems, worst", worst)
    assert_true(worst < Scalar[DT](1e-5), "the integrated sum is not -c —"
                                          " sign, magnitude or step count")

    # a flipped sign would land on x0 + c; make sure that is far away
    var sep = Scalar[DT](0)
    for i in range(N):
        var wrong = x0.data[i] + v.data[i]
        var dd = abs(x.data[i] - wrong)
        if dd > sep:
            sep = dd
    assert_true(sep > Scalar[DT](1e-3), "x0-c and x0+c are indistinguishable"
                                        " on this input — the sign check is"
                                        " vacuous")
    print("       distance from the sign-flipped answer:", sep)

    # ── 3. per-token concat ──────────────────────────────────────────────
    comptime BB = 2
    comptime SEQ = 3
    comptime DA = 4
    comptime DB = 2
    var a = Tensor.alloc(BB * SEQ * DA)
    var b = Tensor.alloc(DB)
    for i in range(BB * SEQ * DA):
        a.data[i] = Scalar[DT](100 + i)
    for i in range(DB):
        b.data[i] = Scalar[DT](-1 - i)
    var dst = Tensor.alloc(BB * SEQ * (DA + DB))
    token_concat["cpu", BB, SEQ, DA, DB](a, b, dst, None)
    var cbad = 0
    for bi in range(BB):
        for t in range(SEQ):
            var o = bi * (SEQ * (DA + DB)) + t * (DA + DB)
            for d in range(DA):
                if dst.data[o + d] != a.data[bi * (SEQ * DA) + t * DA + d]:
                    cbad += 1
            for d in range(DB):
                if dst.data[o + DA + d] != b.data[d]:
                    cbad += 1
    print("  [3] token concat: compared", BB * SEQ * (DA + DB), " wrong", cbad)
    assert_true(cbad == 0, "the per-token concat is wrong")

    # a whole-row concat would put a[0,1] at index DA; assert it does not
    assert_true(
        dst.data[DA] == b.data[0],
        "index DA holds the action stream, not the time vector — this is the"
        " whole-row layout Concat2 would produce",
    )

    # ── 4. GPU parity ────────────────────────────────────────────────────
    var c = DeviceContext()
    var gx = Tensor.alloc(N)
    var gv = Tensor.alloc(N)
    for i in range(N):
        gx.data[i] = x0.data[i]
        gv.data[i] = v.data[i]
    gx.upload(c)
    gv.upload(c)
    for _ in range(STEPS):
        E.advance["gpu", N](gx, gv, Optional(c))
    gx.download(c)
    var gbad = 0
    for i in range(N):
        if abs(gx.data[i] - x.data[i]) > Scalar[DT](1e-5):
            gbad += 1

    var ga = Tensor.alloc(BB * SEQ * DA)
    var gb = Tensor.alloc(DB)
    for i in range(BB * SEQ * DA):
        ga.data[i] = a.data[i]
    for i in range(DB):
        gb.data[i] = b.data[i]
    ga.upload(c)
    gb.upload(c)
    var gd = Tensor.alloc(BB * SEQ * (DA + DB))
    token_concat["gpu", BB, SEQ, DA, DB](ga, gb, gd, Optional(c))
    gd.download(c)
    for i in range(BB * SEQ * (DA + DB)):
        if gd.data[i] != dst.data[i]:
            gbad += 1
    print("  [4] GPU vs CPU (advance + concat): wrong", gbad)
    assert_true(gbad == 0, "the GPU paths disagree with the CPU ones")

    print()
    print("PASSED")
