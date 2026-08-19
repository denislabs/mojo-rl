"""`mju_wrap` against MuJoCo 3.10.0, pose by pose.

WHY THIS EXISTS
===============
`mju_wrap` is the whole of geom tendon wrapping: everything else in the
feature is bookkeeping around it. It is also ~230 lines of branchy 2D geometry
whose wrong answers are PLAUSIBLE — a tangent on the wrong side, or the minor
arc where the major one was meant, gives a tendon length that is simply a bit
different, never a NaN and never a crash. Nothing downstream can tell.

⚠⚠ THE ORACLE IS NOT CALLABLE. `mju_wrap` is absent from the Python bindings
(`dir(mujoco)` has only `MjsWrap` and `mjtWrap`), so the goldens come from a
real one-tendon model: the wrap POINTS are read straight off `d.wrap_xpos`,
and the arc length is `d.ten_length` minus the two straight runs. See
`scripts/dump_mujoco_wrap.py`, which also documents why each pose is there.

⚠ THE POSES ARE CHOSEN BY BRANCH, not by realism. Two of the eighteen do NOT
wrap, and they are as load-bearing as the sixteen that do — a `mju_wrap` that
wrapped everything would pass every length check while quietly shortening
every tendon that should have run straight.

Run: pixi run mojo run -I . tests/physics3d/test_mju_wrap_vs_mujoco.mojo
"""

from mojo_rl.physics3d.dynamics.wrap import mju_wrap, WrapOut
from mojo_rl.physics3d.gpu.constants import WRAP_SPHERE, WRAP_CYLINDER
from tests.physics3d.wrap_goldens import wrap_goldens, wrap_case_labels, WRAP_COLS


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def close(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        var d = got - want
        if d < 0:
            d = -d
        if d <= tol:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want, "|d|", d)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()
    var g = wrap_goldens()
    var labels = wrap_case_labels()
    var ncase = len(g) // WRAP_COLS
    print("=== mju_wrap vs MuJoCo 3.10.0 —", ncase, "poses ===")
    t.truth(ncase == len(labels), String("table is whole: ", ncase, " cases"))

    # ⚠ f64 THROUGHOUT. The goldens carry MuJoCo's full precision and the
    # solve is iterative (`wrap_inside` runs Newton to 1e-6), so an f32 port
    # would be gated against its own rounding rather than against MuJoCo.
    # `tolerance below the float32 noise floor` has cost this tree three
    # separate debugging sessions in the other direction.
    comptime TOL_LEN = 1e-9
    comptime TOL_PNT = 1e-9

    var nwrapped = 0
    var nstraight = 0

    for c in range(ncase):
        var b = c * WRAP_COLS
        var x0x = g[b + 0]
        var x0y = g[b + 1]
        var x0z = g[b + 2]
        var x1x = g[b + 3]
        var x1y = g[b + 4]
        var x1z = g[b + 5]
        var gx = g[b + 6]
        var gy = g[b + 7]
        var gz = g[b + 8]
        var radius = g[b + 18]
        # ⚠⚠ THE GOLDEN COLUMN IS THE GENERATOR'S OWN ENCODING (1 sphere,
        # 2 cylinder), TRANSLATED HERE. It used to be passed to `mju_wrap`
        # raw, which worked only while `WRAP_SPHERE` happened to be 1 — and
        # then `WRAP_SITE` was added to the enum, every value shifted, and 84
        # of these arms failed at once. A table that encodes another module's
        # constant by VALUE is a second spelling of it; keeping the file in a
        # neutral encoding and mapping at the boundary is what stops the next
        # renumber from silently reinterpreting the goldens.
        var wtype = WRAP_SPHERE if Int(g[b + 19]) == 1 else WRAP_CYLINDER
        var has_side = g[b + 20] > 0.5
        var sx = g[b + 21]
        var sy = g[b + 22]
        var sz = g[b + 23]
        var want_len = g[b + 24]

        print("---", labels[c])
        var r = mju_wrap[DType.float64](
            x0x, x0y, x0z,
            x1x, x1y, x1z,
            gx, gy, gz,
            g[b + 9], g[b + 10], g[b + 11],
            g[b + 12], g[b + 13], g[b + 14],
            g[b + 15], g[b + 16], g[b + 17],
            radius, wtype, has_side, sx, sy, sz,
        )

        if want_len < 0:
            nstraight += 1
            # ⚠ THE NEGATIVE ARM. `wlen < 0` is the ONLY signal the caller has
            # that the segment runs straight; a wrap invented here silently
            # shortens the tendon.
            t.truth(r.wlen < 0, "no wrap, and we agree")
        else:
            nwrapped += 1
            t.close(r.wlen, want_len, TOL_LEN, "arc length")
            t.close(r.p0x, g[b + 25], TOL_PNT, "w0.x")
            t.close(r.p0y, g[b + 26], TOL_PNT, "w0.y")
            t.close(r.p0z, g[b + 27], TOL_PNT, "w0.z")
            t.close(r.p1x, g[b + 28], TOL_PNT, "w1.x")
            t.close(r.p1y, g[b + 29], TOL_PNT, "w1.y")
            t.close(r.p1z, g[b + 30], TOL_PNT, "w1.z")

    # ⚠⚠ NON-VACUITY. Without these two the whole file could be a table of
    # `wlen < 0` rows compared against a function that always returns -1.
    print("--- the table exercises both outcomes ---")
    t.truth(nwrapped >= 12, String("poses that WRAP: ", nwrapped))
    t.truth(nstraight >= 2, String("poses that do NOT: ", nstraight))

    # ⚠ A wrap type we do not implement must decline, not guess. MuJoCo calls
    # `mjERROR` here; returning "no wrap" is the graceful equivalent and is
    # what `full_parser` relies on when it refuses a `<pulley>`.
    var bogus = mju_wrap[DType.float64](
        -0.5, 0.02, 0.0, 0.5, 0.02, 0.0, 0.0, 0.0, 0.0,
        1, 0, 0, 0, 1, 0, 0, 0, 1,
        0.1, 0, False, 0, 0, 0,
    )
    t.truth(bogus.wlen < 0, "an unsupported wrap type declines (control)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_mju_wrap_vs_mujoco: " + String(t.fails) + " failed")
