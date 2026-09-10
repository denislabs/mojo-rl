"""`optimal_margin` RETURNS THE MARGIN THAT MAXIMISES THE GRADIENT.

    pixi run mojo run -I . tests/tasks/test_optimal_margin.mojo

## ⚠⚠ THE GATE SWEEPS; IT DOES NOT RE-DERIVE

The claim is about the real `tolerance`, so the check measures the real
`tolerance`: for each shortfall it sweeps margins on a fine grid, finds the
one whose finite-difference gradient is steepest, and requires
`optimal_margin` to land on it. Restating `k / sqrt(2)` in the test would
share the formula with the code and gate nothing — the two would have to be
wrong together, which is exactly how a closed form gets shipped wrong.

## ⚠ WHY IT MATTERS

Both margins ever hand-picked for `lift` were wrong, in opposite directions.
0.02 against a 0.030 m shortfall is a gradient of 1.9/m and a run whose
return never moved; 0.10 is a term already at 0.81 before the policy acts,
and a 1M-step run that bought +0.034 reward per step and left the brick on
the table. The peak is 24.5/m at 0.046.
"""

from std.math import abs as fabs

from mojo_rl.envs.dm_control.rewards import (
    tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN
)
from mojo_rl.tasks.shaping import optimal_margin

comptime DT = DType.float64


def term(d: Float64, radius: Float64, margin: Float64) -> Float64:
    return Float64(
        tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
            Scalar[DT](d), Scalar[DT](0), Scalar[DT](radius),
            Scalar[DT](margin),
        )
    )


def grad_at(d: Float64, radius: Float64, margin: Float64) -> Float64:
    var h = 1e-7
    return fabs(
        (term(d + h, radius, margin) - term(d - h, radius, margin)) / (2.0 * h)
    )


def main() raises:
    print("=== optimal_margin: does it find the steepest margin? ===")
    var fails = 0

    # Shortfalls spanning the three shipped tasks: `On` (~0), `Above` (0.03),
    # `Near` (0.14), plus the reach term's ~0.10.
    var dists = [0.005, 0.01, 0.03, 0.05, 0.098, 0.14, 0.30]
    for i in range(len(dists)):
        var d = dists[i]
        var got = optimal_margin(d)

        # ⚠ THE SWEEP IS THE REFERENCE. 4000 margins from a twentieth of the
        # shortfall to ten times it — wide enough that the peak is interior
        # and a wrong answer has somewhere to be wrong.
        var lo = d * 0.05
        var hi = d * 10.0
        var best_m = lo
        var best_g = -1.0
        for j in range(4001):
            var m = lo + (hi - lo) * Float64(j) / 4000.0
            var g = grad_at(d, 0.0, m)
            if g > best_g:
                best_g = g
                best_m = m
        var rel = fabs(got - best_m) / best_m
        var g_got = grad_at(d, 0.0, got)
        print("  shortfall", d, ": swept peak", best_m, " optimal_margin",
              got, " rel", rel, " gradient", g_got, "vs", best_g)
        if rel > 0.01:
            print("    FAIL: the closed form is not at the swept peak")
            fails += 1
        # ⚠ AND THE GRADIENT ITSELF MUST MATCH, since a flat neighbourhood
        # would make the margin comparison pass on a curve with no peak.
        if g_got < 0.999 * best_g:
            print("    FAIL: the returned margin's gradient is below the"
                  " swept maximum")
            fails += 1
        # The peak lands the term at exp(-1); that is the second half of the
        # claim and is what gives the headroom.
        var t = term(d, 0.0, got)
        if t < 0.363 or t > 0.373:
            print("    FAIL: term at reset", t, "is not exp(-1) = 0.368")
            fails += 1

    # ⚠ ANTI-VACUITY: a wrong margin must actually be caught. Half and double
    # the recommendation and require a strictly worse gradient — if the curve
    # were flat, every leg above would pass on a formula that returned
    # anything.
    var d0 = 0.03
    var m0 = optimal_margin(d0)
    var g0 = grad_at(d0, 0.0, m0)
    var g_half = grad_at(d0, 0.0, m0 * 0.5)
    var g_dbl = grad_at(d0, 0.0, m0 * 2.0)
    print("  discrimination at", d0, ": half", g_half, " opt", g0,
          " double", g_dbl)
    if g_half >= g0 or g_dbl >= g0:
        print("    FAIL: halving or doubling the margin did not reduce the"
              " gradient — the curve has no peak and the check is vacuous")
        fails += 1

    # ⚠ A TERM ALREADY INSIDE ITS RADIUS HAS NOTHING TO CLOSE. `settle`'s
    # goal holds at reset, and a margin recommendation for it is meaningless
    # rather than merely large.
    var raised = False
    try:
        _ = optimal_margin(0.0)
    except:
        raised = True
    if not raised:
        print("    FAIL: a zero shortfall returned a margin instead of"
              " raising")
        fails += 1

    print()
    if fails == 0:
        print("=== PASS ===")
    else:
        raise Error("optimal_margin: " + String(fails) + " check(s) failed")
