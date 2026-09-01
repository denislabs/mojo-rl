# +--------------------------------------------------------------------------+ #
# | The calibration span guard, on numbers from a real session
# +--------------------------------------------------------------------------+ #
"""Gate `robot/so101/calibration.mojo:span_regressions`.

    pixi run mojo run -I . tests/soarm/test_span_guard.mojo

No hardware, no serial port: the guard is a pure function of two calibrations,
which is exactly why it lives in a module rather than inside the tool.

⚠ THIS GATE IMPORTS THE RULE, IT DOES NOT RESTATE IT. A copy of the comparison
here would agree with itself forever while the tool drifted —
`_a_rule_written_inline_twice_drifts` is the most frequent defect shape in
this repo, and a calibration guard is a bad place to demonstrate it again.

## Where the numbers come from

The first case is a REAL session, 2026-09-01. The operator swept five joints
well and under-swept one: `shoulder_lift` came back 2420 -> 1726 ticks, 61
degrees of travel gone, and the absolute "span < 200" check let it through.
`write_goals` clamps to `[range_min, range_max]`, so the follower would simply
have stopped reaching poses it could reach that morning — with nothing
reported. The other five joints' scatter (-10, +8, +13, -12 ticks) is in the
same case, and must NOT flag: a guard that fires on a good sweep is a guard
that gets disabled.
"""

from mojo_rl.robot.so101 import (
    NARROWER_FRACTION, SO101_N, UNLIMITED_MAX, UNLIMITED_MIN,
    CalibrationRecord, joint_name, span_regressions,
)


def _mk(lo: List[Int], hi: List[Int]) -> CalibrationRecord:
    var c = CalibrationRecord()
    for i in range(SO101_N):
        c.rmin[i] = Int32(lo[i])
        c.rmax[i] = Int32(hi[i])
    return c^

def main() raises:
    print("[span-guard] gate")
    var skip = List[Int](); skip.append(4)   # wrist_roll continuous
    var checks = 0

    # ── the REAL session: only shoulder_lift must flag ────────────────
    var old = _mk([592,816,927,875,0,2030], [3317,3236,3130,3212,4095,3513])
    var new = _mk([173,1364,1300,826,0,1534], [2888,3090,3511,3176,4095,3005])
    var r = span_regressions(old, new, skip)
    if len(r) != 1 or r[0] != 1:
        var got = String("")
        for k in range(len(r)): got += joint_name(r[k]) + " "
        raise Error("real session: expected only shoulder_lift, got: " + got)
    print("  real session: flagged exactly shoulder_lift (2420 -> 1726)")
    checks += 1

    # ── an UNCALIBRATED baseline must never flag ──────────────────────
    var un = _mk([0,0,0,0,0,0], [4095,4095,4095,4095,4095,4095])
    if len(span_regressions(un, new, skip)) != 0:
        raise Error("an uncalibrated baseline must not flag: it has no baseline")
    print("  uncalibrated baseline: 0 flagged (correct — nothing to compare)")
    checks += 1

    # ── a good sweep (the session's own scatter) must not flag ────────
    var good = _mk([592,816,927,875,0,2030], [3307,3244,3138,3225,4095,3501])
    if len(span_regressions(old, good, skip)) != 0:
        raise Error("a +/-13 tick scatter must not flag")
    print("  good sweep (-10,+8,+13,-12 ticks): 0 flagged")
    checks += 1

    # ── just inside / just outside the 10% line ───────────────────────
    var inside = _mk([592,816,927,875,0,2030], [3317,3236,3130,3212,4095,3513])
    inside.rmax[0] = Int32(592 + 2453)      # 90.02% of 2725
    if len(span_regressions(old, inside, skip)) != 0:
        raise Error("90.02% must pass")
    var outside = _mk([592,816,927,875,0,2030], [3317,3236,3130,3212,4095,3513])
    outside.rmax[0] = Int32(592 + 2451)     # 89.94%
    var ro = span_regressions(old, outside, skip)
    if len(ro) != 1 or ro[0] != 0:
        raise Error("89.94% must flag")
    print("  threshold: 90.02% passes, 89.94% flags")
    checks += 2

    # ── a continuous joint is never flagged ───────────────────────────
    var narrowed_roll = _mk([592,816,927,875,1000,2030], [3317,3236,3130,3212,1100,3513])
    if len(span_regressions(old, narrowed_roll, skip)) != 0:
        raise Error("wrist_roll is skipped; it must never flag")
    print("  continuous joint: never flagged")
    checks += 1

    print("  " + String(checks) + " checks, 0 failures")
    print("[PASS] span-guard")
