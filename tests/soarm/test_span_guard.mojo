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

from noeira.robot.so101 import (
    NARROWER_FRACTION, SO101_N, UNLIMITED_MAX, UNLIMITED_MIN,
    CalibrationRecord, centre_on_middle_pose, frame_position, joint_name,
    span_regressions,
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

    # ── --limited wrist_roll: the range is CENTRED on the middle pose ──
    #
    # ⚠ The value that matters is the MID, not the span: degrees are measured
    # from (min + max) / 2, and a leader still carrying the 0..4095 marker has
    # its mid at 2047. An asymmetric cable-bounded sweep, +170 / -120 deg
    # (1934 / 1365 ticks), must come back mirrored to the tighter side.
    var lim = _mk([592,816,927,875,2047 - 1934,2030], [3317,3236,3130,3212,2047 + 1365,3513])
    var lost = centre_on_middle_pose(lim, 4, 2047)
    if Int(lim.rmin[4]) != 682 or Int(lim.rmax[4]) != 3412:
        raise Error(
            "centred range: expected 682..3412, got " + String(Int(lim.rmin[4]))
            + ".." + String(Int(lim.rmax[4]))
        )
    if Int(lim.rmin[4]) + Int(lim.rmax[4]) != 2 * 2047:
        raise Error("centred range: mid moved off the middle pose")
    if lost != 569:
        raise Error("centred range: expected 569 ticks dropped, got " + String(lost))
    print("  --limited: +170/-120 deg sweep -> 682..3412, mid 2047, 569 ticks dropped")
    checks += 3

    # The tighter side on the OTHER side, so a mutant that always keeps the
    # low half cannot pass both.
    var lim2 = _mk([592,816,927,875,2047 - 500,2030], [3317,3236,3130,3212,2047 + 1500,3513])
    _ = centre_on_middle_pose(lim2, 4, 2047)
    if Int(lim2.rmin[4]) != 1547 or Int(lim2.rmax[4]) != 2547:
        raise Error(
            "centred range (tight low side): expected 1547..2547, got "
            + String(Int(lim2.rmin[4])) + ".." + String(Int(lim2.rmax[4]))
        )
    # The other five joints are untouched.
    for i in range(SO101_N):
        if i == 4:
            continue
        if lim2.rmin[i] != old.rmin[i] or lim2.rmax[i] != old.rmax[i]:
            raise Error("centring wrist_roll changed " + joint_name(i))
    print("  --limited: tight low side -> 1547..2547; other joints untouched")
    checks += 2

    # A sweep that never crossed the middle pose cannot be centred.
    var one_sided = _mk([592,816,927,875,2100,2030], [3317,3236,3130,3212,3000,3513])
    var raised = False
    try:
        _ = centre_on_middle_pose(one_sided, 4, 2047)
    except:
        raised = True
    if not raised:
        raise Error("a sweep not containing the middle pose must be refused")
    print("  --limited: a one-sided sweep is refused")
    checks += 1

    # ── the seam: a limited sweep that reached the wrap is refused ────
    #
    # The 2026-09-14 dry run swept wrist_roll to exactly 0..4095 — the wrap,
    # not a stop — and centring turned it into 0..4094, a limit of nothing.
    var seam = _mk([592,816,927,875,0,2030], [3317,3236,3130,3212,4095,3513])
    var seam_raised = False
    try:
        _ = centre_on_middle_pose(seam, 4, 2047)
    except:
        seam_raised = True
    if not seam_raised:
        raise Error("a sweep reaching the encoder seam must be refused")
    var near = _mk([592,816,927,875,44,2030], [3317,3236,3130,3212,3000,3513])
    var near_raised = False
    try:
        _ = centre_on_middle_pose(near, 4, 2047)
    except:
        near_raised = True
    var inside_seam = _mk([592,816,927,875,45,2030], [3317,3236,3130,3212,3000,3513])
    _ = centre_on_middle_pose(inside_seam, 4, 2047)
    if not near_raised:
        raise Error("a sweep 44 ticks from the seam must be refused")
    print("  seam: 0..4095 refused, 44 ticks refused, 45 ticks accepted")
    checks += 3

    # ── the frame: limits must be recorded under the NEW offset ───────
    #
    # ⚠⚠ The 2026-09-14 follower, elbow_flex. Stored homing +382 (lerobot's),
    # stored limits 927..3130; a dry run read under that offset swept
    # 929..3133 and put the middle pose 19 ticks below 2047. The new offset is
    # therefore 382 - 19 = 363, and the invariant is PHYSICAL: a pose's
    # absolute count (reading + offset) is the same whichever frame read it.
    var old_h = 382
    var delta = -19
    var new_h = old_h + delta
    for reading in [929, 3133, 2047 + delta]:
        var moved = frame_position(reading, delta)
        if moved + new_h != reading + old_h:
            raise Error(
                "frame_position: reading " + String(reading) + " -> "
                + String(moved) + " is not the same pose under offset "
                + String(new_h)
            )
    if frame_position(2047 + delta, delta) != 2047:
        raise Error("frame_position: the middle pose must read 2047 in the new frame")
    # A shift that crosses the seam wraps rather than going negative.
    if frame_position(10, 50) != 4056 or frame_position(4090, -20) != 14:
        raise Error("frame_position must wrap at 4096")
    print("  frame: elbow_flex 929..3133 under +382 -> 948..3152 under +363 (same poses); wraps")
    checks += 3

    print("  " + String(checks) + " checks, 0 failures")
    print("[PASS] span-guard")
