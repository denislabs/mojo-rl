# +--------------------------------------------------------------------------+ #
# | A calibration as DATA — saved, loaded, and checked before it is written
# +--------------------------------------------------------------------------+ #
"""`CalibrationRecord` plus the checks that decide whether it is safe to write.

`arm.mojo`'s `SO101Calibration` is the LIVE calibration, read out of the
servos and used to convert ticks to degrees on every tick of a control loop.
This is the same three numbers as a value that can be written to a file,
compared against another, and rejected — which is a different job, and one
that must not live inside a tool where a gate cannot reach it.

⚠ **THE COMPARISON LIVES HERE SO THERE IS EXACTLY ONE OF IT.** The obvious
alternative — the rule inline in `tools/soarm/so101_calibrate.mojo` and again
in its test — is the shape `_a_rule_written_inline_twice_drifts` names as this
repo's most frequent defect. `tests/soarm/test_span_guard.mojo` imports this
module; it does not restate the rule.

The JSON is `lerobot-calibrate`'s own shape, so a file written here restores
with lerobot's tooling and vice versa.
"""

from mojo_rl.io.fileio import read_file_bytes, write_text_atomic
from mojo_rl.io.json import JsonWriter, parse_json

from mojo_rl.robot.so101.arm import SO101_N, joint_name


comptime UNLIMITED_MIN = 0
comptime UNLIMITED_MAX = 4095
"""⚠ `range_min == 0 and range_max == 4095` is lerobot's marker for a joint
with NO END STOPS, not a measurement. `SO101Calibration.is_unlimited` reads it
the same way."""

comptime NARROWER_FRACTION = 0.90
"""A proposed span below this fraction of the joint's PREVIOUS span is treated
as an under-swept joint rather than a new calibration.

⚠ AN ABSOLUTE "did it move at all" CHECK IS NOT ENOUGH, AND THAT IS MEASURED.
A real session produced `shoulder_lift` 2420 -> 1726 ticks — **61 degrees of
travel gone** — and sailed past a `span < 200` guard. `write_goals` clamps
every goal to `[range_min, range_max]`, so the follower simply stops reaching
poses it used to; nothing errors, the arm just gets smaller. 10% is loose
enough that the scatter of a good sweep never trips it: the same session's
other five joints came in at -10, +8, +13 and -12 ticks."""


struct CalibrationRecord(Copyable, Movable):
    """`Homing_Offset` / `Min_Position_Limit` / `Max_Position_Limit`, per joint."""

    var homing: Array[Int32, SO101_N]
    var rmin: Array[Int32, SO101_N]
    var rmax: Array[Int32, SO101_N]

    def __init__(out self):
        self.homing = Array[Int32, SO101_N](fill=0)
        self.rmin = Array[Int32, SO101_N](fill=0)
        self.rmax = Array[Int32, SO101_N](fill=0)

    def __init__(out self, *, copy: Self):
        self.homing = copy.homing.copy()
        self.rmin = copy.rmin.copy()
        self.rmax = copy.rmax.copy()

    def __init__(out self, *, deinit move: Self):
        self.homing = move.homing^
        self.rmin = move.rmin^
        self.rmax = move.rmax^

    def span(self, i: Int) -> Int:
        return Int(self.rmax[i]) - Int(self.rmin[i])

    def is_unlimited(self, i: Int) -> Bool:
        return (
            Int(self.rmin[i]) == UNLIMITED_MIN
            and Int(self.rmax[i]) == UNLIMITED_MAX
        )


def span_regressions(
    ref previous: CalibrationRecord,
    ref proposed: CalibrationRecord,
    ref skip: List[Int],
) -> List[Int]:
    """Joints whose proposed travel is materially smaller than before.

    ⚠ ONLY WHERE THERE IS A BASELINE. An arm that was UNCALIBRATED reads
    `0..4095` on every joint, so every honest new span would look like a
    catastrophic regression — the guard would fire hardest exactly when
    calibration is most needed. A joint in `skip` (a continuous one) is
    excluded for the same reason: its `0..4095` is a marker, not a range.
    """
    var out = List[Int]()
    for i in range(SO101_N):
        var skipped = False
        for k in range(len(skip)):
            if skip[k] == i:
                skipped = True
        if skipped:
            continue
        if previous.is_unlimited(i):
            continue  # no baseline to compare against
        var old_span = previous.span(i)
        if old_span <= 0:
            continue
        if Float64(proposed.span(i)) < NARROWER_FRACTION * Float64(old_span):
            out.append(i)
    return out^


comptime SEAM_MARGIN = 45
"""Ticks (~4 deg) short of the encoder seam a limited sweep may reach. See
`centre_on_middle_pose`."""


def frame_position(reading: Int, shift: Int) -> Int:
    """A position read under one `Homing_Offset`, re-expressed under an offset
    `shift` ticks larger. Wraps at the 4096-tick turn.

    ⚠⚠ LIMITS LIVE IN THE HOMED FRAME, AND THAT IS WHAT THIS IS FOR. The servo
    reports `Present_Position = Actual - Homing_Offset`, and lerobot records
    `Min`/`Max_Position_Limit` from those homed readings. A sweep read under a
    DIFFERENT offset — zeroed, or the old calibration's in a dry run — gives
    limits shifted by the difference, and a shift keeps the span, so no span
    check can see it. Every sweep reading goes through here first.

    ⚠ THE WRAP IS REAL, NOT DEFENSIVE. A continuous joint crosses the seam, and
    `Present_Position` wraps with it; so does a joint whose old offset put the
    seam inside its travel.
    """
    return ((reading - shift) % 4096 + 4096) % 4096


def centre_on_middle_pose(
    mut c: CalibrationRecord, i: Int, centre: Int
) raises -> Int:
    """Limit joint `i` to a range SYMMETRIC about `centre`. Returns the ticks
    of swept travel given up to make it symmetric.

    For a normally continuous joint calibrated as limited (`--limited
    wrist_roll`, when a camera cable must not wind round the wrist).
    `c.rmin[i]` / `c.rmax[i]` hold the swept extremes on entry.

    ⚠⚠ A LIMITED RANGE MOVES THE JOINT'S ZERO UNLESS IT IS CENTRED.
    `SO101Calibration.degrees` measures from `mid = (range_min + range_max) /
    2`. A continuous joint's marker `0..4095` puts that at 2047 — the middle
    pose. A cable-bounded sweep of +170/-120 degrees would put it 25 degrees
    away, and teleop maps LEADER degrees to FOLLOWER degrees: a leader still
    carrying the marker would drive the follower's wrist 25 degrees off its
    own, silently, on every frame. Recording would capture that offset too.

    So the range is cut to the TIGHTER side, mirrored. The travel lost is
    returned so the tool can print it rather than hide it.
    """
    var lo = Int(c.rmin[i])
    var hi = Int(c.rmax[i])
    # ⚠⚠ THE SEAM. In the homed frame the middle pose reads 2047, so the
    # encoder wraps exactly half a turn away on either side. A sweep that got
    # within `SEAM_MARGIN` of 0 or 4095 turned (or nearly turned) past half a
    # turn, its extremes are the WRAP and not the operator's stop, and centring
    # them would produce 0..4094: a "limit" that limits nothing.
    if lo < SEAM_MARGIN or hi > 4095 - SEAM_MARGIN:
        raise Error(
            joint_name(i) + " swept " + String(lo) + ".." + String(hi)
            + ": it went (nearly) half a turn from the middle pose, where the"
            " encoder wraps, so these extremes are not stops. A limited joint"
            " must stay within +/-176 deg of the middle pose. Sweep it again,"
            " less far."
        )
    if lo > centre or hi < centre:
        raise Error(
            joint_name(i) + " swept " + String(lo) + ".." + String(hi)
            + ", which does not contain the middle pose (" + String(centre)
            + "). Sweep it to both sides of the middle pose."
        )
    var half = min(centre - lo, hi - centre)
    c.rmin[i] = Int32(centre - half)
    c.rmax[i] = Int32(centre + half)
    return (hi - lo) - 2 * half


def save_calibration_json(path: String, ref c: CalibrationRecord) raises:
    """Write `lerobot-calibrate`'s own JSON shape."""
    var w = JsonWriter()
    w.begin_object()
    for i in range(SO101_N):
        w.key(joint_name(i))
        w.begin_object()
        w.member(String("id"), i + 1)
        w.member(String("drive_mode"), 0)
        w.member(String("homing_offset"), Int(c.homing[i]))
        w.member(String("range_min"), Int(c.rmin[i]))
        w.member(String("range_max"), Int(c.rmax[i]))
        w.end_object()
    w.end_object()
    var text = w.done()
    write_text_atomic(path, text)


def load_calibration_json(path: String) raises -> CalibrationRecord:
    var doc = parse_json(read_file_bytes(path))
    var r = doc.root()
    var c = CalibrationRecord()
    for i in range(SO101_N):
        var node = doc.field(r, joint_name(i))
        if node < 0:
            raise Error(
                "calibration: " + path + " has no entry for " + joint_name(i)
            )
        c.homing[i] = Int32(
            doc.integer(doc.field(node, String("homing_offset")))
        )
        c.rmin[i] = Int32(doc.integer(doc.field(node, String("range_min"))))
        c.rmax[i] = Int32(doc.integer(doc.field(node, String("range_max"))))
    return c^
