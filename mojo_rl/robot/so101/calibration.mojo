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

from mojo_rl.io.fileio import read_file_bytes, write_file_atomic
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

    var homing: InlineArray[Int32, SO101_N]
    var rmin: InlineArray[Int32, SO101_N]
    var rmax: InlineArray[Int32, SO101_N]

    def __init__(out self):
        self.homing = InlineArray[Int32, SO101_N](fill=0)
        self.rmin = InlineArray[Int32, SO101_N](fill=0)
        self.rmax = InlineArray[Int32, SO101_N](fill=0)

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
    var b = List[UInt8]()
    for i in range(text.byte_length()):
        b.append(text.as_bytes()[i])
    write_file_atomic(path, b)


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
