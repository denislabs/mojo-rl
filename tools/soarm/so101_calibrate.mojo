# +--------------------------------------------------------------------------+ #
# | Calibrate an SO-101 arm — the write half of `read_calibration`
# +--------------------------------------------------------------------------+ #
"""Set `Homing_Offset` / `Min_Position_Limit` / `Max_Position_Limit` in EEPROM.

    # 1. ALWAYS back up first. Reads only; writes a JSON file.
    pixi run mojo run -I . tools/soarm/so101_calibrate.mojo \\
        --port /dev/cu.usbmodem5B8E1139971 --backup follower.json

    # 2. A dry run: sweeps, computes, prints what it WOULD write. No EEPROM.
    pixi run mojo run -I . tools/soarm/so101_calibrate.mojo \\
        --port /dev/cu.usbmodem5B8E1139971

    # 3. The real thing.
    pixi run mojo run -I . tools/soarm/so101_calibrate.mojo \\
        --port /dev/cu.usbmodem5B8E1139971 --write --backup follower.json

    # and if it goes wrong:
    pixi run mojo run -I . tools/soarm/so101_calibrate.mojo \\
        --port ... --restore follower.json --write

The Mojo equivalent of `lerobot-calibrate`. `mojo_rl/robot/so101/arm.mojo`
already READS this out of the servos — deliberately, so no JSON parser is
needed at run time — and this is the other direction.

⚠⚠ **THIS WRITES EEPROM, AND A WRONG VALUE IS INVISIBLE AFTERWARDS.** Every
joint angle this repo computes comes from these three numbers
(`deg = (raw - mid) * 360 / 4095`, `mid = (min + max) / 2`). Corrupt them and
nothing errors — every recording, every policy and every sim-to-real
comparison is quietly wrong by an offset. **`--backup` before `--write` is not
advice.** The backup is plain JSON in `lerobot-calibrate`'s own format, so it
also restores with lerobot's tooling if this one is unavailable.

⚠ **WRITING IS OPT-IN (`--write`).** The default sweeps and prints. This is
the same lesson `examples/so101/record.mojo` carries: a safety property that
depends on remembering a flag is not a safety property.

⚠ **NOTHING HERE ENERGISES A MOTOR.** Torque is disabled throughout — the
operator moves the arm BY HAND, which is the only way to find its real end
stops. `set_torque(False)` also clears `Lock`, which is what makes the EEPROM
writable at all (`arm.mojo:311`).

## What the numbers mean

1. `Homing_Offset` is zeroed, so `Present_Position` reads the servo's ABSOLUTE
   count. The arm is then placed in its middle pose and the offset is set to
   `present - 2047`, which makes that pose read 2047 — the centre of the turn.
   ⚠ The magnitude is limited to 2047 by the direction bit at 11, so a pose
   further than half a turn from centre cannot be encoded; this refuses rather
   than wrapping.
2. The operator sweeps every joint to both end stops. The extremes of the
   OFFSET-APPLIED positions become `range_min` / `range_max`.
3. ⚠ **A CONTINUOUS JOINT KEEPS `0 .. 4095`.** `wrist_roll` on this arm turns
   without end stops, and `SO101Calibration.is_unlimited` treats that exact
   pair as the marker. A sweep would record whatever arc the operator happened
   to turn through and silently convert a continuous joint into a limited one.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.io.fileio import StdinReader, read_file_bytes, write_file_atomic
from mojo_rl.io.json import JsonWriter, parse_json
from mojo_rl.robot.feetech.control_table import (
    SIZE_2, STS_HOMING_OFFSET, STS_MAX_POSITION_LIMIT, STS_MIN_POSITION_LIMIT,
)
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name, joint_short
from mojo_rl.utils.fmt import fixed


comptime CENTRE = 2047
"""Half of the 0..4095 turn. `Homing_Offset` is set so the middle pose reads
this, matching `lerobot-calibrate`."""

comptime MAX_OFFSET_MAG = 2047
"""The direction bit sits at 11, so the magnitude field is 11 bits."""

comptime UNLIMITED_MIN = 0
comptime UNLIMITED_MAX = 4095

comptime SWEEP_POLL_MS = 20


struct Cal(Copyable, Movable):
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


def _read_cal(mut arm: SO101Arm) raises -> Cal:
    var c = Cal()
    for i in range(SO101_N):
        c.homing[i] = arm.cal.homing_offset[i]
        c.rmin[i] = arm.cal.range_min[i]
        c.rmax[i] = arm.cal.range_max[i]
    return c^


def _print_cal(label: String, ref c: Cal) raises:
    print("  " + label)
    print("    joint    homing   range_min   range_max   span")
    for i in range(SO101_N):
        var span = Int(c.rmax[i]) - Int(c.rmin[i])
        var note = String("")
        if Int(c.rmin[i]) == UNLIMITED_MIN and Int(c.rmax[i]) == UNLIMITED_MAX:
            note = "   (continuous)"
        print(
            "    " + joint_short(i) + "   " + String(Int(c.homing[i]))
            + "        " + String(Int(c.rmin[i])) + "        "
            + String(Int(c.rmax[i])) + "     " + String(span) + note
        )


def _save(path: String, ref c: Cal) raises:
    """Write `lerobot-calibrate`'s own JSON shape, so it restores either way."""
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


def _load(path: String) raises -> Cal:
    var doc = parse_json(read_file_bytes(path))
    var r = doc.root()
    var c = Cal()
    for i in range(SO101_N):
        var node = doc.field(r, joint_name(i))
        if node < 0:
            raise Error(
                "calibrate: " + path + " has no entry for " + joint_name(i)
            )
        c.homing[i] = Int32(
            doc.integer(doc.field(node, String("homing_offset")))
        )
        c.rmin[i] = Int32(doc.integer(doc.field(node, String("range_min"))))
        c.rmax[i] = Int32(doc.integer(doc.field(node, String("range_max"))))
    return c^


def _apply(mut arm: SO101Arm, ref c: Cal, write: Bool) raises:
    """Write a calibration into EEPROM and read it back to prove it landed."""
    if not write:
        print("  (dry run — nothing written)")
        return
    # `set_torque(False)` also clears `Lock`, which is what unlocks the EEPROM.
    arm.set_torque(False)
    for i in range(SO101_N):
        var id = arm.ids[i]
        arm.bus.write_register(
            id, STS_HOMING_OFFSET, Int(c.homing[i]), SIZE_2
        )
        arm.bus.write_register(
            id, STS_MIN_POSITION_LIMIT, Int(c.rmin[i]), SIZE_2
        )
        arm.bus.write_register(
            id, STS_MAX_POSITION_LIMIT, Int(c.rmax[i]), SIZE_2
        )

    # ⚠ READ BACK, ALWAYS. A write that the servo dropped — because `Lock` was
    # still set, or the packet was lost — reports nothing. The only way to
    # know the EEPROM changed is to read it.
    arm.read_calibration()
    var back = _read_cal(arm)
    var bad = 0
    for i in range(SO101_N):
        if (
            back.homing[i] != c.homing[i]
            or back.rmin[i] != c.rmin[i]
            or back.rmax[i] != c.rmax[i]
        ):
            print(
                "    ⚠ " + joint_name(i) + ": wrote ("
                + String(Int(c.homing[i])) + ", " + String(Int(c.rmin[i]))
                + ", " + String(Int(c.rmax[i])) + ") read back ("
                + String(Int(back.homing[i])) + ", "
                + String(Int(back.rmin[i])) + ", "
                + String(Int(back.rmax[i])) + ")"
            )
            bad += 1
    if bad != 0:
        raise Error(
            "calibrate: " + String(bad) + " joint(s) did not take the write —"
            " the arm's calibration may now be inconsistent. Restore the"
            " backup before using it."
        )
    print("  written and verified on all " + String(SO101_N) + " joints")


def main() raises:
    var port = String("")
    var backup = String("")
    var restore = String("")
    var write = False
    var sweep_s = 30
    var continuous = List[Int]()

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--port" and i + 1 < len(args):
            port = String(args[i + 1])
        elif a == "--backup" and i + 1 < len(args):
            backup = String(args[i + 1])
        elif a == "--restore" and i + 1 < len(args):
            restore = String(args[i + 1])
        elif a == "--seconds" and i + 1 < len(args):
            sweep_s = Int(String(args[i + 1]))
        elif a == "--write":
            write = True
    # `wrist_roll` turns without end stops on this arm; see the header.
    continuous.append(4)

    if port == "":
        raise Error(
            "calibrate: --port </dev/cu.usbmodem...> is required. Both arms"
            " look alike on the bus and calibrating the wrong one is"
            " silently destructive."
        )

    print("=" * 72)
    print("SO-101 calibration — " + port)
    print("=" * 72)
    if write:
        print("  ⚠⚠ --write: THIS WILL OVERWRITE THE ARM'S EEPROM")
    else:
        print("  dry run (no --write): sweeps and prints, changes nothing")
    print("")

    var arm = SO101Arm(port.copy(), max_step_ticks=0)
    arm.bus.timeout_ms = 50
    arm.set_torque(False)

    var current = _read_cal(arm)
    _print_cal(String("current calibration, as stored:"), current)
    print("")

    if backup != "":
        _save(backup, current)
        print("  backed up to " + backup)
        print("")

    # ── restore mode: put a saved calibration back and stop ───────────
    if restore != "":
        var saved = _load(restore)
        _print_cal(String("restoring from " + restore + ":"), saved)
        if write and backup == "":
            raise Error(
                "calibrate: --restore --write without --backup. Write down"
                " what is on the arm now before replacing it."
            )
        _apply(arm, saved, write)
        return

    var stdin = StdinReader()

    # ── step 1: zero the offsets so positions read absolute ──────────
    print("── 1. middle pose ─────────────────────────────────────────")
    print(
        "  Move the arm BY HAND to its middle/rest pose — every joint near"
        " the centre of its travel — then press Enter."
    )
    stdin.discard_pending()
    _ = stdin.line()

    var zero = Cal()
    for i in range(SO101_N):
        zero.homing[i] = 0
        zero.rmin[i] = Int32(UNLIMITED_MIN)
        zero.rmax[i] = Int32(UNLIMITED_MAX)
    # ⚠ The offsets must actually be zero on the SERVO for the next read to be
    # absolute. In a dry run they are not, so the numbers below are relative
    # to the OLD offsets and are printed as an estimate, not a result.
    _apply(arm, zero, write)

    var raw = InlineArray[Int32, SO101_N](fill=0)
    if arm.read_positions(Span(raw)) != SO101_N:
        raise Error("calibrate: could not read all six positions")

    var next_cal = Cal()
    for i in range(SO101_N):
        var off = Int(raw[i]) - CENTRE
        if off > MAX_OFFSET_MAG or off < -MAX_OFFSET_MAG:
            raise Error(
                "calibrate: " + joint_name(i) + " sits at " + String(Int(raw[i]))
                + ", which needs a homing offset of " + String(off)
                + " — outside the +/-" + String(MAX_OFFSET_MAG)
                + " the 11-bit direction-bit encoding can hold. Move that"
                " joint closer to the middle of its travel and start again."
            )
        next_cal.homing[i] = Int32(off)
    print("")

    # ── step 2: sweep ────────────────────────────────────────────────
    print("── 2. sweep ───────────────────────────────────────────────")
    print(
        "  Move EVERY joint slowly to BOTH end stops, by hand. "
        + String(sweep_s) + " s. Press Enter to start."
    )
    stdin.discard_pending()
    _ = stdin.line()

    var lo = InlineArray[Int32, SO101_N](fill=0)
    var hi = InlineArray[Int32, SO101_N](fill=0)
    var seeded = False
    var t_end = perf_counter_ns() + sweep_s * 1_000_000_000
    var samples = 0
    var partial = 0
    while perf_counter_ns() < t_end:
        if arm.read_positions(Span(raw)) != SO101_N:
            partial += 1
            continue
        if not seeded:
            for i in range(SO101_N):
                lo[i] = raw[i]
                hi[i] = raw[i]
            seeded = True
        else:
            for i in range(SO101_N):
                if raw[i] < lo[i]:
                    lo[i] = raw[i]
                if raw[i] > hi[i]:
                    hi[i] = raw[i]
        samples += 1
        if samples % 50 == 0:
            var line = String("    ")
            for i in range(SO101_N):
                line += (
                    joint_short(i) + " " + String(Int(hi[i]) - Int(lo[i])) + "  "
                )
            print(line)

    if not seeded or samples < 10:
        raise Error(
            "calibrate: only " + String(samples) + " good reads during the"
            " sweep — check the bus before trusting anything here"
        )
    print(
        "  " + String(samples) + " samples, " + String(partial)
        + " partial reads"
    )
    print("")

    # ── step 3: assemble ─────────────────────────────────────────────
    for i in range(SO101_N):
        next_cal.rmin[i] = lo[i]
        next_cal.rmax[i] = hi[i]
    for k in range(len(continuous)):
        var i = continuous[k]
        # See the header: a continuous joint must keep the 0..4095 marker.
        next_cal.rmin[i] = Int32(UNLIMITED_MIN)
        next_cal.rmax[i] = Int32(UNLIMITED_MAX)
        print(
            "  " + joint_name(i) + " kept at 0..4095 (continuous joint, its"
            " swept arc is not a limit)"
        )

    var narrow = 0
    for i in range(SO101_N):
        var span = Int(next_cal.rmax[i]) - Int(next_cal.rmin[i])
        if span < 200:
            print(
                "  ⚠ " + joint_name(i) + " swept only " + String(span)
                + " ticks (" + fixed(Float64(span) * 360.0 / 4095.0, 1)
                + " deg) — that joint was probably not moved"
            )
            narrow += 1
    print("")
    _print_cal(String("proposed calibration:"), next_cal)
    print("")

    if narrow > 0 and write:
        raise Error(
            "calibrate: " + String(narrow) + " joint(s) barely moved, so this"
            " sweep does not describe the arm. Refusing to write it. Run"
            " again and move every joint to both stops."
        )

    if not write:
        print(
            "  dry run — nothing was written. Re-run with --write --backup"
            " <file> to apply."
        )
        return

    print("  apply this to the arm? [yes/N] ")
    stdin.discard_pending()
    var confirm = stdin.line()
    if confirm != "yes":
        print("  not applied. The arm still has the offsets zeroed by step 1 —")
        print("  restore with:  --restore " + backup + " --write")
        return
    _apply(arm, next_cal, write)
    print("\ndone. Verify with:  pixi run soarm-diag")
