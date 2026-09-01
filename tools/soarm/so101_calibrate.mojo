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

from mojo_rl.io.fileio import (
    StdinReader, read_file_bytes, stdout_is_tty, write_file_atomic,
)
from mojo_rl.io.hf import mojo_rl_cache
from mojo_rl.io.json import JsonWriter, parse_json
from std.os import makedirs
from mojo_rl.robot.feetech.control_table import (
    SIZE_2, STS_HOMING_OFFSET, STS_MAX_POSITION_LIMIT, STS_MIN_POSITION_LIMIT,
)
from mojo_rl.robot.so101 import (
    NARROWER_FRACTION, SO101Arm, SO101_N, UNLIMITED_MAX, UNLIMITED_MIN,
    CalibrationRecord, joint_name, joint_short, load_calibration_json,
    save_calibration_json, span_regressions,
)
from mojo_rl.utils.fmt import fixed


comptime CENTRE = 2047
"""Half of the 0..4095 turn. `Homing_Offset` is set so the middle pose reads
this, matching `lerobot-calibrate`."""

comptime MAX_OFFSET_MAG = 2047
"""The direction bit sits at 11, so the magnitude field is 11 bits."""

comptime SWEEP_POLL_MS = 20


def _read_cal(mut arm: SO101Arm) raises -> CalibrationRecord:
    var c = CalibrationRecord()
    for i in range(SO101_N):
        c.homing[i] = arm.cal.homing_offset[i]
        c.rmin[i] = arm.cal.range_min[i]
        c.rmax[i] = arm.cal.range_max[i]
    return c^


def _rj(v: Int, w: Int) -> String:
    var s = String(v)
    while s.byte_length() < w:
        s = " " + s
    return s^


def _lj(s: String, w: Int) -> String:
    var out = s.copy()
    while out.byte_length() < w:
        out += " "
    return out^


def _rows(
    ref lo: InlineArray[Int32, SO101_N],
    ref pos: InlineArray[Int32, SO101_N],
    ref hi: InlineArray[Int32, SO101_N],
    ref skip: List[Int],
    live: Bool,
) raises -> List[String]:
    """The table's lines. `live` shows POS; a final table does not need it."""
    var lines = List[String]()
    lines.append(_lj(String("NAME"), 16) + "|" + _rj_head(live))
    for i in range(SO101_N):
        var skipped = False
        for k in range(len(skip)):
            if skip[k] == i:
                skipped = True
        var row = _lj(joint_name(i), 16) + "|" + _rj(Int(lo[i]), 7) + " |"
        if live:
            row += _rj(Int(pos[i]), 7) + " |"
        row += _rj(Int(hi[i]), 7) + " |" + _rj(Int(hi[i]) - Int(lo[i]), 8)
        if skipped:
            row += "   continuous, leave it alone"
        lines.append(row^)
    return lines^


def _sweep_table(
    ref lo: InlineArray[Int32, SO101_N],
    ref pos: InlineArray[Int32, SO101_N],
    ref hi: InlineArray[Int32, SO101_N],
    ref skip: List[Int],
    redraw_lines: Int,
    plain: Bool,
) raises -> Int:
    """One frame of the live sweep table. Returns the line count it printed.

    ⚠ **REDRAWN IN PLACE, NOT APPENDED.** A row of numbers every 50 samples
    scrolls the terminal while the operator is watching a joint approach its
    stop — the thing lerobot's fixed table gets right.

    ⚠ **`\x1b[nF` THEN `\x1b[0J`, NOT `\x1b[nA` ALONE.** The first version
    moved up with `A`, which preserves the COLUMN and erases nothing; on a
    real terminal it left a duplicated header and, worse, put the cursor above
    where the tool believed it was — so the "proposed calibration" printed
    afterwards OVERWROTE earlier lines and came out missing two joints. `F`
    moves up *and* to column 0, and `0J` erases everything from there to the
    end of the screen, so residue is impossible whatever the line count did.

    ⚠ Only on a TTY, and `--plain` opts out entirely. Piped, there is no
    cursor to move and every redraw would append.
    """
    var live = stdout_is_tty() and not plain
    var lines = _rows(lo, pos, hi, skip, True)
    var block = String("")
    if live and redraw_lines > 0:
        block += "\x1b[" + String(redraw_lines) + "F" + "\x1b[0J"
    for i in range(len(lines)):
        block += lines[i]
        if i + 1 < len(lines):
            block += "\n"
    print(block)
    return len(lines)


def _rj_head(live: Bool) -> String:
    var h = _rj_str(String("MIN"), 7) + " |"
    if live:
        h += _rj_str(String("POS"), 7) + " |"
    return h + _rj_str(String("MAX"), 7) + " |" + _rj_str(String("SPAN"), 8)


def _rj_str(s: String, w: Int) -> String:
    var out = s.copy()
    while out.byte_length() < w:
        out = " " + out
    return out^


def _print_cal(label: String, ref c: CalibrationRecord) raises:
    """Aligned, full joint names — the same shape as the live sweep table.

    ⚠ The first version used `joint_short` and space padding by hand, and a
    single overwritten line came out with no label at all. Columns that line
    up make a corrupted row obvious instead of merely wrong.
    """
    print("  " + label)
    print(
        "    " + _lj(String("NAME"), 16) + "|" + _rj_str(String("HOMING"), 8)
        + " |" + _rj_str(String("MIN"), 7) + " |" + _rj_str(String("MAX"), 7)
        + " |" + _rj_str(String("SPAN"), 8)
    )
    for i in range(SO101_N):
        var span = Int(c.rmax[i]) - Int(c.rmin[i])
        var note = String("")
        if Int(c.rmin[i]) == UNLIMITED_MIN and Int(c.rmax[i]) == UNLIMITED_MAX:
            note = "   continuous"
        print(
            "    " + _lj(joint_name(i), 16) + "|" + _rj(Int(c.homing[i]), 8)
            + " |" + _rj(Int(c.rmin[i]), 7) + " |" + _rj(Int(c.rmax[i]), 7)
            + " |" + _rj(span, 8) + note
        )


def _auto_backup_path(port: String) raises -> String:
    """Where the automatic pre-write backup goes.

    Under the mojo-rl cache with the port and a timestamp, so a second
    calibration attempt cannot overwrite the copy that would restore the
    first.
    """
    var slug = String("")
    for i in range(port.byte_length()):
        var c = chr(Int(port.as_bytes()[i]))
        if c == "/" or c == "." or c == " ":
            slug += "_"
        else:
            slug += c
    var dir = mojo_rl_cache() + "/so101_calibration"
    makedirs(dir, exist_ok=True)
    return (
        dir + "/" + slug + "-" + String(perf_counter_ns() // 1_000_000_000)
        + ".json"
    )


def _apply(mut arm: SO101Arm, ref c: CalibrationRecord, write: Bool) raises:
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
    var plain = False
    var allow_narrower = False
    # A CAP on the sweep, not its length — ENTER ends it. Ten minutes is long
    # enough that an unhurried calibration never reaches it.
    var sweep_s = 600
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
        elif a == "--allow-narrower":
            # Deliberately shrinking a joint's range is legitimate — a new
            # physical limit, a changed gripper. It just must not be silent.
            allow_narrower = True
        elif a == "--plain":
            # Escape hatch: no cursor tricks, one table per redraw interval.
            plain = True
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

    # ⚠ **THE BACKUP IS AUTOMATIC, NOT OPT-IN.** Step 1 ZEROES the arm's
    # calibration so positions read absolute — which means the moment the
    # operator says "go", the previous calibration is gone from the servos and
    # exists only in a file. Making that file conditional on remembering
    # `--backup` is the same mistake `--dry-run` was in `record.mojo`.
    var auto_backup = _auto_backup_path(port)
    save_calibration_json(auto_backup, current)
    print("  backed up to " + auto_backup)
    if backup != "":
        save_calibration_json(backup, current)
        print("  backed up to " + backup)
    print("")

    # ── restore mode: put a saved calibration back and stop ───────────
    if restore != "":
        var saved = load_calibration_json(restore)
        # ⚠ The path on its own line. Inlined into the table's label it made
        # a ~120-character heading that WRAPS in a normal terminal, and a
        # wrapped heading directly above a fixed-width table is both hard to
        # read and hard to copy out intact.
        print("  restoring from:")
        print("    " + restore)
        _print_cal(String("values to write:"), saved)
        # ⚠ NO `--backup` REQUIREMENT HERE ANY MORE. This guard predates the
        # automatic backup and kept firing AFTER one had already been written
        # two lines above — refusing the restore and leaving the operator with
        # the calibration they were trying to replace. A precondition that the
        # code itself now always satisfies is not a guard, it is a bug.
        _apply(arm, saved, write)
        return

    var stdin = StdinReader()
    var committed = False

    # ⚠ **AN ABORTED CALIBRATION MUST NOT LEAVE THE ARM UNCALIBRATED.** Step 1
    # zeroes the offsets before the operator has committed to replacing them,
    # so every exit that is not a completed calibration — an exception, a
    # refused confirmation, a sweep that found nothing — has to put the old
    # values back. This happened for real on 2026-09-01: the operator ran out
    # of time mid-sweep and the follower was left with homing 0 and ranges
    # 0..4095 on all six joints.
    #
    # ⚠ A `finally` DOES NOT COVER Ctrl-C OR A KILL. If that happens, restore
    # by hand from the backup path printed above.
    try:
        committed = _calibrate(
            arm, stdin, continuous, sweep_s, write, auto_backup, plain,
            current, allow_narrower,
        )
    finally:
        if write and not committed:
            print(
                "\n  calibration did not complete — putting the previous"
                " values back ..."
            )
            try:
                _apply(arm, current, True)
                print("  restored.")
            except:
                print(
                    "  ⚠ COULD NOT RESTORE. The arm is UNCALIBRATED. Run:\n"
                    "    pixi run soarm-calibrate -- --port " + port
                    + " --restore " + auto_backup + " --write"
                )


def _calibrate(
    mut arm: SO101Arm,
    mut stdin: StdinReader,
    ref continuous: List[Int],
    sweep_s: Int,
    write: Bool,
    auto_backup: String,
    plain: Bool,
    ref previous: CalibrationRecord,
    allow_narrower: Bool,
) raises -> Bool:
    """Returns True only when a new calibration was actually applied.

    ⚠ EVERY OTHER EXIT MUST RESTORE. A dry run never wrote, so there is
    nothing to undo; a declined confirmation and a raised error both leave the
    arm with step 1's zeroed offsets, and the caller's `finally` puts the old
    values back. Returning True on a decline is how the arm gets left
    uncalibrated by a tool that reported success.
    """
    var raw = InlineArray[Int32, SO101_N](fill=0)

    # ── step 1: zero the offsets so positions read absolute ──────────
    print("── 1. middle pose ─────────────────────────────────────────")
    print(
        "  Move the arm BY HAND to its middle/rest pose — every joint near"
        " the centre of its travel — then press Enter."
    )
    stdin.discard_pending()
    _ = stdin.line()

    var zero = CalibrationRecord()
    for i in range(SO101_N):
        zero.homing[i] = 0
        zero.rmin[i] = Int32(UNLIMITED_MIN)
        zero.rmax[i] = Int32(UNLIMITED_MAX)
    # ⚠ The offsets must actually be zero on the SERVO for the next read to be
    # absolute. In a dry run they are not, so the numbers below are relative
    # to the OLD offsets and are printed as an estimate, not a result.
    _apply(arm, zero, write)

    if arm.read_positions(Span(raw)) != SO101_N:
        raise Error("calibrate: could not read all six positions")

    var next_cal = CalibrationRecord()
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
    var skip_names = String("")
    for k in range(len(continuous)):
        if k > 0:
            skip_names += ", "
        skip_names += joint_name(continuous[k])
    print(
        "  Move all joints EXCEPT " + skip_names + " through their FULL range"
        " of motion, one at a time, by hand."
    )
    print("  Recording positions. Press ENTER to stop.")
    stdin.discard_pending()
    print("")

    var lo = InlineArray[Int32, SO101_N](fill=0)
    var hi = InlineArray[Int32, SO101_N](fill=0)
    # ⚠ SEED BEFORE THE LOOP. Left at zero, a sweep that ends immediately
    # renders MIN and MAX as 0 beside a real POS — a table that looks like the
    # arm is at one end of a range it never had.
    var seeded = False
    if arm.read_positions(Span(raw)) == SO101_N:
        for i in range(SO101_N):
            lo[i] = raw[i]
            hi[i] = raw[i]
        seeded = True
    # ⚠ A CAP, NOT A SCHEDULE. The operator decides when the sweep is done —
    # the first version ran for a fixed `--seconds` and simply stopped
    # mid-calibration, which is not enough time to reach six pairs of end
    # stops. This only exists so a piped or unattended run terminates.
    var t_end = perf_counter_ns() + sweep_s * 1_000_000_000
    var samples = 0
    var partial = 0
    var drawn = 0
    var last_draw = perf_counter_ns()

    while True:
        if stdin.has_input():
            _ = stdin.line()
            break
        if perf_counter_ns() > t_end:
            print(
                "\n  reached the " + String(sweep_s) + " s cap (--seconds)"
            )
            break
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
        # ~10 Hz: fast enough to follow a joint, slow enough not to flicker.
        var now = perf_counter_ns()
        # ⚠ Only redraw on a TTY. Piped, each "redraw" would append another
        # full table — 10 a second of them.
        if now - last_draw > 100_000_000:
            if stdout_is_tty() and not plain:
                drawn = _sweep_table(lo, raw, hi, continuous, drawn, plain)
                last_draw = now
            elif plain:
                drawn = _sweep_table(lo, raw, hi, continuous, 0, plain)
                last_draw = now

    _ = _sweep_table(lo, raw, hi, continuous, drawn, plain)
    # ⚠ LEAVE THE CURSOR ON A FRESH LINE. Everything printed after this is a
    # normal `print`; if the redraw left the cursor inside the table, those
    # lines overwrite it — which is exactly how the proposed calibration came
    # out missing two joints.
    print("")

    if not seeded or samples < 10:
        raise Error(
            "calibrate: only " + String(samples) + " good reads during the"
            " sweep — check the bus before trusting anything here"
        )
    print(
        "\n  " + String(samples) + " samples, " + String(partial)
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
        var skipped = False
        for k in range(len(continuous)):
            if continuous[k] == i:
                skipped = True
        if skipped:
            continue
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

    # ⚠ A JOINT THAT LOST TRAVEL IS THE FAILURE THIS TOOL CANNOT SEE ITSELF.
    # An under-swept joint produces a perfectly well-formed calibration whose
    # only symptom is an arm that quietly stops reaching.
    var regressed = span_regressions(previous, next_cal, continuous)
    for k in range(len(regressed)):
        var i = regressed[k]
        var was = Int(previous.rmax[i]) - Int(previous.rmin[i])
        var now_span = Int(next_cal.rmax[i]) - Int(next_cal.rmin[i])
        print(
            "  ⚠ " + joint_name(i) + " swept " + String(was - now_span)
            + " ticks LESS than its previous calibration ("
            + fixed(Float64(was - now_span) * 360.0 / 4095.0, 1)
            + " deg of travel): " + String(was) + " -> " + String(now_span)
        )
    if len(regressed) > 0:
        print(
            "    `write_goals` clamps to this range, so the follower would"
            " stop reaching poses it can reach today."
        )
        if write and not allow_narrower:
            raise Error(
                "calibrate: " + String(len(regressed)) + " joint(s) lost more"
                " than " + fixed((1.0 - NARROWER_FRACTION) * 100.0, 0)
                + "% of their travel. Sweep them to BOTH end stops and run"
                " again — or pass --allow-narrower if the range really did"
                " shrink."
            )

    if narrow > 0 and write:
        raise Error(
            "calibrate: " + String(narrow) + " joint(s) barely moved, so this"
            " sweep does not describe the arm. Refusing to write it. Run"
            " again and move every joint to both stops."
        )

    if not write:
        print(
            "  dry run — nothing was written. Re-run with --write to apply."
        )
        # Nothing was written, so there is nothing for the caller to undo.
        return True

    print("  apply this to the arm? [yes/N] ")
    stdin.discard_pending()
    var confirm = stdin.line()
    if confirm != "yes":
        print("  not applied.")
        return False
    _apply(arm, next_cal, write)
    print("\ndone. Verify with:  pixi run soarm-diag")
    return True
