"""Gate ticks <-> lerobot units against lerobot's OWN SOURCE ARITHMETIC.

    pixi run python tools/act/dump_lerobot_units_reference.py   # once
    pixi run mojo run -I . tests/robot/test_so101_lerobot_units.mojo

⚠⚠ WHY THIS GATE EXISTS. ACT is trained on `observation.state` / `action`
columns that a LeRobot recording wrote in lerobot's units. Deploying the policy
means producing those same numbers from servo ticks — and
`mojo_rl/robot/so101/sim_map.mojo`, which maps ticks to SIM RADIANS, is the
WRONG map for that. It is the right map for the reach policy. Using either in
the other's place feeds a network numbers it has never seen and raises nothing.

⚠ THE FIXTURE IS EXTRACTED FROM lerobot, NOT TRANSCRIBED. The generator reads
the expressions out of `MotorsBus._normalize` / `_unnormalize` and evaluates
them, so this compares our arithmetic against lerobot's, not against a second
copy of my reading of it.
"""

from std.pathlib import Path

from mojo_rl.robot.so101 import SO101_N
from mojo_rl.robot.so101.arm import SO101Calibration

comptime FIX = "tests/fixtures/robot/lerobot_units.txt"
comptime TOL = 1e-9


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var out = String("")
        for i in range(len(raw)):
            out += chr(Int(raw[i]))
        return out^


def cal_for(
    lo: Int, hi: Int, gripper_lo: Int, gripper_hi: Int
) -> SO101Calibration:
    """One joint under test in slot 0, the gripper in slot 5.

    Two slots because `degrees()` branches on the GRIPPER index, so a body
    joint and the gripper cannot share one.
    """
    var ofs = InlineArray[Int32, SO101_N](fill=0)
    var mn = InlineArray[Int32, SO101_N](fill=0)
    var mx = InlineArray[Int32, SO101_N](fill=4095)
    mn[0] = Int32(lo)
    mx[0] = Int32(hi)
    mn[5] = Int32(gripper_lo)
    mx[5] = Int32(gripper_hi)
    return SO101Calibration(ofs^, mn^, mx^)


def main() raises:
    print("=" * 70)
    print("SO-101 ticks <-> lerobot units, against lerobot's own expressions")
    print("=" * 70)

    if not Path(String(FIX)).exists():
        print("SKIP: no fixture. Run:")
        print("  pixi run python tools/act/dump_lerobot_units_reference.py")
        return

    var text = read_text(String(FIX))
    var lines = text.split("\n")

    # joint name -> (min, max, is_gripper)
    var names = List[String]()
    var mins = List[Int]()
    var maxs = List[Int]()
    var grips = List[Bool]()
    for li in range(len(lines)):
        var parts = lines[li].split()
        if len(parts) == 5 and parts[0] == "joint":
            names.append(String(parts[1]))
            mins.append(Int(String(parts[2])))
            maxs.append(Int(String(parts[3])))
            grips.append(String(parts[4]) == "gripper")
    if len(names) != 6:
        print("  FAIL: fixture declares", len(names), "joints, expected 6")
        return
    print("  joints:", len(names))

    var failures = 0
    var n_norm = 0
    var n_unnorm = 0
    var n_m100 = 0
    var worst = 0.0
    var beyond_100 = 0

    for li in range(len(lines)):
        var parts = lines[li].split()
        if len(parts) == 0:
            continue

        if String(parts[0]) == "norm" and len(parts) == 6:
            var j = -1
            for k in range(len(names)):
                if names[k] == String(parts[1]):
                    j = k
            if j < 0:
                continue
            var raw = Int32(Int(String(parts[2])))
            var want_deg = Float64(String(parts[3]))
            var want_m100 = Float64(String(parts[4]))
            var want_p100 = Float64(String(parts[5]))
            var cal = cal_for(mins[j], maxs[j], mins[5], maxs[5])

            if grips[j]:
                # The gripper is RANGE_0_100 in every SO-101 configuration.
                var got = cal.degrees(5, raw)
                var d = abs(got - want_p100)
                if d > worst:
                    worst = d
                if d > TOL:
                    print("  FAIL: gripper", raw, "->", got, "want", want_p100)
                    failures += 1
                n_norm += 1
            else:
                var got = cal.degrees(0, raw)
                var d = abs(got - want_deg)
                if d > worst:
                    worst = d
                if d > TOL:
                    print(
                        "  FAIL:",
                        parts[1],
                        raw,
                        "deg ->",
                        got,
                        "want",
                        want_deg,
                    )
                    failures += 1
                n_norm += 1
                # ⚠ THE OTHER MODE, GATED TOO, so it cannot rot unnoticed while
                # being the thing a future dataset needs.
                var got2 = cal.range_m100_100(0, raw)
                var d2 = abs(got2 - want_m100)
                if d2 > worst:
                    worst = d2
                if d2 > TOL:
                    print(
                        "  FAIL:",
                        parts[1],
                        raw,
                        "m100 ->",
                        got2,
                        "want",
                        want_m100,
                    )
                    failures += 1
                n_m100 += 1
                # ⚠ THE DISCRIMINATOR, made explicit: DEGREES is unbounded and
                # RANGE_M100_100 is not. Count the rows where they could not be
                # confused, so "the two modes agree everywhere" can never pass
                # silently.
                if want_deg < -100.0 or want_deg > 100.0:
                    beyond_100 += 1

        elif String(parts[0]) == "unnorm" and len(parts) == 4:
            var j = -1
            for k in range(len(names)):
                if names[k] == String(parts[1]):
                    j = k
            if j < 0:
                continue
            var value = Float64(String(parts[2]))
            var want = Int32(Int(String(parts[3])))
            var cal = cal_for(mins[j], maxs[j], mins[5], maxs[5])
            var got = cal.raw_from_degrees(5 if grips[j] else 0, value)
            if got != want:
                print(
                    "  FAIL:",
                    parts[1],
                    "unnorm",
                    value,
                    "->",
                    got,
                    "want",
                    want,
                )
                failures += 1
            n_unnorm += 1

    print("  normalize   :", n_norm, "rows")
    print("  m100_100    :", n_m100, "rows")
    print("  unnormalize :", n_unnorm, "rows")
    print("  worst abs error:", worst)

    # ⚠ VACUITY. A fixture that never leaves the calibrated range would compare
    # two modes that happen to agree at the ends and prove nothing about which
    # one a dataset used.
    if beyond_100 == 0:
        print("  FAIL: no row exceeds +-100, so the two modes are")
        print("        indistinguishable here — the fixture is vacuous")
        failures += 1
    else:
        print("  rows where degrees exceeds +-100 (unreachable for")
        print("  RANGE_M100_100, which is how the 50-demo store is known")
        print("  to be in DEGREES):", beyond_100)

    print("-" * 70)
    if failures == 0:
        print("PASS —", n_norm + n_m100 + n_unnorm, "rows against lerobot")
    else:
        print("FAIL —", failures, "rows")
    print("=" * 70)
