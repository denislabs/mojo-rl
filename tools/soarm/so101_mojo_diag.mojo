# +--------------------------------------------------------------------------+ #
# | SO-101 register dump — the Mojo half of the hardware gate
# +--------------------------------------------------------------------------+ #
"""Read-only dump of both arms, laid out to be diffed against
`tools/soarm/so101_diag.py` (Python + lerobot + scservo_sdk).

**This is the gate for `mojo_rl/robot/`.** Run the two back to back on the
same arms in the same pose; every integer in the shared columns must match
exactly. On 2026-08-25 they did: the follower read
`1931 812 3125 2901 2102 2559` from both stacks.

Writes NOTHING — no torque change, no EEPROM write — so it is safe with the
arms powered and holding a pose.

    pixi run build-serial
    pixi run mojo run -I . tools/soarm/so101_mojo_diag.mojo

Compare with:

    /path/to/lerobot-env/bin/python tools/soarm/so101_diag.py
"""

from std.time import perf_counter_ns

from mojo_rl.robot.feetech.control_table import (
    SIZE_1,
    SIZE_2,
    STS_FIRMWARE_MAJOR,
    STS_FIRMWARE_MINOR,
    STS_GOAL_POSITION,
    STS_HOMING_OFFSET,
    STS_MAX_POSITION_LIMIT,
    STS_MIN_POSITION_LIMIT,
    STS_OPERATING_MODE,
    STS_PRESENT_LOAD,
    STS_PRESENT_POSITION,
    STS_PRESENT_TEMPERATURE,
    STS_PRESENT_VOLTAGE,
    STS_STATUS,
    STS_TORQUE_ENABLE,
)
from mojo_rl.robot.feetech.packet import error_names
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.utils.fmt import col, fixed, pad_left, pad_right

comptime FOLLOWER = "/dev/cu.usbmodem5B8E1139971"
comptime LEADER = "/dev/cu.usbmodem5B910455171"


def dump(var path: String, label: String) raises:
    print("\n" + "=" * 92)
    print(label + "  (" + path + ")")
    print("=" * 92)

    # max_step_ticks=0: this tool never writes, and a clamp that costs an
    # extra sync_read would only slow the dump down.
    var arm = SO101Arm(path^, max_step_ticks=0)

    var header = pad_right(String("MOTOR"), 15) + pad_left(String("id"), 3)
    header += pad_left(String("pres"), 7) + pad_left(String("goal"), 7)
    header += pad_left(String("ofs"), 7) + pad_left(String("lo"), 6)
    header += pad_left(String("hi"), 6) + pad_left(String("load"), 7)
    header += pad_left(String("V"), 6) + pad_left(String("degC"), 5)
    header += pad_left(String("trq"), 4) + pad_left(String("mode"), 5)
    header += "  fw     status"
    print(header)
    print("-" * header.byte_length())

    for i in range(SO101_N):
        var id = arm.ids[i]
        var pres = arm.bus.read_register(id, STS_PRESENT_POSITION, SIZE_2)
        var goal = arm.bus.read_register(id, STS_GOAL_POSITION, SIZE_2)
        var load = arm.bus.read_register(id, STS_PRESENT_LOAD, SIZE_2)
        var volt = arm.bus.read_register(id, STS_PRESENT_VOLTAGE, SIZE_1)
        var temp = arm.bus.read_register(id, STS_PRESENT_TEMPERATURE, SIZE_1)
        var trq = arm.bus.read_register(id, STS_TORQUE_ENABLE, SIZE_1)
        var mode = arm.bus.read_register(id, STS_OPERATING_MODE, SIZE_1)
        var stat = arm.bus.read_register(id, STS_STATUS, SIZE_1)
        var fwma = arm.bus.read_register(id, STS_FIRMWARE_MAJOR, SIZE_1)
        var fwmi = arm.bus.read_register(id, STS_FIRMWARE_MINOR, SIZE_1)

        var row = pad_right(joint_name(i), 15) + pad_left(String(Int(id)), 3)
        row += pad_left(String(pres), 7) + pad_left(String(goal), 7)
        row += pad_left(String(Int(arm.cal.homing_offset[i])), 7)
        row += pad_left(String(Int(arm.cal.range_min[i])), 6)
        row += pad_left(String(Int(arm.cal.range_max[i])), 6)
        row += pad_left(String(load), 7)
        row += col(Float64(volt) / 10.0, 6, 1)
        row += pad_left(String(temp), 5) + pad_left(String(trq), 4)
        row += pad_left(String(mode), 5)
        row += "  " + String(fwma) + "." + String(fwmi)
        row += "    " + error_names(UInt8(stat))
        print(row)

    # Calibrated units beside the raw ticks: a sign-magnitude or a 4096-vs-4095
    # slip is invisible in ticks and obvious in degrees.
    print("")
    var raw = InlineArray[Int32, SO101_N](fill=0)
    var got = arm.read_positions(Span(raw))
    var units = pad_right(String("in lerobot units:"), 20)
    for i in range(SO101_N):
        var v = arm.cal.degrees(i, raw[i])
        units += pad_right(
            String(joint_name(i)[byte=0:4]) + "=" + fixed(v, 2), 14
        )
    print(units)
    print("  motors answering sync_read: " + String(got) + " / " + String(SO101_N))

    # Throughput, because the number that matters for teleop is round trips
    # per second, not bytes per second.
    var t0 = perf_counter_ns()
    var ok = 0
    for _ in range(100):
        ok += arm.read_positions(Span(raw))
    var dt = Float64(perf_counter_ns() - t0) / 1e6
    print(
        "  100 sync_reads in "
        + fixed(dt, 2)
        + " ms -> "
        + fixed(100000.0 / dt, 1)
        + " Hz  ("
        + String(ok)
        + " / "
        + String(100 * SO101_N)
        + " motor rows)"
    )


def main() raises:
    dump(String(FOLLOWER), String("FOLLOWER"))
    dump(String(LEADER), String("LEADER"))
