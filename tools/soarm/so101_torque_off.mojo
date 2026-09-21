# +--------------------------------------------------------------------------+ #
# | Release torque on both SO-101 arms
# +--------------------------------------------------------------------------+ #
"""Disable torque on every servo of both arms, and clear `Lock`.

The recovery tool for a control loop that died without running its cleanup —
a `kill`, a crash, a debugger detach. A follower left holding a pose is both a
safety problem (it resists being moved) and a thermal one (the servos heat up
under a static load).

    pixi run soarm-torque-off

⚠ EACH ARM IS RELEASED INDEPENDENTLY. A recovery tool that gives up because
the OTHER arm is unplugged leaves the energised one energised, which is the
exact situation this exists for.
"""

from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.ports import (
    follower_port, leader_port, port_refusal,
)
from noeira.robot.feetech.control_table import (
    SIZE_1,
    STS_PRESENT_TEMPERATURE,
    STS_TORQUE_ENABLE,
)



def release(var path: String, label: String) raises:
    var arm = SO101Arm(path^, max_step_ticks=0)
    var line = String("")
    for i in range(SO101_N):
        line += (
            String(Int(arm.bus.read_register(
                arm.ids[i], STS_TORQUE_ENABLE, SIZE_1
            )))
            + " "
        )
    print(label + ": torque was [" + line + "]")

    arm.set_torque(False)

    var after = String("")
    var temps = String("")
    for i in range(SO101_N):
        after += (
            String(Int(arm.bus.read_register(
                arm.ids[i], STS_TORQUE_ENABLE, SIZE_1
            )))
            + " "
        )
        temps += (
            String(Int(arm.bus.read_register(
                arm.ids[i], STS_PRESENT_TEMPERATURE, SIZE_1
            )))
            + " "
        )
    print(label + ": torque now [" + after + "]  degC [" + temps + "]")


def main() raises:
    var failures = 0
    for pair in [
        (follower_port(), String("FOLLOWER"), String("follower")),
        (leader_port(), String("LEADER"), String("leader")),
    ]:
        var why = port_refusal(pair[0], pair[2])
        if why.byte_length() > 0:
            print(pair[1] + ": skipped — " + why)
            failures += 1
            continue
        try:
            release(pair[0], pair[1])
        except e:
            # ⚠ REPORTED, NOT RAISED: see the header. The other arm still has
            # to be released.
            print(pair[1] + ": FAILED — " + String(e))
            failures += 1
    if failures == 2:
        raise Error(
            "soarm-torque-off: neither arm could be released — nothing was"
            " disarmed. Check the cables and the power switch."
        )
