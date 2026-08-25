# +--------------------------------------------------------------------------+ #
# | Release torque on both SO-101 arms
# +--------------------------------------------------------------------------+ #
"""Disable torque on every servo of both arms, and clear `Lock`.

The recovery tool for a control loop that died without running its cleanup —
a `kill`, a crash, a debugger detach. A follower left holding a pose is both a
safety problem (it resists being moved) and a thermal one (the servos heat up
under a static load).

    pixi run soarm-torque-off
"""

from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.robot.feetech.control_table import (
    SIZE_1,
    STS_PRESENT_TEMPERATURE,
    STS_TORQUE_ENABLE,
)

comptime FOLLOWER = "/dev/cu.usbmodem5B8E1139971"
comptime LEADER = "/dev/cu.usbmodem5B910455171"


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
    release(String(FOLLOWER), String("FOLLOWER"))
    release(String(LEADER), String("LEADER"))
