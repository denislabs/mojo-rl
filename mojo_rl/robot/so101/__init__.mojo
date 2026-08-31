# +--------------------------------------------------------------------------+ #
# | SO-ARM101
# +--------------------------------------------------------------------------+ #
"""The SO-ARM101 leader/follower pair over the Feetech bus.

`SO101Arm` is both roles — a leader is simply an arm nobody writes goals to.
Calibration is read from the servos' EEPROM, so no calibration file is needed;
units match lerobot's `DEGREES` / `RANGE_0_100` exactly, which is the contract
a policy has to speak at both ends of sim-to-real.

⚠ Needs the serial shim: `pixi run build-serial`.
"""

from mojo_rl.robot.so101.sim_map import SimJointMap
from mojo_rl.robot.so101.arm import (
    SO101Arm,
    SO101Calibration,
    SO101_N,
    GRIPPER,
    joint_name, joint_short,
)
