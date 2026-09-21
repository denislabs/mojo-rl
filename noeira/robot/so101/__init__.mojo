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

from noeira.robot.so101.sim_map import SimJointMap
from noeira.robot.so101.arm import (
    ALIGN_TICKS,
    SO101Arm,
    SO101Calibration,
    SO101_N,
    GRIPPER,
    is_aligned, joint_name, joint_short, step_limit,
)

from noeira.robot.so101.calibration import (
    NARROWER_FRACTION,
    SEAM_MARGIN,
    UNLIMITED_MAX,
    UNLIMITED_MIN,
    CalibrationRecord,
    centre_on_middle_pose,
    frame_position,
    load_calibration_json,
    save_calibration_json,
    span_regressions,
)
