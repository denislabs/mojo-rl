# +--------------------------------------------------------------------------+ #
# | Feetech SCS/STS bus servos
# +--------------------------------------------------------------------------+ #
"""Protocol and bus for Feetech bus servos (the SO-101 runs six STS3215).

`packet` is pure bytes-to-bytes and is gated in CI against byte strings
captured from the reference `scservo_sdk`; `bus` adds the serial port,
retries and timeouts. Register addresses live in `control_table`.
"""

from mojo_rl.robot.feetech.control_table import (
    STS_TORQUE_ENABLE,
    STS_GOAL_POSITION,
    STS_PRESENT_POSITION,
    STS_HOMING_OFFSET,
    STS_MIN_POSITION_LIMIT,
    STS_MAX_POSITION_LIMIT,
    STS_RESOLUTION,
)
from mojo_rl.robot.feetech.bus import FeetechBus
