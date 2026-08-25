# +--------------------------------------------------------------------------+ #
# | SO-ARM101 — six STS3215 on one bus
# +--------------------------------------------------------------------------+ #
"""`SO101Arm`: the leader or the follower, driven over `FeetechBus`.

A leader and a follower are the same hardware and the same code; the only
difference is that nobody writes goals to a leader. So there is one struct
here, not two.

**Calibration comes off the servos, not off disk.** `lerobot-calibrate` writes
`Homing_Offset`, `Min_Position_Limit` and `Max_Position_Limit` into each
servo's EEPROM (`feetech.py:268 write_calibration`), so an arm carries its own
calibration and this layer needs no JSON parser — which matters, because the
repo has none and hand-rolling one for a robot driver would be absurd.

⚠ **Units follow lerobot exactly**, because a policy trained in sim has to
speak the same numbers a LeRobot dataset recorded (`so101-nexus`'s
`lerobot_adapter/normalization.py` is the same table):

* body joints — `MotorNormMode.DEGREES`:
  `deg = (raw - mid) * 360 / (4096 - 1)`, `mid = (range_min + range_max) / 2`,
  **unclamped**;
* gripper — `MotorNormMode.RANGE_0_100`:
  `pct = (clamp(raw) - min) / (max - min) * 100`.

Note the `4096 - 1`: the inclusive tick range 0..4095 spans one turn. Using
4096 is a systematic ~0.09 degree error, small enough to survive a review and
large enough to sit in every sim-to-real comparison afterwards.
"""

from std.math import pi

from mojo_rl.robot.feetech.bus import FeetechBus
from mojo_rl.robot.feetech.control_table import (
    MODE_POSITION,
    SIZE_1,
    SIZE_2,
    STS_GOAL_POSITION,
    STS_HOMING_OFFSET,
    STS_LOCK,
    STS_MAX_POSITION_LIMIT,
    STS_MIN_POSITION_LIMIT,
    STS_OPERATING_MODE,
    STS_PRESENT_POSITION,
    STS_RESOLUTION,
    STS_TORQUE_ENABLE,
    TORQUE_DISABLED,
    TORQUE_ENABLED,
)

comptime SO101_N = 6
"""Joint count: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper —
servo ids 1..6 in that order, as `lerobot-setup-motors` assigns them."""

comptime GRIPPER = 5
"""Index of the gripper, the one joint normalised 0..100 instead of degrees."""

comptime TICKS_PER_TURN = STS_RESOLUTION - 1
"""4095. See the module docstring — NOT 4096."""


def joint_name(i: Int) -> String:
    if i == 0:
        return String("shoulder_pan")
    if i == 1:
        return String("shoulder_lift")
    if i == 2:
        return String("elbow_flex")
    if i == 3:
        return String("wrist_flex")
    if i == 4:
        return String("wrist_roll")
    return String("gripper")


@fieldwise_init
struct SO101Calibration(Copyable, Movable):
    """What `lerobot-calibrate` left in the servos' EEPROM."""

    var homing_offset: InlineArray[Int32, SO101_N]
    var range_min: InlineArray[Int32, SO101_N]
    var range_max: InlineArray[Int32, SO101_N]

    def mid(self, i: Int) -> Float64:
        return 0.5 * (Float64(self.range_min[i]) + Float64(self.range_max[i]))

    def span(self, i: Int) -> Int:
        return Int(self.range_max[i]) - Int(self.range_min[i])

    def degrees(self, i: Int, raw: Int32) -> Float64:
        """Ticks to the units lerobot records — degrees, or 0..100 for the
        gripper."""
        if i == GRIPPER:
            var lo = Int(self.range_min[i])
            var hi = Int(self.range_max[i])
            var v = min(hi, max(lo, Int(raw)))
            return Float64(v - lo) / Float64(hi - lo) * 100.0
        return (Float64(raw) - self.mid(i)) * 360.0 / Float64(TICKS_PER_TURN)

    def radians(self, i: Int, raw: Int32) -> Float64:
        """Body joints in radians; the gripper stays 0..100 (it is an opening
        fraction, not an angle, and pretending otherwise would put a unit
        error into every observation)."""
        if i == GRIPPER:
            return self.degrees(i, raw)
        return self.degrees(i, raw) * pi / 180.0

    def raw_from_degrees(self, i: Int, value: Float64) -> Int32:
        if i == GRIPPER:
            var lo = Float64(self.range_min[i])
            var hi = Float64(self.range_max[i])
            var pct = min(100.0, max(0.0, value))
            return Int32(Int(pct / 100.0 * (hi - lo) + lo))
        return Int32(
            Int(value * Float64(TICKS_PER_TURN) / 360.0 + self.mid(i))
        )

    def raw_from_radians(self, i: Int, value: Float64) -> Int32:
        if i == GRIPPER:
            return self.raw_from_degrees(i, value)
        return self.raw_from_degrees(i, value * 180.0 / pi)


struct SO101Arm(Movable):
    var bus: FeetechBus
    var cal: SO101Calibration
    var ids: InlineArray[UInt8, SO101_N]
    var max_step_ticks: Int
    """Largest change from the CURRENT position a single `write_goals` may
    command, per joint.

    ⚠ lerobot's own config for these arms records `max_relative_target: None`
    — no clamp at all — so the first bad goal a policy emits is a full-speed
    slam into the table. 200 ticks is ~17 degrees. Set it to 0 to disable the
    clamp deliberately; do not leave it off by accident.
    """

    def __init__(
        out self,
        var path: String,
        baud: Int = 1000000,
        max_step_ticks: Int = 200,
    ) raises:
        self.bus = FeetechBus(path^, baud)
        self.max_step_ticks = max_step_ticks
        self.ids = InlineArray[UInt8, SO101_N](fill=0)
        for i in range(SO101_N):
            self.ids[i] = UInt8(i + 1)
        self.cal = SO101Calibration(
            InlineArray[Int32, SO101_N](fill=0),
            InlineArray[Int32, SO101_N](fill=0),
            InlineArray[Int32, SO101_N](fill=0),
        )

        # Ping every servo before reading anything: a missing motor otherwise
        # surfaces as a confusing timeout inside calibration.
        for i in range(SO101_N):
            if not self.bus.ping(self.ids[i]):
                raise Error(
                    "so101: no servo answered id "
                    + String(i + 1)
                    + " ("
                    + joint_name(i)
                    + ") — check power and the daisy chain"
                )
        self.read_calibration()

    def read_calibration(mut self) raises:
        """Pull `Homing_Offset` / `Min` / `Max` out of each servo's EEPROM."""
        for i in range(SO101_N):
            var id = self.ids[i]
            self.cal.homing_offset[i] = Int32(
                self.bus.read_register(id, STS_HOMING_OFFSET, SIZE_2)
            )
            self.cal.range_min[i] = Int32(
                self.bus.read_register(id, STS_MIN_POSITION_LIMIT, SIZE_2)
            )
            self.cal.range_max[i] = Int32(
                self.bus.read_register(id, STS_MAX_POSITION_LIMIT, SIZE_2)
            )
            if self.cal.span(i) == 0:
                raise Error(
                    "so101: "
                    + joint_name(i)
                    + " has range_min == range_max ("
                    + String(Int(self.cal.range_min[i]))
                    + "), which is the UNCALIBRATED marker — run"
                    " `lerobot-calibrate` for this arm"
                )

    # ── reading ────────────────────────────────────────────────────────────

    def read_positions[
        o: MutOrigin
    ](mut self, out_raw: Span[Int32, o]) raises -> Int:
        """All six present positions, in ticks, in one round trip.

        Returns how many answered. **Check it** — a partial read means a motor
        dropped off the bus, and treating the untouched entries as current is
        how a teleop loop commands last second's pose.
        """
        return self.bus.sync_read(
            STS_PRESENT_POSITION, SIZE_2, Span(self.ids), out_raw
        )

    # ── writing ────────────────────────────────────────────────────────────

    def set_torque(mut self, on: Bool) raises:
        var v = TORQUE_ENABLED if on else TORQUE_DISABLED
        for i in range(SO101_N):
            self.bus.write_register(
                self.ids[i], STS_TORQUE_ENABLE, v, SIZE_1
            )
            if not on:
                # `Lock` guards the EEPROM; lerobot clears it alongside torque
                # (`feetech.py:291 disable_torque`) so a subsequent
                # calibration write is not silently dropped.
                self.bus.write_register(self.ids[i], STS_LOCK, 0, SIZE_1)

    def set_position_mode(mut self) raises:
        for i in range(SO101_N):
            self.bus.write_register(
                self.ids[i], STS_OPERATING_MODE, MODE_POSITION, SIZE_1
            )

    def write_goals[
        mut: Bool, //, og: Origin[mut=mut]
    ](mut self, goals: Span[Int32, og]) raises:
        """Command all six goal positions in ONE packet, clamped twice.

        Clamped to each joint's calibrated `[range_min, range_max]`, and — if
        `max_step_ticks > 0` — to `present ± max_step_ticks`, which costs one
        extra `sync_read` (~1.3 ms of a ~20 ms tick) and is the difference
        between a bad goal being a jerk and being a slam.
        """
        if len(goals) != SO101_N:
            raise Error(
                "so101: write_goals expects "
                + String(SO101_N)
                + " goals, got "
                + String(len(goals))
            )

        var safe = InlineArray[Int32, SO101_N](fill=0)
        for i in range(SO101_N):
            var lo = Int(self.cal.range_min[i])
            var hi = Int(self.cal.range_max[i])
            safe[i] = Int32(min(hi, max(lo, Int(goals[i]))))

        if self.max_step_ticks > 0:
            var present = InlineArray[Int32, SO101_N](fill=0)
            var got = self.read_positions(Span(present))
            if got != SO101_N:
                raise Error(
                    "so101: refusing to write goals — only "
                    + String(got)
                    + " of "
                    + String(SO101_N)
                    + " motors reported a position, so the step clamp cannot"
                    " be applied"
                )
            for i in range(SO101_N):
                var p = Int(present[i])
                var step = Int(safe[i]) - p
                if step > self.max_step_ticks:
                    safe[i] = Int32(p + self.max_step_ticks)
                elif step < -self.max_step_ticks:
                    safe[i] = Int32(p - self.max_step_ticks)

        self.bus.sync_write(
            STS_GOAL_POSITION, SIZE_2, Span(self.ids), Span(safe)
        )
