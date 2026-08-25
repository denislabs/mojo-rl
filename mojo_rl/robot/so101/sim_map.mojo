# +--------------------------------------------------------------------------+ #
# | Real SO-101 joints <-> simulated SO-101 joints
# +--------------------------------------------------------------------------+ #
"""The mapping between a servo's calibrated angle and the MJCF model's joint.

⚠⚠ **THIS MAPPING HAS NOT BEEN MEASURED.** `SimJointMap.identity()` assumes
`sign = +1` and `offset = 0` for every joint, and that assumption is almost
certainly wrong for at least one of them. It is the honest starting point, not
a result: `examples/so101/teleop_sim.mojo` exists to *find* the real values by
driving the sim from the leader and watching which joints move backwards.

Three separate things stand between "our ticks equal lerobot's ticks" (proven,
`docs/SO101_SERIAL_LAYER.md` §4) and "our radians equal the model's radians"
(not proven at all):

1. **Zero.** `SO101Calibration.radians` is referenced to the servo's
   calibrated mid-point, `(range_min + range_max) / 2`. The MJCF's zero is
   wherever the URDF put it. Two different zeros, one constant offset each.
2. **Sign.** Every joint in `so_arm101.xml` is `axis="0 0 1"` *in its own body
   frame*, so whether a rising tick count is a rising model angle is a
   per-joint coin flip that no amount of reading the XML settles.
3. **Range.** The two do not agree, and this is measurable today — see
   `range_report`. On the arms here, five of six joints have more calibrated
   travel than the model will accept.

⚠ The gripper is not an angle on either side. The servo reports it 0..100
(`MotorNormMode.RANGE_0_100`) and the model has it as a hinge in radians, so
it maps by FRACTION OF RANGE, and `offset_rad` does not apply to it. Treating
it as an angle is the single easiest way to get a gripper that closes when it
should open.
"""

from mojo_rl.robot.so101.arm import GRIPPER, SO101Calibration, SO101_N, joint_name
from mojo_rl.utils.fmt import col, fixed, pad_left, pad_right


@fieldwise_init
struct SimJointMap(Copyable, Movable):
    """Per-joint sign, zero offset and the model's own limits.

    `sim_lo` / `sim_hi` come from the MODEL (a `<position>` servo's
    `ctrlrange` is its joint range), never from a copy of the numbers — a
    second copy of a limit is a second thing to drift.
    """

    var sign: InlineArray[Float64, SO101_N]
    var offset_rad: InlineArray[Float64, SO101_N]
    var sim_lo: InlineArray[Float64, SO101_N]
    var sim_hi: InlineArray[Float64, SO101_N]

    @staticmethod
    def identity(
        var sim_lo: InlineArray[Float64, SO101_N],
        var sim_hi: InlineArray[Float64, SO101_N],
    ) -> Self:
        """⚠ UNMEASURED: every sign +1, every offset 0. See the module doc."""
        var s = InlineArray[Float64, SO101_N](fill=1.0)
        var o = InlineArray[Float64, SO101_N](fill=0.0)
        return Self(s^, o^, sim_lo^, sim_hi^)

    def measured(self) -> Bool:
        """False while this is still the identity — i.e. while nobody has
        driven the sim from the real arm and written the numbers down.

        Exists so a caller can SAY so rather than presenting an unmeasured
        mapping as a calibrated one.
        """
        for i in range(SO101_N):
            if self.sign[i] != 1.0 or self.offset_rad[i] != 0.0:
                return True
        return False

    # ── real -> sim ────────────────────────────────────────────────────────

    def to_sim_unclamped(
        self, cal: SO101Calibration, i: Int, raw: Int32
    ) -> Float64:
        """Servo ticks to model radians, BEFORE the model's limits apply.

        Unclamped on purpose: `clamped_by` needs to see how far outside the
        model's range the real arm actually went, and a function that clamps
        silently cannot answer that.
        """
        if i == GRIPPER:
            # Fraction of the servo's calibrated opening, mapped onto the
            # model's hinge range. `sign` flips the fraction; `offset_rad`
            # does not apply — this is not an angle on the servo side.
            var frac = cal.degrees(i, raw) / 100.0
            if self.sign[i] < 0.0:
                frac = 1.0 - frac
            return self.sim_lo[i] + frac * (self.sim_hi[i] - self.sim_lo[i])
        return self.sign[i] * cal.radians(i, raw) + self.offset_rad[i]

    def to_sim(self, cal: SO101Calibration, i: Int, raw: Int32) -> Float64:
        var v = self.to_sim_unclamped(cal, i, raw)
        return min(self.sim_hi[i], max(self.sim_lo[i], v))

    def clamped_by(
        self, cal: SO101Calibration, i: Int, raw: Int32
    ) -> Float64:
        """Radians of overshoot past the model's limit, 0 when inside.

        A teleop loop reports this rather than hiding it: a joint pinned at
        its simulated limit while the real one keeps moving looks exactly like
        a broken mapping, and this is what tells the two apart.
        """
        var v = self.to_sim_unclamped(cal, i, raw)
        if v > self.sim_hi[i]:
            return v - self.sim_hi[i]
        if v < self.sim_lo[i]:
            return self.sim_lo[i] - v
        return 0.0

    # ── reporting ──────────────────────────────────────────────────────────

    def range_report(self, cal: SO101Calibration) -> String:
        """Calibrated servo travel vs what the model will accept, per joint.

        Measured from the two sources rather than transcribed, so it stays
        true when either side changes. A positive `gap` is real travel the
        simulation cannot represent — the leader can reach a pose the model
        refuses, and the sim joint will sit at its limit.
        """
        var out = String(
            pad_right(String("JOINT"), 15)
            + pad_left(String("real span"), 11)
            + pad_left(String("sim span"), 11)
            + pad_left(String("gap"), 9)
            + "\n"
        )
        out += "-" * 46 + "\n"
        for i in range(SO101_N):
            var real_span = Float64(cal.span(i)) * 360.0 / 4095.0
            var sim_span = (self.sim_hi[i] - self.sim_lo[i]) * 180.0 / 3.141592653589793
            if i == GRIPPER:
                # Both sides are rescaled onto each other by construction, so
                # a "gap" here would be meaningless rather than zero.
                out += (
                    pad_right(joint_name(i), 15)
                    + col(real_span, 11, 1)
                    + col(sim_span, 11, 1)
                    + pad_left(String("n/a"), 9)
                    + "   (fraction-mapped)\n"
                )
                continue
            var gap = real_span - sim_span
            out += (
                pad_right(joint_name(i), 15)
                + col(real_span, 11, 1)
                + col(sim_span, 11, 1)
                + col(gap, 9, 1)
                + ("   <-- real exceeds sim" if gap > 1.0 else "")
                + "\n"
            )
        return out^

    def describe(self) -> String:
        if not self.measured():
            return String(
                "sim map: IDENTITY (sign +1, offset 0) — UNMEASURED, see"
                " mojo_rl/robot/so101/sim_map.mojo"
            )
        var out = String("sim map:")
        for i in range(SO101_N):
            out += (
                " "
                + String(joint_name(i)[byte=0:4])
                + ("+" if self.sign[i] > 0.0 else "-")
                + fixed(self.offset_rad[i], 2)
            )
        return out^
