# +--------------------------------------------------------------------------+ #
# | Real SO-101 joints <-> simulated SO-101 joints
# +--------------------------------------------------------------------------+ #
"""The mapping between a servo's calibrated angle and the MJCF model's joint.

**This mapping is the reference implementation's**, not a guess.
`so101-nexus`'s `lerobot_adapter/normalization.py::motor_ticks_to_sim_rad` —
the function a working LeRobot-on-MuJoCo stack uses for these arms — is:

```python
mid  = (range_min + range_max) / 2
sign = -1 if cal.drive_mode else 1
qpos = sign * (ticks - mid) / TICKS_PER_RADIAN          # body joints
frac = (ticks - range_min) / (range_max - range_min)    # gripper
qpos = lower + frac * (upper - lower)
```

which is exactly `to_sim_unclamped` below, with `offset_rad = 0` (the
reference has no offset term at all) and `sign = +1`.

⚠ **`sign = +1` is a property of the platform, not an assumption.** lerobot
HARD-CODES `drive_mode=0` for every SO-101 joint — `robots/so_follower/
so_follower.py:149`, `teleoperators/so_leader/so_leader.py:117` and
`motors/feetech/feetech.py:260` — so there is no inverted joint to discover on
this arm. An earlier version of this file claimed the identity was "almost
certainly wrong for at least one joint"; that was wrong, and reading the
reference rather than guessing is what settled it.

`sign` and `offset_rad` stay as fields because they are the knobs a DIFFERENT
arm would need (a Koch follower does use `drive_mode=1`), and because a
mis-bolted horn is a real, physical thing that no table can predict.

⚠ The gripper is not an angle on either side. The servo reports it 0..100
(`MotorNormMode.RANGE_0_100`) and the model has it as a hinge in radians, so
it maps by FRACTION OF RANGE, and `offset_rad` does not apply to it. The
reference's gripper limits — `SO101_GRIPPER_LIMITS_RAD = (-10 deg, 100 deg)` —
agree with our model's `ctrlrange` to 1e-16, which is a pleasing independent
confirmation that we are pointing at the same joint.

⚠ **Range still disagrees, and that is measured** — see `range_report`. Our
MJCF ranges are byte-identical to
`references/SO-ARM100-main/Simulation/SO101/so101_new_calib.xml`, so the model
is a faithful port; three body joints simply have more CALIBRATED travel than
it accepts, and `wrist_roll` is a continuous joint meeting a bounded one.
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
        """The SO-101 mapping: every sign +1, every offset 0.

        Not a placeholder — this IS `motor_ticks_to_sim_rad` for
        `drive_mode=0`, which lerobot hard-codes for this arm. See the module
        docstring.
        """
        var s = InlineArray[Float64, SO101_N](fill=1.0)
        var o = InlineArray[Float64, SO101_N](fill=0.0)
        return Self(s^, o^, sim_lo^, sim_hi^)

    def differs_from_lerobot(self) -> Bool:
        """True once someone has moved a sign or an offset off the reference.

        Named for what it means. The previous name, `measured()`, said the
        opposite of the truth: the default is not an unmeasured guess, it is
        the reference implementation's mapping, and a `True` here means we
        have DEPARTED from it — which is the thing worth announcing.
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
            #
            # ⚠⚠ THE FRACTION IS COMPUTED HERE, NOT VIA `cal.degrees`, WHICH
            # CLAMPS. `degrees()` opens with `min(hi, max(lo, raw))` for the
            # gripper, so this function — whose entire contract is to be
            # unclamped, because `clamped_by` cannot report an overshoot it
            # cannot see — was silently clamped for exactly one joint. A
            # gripper parked below its CALIBRATED minimum then reported zero
            # overshoot while `to_sim`/`from_sim` disagreed by the whole
            # shortfall, which `deploy_reach_real.mojo`'s round-trip check
            # reported as a sign error in the mapping. Measured: 11 ticks.
            var lo_t = Float64(cal.range_min[i])
            var hi_t = Float64(cal.range_max[i])
            var span_t = hi_t - lo_t
            var frac = (Float64(raw) - lo_t) / span_t if span_t != 0.0 else 0.0
            if self.sign[i] < 0.0:
                frac = 1.0 - frac
            return self.sim_lo[i] + frac * (self.sim_hi[i] - self.sim_lo[i])
        return self.sign[i] * cal.radians(i, raw) + self.offset_rad[i]

    def to_sim(self, cal: SO101Calibration, i: Int, raw: Int32) -> Float64:
        var v = self.to_sim_unclamped(cal, i, raw)
        return min(self.sim_hi[i], max(self.sim_lo[i], v))

    def from_sim(
        self, cal: SO101Calibration, i: Int, value: Float64
    ) -> Int32:
        """Model radians back to servo ticks — the inverse of `to_sim`.

        What a POLICY's action has to go through to reach the hardware: the
        net was trained in the model's joint space, and the bus speaks ticks.
        Exact inverse of `to_sim_unclamped`, gripper fraction included, so a
        round trip through both is the identity up to tick quantisation.
        """
        if i == GRIPPER:
            var span = self.sim_hi[i] - self.sim_lo[i]
            var frac = (value - self.sim_lo[i]) / span if span != 0.0 else 0.0
            if self.sign[i] < 0.0:
                frac = 1.0 - frac
            return cal.raw_from_degrees(i, frac * 100.0)
        return cal.raw_from_radians(i, (value - self.offset_rad[i]) / self.sign[i])

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
            var sim_span = (
                self.sim_hi[i] - self.sim_lo[i]
            ) * 180.0 / 3.141592653589793

            if cal.is_unlimited(i):
                # ⚠ NOT a gap. `0..4095` is lerobot's unlimited marker, so
                # there is no measured travel to compare against — the joint
                # turns freely and the model bounds it. Reporting a number
                # here would invent one.
                out += (
                    pad_right(joint_name(i), 15)
                    + pad_left(String("free"), 11)
                    + col(sim_span, 11, 1)
                    + pad_left(String("n/a"), 9)
                    + "   <-- CONTINUOUS joint, bounded model\n"
                )
                continue

            var real_span = Float64(cal.span(i)) * 360.0 / 4095.0
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
        if not self.differs_from_lerobot():
            return String(
                "sim map: lerobot reference (drive_mode=0 => sign +1, no"
                " offset) — matches so101-nexus motor_ticks_to_sim_rad"
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
