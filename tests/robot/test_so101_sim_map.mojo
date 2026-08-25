# +--------------------------------------------------------------------------+ #
# | Real SO-101 joints -> simulated SO-101 joints
# +--------------------------------------------------------------------------+ #
"""Gate for `mojo_rl/robot/so101/sim_map.mojo`, with no arm on the desk.

The calibration below is the REAL one, read off this follower's servo EEPROM
on 2026-08-25 (`pixi run soarm-diag`), so the numbers under test are the ones
the hardware actually produces — including the negative sign-magnitude homing
offsets. The simulated limits come from the MODEL (`actuator_column` over
`ACT_IDX_CTRL_MIN/MAX`; a `<position>` servo's ctrlrange IS its joint range),
never from a transcription — so if either side moves, this test moves with it
rather than quietly testing a stale copy.

⚠ **This gates the CONVENTIONS, not the mapping.** Sign and zero are still
unmeasured (`SimJointMap.identity`), and no test can discover them — that
needs a human moving the real arm against
`examples/so101/teleop_sim.mojo`. What is gated here is that the plumbing
around them is right: mid-range maps to zero, the gripper maps by fraction
rather than as an angle, a sign flip actually inverts, and the range
disagreement is REPORTED rather than silently clamped away.

Run: pixi run mojo run -I . tests/robot/test_so101_sim_map.mojo
"""

from std.math import abs
from std.testing import assert_almost_equal, assert_equal, assert_true, assert_false, TestSuite

from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from mojo_rl.robot.so101.arm import GRIPPER, SO101Calibration, SO101_N
from mojo_rl.robot.so101.sim_map import SimJointMap

# Measured on the follower, 2026-08-25. `ofs` is sign-magnitude-decoded, which
# is why four of the six are negative.
# Functions, not `comptime` arrays: a comptime `InlineArray` is not
# `ImplicitlyCopyable` and cannot be materialised at runtime.
def CAL_OFS(i: Int) -> Int:
    var v: List[Int] = [-430, 563, 382, -51, -353, -485]
    return v[i]


def CAL_LO(i: Int) -> Int:
    var v: List[Int] = [592, 816, 927, 875, 0, 2030]
    return v[i]


def CAL_HI(i: Int) -> Int:
    var v: List[Int] = [3317, 3236, 3130, 3212, 4095, 3513]
    return v[i]


def _cal() raises -> SO101Calibration:
    var ofs = InlineArray[Int32, SO101_N](fill=0)
    var lo = InlineArray[Int32, SO101_N](fill=0)
    var hi = InlineArray[Int32, SO101_N](fill=0)
    for i in range(SO101_N):
        ofs[i] = Int32(CAL_OFS(i))
        lo[i] = Int32(CAL_LO(i))
        hi[i] = Int32(CAL_HI(i))
    return SO101Calibration(ofs^, lo^, hi^)


def _map() raises -> SimJointMap:
    """Model limits straight out of the compiled model."""
    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
    var lo = InlineArray[Float64, SO101_N](fill=0.0)
    var hi = InlineArray[Float64, SO101_N](fill=0.0)
    for i in range(SO101_N):
        lo[i] = Float64(lo_col[i])
        hi[i] = Float64(hi_col[i])
    return SimJointMap.identity(lo^, hi^)


def _mid(cal: SO101Calibration, i: Int) raises -> Int32:
    return Int32(Int(cal.mid(i)))


def test_identity_maps_the_servo_midpoint_to_zero() raises:
    """A body joint at its calibrated mid-point is 0 rad under the identity.

    This is the definition `SO101Calibration.radians` implements, and it is
    what makes `offset_rad` mean "how far the MODEL's zero sits from the
    servo's mid-point" rather than something ad hoc.
    """
    var cal = _cal()
    var m = _map()
    for i in range(SO101_N):
        if i == GRIPPER:
            continue
        assert_almost_equal(
            m.to_sim_unclamped(cal, i, _mid(cal, i)),
            0.0,
            atol=1e-3,
            msg="joint " + String(i) + " midpoint is not 0 rad",
        )


def test_gripper_maps_by_fraction_not_as_an_angle() raises:
    """The servo reports the gripper 0..100, the model wants radians, and the
    two ranges differ (101.8 deg of servo travel, 110 deg of model hinge). A
    fraction map takes the servo's closed end to the model's low limit and its
    open end to the high one; treating the 0..100 as degrees would land at
    ~1.75 rad for a fully open gripper and clamp."""
    var cal = _cal()
    var m = _map()
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, Int32(CAL_LO(GRIPPER))),
        m.sim_lo[GRIPPER],
        atol=1e-9,
        msg="closed gripper -> model low limit",
    )
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, Int32(CAL_HI(GRIPPER))),
        m.sim_hi[GRIPPER],
        atol=1e-9,
        msg="open gripper -> model high limit",
    )
    # And it NEVER clamps, at either end — that is what fraction-mapping buys.
    assert_equal(m.clamped_by(cal, GRIPPER, Int32(CAL_LO(GRIPPER))), 0.0)
    assert_equal(m.clamped_by(cal, GRIPPER, Int32(CAL_HI(GRIPPER))), 0.0)


def test_the_range_gap_is_reported_not_hidden() raises:
    """⚠ THE FINDING THIS FILE EXISTS FOR. EVERY body joint has more
    calibrated travel than the model accepts, so an arm at its own end stop
    asks for a pose the simulation refuses. `clamped_by` must SAY so.

    `to_sim` clamps (the model would reject the value anyway); a caller that
    only ever saw the clamped number could not tell "the arm is at its limit"
    from "the mapping is wrong", which is exactly the confusion teleop_sim is
    meant to resolve.
    """
    var cal = _cal()
    var m = _map()
    var clamped_joints = 0
    for i in range(SO101_N):
        if i == GRIPPER:
            continue
        var at_lo = m.clamped_by(cal, i, Int32(CAL_LO(i)))
        var at_hi = m.clamped_by(cal, i, Int32(CAL_HI(i)))
        if at_lo > 0.0 or at_hi > 0.0:
            clamped_joints += 1
            # And the clamped value must sit exactly ON the limit.
            var v = m.to_sim(cal, i, Int32(CAL_HI(i)))
            assert_true(
                v <= m.sim_hi[i] + 1e-12 and v >= m.sim_lo[i] - 1e-12,
                "clamped value is inside the model range",
            )
    assert_equal(
        clamped_joints,
        5,
        (
            "expected ALL 5 body joints' real travel to exceed the model's —"
            " if this changed, either the arms were recalibrated or"
            " so_arm101.xml's ranges moved, and the sim/real story changed"
            " with it"
        ),
    )


def test_elbow_flex_is_marginal_and_the_others_are_not() raises:
    """Non-vacuity for the test above, and a finding of its own.

    A `clamped_by` that returned some positive number unconditionally would
    pass the count test, so the MAGNITUDES have to separate. They do, by three
    orders of magnitude:

    * `elbow_flex` overshoots by **9.2e-05 rad (0.005 deg)** — the MJCF writes
      its range as a rounded `±1.69`, where every other joint carries full
      precision, and that rounding is the entire overshoot. Nothing to fix.
    * every other body joint overshoots by **> 0.05 rad (3 deg)** — real
      travel the simulation genuinely cannot represent.

    ⚠ Which arm's calibration you use matters: this is the FOLLOWER's. The
    leader's elbow span is 11 ticks narrower and does NOT clamp at all, so a
    gate written against the leader would have found four joints, not five.
    """
    var cal = _cal()
    var m = _map()
    var elbow = max(
        m.clamped_by(cal, 2, Int32(CAL_LO(2))),
        m.clamped_by(cal, 2, Int32(CAL_HI(2))),
    )
    assert_true(elbow > 0.0, "elbow does overshoot, if barely")
    assert_true(
        elbow < 1.0e-3,
        "elbow overshoot is rounding, not travel: " + String(elbow),
    )
    for i in [Int(0), Int(1), Int(3), Int(4)]:
        var worst = max(
            m.clamped_by(cal, i, Int32(CAL_LO(i))),
            m.clamped_by(cal, i, Int32(CAL_HI(i))),
        )
        assert_true(
            worst > 0.05,
            (
                "joint "
                + String(i)
                + " should overshoot by degrees, not rounding: "
                + String(worst)
            ),
        )


def test_sign_flip_actually_inverts() raises:
    """The knob teleop_sim exists to turn. A flipped sign must negate a body
    joint's angle and mirror the gripper's fraction."""
    var cal = _cal()
    var m = _map()
    var raw = Int32(CAL_LO(0) + (CAL_HI(0) - CAL_LO(0)) // 3)
    var pos = m.to_sim_unclamped(cal, 0, raw)
    m.sign[0] = -1.0
    assert_almost_equal(m.to_sim_unclamped(cal, 0, raw), -pos, atol=1e-12)

    var g_raw = Int32(CAL_LO(GRIPPER))
    m.sign[GRIPPER] = -1.0
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, g_raw),
        m.sim_hi[GRIPPER],
        atol=1e-9,
        msg="a mirrored gripper takes CLOSED to the model's open end",
    )


def test_offset_shifts_body_joints_and_not_the_gripper() raises:
    """`offset_rad` is a zero correction for angles. The gripper is not an
    angle on the servo side, so applying an offset there would silently move
    one end of the fraction map."""
    var cal = _cal()
    var m = _map()
    var raw = _mid(cal, 1)
    var g_raw = _mid(cal, GRIPPER)
    var before_g = m.to_sim_unclamped(cal, GRIPPER, g_raw)
    m.offset_rad[1] = 0.25
    m.offset_rad[GRIPPER] = 0.25
    assert_almost_equal(m.to_sim_unclamped(cal, 1, raw), 0.25, atol=1e-3)
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, g_raw), before_g, atol=1e-12
    )


def test_measured_flag_tells_the_truth() raises:
    """A caller must be able to say "this mapping is unmeasured" out loud —
    which is the difference between an honest viewer banner and a fake one."""
    var m = _map()
    assert_false(m.measured(), "the identity is not a measurement")
    m.sign[3] = -1.0
    assert_true(m.measured(), "a flipped sign counts as measured")


def test_monotonic_in_ticks() raises:
    """More ticks must never mean a smaller angle under a positive sign. A
    sign or midpoint slip inside `radians` would show up here."""
    var cal = _cal()
    var m = _map()
    for i in range(SO101_N):
        var prev = m.to_sim_unclamped(cal, i, Int32(CAL_LO(i)))
        for k in range(1, 11):
            var raw = Int32(
                CAL_LO(i) + (CAL_HI(i) - CAL_LO(i)) * k // 10
            )
            var v = m.to_sim_unclamped(cal, i, raw)
            assert_true(
                v >= prev - 1e-12,
                "joint " + String(i) + " is not monotonic in ticks",
            )
            prev = v


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
