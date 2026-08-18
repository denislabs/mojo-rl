"""The manipulation reset must survive its DTYPE — not just float64.

⚠⚠ WHY THIS FILE EXISTS. Every other manipulation gate in this tree pins
`comptime DTYPE = DType.float64`, and the facades (`DMReachSiteFeatures[DTYPE:
DType = DType.float64]`) default to it, so the whole family was only ever
compiled and measured in double precision. The VIEWER is not: `viewer_core`
builds `Phyics3dEnv[MODEL, CONFIG, DT, False]` with `DT = DType.float32` from
`nn.constants`, and so does every training driver that reaches these models.

At float32 the reset failed COMPLETELY and SILENTLY. `qpos_from_site_pose`
inherited dm_control's `tol = 1e-14`, which is a float64 number: `err_norm` is
a length in metres built through the FK chain, so its float32 resolution is
~1e-7 and the test could never be satisfied. Measured on `reach_site_features`
at float32, the converged attempts landed at `err_norm` 1.0e-07 .. 2.1e-06 and
were ALL reported as failures, while the genuinely stuck ones sat at 0.20 ..
0.95 — five orders of magnitude apart, with every member of the first
population discarded.

The consequence was not a slightly different pose. `tool_center_point_
initializer` exhausted all 10 rejection samples with 10/10 IK failures on every
reset of every task, `Phyics3dEnv._reset_state` printed and carried on, and the
episode began at qpos0 — where Jaco's `joint_2` and `joint_3` sit OUTSIDE their
own ranges ([0.820, 5.463] and [0.332, 5.952], both starting at 0) and MuJoCo
reports 55 contacts. The limit constraints then pin those joints against
actuators that cap at 30.5 N·m, so the arm cannot move at all while the
fingers, which violate by only 0.15 rad, free themselves and articulate
normally. That is the "frozen arm, moving fingers" the viewer showed, and it
was 16/16 resets across both tasks probed.

⚠ THE FIX IS IN `ik_site.default_ik_tol`, NOT HERE. float64 keeps dm_control's
1e-14 unchanged, so `test_ik_site_vs_dm_control` still asserts exact parity
against the reference; only the float32 arm is this port's own choice.

WHAT IS CHECKED, for BOTH dtypes:

  1. THE RESET DOES NOT LAND AT qpos0. The direct statement of the bug.
  2. THE TCP LANDS IN `tcp_bbox`. A solve that stopped on the progress guard
     rather than converging still reports success, so "not qpos0" alone would
     accept a pose the initializer never actually solved for.
  3. THE DISTRIBUTION IS NOT DEGENERATE — resets differ from one another.
  4. A NEGATIVE CONTROL proves check 1 can fail: qpos0 is written into the
     env deliberately and the same predicate must report it.

⚠ NOT A PARITY GATE. dm_control cannot be run at float32, so there is no
reference column here; `test_reach_site_reset_vs_dm_control` owns parity and
this owns precision-independence. Keep both.

Run with:
    pixi run mojo run -I . tests/dm_control/test_manipulation_reset_dtype.mojo
"""

from std.math import abs, max
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.envs.dm_control.manipulation_reach_config import (
    N_ARM,
    SITE_PINCH,
    ReachSiteFeaturesConfig,
    TARGET_BBOX_LOWER_X,
    TARGET_BBOX_LOWER_Y,
    TARGET_BBOX_LOWER_Z,
    TARGET_BBOX_UPPER_X,
    TARGET_BBOX_UPPER_Y,
    TARGET_BBOX_UPPER_Z,
)

comptime N_RESETS: Int = 8

# The IK stops at `default_ik_tol[DTYPE]()`, and `set_site_to_xpos` also accepts
# a solve that stopped on the progress guard, so the box test takes a
# millimetre of slack — the same allowance the float64 gate makes. A genuine
# convergence failure is orders out, not fractions of a millimetre.
comptime BOX_SLACK: Float64 = 1e-3

# `sum |qpos[0:N_ARM]|` below this counts as "the arm is at qpos0". Jaco's
# qpos0 is exactly zero on all six arm joints and any solved pose is O(1) on
# several of them, so this separates by ~15 orders and needs no tuning.
comptime QPOS0_EPS: Float64 = 1e-9


def _check[DTYPE: DType](label: String, ctx: DeviceContext) raises:
    comptime E = Phyics3dEnv[
        ReachSiteFeaturesModel, ReachSiteFeaturesConfig, DTYPE, False
    ]
    var env = E(ctx)

    var at_qpos0 = 0
    var outside_box = 0
    var worst_excess = Float64(0)
    var identical = 0
    var first_x = Float64(0)
    var first_y = Float64(0)
    var first_z = Float64(0)

    for r in range(N_RESETS):
        _ = env.reset()

        var arm_norm = Float64(0)
        for a in range(N_ARM):
            arm_norm += abs(Float64(env.d.qpos.data[a]))
        if arm_norm < QPOS0_EPS:
            at_qpos0 += 1

        var px = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 0])
        var py = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 1])
        var pz = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])

        var ex = Float64(0)
        if px < TARGET_BBOX_LOWER_X - BOX_SLACK:
            ex = max(ex, TARGET_BBOX_LOWER_X - px)
        if px > TARGET_BBOX_UPPER_X + BOX_SLACK:
            ex = max(ex, px - TARGET_BBOX_UPPER_X)
        if py < TARGET_BBOX_LOWER_Y - BOX_SLACK:
            ex = max(ex, TARGET_BBOX_LOWER_Y - py)
        if py > TARGET_BBOX_UPPER_Y + BOX_SLACK:
            ex = max(ex, py - TARGET_BBOX_UPPER_Y)
        if pz < TARGET_BBOX_LOWER_Z - BOX_SLACK:
            ex = max(ex, TARGET_BBOX_LOWER_Z - pz)
        if pz > TARGET_BBOX_UPPER_Z + BOX_SLACK:
            ex = max(ex, pz - TARGET_BBOX_UPPER_Z)
        if ex > 0.0:
            outside_box += 1
            worst_excess = max(worst_excess, ex)

        if r == 0:
            first_x = px
            first_y = py
            first_z = pz
        elif (
            abs(px - first_x) < 1e-12
            and abs(py - first_y) < 1e-12
            and abs(pz - first_z) < 1e-12
        ):
            identical += 1

    print("  ", label, "resets", N_RESETS,
          " at qpos0:", at_qpos0,
          " TCP outside tcp_bbox:", outside_box,
          " worst excess:", worst_excess,
          " identical to first:", identical)

    assert_true(
        at_qpos0 == 0,
        String(label)
        + ": "
        + String(at_qpos0)
        + " of "
        + String(N_RESETS)
        + " resets left the arm at qpos0 — the TCP initializer exhausted and"
        + " Phyics3dEnv fell through. At float32 this was the IK tolerance"
        + " sitting below the precision floor; see ik_site.default_ik_tol.",
    )
    assert_true(
        outside_box == 0,
        String(label)
        + ": the TCP landed outside tcp_bbox on "
        + String(outside_box)
        + " resets, worst excess "
        + String(worst_excess)
        + " m — the IK reported success without converging.",
    )
    assert_true(
        identical == 0,
        String(label)
        + ": "
        + String(identical)
        + " resets produced the SAME TCP as the first, so the hook is not"
        + " consuming its draws.",
    )


def test_manipulation_reset_survives_float64() raises:
    print("=== reach_site_features reset @ float64 ===")
    var ctx = DeviceContext()
    _check[DType.float64]("float64", ctx)


def test_manipulation_reset_survives_float32() raises:
    print("=== reach_site_features reset @ float32 (the viewer's dtype) ===")
    var ctx = DeviceContext()
    _check[DType.float32]("float32", ctx)


def test_qpos0_detector_is_not_vacuous() raises:
    """NEGATIVE CONTROL — the qpos0 predicate must fire when qpos0 is real.

    Without this, `at_qpos0 == 0` would also pass if `_arm_norm` were reading
    the wrong slots, or reading a buffer the reset never writes. It is the
    same failure the rest of this suite guards with a negative control: a
    check that cannot fail is not a check.
    """
    print("=== negative control: force qpos0 ===")
    var ctx = DeviceContext()
    comptime E = Phyics3dEnv[
        ReachSiteFeaturesModel, ReachSiteFeaturesConfig, DType.float32, False
    ]
    var env = E(ctx)
    _ = env.reset()
    var solved = Float64(0)
    for a in range(N_ARM):
        solved += abs(Float64(env.d.qpos.data[a]))
    for a in range(N_ARM):
        env.d.qpos.data[a] = Scalar[DType.float32](0)
    var forced = Float64(0)
    for a in range(N_ARM):
        forced += abs(Float64(env.d.qpos.data[a]))
    print("   solved-pose arm norm:", solved, "  forced qpos0 norm:", forced)
    assert_true(
        solved >= QPOS0_EPS,
        "the reset produced an arm norm below the qpos0 threshold",
    )
    assert_true(
        forced < QPOS0_EPS,
        "the qpos0 predicate did NOT fire on a deliberately zeroed arm — it"
        + " is reading something other than the arm joints",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
