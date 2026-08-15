"""Per-actuator action bounds vs dm_control's `action_spec`.

`BoxContinuousActionEnv.action_low/action_high` are SCALARS by contract, and
`MODEL_DEF.CTRL_MIN/CTRL_MAX` — what they return — come from a ROOT
`<default><motor ctrlrange>`. A model that keeps its ranges per actuator or
per default class silently gets (-1, 1) instead. `ctrl_min_at`/`ctrl_max_at`
are the per-actuator answer; this gates them against dm_control's own spec.

⚠ THE SIMULATION WAS NEVER WRONG. `apply_actions` already clamped each
actuator to its own range — verified here as the same numbers. What was
missing was any way for an env to TELL a policy what that range is, so a
policy sampling the advertised (-1, 1) had part of its output clamped away on
tight actuators and could never reach the limit of loose ones.

TWO MODELS, chosen so the gate cannot pass by accident:

  * `reach_site_features` — THREE distinct ranges (+/-0.6283 x3, +/-0.8378 x3,
    +/-5.0 x3), none of them the (-1, 1) default and one of them 5x it. Also
    the case that motivated this: its actuators are `<velocity>`, which
    `_xml_default_motor_ctrlrange` does not even look at.
  * `quadruped walk` — ALREADY SHIPPED and already non-uniform (lo in
    [-1, -0.8], hi in [0.8, 1.1]), which is what makes this a live bug rather
    than a manipulation-only one. Its ranges live in a default CLASS.

⚠ A UNIFORM MODEL WOULD PROVE NOTHING HERE. dog, humanoid, walker, cheetah and
finger are all uniform +/-1, so on any of them the broken scalar path and the
correct per-actuator path return the same numbers.

Run with:
    pixi run mojo run -I . tests/dm_control/test_per_actuator_action_bounds.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    DMQuadrupedWalkModel,
)

comptime TOL: Float64 = 1e-12


def test_reach_site_features_per_actuator_bounds() raises:
    print("=== reach_site_features: per-actuator action bounds ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var refmod = Python.import_module("manipulation_ref")

    var spec = refmod.action_spec_reference("reach_site_features")
    var lo = spec[0]
    var hi = spec[1]
    var nu = Int(py=Python.evaluate("len")(lo))

    comptime M = ReachSiteFeaturesModel
    var sf = M.make_spec_fields[DType.float64]()
    print("  model-wide CTRL_MIN/CTRL_MAX:", M.CTRL_MIN, M.CTRL_MAX)
    var worst = 0.0
    var n_off_scalar = 0
    for a in range(nu):
        var mlo = Float64(py=lo[a])
        var mhi = Float64(py=hi[a])
        var e0 = abs(M.ctrl_min_at[DType.float64](sf, a) - mlo)
        var e1 = abs(M.ctrl_max_at[DType.float64](sf, a) - mhi)
        if e0 > worst:
            worst = e0
        if e1 > worst:
            worst = e1
        if abs(M.CTRL_MIN - mlo) > 1e-9 or abs(M.CTRL_MAX - mhi) > 1e-9:
            n_off_scalar += 1
        print("    act", a, " ours [", M.ctrl_min_at[DType.float64](sf, a), ",",
              M.ctrl_max_at[DType.float64](sf, a), "]  spec [", mlo, ",", mhi, "]")
    print("  worst |d| vs action_spec:", worst)
    print("  actuators the SCALAR bound gets wrong:", n_off_scalar, "of", nu)

    assert_true(
        n_off_scalar == nu,
        "the scalar bound happens to be right here, so this model cannot"
        " show the difference between the two paths — pick another",
    )
    assert_true(
        worst < TOL,
        "per-actuator bounds disagree with dm_control's action_spec",
    )


def test_quadruped_per_actuator_bounds() raises:
    print("=== quadruped walk: per-actuator action bounds (SHIPPED env) ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var refmod = Python.import_module("manipulation_ref")

    var spec = refmod.suite_action_spec_reference("quadruped", "walk")
    var lo = spec[0]
    var hi = spec[1]
    var nu = Int(py=Python.evaluate("len")(lo))

    comptime M = DMQuadrupedWalkModel
    var sf = M.make_spec_fields[DType.float64]()
    print("  model-wide CTRL_MIN/CTRL_MAX:", M.CTRL_MIN, M.CTRL_MAX)
    var worst = 0.0
    var n_distinct = 0
    for a in range(nu):
        var mlo = Float64(py=lo[a])
        var mhi = Float64(py=hi[a])
        var e0 = abs(M.ctrl_min_at[DType.float64](sf, a) - mlo)
        var e1 = abs(M.ctrl_max_at[DType.float64](sf, a) - mhi)
        if e0 > worst:
            worst = e0
        if e1 > worst:
            worst = e1
        if abs(mlo - Float64(py=lo[0])) > 1e-9 or abs(
            mhi - Float64(py=hi[0])
        ) > 1e-9:
            n_distinct += 1
    print("  nu:", nu, " actuators differing from actuator 0:", n_distinct)
    print("  worst |d| vs action_spec:", worst)

    # ⚠ NON-VACUITY: this env is only worth gating BECAUSE it is non-uniform.
    # If dm_control ever made quadruped uniform, this test would still pass
    # while proving nothing, so the non-uniformity is asserted too.
    assert_true(
        n_distinct > 0,
        "quadruped is uniform after all — then it cannot distinguish the"
        " scalar path from the per-actuator one, and the claim that this bug"
        " is live on a shipped env needs re-checking",
    )
    assert_true(
        worst < TOL,
        "quadruped's per-actuator bounds disagree with dm_control's"
        " action_spec — this is a SHIPPED env, so a mismatch here is a live"
        " action-space bug, not a manipulation-only one",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
