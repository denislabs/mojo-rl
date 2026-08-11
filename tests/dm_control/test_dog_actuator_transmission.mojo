"""dog's actuator TRANSMISSION against MuJoCo — the last untested factor.

WHY THIS FILE EXISTS

Every other piece of the dog step is now measured exact: the contact set, the
mixed parameters, the row constants, the Jacobian, the mass matrix, the bias,
the whole solve (2.99e-11), an applied force on every dof (3.41e-11), and five
env steps through the production `Phyics3dEnv` path (1e-13) — but only with
`ctrl = act = 0`. The rollout, which drives `ctrl != 0`, is still red.

So the residual is in the ACTUATOR path, and that path is exactly three
factors:

    qfrc_actuator[dof] = moment[act, dof] * gain[act] * act[act]
                         `-- transmission --'  `--- gainprm -----'

`gainprm` is now gated (`test_dog_actuator_gain.mojo`, defect 16) and the
activation is gated by the rollout itself (`|d(act)| = 0.0`). The transmission
has NEVER been compared, and it is the factor most likely to be wrong: 8 of
dog's 38 actuators drive FIXED TENDONS, whose moment is `gear * coef_k`
scattered over every joint the tendon wraps — up to 11 of them.

⚠ AND A TRUNCATION LIVED THERE. Defect 17 was a bare `while n < 4` in the
tendon wrap loop, so `caudal_extend` (11 wraps) and `caudal_bend` (10) drove
their first four joints and nothing else. That is fixed; this file is what
should have caught it, and what will catch the next one.

WHAT IS COMPARED. MuJoCo's `d.actuator_moment` is sparse (`nnz` = 73 for dog,
not `nu*nv`), so it is densified with `mju_sparse2dense` on the reference side.
Ours is built from the comptime tables the engine actually reads in
`apply_actions`: `gear * coef_k` written at `dadr_k`. Comparing the assembled
MATRIX rather than the raw tables is deliberate — it is invariant to how either
side chooses to store the transmission, and a wrong dof index shows up as two
large errors rather than as a silently-reordered table that still sums right.

⚠ `_acd` RE-MATERIALIZES ON EVERY READ. Reading it field-by-field inside a loop
yields garbage; §8 of the plan requires one explicit `materialize` into a local.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_actuator_transmission.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.dog import DMDogStandWalkModel
from mojo_rl.physics3d.parser.xml_parser import MAX_COMPTIME_TENDON_WRAPS

comptime M = DMDogStandWalkModel
comptime NV = M.NV
comptime NACT = M.nact
comptime TEST_PATH = "tests/dm_control"


def _ref() raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var mujoco = Python.import_module("mujoco")
    var builder = Python.import_module("dog_ref")
    var m = builder.model()
    var d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return (mujoco, m, d)


def test_dog_actuator_moments_match_mujoco() raises:
    """The full `nu x nv` transmission matrix, element by element."""
    print("--- dog: actuator transmission vs MuJoCo ---")
    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var d = h[2]
    var np = Python.import_module("numpy")

    var nu = Int(py=m.nu)
    var nv = Int(py=m.nv)
    assert_true(
        NACT == nu and NV == nv,
        "actuator or dof COUNT differs — every per-index comparison below"
        " would be comparing different actuators",
    )

    # MuJoCo stores the moment SPARSE (nnz, not nu*nv). Densify on its side.
    var mom = np.zeros(Python.tuple(nu, nv))
    mujoco.mju_sparse2dense(
        mom, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
    )

    # ⚠ ONE materialize, per §8 — not a read per element.
    var acd = materialize[M._acd]()

    var worst = 0.0
    var worst_a = -1
    var worst_dof = -1
    var worst_ours = 0.0
    var worst_mj = 0.0
    var n_tendon_act = 0
    var max_wraps = 0
    var total_nnz_ours = 0
    var total_nnz_mj = 0

    for a in range(nu):
        # Assemble OUR row exactly as `apply_actions` scatters it.
        var row = List[Float64]()
        for _ in range(nv):
            row.append(0.0)
        var n = acd.motor_trn_n[a]
        var gear = acd.motor_gears[a]
        if n > 1:
            n_tendon_act += 1
        if n > max_wraps:
            max_wraps = n
        for k in range(n):
            var dadr = acd.motor_trn_dadr[a * MAX_COMPTIME_TENDON_WRAPS + k]
            var coef = acd.motor_trn_coef[a * MAX_COMPTIME_TENDON_WRAPS + k]
            if dadr >= 0 and dadr < nv:
                row[dadr] += gear * coef

        for j in range(nv):
            var want = Float64(py=mom[a][j])
            if abs(row[j]) > 1e-12:
                total_nnz_ours += 1
            if abs(want) > 1e-12:
                total_nnz_mj += 1
            var e = abs(row[j] - want)
            if e > worst:
                worst = e
                worst_a = a
                worst_dof = j
                worst_ours = row[j]
                worst_mj = want

    print("  actuators driving a tendon (n_wraps > 1):", n_tendon_act)
    print("  widest transmission:", max_wraps, "wraps")
    print("  nonzeros: ours", total_nnz_ours, " MuJoCo", total_nnz_mj)
    print("  max |d(moment)| =", worst, " at actuator", worst_a, "dof", worst_dof)
    if worst_a >= 0:
        print("      ours", worst_ours, " MuJoCo", worst_mj)

    # DIAGNOSTIC: which dofs each side drives, for every actuator that misses.
    # A wrong INDEX and a wrong COEFFICIENT are different defects and the
    # scalar above cannot tell them apart.
    for a in range(nu):
        var n = acd.motor_trn_n[a]
        var ours_s = String("")
        for k in range(n):
            ours_s += String(acd.motor_trn_dadr[a * MAX_COMPTIME_TENDON_WRAPS + k]) + " "
        var mj_s = String("")
        var bad = False
        for j in range(nv):
            if abs(Float64(py=mom[a][j])) > 1e-12:
                mj_s += String(j) + " "
        for j in range(nv):
            var got = 0.0
            for k in range(n):
                if acd.motor_trn_dadr[a * MAX_COMPTIME_TENDON_WRAPS + k] == j:
                    got += acd.motor_gears[a] * acd.motor_trn_coef[
                        a * MAX_COMPTIME_TENDON_WRAPS + k
                    ]
            if abs(got - Float64(py=mom[a][j])) > 1e-9:
                bad = True
        if bad:
            print("  act", a, "MISS")
            print("      ours dofs:", ours_s)
            print("      mj   dofs:", mj_s)

    # NON-VACUITY. If every actuator were a single-joint transmission this file
    # could not tell a correct tendon path from an absent one — and the tendon
    # path is the whole reason it exists.
    assert_true(
        n_tendon_act >= 4 and max_wraps >= 8,
        "MuJoCo compiled no wide tendon transmissions for dog — the truncation"
        " this test targets is not expressible in the model, so a match here"
        " would prove nothing",
    )
    assert_true(
        total_nnz_ours == total_nnz_mj,
        "the transmission has a different NUMBER of nonzeros than MuJoCo's —"
        " a truncated or over-long wrap list. Defect 17's shape: a bare"
        " `while n < 4` capped every fixed tendon at four joints.",
    )
    assert_true(
        worst <= 1e-9,
        "actuator moment differs from MuJoCo — the actuator force reaches the"
        " wrong dofs or with the wrong arm. `qfrc_actuator = moment * gain *"
        " act`, and gain and act are both already gated, so this is the whole"
        " remaining actuator residual.",
    )


def test_dog_actuator_dynamics_match_mujoco() raises:
    """`dynprm` and `ctrlrange` — the other two things the class chain sets.

    Defect 16 was that nested `<default>` classes were not walked for actuator
    attributes. `gainprm` was the loud casualty, but the SAME resolution feeds
    `dynprm` (the filter time constant, which decides how fast `act` tracks
    `ctrl`) and `ctrlrange` (which clamps the drive). A gain-only gate would
    leave two thirds of the fix ungated.
    """
    print("--- dog: actuator dynprm/ctrlrange vs MuJoCo ---")
    var h = _ref()
    var m = h[1]
    var acd = materialize[M._acd]()

    var nu = Int(py=m.nu)
    var worst_tau = 0.0
    var worst_lo = 0.0
    var worst_hi = 0.0
    var n_filter = 0
    for i in range(nu):
        var tau_mj = Float64(py=m.actuator_dynprm[i][0])
        var dyntype = Int(py=m.actuator_dyntype[i])
        if dyntype != 0:
            n_filter += 1
        var e = abs(acd.motor_dyn_tau[i] - tau_mj)
        if e > worst_tau:
            worst_tau = e
        var lo = Float64(py=m.actuator_ctrlrange[i][0])
        var hi = Float64(py=m.actuator_ctrlrange[i][1])
        var el = abs(acd.motor_ctrl_min[i] - lo)
        var eh = abs(acd.motor_ctrl_max[i] - hi)
        if el > worst_lo:
            worst_lo = el
        if eh > worst_hi:
            worst_hi = eh

    print("  actuators with a dyntype:", n_filter, "/", nu)
    print("  max |d(dynprm[0])| =", worst_tau)
    print("  max |d(ctrlrange)| =", worst_lo, worst_hi)

    assert_true(
        n_filter == nu,
        "dog's actuators are all `dyntype=\"filter\"` — if MuJoCo compiled any"
        " without an activation the filter comparison is measuring nothing",
    )
    assert_true(worst_tau <= 1e-12, "actuator dynprm differs from MuJoCo")
    assert_true(
        worst_lo <= 1e-12 and worst_hi <= 1e-12,
        "actuator ctrlrange differs from MuJoCo — the drive is clamped"
        " differently, which only a saturating rollout would reveal",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
