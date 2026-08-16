"""SO-ARM100 — layer-2 parity: our `fields.Model` and rollout vs MuJoCo.

    pixi run mojo run -I . tests/robots/test_so_arm100_vs_mujoco.mojo

⚠ THIS IS LAYER 2 AND DOES NOT REPLACE LAYER 1. Both sides here compile from
OUR OWN XML string, so a defect in that string is invisible by construction.
`tests/robots/so_arm_ref.py` is the layer-1 gate that proves the string IS the
Menagerie model — run it too, and run it FIRST.

⚠ RUN FROM THE REPO ROOT. Mesh `file=` paths are repo-root-relative (our parser
does not implement `<compiler meshdir>`; see `so_arm_bake.py`), so both MuJoCo
and our loader resolve them against the cwd.

WHAT EACH GATE IS FOR, and what it caught:

  · `test_model_counts` — dimensions, INCLUDING `nexclude`. `ModelDefFromXML`
    defaults `nexclude`/`npair` to 0 and omitting them is SILENT: `parse_xml`
    reported NEXCLUDE 1 while the built model carried 0, so the two adjacent
    base geoms would have collided forever. Caught by asserting the model's
    counts and not just the parser's.

  · `test_actuator_law` — kp / kv / forcerange / ctrlrange per actuator. ⚠⚠
    THIS IS THE ONE THAT MATTERS. Our comptime parser resolves an attribute as
    element -> named class -> ROOT default, and does NOT walk the class chain
    in between. SO-100 declares `kp="50"` in a `<default>` class, so every servo
    silently ran at **kp = 1.0**, MuJoCo's default — a 50x weak controller that
    reads as bad tuning, not as a parse failure. The bake now
    writes every gain onto the element; this gate is what keeps it that way.

  · `test_zero_ctrl_rollout` — the discriminating one. With kp wrong the arm
    still moved TOWARD its target, just ~50x too slowly, and every static gate
    passed. Only stepping both sides from the same state showed it.

⚠ A ZERO-CONTACT MODEL. Measured: `ncon = 0` for the bare arm at every pose
these gates visit. So nothing here exercises the mesh collision path, however
green it is. Contacts need a manipuland; see `so_arm100_xml.mojo`.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.robots.so_arm100 import SoArm100Reach
from mojo_rl.envs.robots.so_arm100_xml import (
    SoArm100Model,
    MOVING_JAW_BODY_IDX,
    TARGET_BODY_IDX,
)
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_FORCE_MAX,
    ACT_IDX_FORCE_MIN,
    ACT_IDX_KP,
    ACT_IDX_KV,
)

comptime NQ = SoArm100Model.NQ
comptime NU = 6
comptime FRAME_SKIP = 10

# The two rollout regimes, measured on the reference and NOT assumed:
#
#   commanded pose -> max nefc 6   (the six `dof_frictionloss` rows only)
#   ctrl = 0       -> max nefc 15  (joint LIMIT rows engage: `Pitch` reaches
#                                   0.1609 against a limit of 0.174, `Elbow`
#                                   0.0097 against -0.174, `Jaw` 0.0002
#                                   against -0.174)
#
# They are gated separately because they exercise different code and agree to
# very different tolerances. Collapsing them into one gate would either hide
# the exact one or fail the constrained one.
comptime NEFC_UNCONSTRAINED = 6


def _pose(i: Int) -> Float64:
    if i == 0:
        return 0.35
    if i == 1:
        return -1.10
    if i == 2:
        return 0.90
    if i == 3:
        return 0.40
    if i == 4:
        return -0.60
    return 0.25


def _ctrl(i: Int) -> Float64:
    """A commanded pose distinct from `_pose`, so tracking is observable."""
    if i == 0:
        return -0.20
    if i == 1:
        return -1.60
    if i == 2:
        return 1.40
    if i == 3:
        return 0.80
    if i == 4:
        return 0.30
    return 1.00


def _mj() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_path("mojo_rl/envs/robots/assets/so_arm100.xml")


def test_model_counts() raises:
    """Dimensions, and `nexclude` — the one that was silently zero."""
    var m = _mj()
    assert_true(Int(py=m.nbody) == SoArm100Model.NBODY, "nbody")
    assert_true(Int(py=m.njnt) == SoArm100Model.NJOINT, "njnt")
    assert_true(Int(py=m.nq) == SoArm100Model.NQ, "nq")
    assert_true(Int(py=m.nv) == SoArm100Model.NV, "nv")
    assert_true(Int(py=m.ngeom) == SoArm100Model.NGEOM, "ngeom")
    assert_true(
        Int(py=m.nexclude) == SoArm100Model.NEXCLUDE,
        "nexclude: MuJoCo has the Base/Rotation_Pitch <exclude>; if ours is 0"
        " the ModelDefFromXML `nexclude=` parameter was omitted and the"
        " exclusion is silently absent",
    )
    assert_true(Int(py=m.nexclude) == 1, "the model should have 1 exclusion")
    print("  counts OK — nbody", Int(py=m.nbody), " ngeom", Int(py=m.ngeom),
          " nexclude", Int(py=m.nexclude))


def test_body_indices() raises:
    """The indices the config indexes by NAME, so a reorder cannot pass.

    ⚠ `MOVING_JAW_BODY_IDX` and `TARGET_BODY_IDX` are plain integers in the
    reward and the observation. If a future task fragment inserts a body, the
    reward silently starts measuring a different link and every other gate
    stays green.
    """
    var mujoco = Python.import_module("mujoco")
    var m = _mj()
    var ee = mujoco.mj_id2name(
        m, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY_IDX
    )
    var tg = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_IDX)
    assert_true(String(ee) == "Moving_Jaw", "EE body index names " + String(ee))
    assert_true(String(tg) == "target", "target body index names " + String(tg))
    print("  body indices OK —", MOVING_JAW_BODY_IDX, "=", String(ee), ",",
          TARGET_BODY_IDX, "=", String(tg))


def test_actuator_law() raises:
    """kp / kv / forcerange / ctrlrange, per actuator, against `mjModel`.

    MuJoCo's `<position>` compiles to `gainprm = [kp,0,0]`,
    `biasprm = [0,-kp,-kv]`. Ours stores `motor_kp` = `gainprm[0]` and
    `motor_kv` = `-biasprm[2]`, so this is a direct comparison and the
    tolerance is 0.0 — both numbers came from the same XML text.
    """
    var m = _mj()
    var sf = SoArm100Model.make_spec_fields[DType.float64]()
    var kp = actuator_column(sf, ACT_IDX_KP, NU)
    var kv = actuator_column(sf, ACT_IDX_KV, NU)
    var fmin = actuator_column(sf, ACT_IDX_FORCE_MIN, NU)
    var fmax = actuator_column(sf, ACT_IDX_FORCE_MAX, NU)
    var cmin = actuator_column(sf, ACT_IDX_CTRL_MIN, NU)
    var cmax = actuator_column(sf, ACT_IDX_CTRL_MAX, NU)

    var worst = 0.0
    for i in range(NU):
        var ref_kp = Float64(py=m.actuator_gainprm[i][0])
        var ref_kv = -Float64(py=m.actuator_biasprm[i][2])
        var ref_fl = Float64(py=m.actuator_forcerange[i][0])
        var ref_fh = Float64(py=m.actuator_forcerange[i][1])
        var ref_cl = Float64(py=m.actuator_ctrlrange[i][0])
        var ref_ch = Float64(py=m.actuator_ctrlrange[i][1])
        worst = max(worst, abs(kp[i] - ref_kp))
        worst = max(worst, abs(kv[i] - ref_kv))
        worst = max(worst, abs(fmin[i] - ref_fl))
        worst = max(worst, abs(fmax[i] - ref_fh))
        worst = max(worst, abs(cmin[i] - ref_cl))
        worst = max(worst, abs(cmax[i] - ref_ch))
        # ⚠ Asserted separately AND loudly: kp falling back to 1.0 is the
        # documented failure, and a max-over-everything would report it as a
        # 49.0 residual without naming the cause.
        assert_true(
            abs(kp[i] - ref_kp) == 0.0,
            "actuator " + String(i) + " kp is " + String(kp[i])
            + " but MuJoCo compiles " + String(ref_kp)
            + " — a value of 1.0 means the class-default lookup missed"
            " `<position kp>` and the servo is running at the fallback gain",
        )
    # ⚠ 1 ULP, NOT 0.0. `kp` IS exact (asserted above, and it is the value
    # that broke), but the 17-significant-digit literals the bake writes for
    # `kv` / `ctrlrange` come back ~2.8e-17 different: MuJoCo parses them with
    # a correctly-rounded `strtod` and our `_parse_float` is not, so the two
    # land one unit apart in the last place. Real, tiny, and worth a bound
    # rather than a pretend zero.
    assert_true(worst < 1e-15, "actuator law residual " + String(worst))
    print("  actuator law matches — worst |d| =", worst,
          " (kp exact; kv/ctrlrange within 1 ULP)")


def _load_reference(mujoco: PythonObject, m: PythonObject,
                    d: PythonObject) raises:
    """Both sides from the SAME state — the shared-state protocol.

    ⚠ `custom_reset_cpu` adds uniform noise, so `env.reset()` alone gives the
    two sides different initial conditions and any residual is meaningless.
    `set_state` overwrites it on our side; this writes the same numbers here.
    """
    mujoco.mj_resetData(m, d)
    for i in range(NQ):
        d.qpos[i] = _pose(i)
    mujoco.mj_forward(m, d)


def _rollout_residual(ctrl_from_pose: Bool) raises -> Float64:
    var mujoco = Python.import_module("mujoco")
    var m = _mj()
    var d = mujoco.MjData(m)
    _load_reference(mujoco, m, d)

    var env = SoArm100Reach[DType.float64]()
    _ = env.reset()
    var qp = List[Float64]()
    var qv = List[Float64]()
    for i in range(NQ):
        qp.append(_pose(i))
        qv.append(0.0)
    env.set_state(qp, qv)

    var a = ContAction[SoArm100Model.ACTION_DIM]()
    for i in range(NU):
        var c = _ctrl(i) if ctrl_from_pose else 0.0
        a.data[i] = c
        d.ctrl[i] = c

    var worst = 0.0
    for _ in range(200):
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        _ = env.step(a)
        for i in range(NQ):
            var e = abs(Float64(env.d.qpos.data[i]) - Float64(py=d.qpos[i]))
            worst = max(worst, e)
    return worst


def test_limit_free_rollout_is_exact() raises:
    """200 control steps under a commanded pose — the STRONG gate.

    Measured on the reference: this trajectory never engages a joint limit
    (max `nefc` 6, the frictionloss rows), so it isolates FK + CRBA + RNE +
    the `<position>` servo + Euler with nothing else in the loop. It comes out
    at **2.2e-16**, i.e. bit-exact, and the bound below is set from that rather
    than inherited.

    ⚠ THIS IS THE GATE THAT FOUND THE kp FALLBACK. With `kp = 1.0` the arm
    still moved toward its target, just ~50x too slowly, and every static
    comparison in this file passed. A servo that ignores `ctrl` entirely also
    survives a zero-ctrl gate, because zero is where it would sit anyway —
    only a NON-ZERO command discriminates.
    """
    var worst = _rollout_residual(True)
    print("  commanded 200-step worst |dqpos| =", worst)
    assert_true(worst < 1e-12, "commanded rollout residual " + String(worst))


def test_rollout_into_joint_limits() raises:
    """The same rollout at ctrl = 0, which DOES engage joint limits.

    ⚠⚠ THE TOLERANCE HERE IS 4 000x LOOSER THAN THE GATE ABOVE, AND THAT IS
    THE FINDING, not an accommodation. Driving every servo to zero pushes
    `Pitch` to 0.1609 against its 0.174 limit and `Elbow`/`Jaw` down onto
    -0.174; `nefc` goes 6 -> 15. The limit-free trajectory agrees to 2.2e-16
    and this one to ~2.4e-4, so the entire residual belongs to the joint-limit
    constraint path — nothing else differs between the two runs.

    That makes these two arms a clean probe of that path: no contacts exist
    anywhere in the model (`test_arm_is_contact_free`), so a limit row is the
    only constraint in the system. Worth a narrower fixture if the number ever
    needs to come down.
    """
    var worst = _rollout_residual(False)
    print("  zero-ctrl (limits ACTIVE) 200-step worst |dqpos| =", worst)
    assert_true(
        worst < 1e-3,
        "zero-ctrl rollout residual " + String(worst)
        + " — this trajectory rides three joint limits; a regression here is"
        " the limit constraint, not the servo (which the limit-free gate"
        " pins at 2.2e-16)",
    )


def test_arm_is_contact_free() raises:
    """Pinned, because it bounds what every other gate here can prove.

    MuJoCo produces `ncon = 0` for the bare arm at these poses. If that ever
    stops being true the gates above start covering the mesh path — and if it
    stays true, nothing here does. Either way it should be a measurement.
    """
    var mujoco = Python.import_module("mujoco")
    var m = _mj()
    var d = mujoco.MjData(m)
    _load_reference(mujoco, m, d)
    var worst_ncon = 0
    for i in range(2000):
        for k in range(NU):
            d.ctrl[k] = _ctrl(k)
        mujoco.mj_step(m, d)
        worst_ncon = max(worst_ncon, Int(py=d.ncon))
    print("  MuJoCo max ncon over the commanded rollout:", worst_ncon)
    assert_true(
        worst_ncon == 0,
        "the arm now makes contact (ncon " + String(worst_ncon) + ") — the"
        " gates in this file no longer bound the mesh path the way the header"
        " claims; re-read it",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
