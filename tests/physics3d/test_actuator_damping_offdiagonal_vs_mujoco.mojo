"""`mjd_actuator_vel` writes an OUTER PRODUCT, and only where `qDeriv` has room.

    pixi run mojo run -I . tests/physics3d/test_actuator_damping_offdiagonal_vs_mujoco.mojo

WHAT WAS MISSING. `Model.dof_actdamp` is `sum_a kv_a * trn_a^2` — the DIAGONAL
of `J^T diag(kv) J`. MuJoCo adds the whole thing:

    addJTBJSparse(m, d, d->actuator_moment, &bias_vel, 1, i, ...)
        -> qDeriv += moment^T * biasprm[2] * moment

For a JOINT transmission the two are the same object — one dof, one entry —
so nothing was lost on any legged model in the tree. A TENDON transmission
across several dofs has off-diagonal terms too, and the implicit integrators
then solve against a different `M_hat`.

⚠ MEASURED. Forcing BOTH engines to Euler — which never touches `qDeriv` —
took `hello_robot_stretch` from **4.406e-05 to 1.823e-10** while 49 of the
other 50 `implicitfast` scenes in Menagerie did not move. Its `arm_extend`
drives the four telescoping links through one fixed tendon at `kv = 10`.
Landing this took the 1-step sweep from 74/85 to **75/85**.

⚠⚠⚠ AND THE OUTER PRODUCT IS NOT WRITTEN IN FULL. `addJTBJSparse` accumulates
through `mju_addToSclSparseInc` (`engine_derivative.c:768`), which writes only
the columns `qDeriv`'s SPARSE `D` pattern lists for that row and **silently
drops the rest**. `D` is the mass matrix's pattern, and a tree's mass matrix
couples two dofs only when one's body is an ANCESTOR of the other's.

So a tendon spanning two SIBLING branches gets its diagonal and nothing else.
Every 2-dof gripper here is exactly that — franka_emika_panda, robotiq_2f85
and v4, stanford_tidybot, ufactory_xarm7 — and MuJoCo's `qDeriv` has no
off-diagonal for any of them (read straight out of `D_colind`: xarm7's
gripper is -2.5 at (7,7) and (10,10) and ABSENT at (7,10)).

⚠⚠ WRITING THE FULL PRODUCT REGARDLESS IS A REGRESSION, AND IT WAS MEASURED
AS ONE: the sweep went 74/85 -> **72/85**, stretch fixed and those grippers
each acquiring a 2.6e-06..3.0e-05 error that had not been there. The rule was
then checked exhaustively rather than argued — over **74,671** dof pairs
across all 85 scenes, "the bodies are ancestor-related" and "(i, j) is in
`D_colind`" disagree **zero** times.

THE FIXTURE HAS BOTH SHAPES IN ONE FILE, which is the whole point:

  * `s1 -> s2 -> s3`, a SERIAL chain, driven by the `chain` tendon. Every
    pair is ancestor-related, so MuJoCo's `qDeriv` block is full:
    -10.2 on the diagonal, **-10 off**.
  * `fa` and `fb`, two SIBLINGS under a shared parent, driven by the
    `siblings` tendon at the same `kv`. MuJoCo writes **-10.2 on the diagonal
    and nothing off it.**

One model, one `kv`, one tendon type — the only difference is the tree, and a
build that ignores the sparsity gets the second half wrong while passing the
first.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.studio.stepping import (
    StudioImpFastPyr, StudioImpFastEll, StudioIntegPyr, StudioIntegEll,
    studio_cone_of, studio_uses_implicit,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import (
    KEY_IDX_NQPOS, KEY_IDX_NQVEL, TENDON_MAX_WRAPS,
    ACTDAMP_TRN_SIZE, ACTDAMP_IDX_N, ACTDAMP_IDX_DOF_0, ACTDAMP_IDX_PAIR_0,
)

comptime DT = DType.float64

comptime STRETCH = String(
    "references/mujoco_menagerie-main/hello_robot_stretch/scene.xml"
)

comptime XML = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.004" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="c1" pos="0 0 1">
      <joint name="s1" type="slide" axis="1 0 0" damping="0.2"/>
      <geom type="box" size=".05 .05 .05" mass="1"/>
      <body name="c2" pos="0.2 0 0">
        <joint name="s2" type="slide" axis="1 0 0" damping="0.2"/>
        <geom type="box" size=".05 .05 .05" mass="1"/>
        <body name="c3" pos="0.2 0 0">
          <joint name="s3" type="slide" axis="1 0 0" damping="0.2"/>
          <geom type="box" size=".05 .05 .05" mass="1"/>
        </body>
      </body>
    </body>
    <body name="p0" pos="0 1 1">
      <geom type="box" size=".05 .05 .05" mass="2"/>
      <body name="pa" pos="0 0.2 0">
        <joint name="fa" type="slide" axis="1 0 0" damping="0.2"/>
        <geom type="box" size=".04 .04 .04" mass="1"/>
      </body>
      <body name="pb" pos="0 -0.2 0">
        <joint name="fb" type="slide" axis="1 0 0" damping="0.2"/>
        <geom type="box" size=".04 .04 .04" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="chain">
      <joint joint="s1" coef="1"/><joint joint="s2" coef="1"/>
      <joint joint="s3" coef="1"/>
    </fixed>
    <fixed name="siblings">
      <joint joint="fa" coef="1"/><joint joint="fb" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <general name="chain_act" tendon="chain" biastype="affine"
             gainprm="20 0 0" biasprm="0 -20 -10" ctrlrange="-2 2"/>
    <general name="sib_act" tendon="siblings" biastype="affine"
             gainprm="20 0 0" biasprm="0 -20 -10" ctrlrange="-2 2"/>
  </actuator>
  <keyframe>
    <key qpos="0.03 -0.02 0.05  0.04 -0.03" qvel="0.7 -0.5 0.9  0.6 -0.8"/>
  </keyframe>
</mujoco>"""
)


def _ctrl() -> List[Float64]:
    return [0.8, -0.6]


def _mj_qpos() -> List[Float64]:
    """MuJoCo 3.10.0, one `implicitfast` step from keyframe 0."""
    return [
        0.03279616973980463, -0.021991655504413246, 0.053651138734812615,
        0.042241352805534205, -0.0333543428132206,
    ]


# ⚠ THE SAME FILE ON EULER differs by 6.341e-06 — the term is LIVE, and a
# gate that passed with `qDeriv` empty would be measuring nothing.
comptime EULER_GAP = 6.341e-06

comptime STRETCH_CTRL_LEN = 8


def _mj_stretch() -> List[Float64]:
    """MuJoCo `qpos` after one step of hello_robot_stretch at `_st_ctrl()`."""
    return [
        -5.943823218825399e-07, 3.992113756144086e-06,
        -3.265685074028722e-05, 0.9999999999719283,
        -6.409555898583843e-06, 1.642657307083311e-06,
        3.5160523102859783e-06, 0.0022362525819077847,
        -0.0007443011239547746, 3.2452258610017885e-06,
        2.752270125291139e-05, 4.293154059311787e-05,
        9.763565377671382e-05, 9.035236695106334e-05,
        -1.3687883200715192e-05, -6.834839305304523e-05,
        -0.001966948830792444, 0.002365803088012221,
        -0.0022762256964776926, -0.0018402801294967368,
        0.0023655956057304197, 0.0022836970774050068,
        0.0026232522735672734, -0.0008189261626123369,
        -0.02, -0.45, 0.59996076, 1.0, 0.0, 0.0, 0.0,
    ]


def _st_ctrl() -> List[Float64]:
    return [0.1, 0.2, -0.1, 0.4, 0.05, -0.2, 0.15, -0.05]


def _actdamp_record(xml: String, base: String) raises -> List[Float64]:
    """`Model.actdamp_trn`, flattened — the record this gate is really about.
    """
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    # ⚠ `build_model_runtime` ALONE LEAVES THE RECORD EMPTY. It is filled by
    # `build_actuator_damping`, which runs inside `spec_fields_runtime`
    # because it needs the FINAL `kv` — a `<position dampratio>` has none
    # until the mass matrix at qpos0 exists. Reading the record without this
    # call reports `n = 0` for every actuator, which is what a genuinely
    # unfilled record also says.
    var _sf = spec_fields_runtime[DT](fmd, dims, m)
    var out = List[Float64]()
    for i in range(dims.get_nact() * ACTDAMP_TRN_SIZE):
        out.append(Float64(m.actdamp_trn.data[i]))
    return out^


def _step_once(
    xml: String, base: String, ctrl: List[Float64]
) raises -> List[Float64]:
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    if dims.get_nkey() > 0:
        var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
        for i in range(min(nqp, nq)):
            d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    if dims.get_nkey() > 0:
        var nqv = Int(Float64(sf.key_meta.data[KEY_IDX_NQVEL]))
        for i in range(min(nqv, nv)):
            d.qvel.data[i] = sf.key_qvel.data[i]
    var nact = dims.get_nact()
    var act = List[Scalar[DT]](
        length=nact if nact > 0 else 1, fill=Scalar[DT](0)
    )
    for i in range(nv):
        d.qfrc.data[i] = Scalar[DT](0)
    apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
    var use_imp = studio_uses_implicit(fmd)
    assert_true(
        use_imp,
        "this gate is about `qDeriv`, which only an IMPLICIT integrator"
        " reads — the model must ask for one",
    )
    var cone = studio_cone_of(fmd)
    var imp_e = StudioImpFastEll(dims)
    var imp_p = StudioImpFastPyr(dims)
    if cone == ConeType.ELLIPTIC:
        imp_e.step["cpu"](d, m)
    else:
        imp_p.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nq):
        out.append(Float64(d.qpos.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def test_serial_chain_gets_the_block_and_siblings_do_not() raises:
    """The record itself — the discriminator lives here, not in the trajectory.
    """
    print("=== actdamp_trn: a serial chain vs two siblings ===")
    var rec = _actdamp_record(XML, String(""))
    assert_true(
        len(rec) == 2 * ACTDAMP_TRN_SIZE,
        "two actuators expected; the record is "
        + String(len(rec) // ACTDAMP_TRN_SIZE) + " long",
    )
    var W = TENDON_MAX_WRAPS
    # actuator 0 — the CHAIN. Every pair is ancestor-related, so every
    # off-diagonal entry is `kv * 1 * 1` = 10.
    var n0 = Int(rec[ACTDAMP_IDX_N])
    print("  chain_act n =", n0)
    assert_true(
        n0 == 3,
        "chain_act drives 3 dofs; the record says " + String(n0),
    )
    var chain_off = 0
    for p in range(n0):
        for q in range(n0):
            if p == q:
                continue
            var v = rec[ACTDAMP_IDX_PAIR_0 + p * W + q]
            print("    chain pair", p, q, "=", v)
            assert_true(
                abs(v - 10.0) < 1e-12,
                "chain pair (" + String(p) + "," + String(q) + ") is "
                + String(v) + "; MuJoCo's qDeriv has -10 there. A 0 means the"
                " ancestor test rejected a pair that IS on one path.",
            )
            chain_off += 1
    assert_true(chain_off == 6, "a 3-dof chain has 6 ordered off-diagonal"
                " pairs; counted " + String(chain_off))

    # actuator 1 — the SIBLINGS. Same kv, same tendon type, DIFFERENT TREE.
    var ao = ACTDAMP_TRN_SIZE
    var n1 = Int(rec[ao + ACTDAMP_IDX_N])
    print("  sib_act n =", n1)
    assert_true(n1 == 2, "sib_act drives 2 dofs; got " + String(n1))
    for p in range(n1):
        for q in range(n1):
            if p == q:
                continue
            var v = rec[ao + ACTDAMP_IDX_PAIR_0 + p * W + q]
            print("    sibling pair", p, q, "=", v)
            assert_true(
                v == 0.0,
                "sibling pair (" + String(p) + "," + String(q) + ") is "
                + String(v) + " and MuJoCo writes NOTHING there —"
                " `mju_addToSclSparseInc` drops any column `qDeriv`'s sparse"
                " `D` pattern does not have, and two sibling branches are not"
                " coupled in a tree's mass matrix. Writing -10 here is what"
                " took the Menagerie sweep from 74/85 to 72/85.",
            )
    print("  PASS")


def test_one_implicitfast_step_matches_mujoco() raises:
    """Both shapes integrated together, against MuJoCo's own answer."""
    print("=== one implicitfast step on the two-shape fixture ===")
    var want = _mj_qpos()
    var got = _step_once(XML, String(""), _ctrl())
    assert_true(len(got) == 5, "5 dofs expected; got " + String(len(got)))
    for i in range(5):
        print("  qpos", i, " ours", got[i], " mj", want[i])
    var worst = _worst(got, want)
    print("  worst |d(qpos)| =", worst)
    # ⚠ VACUITY. The same file on Euler is 6.341e-06 away, so a build with an
    # empty `qDeriv` cannot pass by accident.
    assert_true(
        worst < 1e-12,
        "the fixture is " + String(worst) + " from MuJoCo. For scale, the"
        " SAME model integrated with Euler is " + String(EULER_GAP)
        + " away — a residual near that size means `qDeriv` is missing the"
        " actuator term rather than getting it slightly wrong.",
    )
    print("  PASS")


def test_hello_robot_stretch_arm_tendon() raises:
    """The real model: four telescoping links on one tendon at kv 10."""
    print("=== hello_robot_stretch, one step ===")
    var src = read_model_source(STRETCH)
    var want = _mj_stretch()
    var got = _step_once(src[0], src[1], _st_ctrl())
    assert_true(
        len(got) == len(want),
        "stretch has " + String(len(want)) + " qpos slots; got "
        + String(len(got)),
    )
    var worst = _worst(got, want)
    var wi = 0
    for i in range(len(want)):
        if abs(got[i] - want[i]) == worst:
            wi = i
    print("  worst |d(qpos)| =", worst, " at", wi)
    print("  ours", got[wi], " mj", want[wi])
    # ⚠ VACUITY. The arm must have EXTENDED — dofs 9..12 are the telescoping
    # links and they are the ones the tendon damps. A model that did not move
    # compares two copies of the keyframe.
    var moved = abs(got[12])
    print("  qpos 12 moved", moved)
    assert_true(
        moved > 1e-6,
        "the arm did not extend; the gate would be comparing the start pose"
        " to itself. One step at this control moves qpos 12 by 4.29e-05 —"
        " that slot is one of the four telescoping links the tendon damps.",
    )
    # ⚠ 1e-9, NOT 1e-12, AND THE MARGIN IS THE POINT. What is left at this
    # control is 7.0e-11 and it is NOT this term: stretch also carries five
    # `<equality joint>` rows and a mesh whose widest face polygon (76)
    # exceeds `MC_MAX_POLYVERT`, both of which the load-time warnings name.
    # The defect this gate exists for is 4.406e-05 — four orders above the
    # bound — so a regression cannot hide under it.
    assert_true(
        worst < 1e-9,
        "hello_robot_stretch is " + String(worst) + " from MuJoCo. Its"
        " sweep figure was 4.406e-05 while `qDeriv` held only the diagonal"
        " of its arm tendon's `kv * moment (x) moment`; anything near that"
        " size is this defect back, not the 7.0e-11 floor.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
