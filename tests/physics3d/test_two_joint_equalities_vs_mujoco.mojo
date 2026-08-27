"""A second `<equality><joint>` must not inherit the first row's Jacobian.

    pixi run mojo run -I . tests/physics3d/test_two_joint_equalities_vs_mujoco.mojo

WHAT WENT WRONG. `build_weld_equality_rows` stages each row in a shared
`J_row` buffer and zeroes it before use. The JOINT branch zeroed it with

    for i in range(V_SIZE):     # V_SIZE == cap[D.NV]()
        J_row[i] = 0

and `cap[D.NV]()` is **0 on a dynamic provider** — the studio's path, and
every runtime-loaded model. So the loop ran ZERO times, `J_row` kept whatever
the previous row left in it, and the second equality's Jacobian was the union
of both rows.

Measured on a two-chain fixture (nv 4, two joint equalities on disjoint dofs),
reading the Newton edge list directly:

    row 0   J = [ 1, -1,  0,  0]      correct
    row 1   J = [ 1, -1,  1, -1]      should be [0, 0, 1, -1]

i.e. a constraint row coupling two chains that share no dof and no body.

⚠⚠ IT IS INVISIBLE WITH ONE EQUALITY, WHICH IS WHY IT SURVIVED. `J_row` is
allocated `fill=0`, so the FIRST row is correct no matter what that loop does.
A model with a single joint equality agrees with MuJoCo to 1e-18 and nothing
is wrong. It takes a SECOND one for the stale entries to appear — and this
tree's dm_control models have at most one, while `agility_cassie`'s four are
CONNECT and `quadruped`'s four are TENDON, both of which go through builders
that were already converted.

⚠ THE FILE ALREADY CARRIED TWO WARNINGS ABOUT THIS EXACT TRAP —
`_weld_jacobian_row` (:171) and the weld branch (:935) both say "`nv`, NOT
`V_SIZE` — the cap is 0 on a dynamic provider". The joint branch was written
the same way and never converted. Fixing a hazard in the places you were
looking is not fixing the hazard; grep the SPELLING.

MEASURED, worst |d(qpos)| against MuJoCo 3.10.0 under a fixed control
sequence fed to both engines (20 steps from each model's keyframe):

    aloha            1.465e-04 -> 5.551e-17
    pal_talos        1.430e-04 -> 3.489e-09
    trossen_wxai     1.328e-04 -> 1.483e-09

15 models in this tree declare two or more joint equalities, `iit_softfoot`
with 45. ⚠ FOUR OF THE FIFTEEN DID NOT MOVE — hello_robot_stretch (7.2e-03),
iit_softfoot (2.8e-03), rainbow_robotics_rby1 (1.4e-01) and toddlerbot
(3.1e-03) each carry a larger, separate divergence that this does not touch.
Reporting the three that moved as "the fix works" without the four that did
not would misdescribe the state of the port.

⚠ NO IN-REPO ENVIRONMENT IS AFFECTED (audited: zero env assets declare two
joint equalities), so this is a Menagerie-port fix and changes no training.
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
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr, StudioIntegEll
from mojo_rl.physics3d.gpu.constants import KEY_META_SIZE, KEY_IDX_NQPOS

comptime DT = DType.float64

comptime ALOHA = String("references/mujoco_menagerie-main/aloha/scene.xml")

# Two INDEPENDENT chains, each with its own joint equality. They share no
# body, no joint and no dof, so a correct solver gives each chain exactly the
# answer it would get alone — which is what makes the stale-Jacobian coupling
# visible as a physical impossibility rather than a tolerance.
comptime _BODIES = String(
    """
  <worldbody>
    <body pos="0 0 1">
      <joint name="a1" type="slide" axis="0 1 0"/>
      <geom type="box" size=".01 .01 .05" mass=".05"/>
      <body pos="0 0 .2">
        <joint name="b1" type="slide" axis="0 1 0"/>
        <geom type="box" size=".01 .01 .05" mass=".05"/>
      </body>
    </body>
    <body pos="1 0 1">
      <joint name="a2" type="slide" axis="0 1 0"/>
      <geom type="box" size=".01 .01 .05" mass=".05"/>
      <body pos="0 0 .2">
        <joint name="b2" type="slide" axis="0 1 0"/>
        <geom type="box" size=".01 .01 .05" mass=".05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="a1" kp="2000" kv="124"/>
    <position joint="b1" kp="2000" kv="124"/>
    <position joint="a2" kp="2000" kv="124"/>
    <position joint="b2" kp="2000" kv="124"/>
  </actuator>
"""
)

comptime XML_TWO_PYR = String(
    '<mujoco><compiler angle="radian"/><option timestep="0.002"/>'
    + _BODIES
    + '<equality>'
    + '<joint joint1="a1" joint2="b1" polycoef="0 1 0 0 0"/>'
    + '<joint joint1="a2" joint2="b2" polycoef="0 1 0 0 0"/>'
    + '</equality></mujoco>'
)
# ⚠ THE SAME FIXTURE UNDER THE OTHER CONE, because the two cones reach
# `build_weld_equality_rows` through DIFFERENT call sites — pyramidal copies
# the rows into the Newton edge list, elliptic runs them as a post-pass. One
# builder, two consumers; a fix that only reached one would pass half of this.
comptime XML_TWO_ELL = String(
    '<mujoco><compiler angle="radian"/>'
    + '<option timestep="0.002" cone="elliptic"/>'
    + _BODIES
    + '<equality>'
    + '<joint joint1="a1" joint2="b1" polycoef="0 1 0 0 0"/>'
    + '<joint joint1="a2" joint2="b2" polycoef="0 1 0 0 0"/>'
    + '</equality></mujoco>'
)
comptime XML_ONE = String(
    '<mujoco><compiler angle="radian"/><option timestep="0.002"/>'
    + _BODIES
    + '<equality>'
    + '<joint joint1="a1" joint2="b1" polycoef="0 1 0 0 0"/>'
    + '</equality></mujoco>'
)

# MuJoCo 3.10.0, 5 steps at ctrl = (0.03, 0.005, 0.02, 0.008).
comptime MJ_TWO_0 = 0.00445238314495672
comptime MJ_TWO_1 = 0.00185453309485829
comptime MJ_TWO_2 = 0.00318731844453924
comptime MJ_TWO_3 = 0.00191028624065524
# The SAME run with only the first equality — chain 1 is bit-identical to the
# two-row case above, which is the invariant the defect broke.
comptime MJ_ONE_0 = 4.4523831449567192e-03
comptime MJ_ONE_1 = 1.8545330948582907e-03


def _run_fixture[
    ELLIPTIC: Bool
](xml: String, nstep: Int) raises -> List[Float64]:
    """Step the fixture with a constant ctrl through the studio's path."""
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions: List[Float64] = [0.03, 0.005, 0.02, 0.008]
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var pyr = StudioIntegPyr(dims)
    var ell = StudioIntegEll(dims)
    for _ in range(nstep):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        comptime if ELLIPTIC:
            ell.step["cpu"](d, m)
        else:
            pyr.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    return out^


def test_two_joint_equalities_do_not_couple() raises:
    """Two rows on disjoint dofs, both cones, against MuJoCo.

    ⚠ EXPECTED VALUES ARE MUJOCO'S OWN `qpos` after 5 `mj_step`s at the same
    constant ctrl — not this engine's output.
    """
    print("=== two joint equalities, disjoint chains ===")
    var want: List[Float64] = [MJ_TWO_0, MJ_TWO_1, MJ_TWO_2, MJ_TWO_3]
    var names: List[String] = ["a1", "b1", "a2", "b2"]

    var got_p = _run_fixture[False](XML_TWO_PYR, 5)
    var got_e = _run_fixture[True](XML_TWO_ELL, 5)
    assert_true(
        len(got_p) == 4 and len(got_e) == 4,
        "the fixture must have four dofs — the gate would be vacuous",
    )
    var worst_p = 0.0
    var worst_e = 0.0
    for i in range(4):
        var ep = abs(got_p[i] - want[i])
        var ee = abs(got_e[i] - want[i])
        if ep > worst_p:
            worst_p = ep
        if ee > worst_e:
            worst_e = ee
        print(
            "  ", names[i], " pyr", got_p[i], " ell", got_e[i],
            " mj", want[i],
        )
    print("  worst |d| pyramidal", worst_p, "  elliptic", worst_e)
    assert_true(
        worst_p < 1e-12,
        "PYRAMIDAL: two joint equalities on disjoint chains must give"
        " MuJoCo's answer; worst |d| = " + String(worst_p)
        + ". A nonzero value here means the second row's Jacobian still"
        " carries the first row's entries — check that `J_row` is zeroed"
        " over `nv` and not over `V_SIZE`, which is 0 on a dynamic provider.",
    )
    assert_true(
        worst_e < 1e-12,
        "ELLIPTIC: same fixture, the OTHER consumer of the same row builder;"
        " worst |d| = " + String(worst_e),
    )
    print("  PASS")


def test_one_equality_alone_is_unchanged() raises:
    """The negative control: one row was never broken, and must stay so.

    ⚠⚠ THIS PASSES BEFORE AND AFTER THE FIX, WHICH IS THE POINT. `J_row` is
    allocated zero-filled, so the first row is correct however the zeroing
    loop behaves. Without this row the file would read as "joint equalities
    were broken", and the actual shape of the defect — invisible until the
    SECOND row — would be lost.

    ⚠ ONLY CHAIN 1 IS COMPARED. With its equality removed, chain 2 is a pair
    of unconstrained stiff servos that runs to ~1e+1 in five steps; asserting
    on a runaway would make this gate about integration, not about the row.
    """
    print("=== one joint equality alone (negative control) ===")
    var got = _run_fixture[False](XML_ONE, 5)
    print("  a1", got[0], " mj", MJ_ONE_0)
    print("  b1", got[1], " mj", MJ_ONE_1)
    assert_true(
        abs(got[0] - MJ_ONE_0) < 1e-12 and abs(got[1] - MJ_ONE_1) < 1e-12,
        "a SINGLE joint equality must match MuJoCo — it always did, and if"
        " this row is red the fix broke the case that was working",
    )
    # ⚠ AND THE INVARIANT THE DEFECT VIOLATED, stated directly: adding a
    # second equality on dofs chain 1 does not touch must not move chain 1.
    var two = _run_fixture[False](XML_TWO_PYR, 5)
    print("  chain 1 with the 2nd equality present:", two[0], two[1])
    assert_true(
        abs(two[0] - got[0]) < 1e-12 and abs(two[1] - got[1]) < 1e-12,
        "adding an equality on a DISJOINT chain moved chain 1 from ("
        + String(got[0]) + ", " + String(got[1]) + ") to ("
        + String(two[0]) + ", " + String(two[1]) + "). No physical coupling"
        " exists between them — this is the stale Jacobian.",
    )
    print("  PASS")


def test_aloha_matches_mujoco() raises:
    """The real model: two arms, each welding its gripper fingers.

    ⚠ EXPECTED VALUES ARE MUJOCO'S `qpos` after 20 `mj_step`s from keyframe 0
    at a constant `ctrl` of 0.5 on all 14 actuators, read off the 3.10.0
    runtime. The two arms are commanded identically and MuJoCo returns
    identical numbers for them, which is a free extra check on the fixture.
    """
    print("=== aloha: two joint equalities, 20 steps ===")
    var src = read_model_source(ALOHA)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    assert_true(
        dims.get_nequality() == 2,
        "aloha declares two joint equalities and this gate needs both — got "
        + String(dims.get_nequality()),
    )

    var nq = dims.get_nq()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.5)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    # aloha says `cone="elliptic"` and no `integrator`, so this is the pair
    # the studio builds for it.
    var integ = StudioIntegEll(dims)
    for _ in range(20):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)

    # MuJoCo's own values. Indices 6/7 and 14/15 are the finger pairs the two
    # equalities tie together.
    var want: List[Float64] = [
        0.10140373024628566, -0.8219601201522581, 1.0913033640194967,
        0.07428721437923008, -0.1430578401141253, 0.14631011576353906,
        0.01744025261523424, 0.0164478656246084,
        0.10140373024628564, -0.8219601201522581, 1.0913033640194967,
        0.07428721437923008, -0.1430578401141253, 0.14631011576353906,
        0.01744025261523424, 0.0164478656246084,
    ]
    assert_true(
        len(want) == nq,
        "the expected vector must cover every qpos slot — nq is " + String(nq),
    )
    var worst = 0.0
    for i in range(nq):
        var e = abs(Float64(d.qpos.data[i]) - want[i])
        if e > worst:
            worst = e
    print("  left fingers ", Float64(d.qpos.data[6]),
          Float64(d.qpos.data[7]))
    print("  right fingers", Float64(d.qpos.data[14]),
          Float64(d.qpos.data[15]))
    print("  worst |d(qpos)| =", worst)
    # ⚠ VACUITY. The fingers must have MOVED off the keyframe, or a frozen
    # model would satisfy the comparison for the wrong reason.
    assert_true(
        abs(Float64(d.qpos.data[6]) - Float64(sf.key_qpos.data[6])) > 1e-4,
        "the gripper did not move — the gate would be comparing a pose"
        " neither engine integrated",
    )
    assert_true(
        worst < 1e-12,
        "aloha must match MuJoCo; worst |d(qpos)| = " + String(worst)
        + ". Before the `V_SIZE` zeroing was fixed this was 1.465e-04, and"
        " the error sat symmetrically on the four finger joints — the dofs"
        " the two equality rows tie.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
