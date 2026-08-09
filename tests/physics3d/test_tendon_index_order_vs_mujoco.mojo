"""Tendon JOINT and SITE references vs MuJoCo — the index twin of element order.

`test_element_order_vs_mujoco` gates that our joint/geom/site ARRAYS are in
MuJoCo's body-grouped order. This file gates the other half: that everything
which REFERS to those arrays by index was updated when they were reordered.

⚠ IT WAS NOT. `full_parser` resolves a tendon's joint and site references with
`_find_joint_index_by_name` / `_find_site_index_by_name`, which counted tags in
raw text order — and `parse_xml_full` calls those fills AFTER `_fill_model` has
already run `_stable_group_by_body_*`. So a text ordinal indexed a permuted
array. The comptime side had the same defect in `_rcd_find_site_index_by_name`,
feeding `sten_sites` for `<spatial>` tendons.

WHY IT SURVIVED THE FIX THAT CREATED IT. Measured across all 19 dm_control
suite XMLs (`tests/dm_control/element_order_probe.py`):

    joints diverge from text order:  dog ONLY
    sites  diverge:                  finger, manipulator, stacker

and the intersection with the models that actually HAVE tendons is empty in
one direction and harmless in the other: ball_in_cup is the only model with a
`<spatial>` tendon and its site order matches, while dog's fixed tendons DID
resolve to the wrong joints but are `limited=False`, `stiffness=0`,
`frictionloss=0` with `neq=0`, so they generate no constraint rows and nothing
ever read the wrong index. Inert for a reason, not because the numbers were
right — which is precisely the state a test is supposed to catch.

THE FIXTURE IS BUILT TO EXPRESS BOTH PERMUTATIONS, because a model that
happened to be in text order would pass this file whether or not the fix works
— the exact shape of dead test that let the original through.

  * `trunk` declares `t_first`, then a child `<body>`, then `t_after`. Text
    order is [t_first, c_hinge, t_after]; MuJoCo's is [t_first, t_after,
    c_hinge]. The fixed tendon wraps `t_first` and `t_after`, so it references
    the one joint whose index MOVES (text 2 -> MuJoCo 1).
  * `s_world` is declared LAST in the text but belongs to the WORLD body, so
    MuJoCo emits it FIRST. Text order is [s_trunk, s_child, s_world]; MuJoCo's
    is [s_world, s_trunk, s_child]. The spatial tendon references `s_world`
    (text 2 -> MuJoCo 0) and `s_child` (text 1 -> MuJoCo 2), so BOTH of its
    waypoints move.

Both tendons are `limited="true"` with a real range: an unlimited tendon on a
spring-free model generates no rows, and then even a completely wrong index
changes no number — that is exactly why dog hid this.

Run with:
    pixi run mojo run -I . tests/physics3d/test_tendon_index_order_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_NUM_SITES,
    TENDON_IDX_SITE_0,
)
from mojo_rl.physics3d.types import ConeType


comptime _RAW = """
<mujoco model="tendon_order">
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"/>
    <body name="trunk" pos="0 0 1">
      <joint name="t_first" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
      <geom name="g_trunk" type="sphere" size=".05"/>
      <site name="s_trunk" pos="0 0 0" size=".01"/>
      <body name="child" pos=".2 0 0">
        <joint name="c_hinge" type="hinge" axis="0 1 0" range="-2 2" limited="true"/>
        <geom name="g_child" type="sphere" size=".04"/>
        <site name="s_child" pos="0 0 0" size=".01"/>
      </body>
      <joint name="t_after" type="hinge" axis="1 0 0" range="-3 3" limited="true"/>
    </body>
    <site name="s_world" pos="0 0 2" size=".01"/>
  </worldbody>
  <tendon>
    <fixed name="fx" limited="true" range="-0.5 0.5">
      <joint joint="t_first" coef="1.0"/>
      <joint joint="t_after" coef="-1.0"/>
    </fixed>
    <spatial name="sp" limited="true" range="0 2">
      <site site="s_world"/>
      <site site="s_child"/>
    </spatial>
  </tendon>
</mujoco>
"""

# ⚠ THROUGH `merge_mjcf`, which is the path every ported model takes.
comptime XML = merge_mjcf(_RAW)
comptime pm = parse_xml(XML)
comptime M = ModelDefFromXML[
    xml=XML,
    nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE,
    max_tendon=pm.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=4,
    max_condim=pm.MAX_CONDIM,
    neq=pm.NEQ,
    nexclude=pm.NEXCLUDE,
    timestep=pm.TIMESTEP,
]


def _build() raises -> Model[
    DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
    M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
]:
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_fixture_permutes_both_joints_and_sites() raises:
    """The fixture must really disagree with text order, or this file is dead.

    Checked against MuJoCo rather than asserted from reading the XML — the same
    discipline `test_element_order_vs_mujoco` uses, and the reason that file's
    fixture is trustworthy.
    """
    print("--- tendon index order: the fixture permutes ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var O = mujoco.mjtObj

    var j1 = String(py=mujoco.mj_id2name(m, O.mjOBJ_JOINT, 1))
    var s0 = String(py=mujoco.mj_id2name(m, O.mjOBJ_SITE, 0))
    var s2 = String(py=mujoco.mj_id2name(m, O.mjOBJ_SITE, 2))
    print("  MuJoCo joint 1 =", j1, " site 0 =", s0, " site 2 =", s2)

    assert_true(
        j1 == "t_after",
        "MuJoCo joint 1 is not `t_after` — the joint interleave is gone from"
        " the fixture, so the fixed tendon no longer references a joint whose"
        " index moves and the joint half of this file gates nothing",
    )
    assert_true(
        s0 == "s_world" and s2 == "s_child",
        "the world-level site is no longer emitted first — the site"
        " permutation is gone from the fixture and the spatial half of this"
        " file gates nothing",
    )

    # And the tendons must be where the index comparisons below assume.
    var t0 = String(py=mujoco.mj_id2name(m, O.mjOBJ_TENDON, 0))
    var t1 = String(py=mujoco.mj_id2name(m, O.mjOBJ_TENDON, 1))
    assert_true(
        t0 == "fx" and t1 == "sp",
        "tendon order is not [fx, sp] — the per-index comparisons below would"
        " be reading the wrong tendon",
    )


def test_fixed_tendon_joint_ids_match_mujoco() raises:
    """`<fixed>` wrap joints — the reference dog resolves wrongly today."""
    print("--- tendon index order: <fixed> joint ids ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var mf = _build()

    var adr = Int(py=m.tendon_adr[0])
    var num = Int(py=m.tendon_num[0])
    var ours_n = Int(mf.tendons.data[0 * MODEL_TENDON_SIZE + TENDON_IDX_NUM_JOINTS])
    print("  wraps: ours", ours_n, " MuJoCo", num)
    assert_true(
        ours_n == num,
        "the fixed tendon has a different number of wraps than MuJoCo's",
    )

    var worst = 0
    for k in range(num):
        var ours = Int(
            mf.tendons.data[0 * MODEL_TENDON_SIZE + TENDON_IDX_JOINT_0 + k]
        )
        var want = Int(py=m.wrap_objid[adr + k])
        print("    wrap", k, ": ours joint", ours, " MuJoCo joint", want)
        if abs(Float64(ours - want)) > Float64(worst):
            worst = ours - want if ours > want else want - ours
    assert_true(
        worst == 0,
        "the fixed tendon references the wrong JOINTS. `_find_joint_index_by"
        "_name` counts `<joint` tags in text order, but `_fill_model` groups"
        " the joint array by body before this lookup runs — so the ordinal"
        " indexes a permuted array.",
    )


def test_spatial_tendon_site_ids_match_mujoco() raises:
    """`<spatial>` waypoints — both of this fixture's move."""
    print("--- tendon index order: <spatial> site ids ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var mf = _build()

    var adr = Int(py=m.tendon_adr[1])
    var num = Int(py=m.tendon_num[1])
    var ours_n = Int(mf.tendons.data[1 * MODEL_TENDON_SIZE + TENDON_IDX_NUM_SITES])
    print("  waypoints: ours", ours_n, " MuJoCo", num)
    assert_true(
        ours_n == num,
        "the spatial tendon has a different number of waypoints than MuJoCo's",
    )

    var worst = 0
    for k in range(num):
        var ours = Int(
            mf.tendons.data[1 * MODEL_TENDON_SIZE + TENDON_IDX_SITE_0 + k]
        )
        var want = Int(py=m.wrap_objid[adr + k])
        print("    waypoint", k, ": ours site", ours, " MuJoCo site", want)
        if ours != want:
            worst = 1
    assert_true(
        worst == 0,
        "the spatial tendon routes through the wrong SITES — the tendon length"
        " and its Jacobian are computed from these positions, so this is a"
        " wrong force, not a wrong label",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
