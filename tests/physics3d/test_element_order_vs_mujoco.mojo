"""Element ORDER vs MuJoCo — joints, geoms and sites grouped by body.

MuJoCo numbers joints, geoms and sites GROUPED BY BODY: all of body 0's, then
body 1's, with declaration order preserved inside each body. Our runtime parser
walks the `<worldbody>` text and used to emit in TEXT order. The two coincide
only when every body declares its own elements BEFORE its nested `<body>`
children — which every ported model happened to do until dm_control's dog,
whose `skull` declares its 42 teeth AFTER its child bodies.

⚠ THIS WAS A REAL BUG, NOT A NUMBERING PREFERENCE.

  * `fields_build` assigns `qpos_adr`/`dof_adr` as running counters over the
    JOINT ARRAY, so a permuted array permutes the whole `qpos` layout. On dog
    that made `joint_angles` — 73 of the 223 observation dims — a permutation
    of dm_control's.
  * SENSORS ARE ADDRESSED BY SITE INDEX, so a permuted site array reads the
    wrong sensor entirely.
  * It also silently invalidates every per-index model comparison. On dog,
    `max|d(jnt_range)| = 1e10` — our joint at that index was an unlimited one
    where MuJoCo's had a real range — and the armature / stiffness /
    `dof_invweight0` "mismatches" were all this single permutation wearing
    three hats.

THE MODEL BELOW IS BUILT TO EXPRESS THE DEFECT. `trunk` declares a joint, a
geom and a site, then a child `<body>`, then ANOTHER joint, geom and site. In
text order that reads

    t_first, c_hinge, t_after          (joints)

and MuJoCo's order is

    t_first, t_after, c_hinge

so a text-order parser mismatches at indices 1 and 2. `test_element_order_is_a
_discriminating_model` asserts that interleaving is really present, because a
model whose bodies all declare their elements first would pass this file
whether or not the grouping works — the exact shape of dead test that let the
original defect through.

Every element carries a UNIQUE marker — joint range, geom condim, site size —
so a permutation shows up per index rather than averaging out.

Run with:
    pixi run mojo run -I . tests/physics3d/test_element_order_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_CONDIM,
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_SITE_SIZE,
    SITE_IDX_SIZE_0,
)
from mojo_rl.physics3d.types import ConeType


comptime _RAW = """
<mujoco model="order">
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"/>
    <body name="trunk" pos="0 0 1">
      <joint name="t_first" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
      <geom name="g_trunk_first" type="sphere" size=".05" condim="1"/>
      <site name="s_trunk_first" type="sphere" size=".011"/>
      <body name="child" pos=".2 0 0">
        <joint name="c_hinge" type="hinge" axis="0 1 0" range="-2 2" limited="true"/>
        <geom name="g_child" type="sphere" size=".04" condim="4"/>
        <site name="s_child" type="sphere" size=".022"/>
      </body>
      <joint name="t_after" type="hinge" axis="1 0 0" range="-3 3" limited="true"/>
      <geom name="g_trunk_after" type="sphere" size=".03" condim="6"/>
      <site name="s_trunk_after" type="sphere" size=".033"/>
    </body>
  </worldbody>
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

comptime TOL = 1e-12


def _build() raises -> Model[DType.float64, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]:
    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_element_order_is_a_discriminating_model() raises:
    """The fixture must actually interleave, or the file gates nothing.

    Checked against MuJoCo rather than asserted from the text: `t_after` is
    declared AFTER `c_hinge` in the XML, so if MuJoCo's joint 1 is `t_after`
    then body grouping and text order genuinely disagree here.
    """
    print("--- element order: the fixture interleaves ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var O = mujoco.mjtObj

    var j1 = String(py=mujoco.mj_id2name(m, O.mjOBJ_JOINT, 1))
    var g2 = String(py=mujoco.mj_id2name(m, O.mjOBJ_GEOM, 2))
    var s1 = String(py=mujoco.mj_id2name(m, O.mjOBJ_SITE, 1))
    print("  MuJoCo joint 1 =", j1, " geom 2 =", g2, " site 1 =", s1)
    assert_true(
        j1 == "t_after",
        "MuJoCo no longer groups joints by body in this fixture — without the"
        " interleave this whole file passes trivially",
    )
    assert_true(
        g2 == "g_trunk_after" and s1 == "s_trunk_after",
        "the geom/site interleave is gone from the fixture",
    )


def test_joint_order_matches_mujoco() raises:
    """`jnt_range` per index — the marker that caught this on dog."""
    print("--- element order: joints ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var mf = _build()

    var worst = 0.0
    for j in range(M.NJOINT):
        var lo = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            - Float64(py=m.jnt_range[j][0])
        )
        var hi = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
            - Float64(py=m.jnt_range[j][1])
        )
        if lo > worst:
            worst = lo
        if hi > worst:
            worst = hi
    print("  max |d(jnt_range)| =", worst)
    assert_true(
        worst <= TOL,
        "our joint order is not MuJoCo's — qpos_adr is a running counter over"
        " this array, so the whole qpos layout is permuted with it",
    )


def test_geom_order_matches_mujoco() raises:
    """`geom_condim` per index — 1 / 4 / 6 / 3 are all distinct here."""
    print("--- element order: geoms ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var mf = _build()

    var bad = 0
    for g in range(M.NGEOM):
        if Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONDIM]) != Int(
            py=m.geom_condim[g]
        ):
            bad += 1
    print("  geoms out of order:", bad, "/", M.NGEOM)
    assert_true(bad == 0, "our geom order is not MuJoCo's")


def test_site_order_matches_mujoco() raises:
    """`site_size` per index. Sensors are addressed BY SITE INDEX."""
    print("--- element order: sites ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var mf = _build()

    var worst = 0.0
    for si in range(M.NSITE):
        var d = abs(
            Float64(mf.sites.data[si * MODEL_SITE_SIZE + SITE_IDX_SIZE_0])
            - Float64(py=m.site_size[si][0])
        )
        if d > worst:
            worst = d
    print("  max |d(site_size)| =", worst)
    assert_true(
        worst <= TOL,
        "our site order is not MuJoCo's — every sensor is addressed by site"
        " index, so this reads the wrong sensor",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
