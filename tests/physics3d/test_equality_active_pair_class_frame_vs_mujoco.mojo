"""AUD-04 (`<equality active>`), AUD-05 (`<pair class>`), AUD-09 (`<frame>`
orientation and childclass) — parse-level gates against MuJoCo 3.12.

    pixi run mojo run -I . tests/physics3d/test_equality_active_pair_class_frame_vs_mujoco.mojo

Each fixture is built so the OLD behaviour gives a different record:

  * active: the inactive weld used to be enforced; MuJoCo keeps it with
    `eq_active0 = 0` and builds no rows, so `nefc` says which weld is live.
  * pair class: the class states condim / friction / solref / solimp /
    margin / gap that no element repeats; the old reader saw the defaults.
  * frame: `<frame euler childclass>` around a joint, a box with
    `axisangle`, a site with `zaxis`, a body with `euler`, a light, a
    camera with `xyaxes` and a `fromto` capsule. The old fold read `quat`
    alone (identity here) and shadowed every child alternative.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf

comptime ACTIVE_XML = String(
    """<mujoco model="active">
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="free"/>
      <geom name="ga" type="sphere" size="0.1"/>
    </body>
    <body name="b" pos="1 0 1">
      <joint name="jb" type="free"/>
      <geom name="gb" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t"><joint joint="ja" coef="1"/></fixed>
  </tendon>
  <equality>
    <weld name="dead" body1="a" body2="world" active="false"/>
    <weld name="live" body1="b" body2="world"/>
    <tendon name="tdead" tendon1="t" active="false"/>
  </equality>
</mujoco>
"""
)

comptime PAIR_XML = String(
    """<mujoco model="pair_class">
  <default>
    <pair condim="4" margin="0.01"/>
    <default class="p">
      <pair solref="0.005 1" friction="2 2 0.01 0.0002 0.0002"
            solimp="0.8 0.9 0.002 0.4 3" gap="0.02"/>
    </default>
  </default>
  <worldbody>
    <geom name="g0" type="sphere" size="0.1"/>
    <body name="a" pos="0.15 0 0"><geom name="g1" type="sphere" size="0.1"/></body>
    <body name="b" pos="0 0.15 0"><geom name="g2" type="sphere" size="0.1"/></body>
    <body name="c" pos="0 0 0.15"><geom name="g3" type="sphere" size="0.1"/></body>
  </worldbody>
  <contact>
    <pair geom1="g0" geom2="g1" class="p"/>
    <pair geom1="g0" geom2="g2"/>
    <pair geom1="g0" geom2="g3" class="p" condim="6" margin="0.05"/>
  </contact>
</mujoco>
"""
)

comptime FRAME_XML = String(
    """<mujoco model="frame">
  <compiler angle="degree"/>
  <default>
    <default class="c">
      <geom margin="0.03"/>
      <site size="0.02"/>
    </default>
  </default>
  <worldbody>
    <body name="root" pos="0 0 1">
      <joint name="jr" type="free"/>
      <geom name="groot" type="sphere" size="0.05"/>
      <frame pos="0.1 0.2 0.3" euler="0 0 90" childclass="c">
        <geom name="gbox" type="box" size="0.1 0.1 0.1" pos="0.2 0 0" axisangle="0 0 1 45"/>
        <geom name="gcap" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <site name="s" pos="0 0.1 0" zaxis="1 0 0"/>
        <light name="l" pos="0 0 1" dir="1 0 0"/>
        <camera name="cam" pos="1 0 0" xyaxes="0 1 0 0 0 1"/>
        <body name="child" pos="0.3 0 0" euler="90 0 0">
          <joint name="jc" type="hinge" axis="1 0 0" pos="0.1 0 0"/>
          <geom name="gchild" type="sphere" size="0.05"/>
        </body>
      </frame>
    </body>
  </worldbody>
</mujoco>
"""
)


def _mj() raises -> PythonObject:
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    return Python.import_module("mujoco")


def _index(names: List[String], name: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == name:
            return i
    raise Error("name not found: " + name)


def _same_quat(
    wx: Float64, xx: Float64, yx: Float64, zx: Float64,
    mq: PythonObject,
) raises -> Float64:
    """1 - |<q, q_mj>| for q = (w,x,y,z) given ours in (x,y,z,w) order."""
    var d = (
        wx * Float64(py=mq[0]) + xx * Float64(py=mq[1])
        + yx * Float64(py=mq[2]) + zx * Float64(py=mq[3])
    )
    return abs(1.0 - abs(d))


def _v3err(ax: Float64, ay: Float64, az: Float64, v: PythonObject) raises -> Float64:
    var e = abs(ax - Float64(py=v[0]))
    e = max(e, abs(ay - Float64(py=v[1])))
    return max(e, abs(az - Float64(py=v[2])))


def test_inactive_equality_builds_nothing() raises:
    print("=== AUD-04: <equality active=\"false\"> ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(ACTIVE_XML)
    var d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    var neq = Int(py=m.neq)
    var nefc = Int(py=d.nefc)
    print("  MuJoCo neq =", neq, " eq_active0 =", m.eq_active0, " nefc =", nefc)
    assert_true(neq == 3 and nefc == 6, "fixture: 3 equalities, only the live weld's 6 rows")
    var fmd = parse_xml_full(ACTIVE_XML, String("."))
    print("  ours equalities =", len(fmd.equalities), " inactive =", fmd.inactive_equalities,
          " tendon is_equality =", fmd.tendons[0].is_equality)
    assert_true(
        len(fmd.equalities) == 1 and fmd.inactive_equalities == 2
        and fmd.tendons[0].is_equality == 0,
        "the inactive weld and tendon equality must be dropped, the live weld kept",
    )
    assert_true(
        fmd.equalities[0].body_a == _index(fmd.body_names, String("b")),  # MuJoCo body id
        "the kept equality is not the live one (body b)",
    )


def test_pair_class_chain_is_read() raises:
    print("=== AUD-05: <pair class> and <default><pair> ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(PAIR_XML)
    var fmd = parse_xml_full(PAIR_XML, String("."))
    assert_true(Int(py=m.npair) == 3 and len(fmd.pairs) == 3, "three pairs on both sides")
    for i in range(3):
        var pd = fmd.pairs[i]
        var mdim = Int(py=m.pair_dim[i])
        var e = abs(pd.friction - Float64(py=m.pair_friction[i][0]))
        e = max(e, abs(pd.friction_spin - Float64(py=m.pair_friction[i][2])))
        e = max(e, abs(pd.friction_roll - Float64(py=m.pair_friction[i][3])))
        e = max(e, abs(pd.solref_0 - Float64(py=m.pair_solref[i][0])))
        e = max(e, abs(pd.solref_1 - Float64(py=m.pair_solref[i][1])))
        e = max(e, abs(pd.solimp_0 - Float64(py=m.pair_solimp[i][0])))
        e = max(e, abs(pd.solimp_1 - Float64(py=m.pair_solimp[i][1])))
        e = max(e, abs(pd.solimp_2 - Float64(py=m.pair_solimp[i][2])))
        e = max(e, abs(pd.solimp_3 - Float64(py=m.pair_solimp[i][3])))
        e = max(e, abs(pd.solimp_4 - Float64(py=m.pair_solimp[i][4])))
        e = max(e, abs(pd.margin - Float64(py=m.pair_margin[i])))
        e = max(e, abs(pd.gap - Float64(py=m.pair_gap[i])))
        print("  pair", i, " condim ours", pd.condim, " mj", mdim,
              " solref ours", pd.solref_0, " mj", Float64(py=m.pair_solref[i][0]),
              " margin ours", pd.margin, " mj", Float64(py=m.pair_margin[i]),
              " worst |d|", e)
        assert_true(pd.condim == mdim and e < 1e-12, "pair " + String(i) + " differs from MuJoCo")
    # non-vacuity: the class values are not the defaults
    assert_true(
        fmd.pairs[0].condim == 4 and fmd.pairs[0].solref_0 == 0.005
        and fmd.pairs[1].condim == 4 and fmd.pairs[1].solref_0 == 0.02
        and fmd.pairs[2].condim == 6 and fmd.pairs[2].margin == 0.05
        and fmd.pairs[2].gap == 0.02,
        "the fixture no longer separates element, class and root values",
    )


def test_frame_orientation_and_childclass_fold_like_mujoco() raises:
    print("=== AUD-09: <frame euler childclass> fold ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(FRAME_XML)
    var expanded = expand_mjcf(FRAME_XML, String("."))
    assert_true(expanded.find("<frame") == -1, "the frame did not fold away")
    var fmd = parse_xml_full(expanded, String("."))
    var worst = Float64(0)

    # body `child`: pos and quat
    var bi = _index(fmd.body_names, String("child"))
    var mb = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "child"))
    # `body_names[0]` is the worldbody, which has no record: bodies[id - 1].
    var b = fmd.bodies[bi - 1]
    var e = _v3err(b.pos_x, b.pos_y, b.pos_z, m.body_pos[mb])
    var eq = _same_quat(b.quat_w, b.quat_x, b.quat_y, b.quat_z, m.body_quat[mb])
    print("  body child  pos ours", b.pos_x, b.pos_y, b.pos_z, " mj", m.body_pos[mb], " |d|", e, " quat 1-|dot|", eq)
    worst = max(worst, max(e, eq))

    # joint jc (inside child: untouched by the fold, but its body moved)
    var ji = _index(fmd.joint_names, String("jc"))
    var mj = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jc"))
    var j = fmd.joints[ji]
    var an = sqrt(j.axis_x * j.axis_x + j.axis_y * j.axis_y + j.axis_z * j.axis_z)
    e = _v3err(j.axis_x / an, j.axis_y / an, j.axis_z / an, m.jnt_axis[mj])
    e = max(e, _v3err(j.pos_x, j.pos_y, j.pos_z, m.jnt_pos[mj]))
    print("  joint jc    axis/pos |d|", e)
    worst = max(worst, e)

    # box with axisangle
    var gi = _index(fmd.geom_names, String("gbox"))
    var mg = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gbox"))
    var g = fmd.geoms[gi]
    e = _v3err(g.pos_x, g.pos_y, g.pos_z, m.geom_pos[mg])
    eq = _same_quat(g.quat_w, g.quat_x, g.quat_y, g.quat_z, m.geom_quat[mg])
    print("  geom gbox   pos |d|", e, " quat 1-|dot|", eq, " margin ours", g.margin, " mj", Float64(py=m.geom_margin[mg]))
    worst = max(worst, max(e, eq))
    assert_true(abs(g.margin - Float64(py=m.geom_margin[mg])) < 1e-12 and g.margin == 0.03,
                "frame childclass did not reach the box's margin")

    # capsule with fromto: pos and axis direction (roll about the axis is free)
    gi = _index(fmd.geom_names, String("gcap"))
    mg = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gcap"))
    g = fmd.geoms[gi]
    e = _v3err(g.pos_x, g.pos_y, g.pos_z, m.geom_pos[mg])
    # our z axis from (x,y,z,w); MuJoCo's from geom_xmat column 2 after forward
    var d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    var zx = 2.0 * (g.quat_x * g.quat_z + g.quat_w * g.quat_y)
    var zy = 2.0 * (g.quat_y * g.quat_z - g.quat_w * g.quat_x)
    var zz = 1.0 - 2.0 * (g.quat_x * g.quat_x + g.quat_y * g.quat_y)
    # geom is in body `root` at world pos (0,0,1) with identity quat, so the
    # local z axis IS the world z axis of geom_xmat
    var xm = d.geom_xmat[mg]
    var ez = abs(zx - Float64(py=xm[2])) + abs(zy - Float64(py=xm[5])) + abs(zz - Float64(py=xm[8]))
    print("  geom gcap   pos |d|", e, " axis |d|", ez)
    worst = max(worst, max(e, ez))

    # site with zaxis
    var si = _index(fmd.site_names, String("s"))
    var ms = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "s"))
    var st = fmd.sites[si]
    e = _v3err(st.pos_x, st.pos_y, st.pos_z, m.site_pos[ms])
    eq = _same_quat(st.quat_w, st.quat_x, st.quat_y, st.quat_z, m.site_quat[ms])
    print("  site s      pos |d|", e, " quat 1-|dot|", eq, " size ours", st.size_0, " mj", Float64(py=m.site_size[ms][0]))
    worst = max(worst, max(e, eq))
    assert_true(abs(st.size_0 - Float64(py=m.site_size[ms][0])) < 1e-12 and st.size_0 == 0.02,
                "frame childclass did not reach the site's size")

    # light: pos + dir
    var l = fmd.lights[0]
    var ln = sqrt(l.dir_x * l.dir_x + l.dir_y * l.dir_y + l.dir_z * l.dir_z)
    e = _v3err(l.pos_x, l.pos_y, l.pos_z, m.light_pos[0])
    e = max(e, _v3err(l.dir_x / ln, l.dir_y / ln, l.dir_z / ln, m.light_dir[0]))
    print("  light l     pos/dir |d|", e)
    worst = max(worst, e)

    # camera with xyaxes
    var c = fmd.cameras[0]
    e = _v3err(c.pos_x, c.pos_y, c.pos_z, m.cam_pos[0])
    eq = _same_quat(c.quat_w, c.quat_x, c.quat_y, c.quat_z, m.cam_quat[0])
    print("  camera cam  pos |d|", e, " quat 1-|dot|", eq)
    worst = max(worst, max(e, eq))

    print("  worst =", worst)
    assert_true(worst < 1e-9, "the frame fold differs from MuJoCo by " + String(worst))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
