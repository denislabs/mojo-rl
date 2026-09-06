"""`<attach>` across `<compiler angle>` units — the splice converts, vs MuJoCo.

WHY THIS EXISTS
===============
MuJoCo compiles each attached model under ITS OWN `<compiler angle>` and
attaches the compiled result, so a radian asset in a degree scene is
unremarkable there. We splice TEXT, and after the splice the HOST's compiler
reads the sub-model's numbers — a radian foot in a scene that says nothing
(MuJoCo's default is DEGREE) came out with every joint range and euler 57x
too small. The expander used to REFUSE that case, which was honest and cost
`iit_softfoot` a board row and two gates (PERFORMANCE.md §13.32). It now
scales the sub-model's angles into the host's units, using the attribute set
MuJoCo's own compiler scales (`user_objects.cc`, 3.10.0):

    euler (all), axisangle (the angle only),
    joint range (HINGE and BALL), joint ref / springref (HINGE only).

⚠ THE JOINT ATTRIBUTES DEPEND ON THE JOINT'S TYPE, and the type is resolved
through `class=`, the enclosing body's `childclass=`, parent classes and the
top-level `<default>`. A slide joint's range is in METRES. The fixture has
every route to a type, and two TRAPS MuJoCo confirms:

  * `s_tip` is a slide joint under a root `<joint ref="0.1">` written for
    hinges: MuJoCo gives it qpos0 = 0.1 RAW (no conversion for slide), while
    the root default itself must be scaled for the hinges that share it.
  * `h_odd` is an explicit hinge in the SLIDE class: its inherited range is
    scaled, though the class block it came from is not.

WHAT IT GATES. Two scenes: a radian asset in a degree scene (twice, so the
prefix and the conversion compose) and a degree asset in a radian scene. For
every joint: `jnt_range`, `qpos0` (the ref) and `qpos_spring` (springref).
For every body, geom, site and camera: the orientation quaternion, and the
body's inertial quaternion. Ours is `parse_xml_full(expand_mjcf(...))`,
MuJoCo's is `MjModel.from_xml_path` on the SAME scene file — two routes to
one model, which is the only comparison that can see a splice go wrong.

Run: pixi run mojo run -I . tests/physics3d/test_attach_angle_units_vs_mujoco.mojo
"""
from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true

from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import read_model_source
from mojo_rl.physics3d.parser.full_parser import parse_xml_full

comptime TOL: Float64 = 1e-12
comptime DIR = String("tests/physics3d/fixtures/attach_units/")


struct Tally:
    var checks: Int
    var fails: Int
    var worst: Float64

    def __init__(out self):
        self.checks = 0
        self.fails = 0
        self.worst = 0.0

    def near(mut self, got: Float64, want: Float64, msg: String):
        self.checks += 1
        var e = abs(got - want)
        if e > self.worst:
            self.worst = e
        if e > TOL:
            self.fails += 1
            print("    FAIL", msg, ": got", got, "want", want, "|d|", e)

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got != want:
            self.fails += 1
            print("    FAIL", msg, ": got", got, "want", want)


def _index_of(names: List[String], name: String) -> Int:
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def _quat_near(
    mut t: Tally, gx: Float64, gy: Float64, gz: Float64, gw: Float64,
    q: PythonObject, msg: String,
) raises:
    """Same rotation: |<ours, MuJoCo's>| = 1. MuJoCo stores (w, x, y, z)."""
    var ww = Float64(py=q[0])
    var wx = Float64(py=q[1])
    var wy = Float64(py=q[2])
    var wz = Float64(py=q[3])
    var d = gw * ww + gx * wx + gy * wy + gz * wz
    t.near(abs(d), 1.0, msg)


def _check_scene(mut t: Tally, scene: String) raises:
    var mujoco = Python.import_module("mujoco")
    var path = DIR + scene
    print("---", path, "---")
    var m = mujoco.MjModel.from_xml_path(path)
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])

    var njnt = Int(py=m.njnt)
    t.eq(len(fmd.joints), njnt, scene + " njnt")
    t.eq(len(fmd.bodies) + 1, Int(py=m.nbody), scene + " nbody")
    t.eq(len(fmd.geoms), Int(py=m.ngeom), scene + " ngeom")
    t.eq(len(fmd.sites), Int(py=m.nsite), scene + " nsite")
    t.eq(len(fmd.cameras), Int(py=m.ncam), scene + " ncam")

    # ── joints: range, ref (qpos0), springref (qpos_spring) ──────────────
    var compared = 0
    for j in range(njnt):
        var name = String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        var i = _index_of(fmd.joint_names, name)
        t.eq(1 if i >= 0 else 0, 1, "joint " + name + " present")
        if i < 0:
            continue
        var jt = Int(py=m.jnt_type[j])
        t.eq(fmd.joints[i].jnt_type, jt, name + " type")
        t.near(fmd.joints[i].range_min, Float64(py=m.jnt_range[j][0]),
               name + " range[0]")
        t.near(fmd.joints[i].range_max, Float64(py=m.jnt_range[j][1]),
               name + " range[1]")
        if jt == 2 or jt == 3:  # slide, hinge: qpos0 IS the ref
            var a = Int(py=m.jnt_qposadr[j])
            t.near(fmd.joints[i].ref_val, Float64(py=m.qpos0[a]),
                   name + " ref (qpos0)")
            t.near(fmd.joints[i].springref, Float64(py=m.qpos_spring[a]),
                   name + " springref (qpos_spring)")
        compared += 1

    # ── orientations ─────────────────────────────────────────────────────
    for b in range(1, Int(py=m.nbody)):
        var name = String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
        var i = _index_of(fmd.body_names, name) - 1  # body_names[0] = world
        t.eq(1 if i >= 0 else 0, 1, "body " + name + " present")
        if i < 0:
            continue
        var bd = fmd.bodies[i]
        _quat_near(t, bd.quat_x, bd.quat_y, bd.quat_z, bd.quat_w,
                   m.body_quat[b], name + " body_quat")
        # The inertial frame is compared only where the fixture DECLARES one
        # (`<inertial euler>` on the `root` bodies): the flat stage carries
        # a computed frame for the others, and that is another gate's job.
        if name.endswith("root"):
            _quat_near(t, bd.iquat_x, bd.iquat_y, bd.iquat_z, bd.iquat_w,
                       m.body_iquat[b], name + " body_iquat (<inertial euler>)")
        compared += 1
    for g in range(Int(py=m.ngeom)):
        var name = String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        var i = _index_of(fmd.geom_names, name)
        t.eq(1 if i >= 0 else 0, 1, "geom " + name + " present")
        if i < 0:
            continue
        var gd = fmd.geoms[i]
        _quat_near(t, gd.quat_x, gd.quat_y, gd.quat_z, gd.quat_w,
                   m.geom_quat[g], name + " geom_quat")
        compared += 1
    for s in range(Int(py=m.nsite)):
        var name = String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s))
        var i = _index_of(fmd.site_names, name)
        t.eq(1 if i >= 0 else 0, 1, "site " + name + " present")
        if i < 0:
            continue
        var sd = fmd.sites[i]
        _quat_near(t, sd.quat_x, sd.quat_y, sd.quat_z, sd.quat_w,
                   m.site_quat[s], name + " site_quat")
        compared += 1
    # cameras carry no name list on our side; both sides are in document
    # order and the count arm above pins them.
    for c in range(Int(py=m.ncam)):
        if c >= len(fmd.cameras):
            break
        var cd = fmd.cameras[c]
        _quat_near(t, cd.quat_x, cd.quat_y, cd.quat_z, cd.quat_w,
                   m.cam_quat[c], scene + " cam " + String(c) + " quat")
        t.near(cd.fovy, Float64(py=m.cam_fovy[c]),
               scene + " cam " + String(c) + " fovy (NOT angle-converted)")
        compared += 1
    print("    compared", compared, "elements, worst |d| so far", t.worst)


def main() raises:
    print("=== <attach> across <compiler angle> units vs MuJoCo 3.10.0 ===")
    var t = Tally()
    _check_scene(t, String("rad_asset.xml"))  # PLAIN, no attach: the parser alone
    _check_scene(t, String("scene_deg.xml"))  # radian asset, degree scene, x2
    _check_scene(t, String("scene_rad.xml"))  # degree asset, radian scene
    print("checks:", t.checks, "fails:", t.fails, "worst |d|:", t.worst)
    assert_true(t.checks >= 100, "vacuous: fewer than 100 checks ran")
    assert_true(t.fails == 0, String(t.fails) + " checks differ from MuJoCo")
    print("PASS")
