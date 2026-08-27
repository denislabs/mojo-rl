"""`sensors/rangefinder.mojo` vs MuJoCo's own `sensordata`.

    pixi run mojo run -I . tests/physics3d/test_rangefinder_vs_mujoco.mojo

⚠⚠ THIS COMPARES AGAINST `d.sensordata`, NOT AGAINST `mj_ray`. `mj_ray` is
already gated by `test_ray_model_vs_mujoco`; what is untested until here is the
SENSOR — which direction the ray leaves the site along, which body it excludes,
and what a miss reports. Calling `mj_ray` on both sides with a direction this
file computed would prove none of those, because the thing under test would be
supplying the reference's inputs.

THE THREE THINGS THAT CAN BE WRONG, and the fixture is built so each shows:

  · **direction.** A rangefinder fires along the site's own **+Z**; a CAMERA
    looks down **-Z**. The two conventions are opposite and sit in the same
    reference file. `down`/`up` are two sites at the same point with opposite
    orientations and DIFFERENT things in front of them, so swapping the sign
    swaps their two readings rather than moving both a little.
  · **body exclusion.** `self_ray` sits on a body that also carries a geom
    straight in front of it. MuJoCo excludes the site's own body, so the
    reading is the FLOOR behind it; a sensor that forgot would read its own
    shell at a few centimetres. `test_the_self_geom_is_in_the_way` proves the
    occluder is really there.
  · **a miss.** `into_the_void` points at nothing. MuJoCo reports **-1**, and
    a caller that substituted a large number or a cutoff would change the
    observation exactly where the robot is in the open.

⚠ The tilted rig is deliberate. A rangefinder on an unrotated body makes
`site_world_quat` one factor instead of two, and a composition-order bug reads
as exact — the blindfold that hid `mj_camlight` for the whole port.

⚠⚠ THE AIM DIRECTIONS ARE COMPUTED, NOT TUNED. `zaxis` on each site is
`R_rig^T * (target - rig_pos)`, worked out numerically and pasted in. The first
draft guessed euler angles and produced a sensor named `at_the_mesh` that
reported -1 and a `void` ray that quietly hit the infinite floor at 18.9 m —
so the mesh and heightfield branches were never reached through the sensor at
all, `n_miss` came from the wrong sensor, and every assertion still passed. The
readings are printed per sensor so that stays visible.

WHAT THIS GATE WAS PROVEN ABLE TO FAIL
======================================
    injected defect                                   worst |d|
    -----------------------------------------------   ---------
    fires along -Z (the CAMERA convention)             9.3e+07
    the site's own body not excluded                   1.06
    body quaternion used instead of the composed one   1.78
    site quat composed against body 0                  4.72
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from mojo_rl.physics3d.fields import Data, Model, DynDims, init_hfield_data
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.sensors.rangefinder import rangefinder_site

comptime DT = DType.float64

comptime SCENE = String(
    """
<mujoco model="rangefinder gate">
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.4 0.4 0.15 0.05"/>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -1.0"/>
    <geom name="ceiling_box" type="box" size="0.6 0.6 0.02" pos="0 0 0.9"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0.5 0.4 -0.7" euler="8 -6 20"/>
    <geom name="a_mesh" type="mesh" mesh="notch" pos="-0.5 0.3 -0.2" euler="15 -25 40"/>
    <geom name="a_capsule" type="capsule" size="0.05 0.12" pos="0.45 -0.35 -0.1" euler="55 10 -20"/>

    <!-- A tilted carrier body, so no site frame is the identity. -->
    <body name="rig" pos="0.05 -0.02 0.15" euler="12 -18 25">
      <freejoint/>
      <geom name="rig_shell" type="sphere" size="0.03" pos="0 0 -0.12"/>
      <!-- ⚠ `zaxis` RATHER THAN `euler`, and the numbers are COMPUTED, not
           tuned: a rangefinder fires along +Z, so `zaxis` says where each ray
           goes and the fixture documents itself. They are BODY-frame
           directions (`R_rig^T * (target - rig_pos)`), because the rig is
           deliberately tilted — see `test_no_site_frame_is_the_identity`.
           The first draft guessed euler angles and produced a sensor named
           `at_the_mesh` that reported -1 and a `void` ray that hit the
           infinite floor at 18.9 m: the names were wrong and the mesh and
           heightfield branches were never reached through the sensor at all,
           while every assertion passed. -->
      <site name="at_the_ceiling" pos="0 0 0" zaxis="0.313013 0.111562 0.943174"/>
      <site name="at_the_floor"   pos="0 0 0" zaxis="-0.313013 -0.111562 -0.943174"/>
      <site name="self_ray"       pos="0 0 -0.06" zaxis="-0.313013 -0.111562 -0.943174"/>
      <!-- ⚠ HORIZONTAL, world +x. Verified to hit nothing: `ray_plane` is
           ONE-SIDED and the floor is below, the ceiling box is above, and
           nothing else lies along that line. A "void" ray that quietly hits
           the infinite floor at 18.9 m — which the first draft did — makes
           `n_miss` zero and stops the -1 contract being tested at all. -->
      <site name="into_the_void"  pos="0 0 0" zaxis="0.861950 -0.401934 -0.309017"/>
      <site name="at_the_mesh"    pos="0 0 0" zaxis="-0.670672 0.677742 -0.301439"/>
      <site name="at_the_terrain" pos="0 0 0" zaxis="0.218683 0.144147 -0.965090"/>
      <site name="at_the_capsule" pos="0 0 0" zaxis="0.238201 -0.829380 -0.505360"/>
    </body>
  </worldbody>
  <sensor>
    <rangefinder name="rf_ceiling" site="at_the_ceiling"/>
    <rangefinder name="rf_floor"   site="at_the_floor"/>
    <rangefinder name="rf_self"    site="self_ray"/>
    <rangefinder name="rf_void"    site="into_the_void"/>
    <rangefinder name="rf_mesh"    site="at_the_mesh"/>
    <rangefinder name="rf_terrain" site="at_the_terrain"/>
    <rangefinder name="rf_capsule" site="at_the_capsule"/>
  </sensor>
</mujoco>
"""
)

comptime NSENS = 7


def _names() -> List[String]:
    var v = List[String]()
    v.append(String("at_the_ceiling"))
    v.append(String("at_the_floor"))
    v.append(String("self_ray"))
    v.append(String("into_the_void"))
    v.append(String("at_the_mesh"))
    v.append(String("at_the_terrain"))
    v.append(String("at_the_capsule"))
    return v^


struct Built(Movable):
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self) raises:
        var fmd = parse_xml_full(SCENE, String("."))
        var dims = dims_from_flat(
            fmd, max_contacts=32, nmesh_verts=256, nmesh_tri=64
        )
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var sf = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        init_hfield_data(d, m)
        for i in range(dims.get_nq()):
            d.qpos.data[i] = sf.qpos0.data[i]
        for i in range(dims.get_nv()):
            d.qvel.data[i] = Scalar[DT](0)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        self.m = m^
        self.d = d^
        self.dims = dims


def _mj() raises -> Tuple[PythonObject, PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(SCENE))
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)
    return (m, d)


def test_the_frame_composition_is_not_trivial() raises:
    """The precondition: `site_world_quat` must do real work here.

    ⚠⚠ THE FIRST VERSION OF THIS GUARD MEASURED THE WRONG THING. It asserted
    that no site's WORLD +Z lined up with world +Z — which says nothing about
    the composition, and once the sites were aimed properly it read 0.9974
    against a 0.999 threshold, i.e. it was one rounding away from failing
    while testing nothing. `at_the_ceiling` legitimately points nearly
    straight up in WORLD terms; what matters is that it gets there through a
    tilted body and a non-identity local quat.

    So this asserts the two operands instead: the rig's world orientation is
    not the identity, and no site's LOCAL orientation is either. If either
    collapses, `xquat[body] * site_quat` reduces to one factor and a
    composition-order bug reads as exact — the blindfold that hid
    `mj_camlight` for the entire port.
    [[feedback_the_identity_commutes_so_the_gate_is_blind]]
    """
    var mujoco = Python.import_module("mujoco")
    var r = _mj()
    var m = r[0]
    var d = r[1]
    var bid = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, String("rig")))
    # `xquat` is (w, x, y, z); |w| == 1 is the identity rotation.
    var body_turn = 1.0 - abs(Float64(py=d.xquat[bid][0]))
    print("  rig body: 1-|w| =", body_turn)
    assert_true(
        body_turn > 1e-3,
        "the carrier body is unrotated — the site composition is one factor",
    )

    var names = _names()
    var least_site_turn = 1.0
    for i in range(len(names)):
        var sid = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, names[i])
        )
        var w = abs(Float64(py=m.site_quat[sid][0]))
        least_site_turn = min(least_site_turn, 1.0 - w)
    print("  least site local 1-|w| =", least_site_turn)
    assert_true(
        least_site_turn > 1e-3,
        "some site's LOCAL orientation is the identity, so its ray direction"
        " is the body's +Z and a site-quat bug would not show on it",
    )


def test_the_self_geom_is_in_the_way() raises:
    """`rig_shell` must occlude `self_ray`, or body exclusion is untested."""
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var r = _mj()
    var m = r[0]
    var d = r[1]
    var sid = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, String("self_ray"))
    )
    var pnt = np.zeros(3)
    var vec = np.zeros(3)
    for k in range(3):
        pnt[k] = d.site_xpos[sid][k]
        vec[k] = d.site_xmat[sid][3 * k + 2]
    var gid = np.zeros(1, np.int32)
    # bodyexclude = -1: nothing excluded, so the shell should win.
    var t = Float64(py=mujoco.mj_ray(m, d, pnt, vec, None, True, -1, gid, None))
    var name = String(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, Int(py=gid[0]))
    )
    print("  with nothing excluded, self_ray hits:", name, "at", t)
    assert_true(
        name == "rig_shell",
        "self_ray hits " + name + ", not its own shell — body exclusion is"
        " not exercised by this fixture",
    )


def test_rangefinder_vs_mujoco_sensordata() raises:
    var mujoco = Python.import_module("mujoco")
    var r = _mj()
    var m = r[0]
    var d = r[1]
    var b = Built()


    var names = _names()
    var worst = 0.0
    var n_hit = 0
    var n_miss = 0
    print("    sensor            ours            MuJoCo          |d|")
    for i in range(len(names)):
        var sid = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, names[i])
        )
        var ours = rangefinder_site[DT, DynDims, 1](b.d, b.m, sid)
        # ⚠ MuJoCo's OWN sensor output, not a second `mj_ray` call.
        var theirs = Float64(py=d.sensordata[i])
        var e = abs(ours - theirs)
        worst = max(worst, e)
        if theirs < 0:
            n_miss += 1
        else:
            n_hit += 1
        print("    " + names[i], "  ", ours, "  ", theirs, "  ", e)

    print("  hits", n_hit, " misses", n_miss, " worst |d|", worst)
    assert_true(
        n_hit >= 4,
        "only " + String(n_hit) + " of " + String(NSENS) + " sensors hit"
        " anything — the fixture stopped pointing at the scene",
    )
    assert_true(
        n_miss >= 1,
        "no sensor MISSES, so -1 is never exercised and a caller that"
        " substituted a cutoff would pass this gate",
    )
    # The heightfield's float32 grid is the loosest thing a ray here touches.
    assert_true(worst < 1e-6, "worst |d| " + String(worst))


def test_site_index_matches_sensor_order() raises:
    """The sensors are declared in the same order `_names()` lists them.

    ⚠ `d.sensordata[i]` is indexed by SENSOR, and `rangefinder_site` by SITE.
    They coincide here only because the `<sensor>` block is written in site
    order; asserting it keeps a reordered fixture from silently comparing
    different rays.
    """
    var mujoco = Python.import_module("mujoco")
    var r = _mj()
    var m = r[0]
    var names = _names()
    for i in range(len(names)):
        var objid = Int(py=m.sensor_objid[i])
        var sid = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, names[i])
        )
        assert_true(
            objid == sid,
            "sensor " + String(i) + " reads site " + String(objid)
            + ", expected " + String(sid),
        )
    print("  sensor order matches site order for all", len(names))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
