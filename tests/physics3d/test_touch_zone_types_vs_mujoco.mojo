"""`<touch>` over ALL SIX site zone types vs live MuJoCo (AUD-45).

`sensors/touch.mojo` used to test the zone with two private ray routines:
`_ray_hits_box` and `_ray_hits_sphere`. A CAPSULE or CYLINDER zone raised on
the CPU path and returned `TOUCH_UNSUPPORTED_ZONE` (-1.0) on the GPU one,
which an environment's `log1p` turns into `-inf`; an ELLIPSOID zone was
measured as a SPHERE of radius `size[0]`.

Both routines are gone. MuJoCo tests the zone with
`mju_rayGeom(site_xpos, site_xmat, site_size, con->pos, conray, site_type,
NULL) >= 0` (`engine_sensor.c`, `case mjSENS_TOUCH`), and `ray/geom.ray_geom`
IS `mju_rayGeom` — already swept against it over all six types by
`test_ray_geom_vs_mujoco`, which asserts the residual AND the hit/miss split
at zero. So the zone PREDICATE is gated there and is not re-gated here.

⚠⚠ WHAT THIS FILE ADDS IS THE ROUTING, WHICH THAT SWEEP CANNOT SEE. It checks
that a site of each type reaches `ray_geom` with the right pose, size and type
— and lands on MuJoCo's own `sensordata` value. A sweep over `ray_geom` proves
the ray test; it says nothing about whether `touch.mojo` hands it `size` in the
right order, composes the site's world quaternion correctly, or dispatches the
type at all. The capsule and cylinder rows are the ones that used to raise.

⚠ ONE CONTACT, ON PURPOSE. Touch sums normal forces over contacts, so a scene
where the two engines disagree about the contact SET would fail here for a
reason that has nothing to do with zones. A single sphere resting on a plane
gives exactly one contact in both engines, and the gate asserts that before
anything else — otherwise a mismatch would be unattributable.

⚠⚠ AND THE FORCES ARE NOT COMPARED ACROSS ENGINES, DELIBERATELY. At a settled
resting contact the two solvers disagree about the normal force by ~7% (328.7
vs 307.3 N on this fixture) — a stiff near-static contact is exactly where
that gap lives, and it is a SOLVER difference with its own gates. A tolerance
tight enough to catch a zone bug would fail on it, and one loose enough to
pass would catch nothing. So each engine is compared against ITSELF:

  * the HIT/MISS decision per zone must agree with MuJoCo (that IS the zone
    test, and with one contact MuJoCo's reading is `f_normal` or exactly 0);
  * a zone we call hit must read exactly OUR contact's normal force, which is
    what says the sum is over the right contacts and unscaled.

A wrong `size` order, a mis-composed quaternion or a missing type branch all
flip a hit to a miss on at least one of the six shapes here, and that is what
this catches.

Run with:
    pixi run mojo run -I . tests/physics3d/test_touch_zone_types_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
)

comptime DTYPE = DType.float64

# A sphere resting on a plane, with SIX touch sites on the sphere's body —
# one per zone type, every one of them large enough to contain the contact
# point beneath the sphere. They overlap; that is fine and deliberate, since
# each sensor is evaluated independently and the point is to drive all six
# type branches from the SAME contact.
#
# ⚠ THE SIZES ARE NOT ALL THE SAME SHAPE. `size` means different things per
# type — radius for a sphere; radius + half-length for a capsule and cylinder;
# three half-extents for a box and ellipsoid — so passing them through in the
# wrong order is a real bug this fixture can catch, and one a sweep over
# `ray_geom` alone could not.
comptime TOUCH_XML = """
<mujoco model="touch zones">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="ball" pos="0 0 0.2">
      <joint name="jb" type="free"/>
      <geom name="gball" type="sphere" size="0.2"/>
      <site name="z_sphere"    type="sphere"    pos="0 0 -0.18" size="0.12"/>
      <site name="z_box"       type="box"       pos="0 0 -0.18" size="0.1 0.1 0.1"/>
      <site name="z_ellipsoid" type="ellipsoid" pos="0 0 -0.18" size="0.13 0.09 0.11"/>
      <site name="z_capsule"   type="capsule"   pos="0 0 -0.18" size="0.08 0.07"/>
      <site name="z_cylinder"  type="cylinder"  pos="0 0 -0.18" size="0.11 0.06"/>
      <site name="z_far"       type="capsule"   pos="0 0 0.55"  size="0.04 0.03"/>
    </body>
  </worldbody>
  <sensor>
    <touch name="t_sphere"    site="z_sphere"/>
    <touch name="t_box"       site="z_box"/>
    <touch name="t_ellipsoid" site="z_ellipsoid"/>
    <touch name="t_capsule"   site="z_capsule"/>
    <touch name="t_cylinder"  site="z_cylinder"/>
    <touch name="t_far"       site="z_far"/>
  </sensor>
</mujoco>
"""

comptime tp = parse_xml(TOUCH_XML)
comptime TM = ModelDefFromXML[
    xml=TOUCH_XML,
    nbody=tp.NBODY, njoint=tp.NJOINT, nq=tp.NQ, nv=tp.NV,
    ngeom=tp.NGEOM, nact=tp.NACT, ntex=tp.NTEX, nmat=tp.NMAT,
    nlight=tp.NLIGHT, ncam=tp.NCAM, nsite=tp.NSITE,
    # ⚠ SPELLED OUT, NOT `tp.NSENSOR`. `parse_xml` is the lightweight
    # comptime scanner and does not count `<sensor>` — only
    # `tools/gen_model_dims.py`, which reads a real `MjModel`, produces these,
    # and it runs over the SHIPPED assets, not over an inline fixture. Six
    # touch sensors, one value each.
    nsensor=6, nsensordata=6,
    max_tendon=tp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=tp.TIMESTEP,
]
comptime TMD = ModelDims[TM]
comptime Dat = Data[DTYPE, TMD, 1]
comptime Mod = Model[DTYPE, TMD]
comptime Integ = EulerIntegrator[DTYPE, TMD, TM.CONE_TYPE, 1]

# Declaration order of the six sensors, which is also our site order here.
def _zone_names() -> List[String]:
    return [
        String("t_sphere"), String("t_box"), String("t_ellipsoid"),
        String("t_capsule"), String("t_cylinder"), String("t_far"),
    ]


def _mj_sensor(mujoco: PythonObject, m: PythonObject, dat: PythonObject,
               name: String) raises -> Float64:
    var sid = Int(py=mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SENSOR, PythonObject(name)))
    assert_true(sid >= 0, "no such sensor: " + name)
    return Float64(py=dat.sensordata[Int(py=m.sensor_adr[sid])])


def test_all_six_zone_types_match_mujoco() raises:
    print("=== <touch> over all six site zone types vs MuJoCo ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var sf = TM.make_spec_fields[DTYPE]()
    TM.init_fields[DTYPE](ctx, mf)
    TM.reset_data(sf, d)

    # Drop the sphere onto the plane and let it settle so the contact is
    # squarely under the body and well inside every zone.
    var integ = Integ()
    for _ in range(220):
        integ.step["cpu"](d, mf)

    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    print("  our contacts after settling:", ncon)

    # Same state into MuJoCo, then `mj_forward` so its sensors are filled at
    # exactly the state we measured ours at.
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(TOUCH_XML)))
    var dat = mujoco.MjData(m)
    for i in range(TM.NQ):
        dat.qpos[i] = Float64(d.qpos.data[i])
    for i in range(TM.NV):
        dat.qvel[i] = Float64(d.qvel.data[i])
    mujoco.mj_forward(m, dat)
    var mj_ncon = Int(py=dat.ncon)
    print("  MuJoCo contacts at the same state:", mj_ncon)

    # ⚠ ATTRIBUTABILITY, ASSERTED BEFORE ANY FORCE IS COMPARED.
    assert_true(
        ncon == 1 and mj_ncon == 1,
        "this fixture is built so BOTH engines see exactly one contact;"
        " ours " + String(ncon) + ", MuJoCo " + String(mj_ncon)
        + ". With a differing contact set a touch mismatch says nothing"
        " about zone types.",
    )

    # Our one contact's normal force — the value a hit zone must read back.
    var our_fn = Float64(d.contacts.data[0 * CONTACT_SIZE + CONTACT_IDX_FORCE_N])
    print("  our contact normal force:", our_fn)

    var names = _zone_names()
    var loaded = 0
    print("  sensor        ours          MuJoCo        ourHit  mjHit")
    for s in range(len(names)):
        # Site index == sensor index here: the six sites are declared in the
        # same order as the six sensors, all on one body.
        var ours = touch_sphere_site[DTYPE, TMD](d, mf.sites.data, s, 1.0)
        var expect = _mj_sensor(mujoco, m, dat, names[s])
        var our_hit = ours > 0.0
        var mj_hit = expect > 0.0
        print("  ", names[s], ours, expect, our_hit, mj_hit)
        assert_true(
            our_hit == mj_hit,
            names[s] + ": the ZONE TEST disagrees — ours "
            + ("hit" if our_hit else "miss") + ", MuJoCo "
            + ("hit" if mj_hit else "miss") + " (ours " + String(ours)
            + ", MuJoCo " + String(expect) + ")",
        )
        if our_hit:
            # A hit zone sums exactly the one contact, unscaled.
            assert_true(
                abs(ours - our_fn) <= 1e-12 * (our_fn + 1.0),
                names[s] + ": hit, but read " + String(ours)
                + " where our single contact carries " + String(our_fn)
                + " — the sum is over the wrong contacts or is scaled",
            )
            loaded += 1

    print("  zones carrying force:", loaded, "/ 6  (hit/miss agreed with"
          " MuJoCo on all 6)")

    # ⚠ NON-VACUITY, TWO WAYS.
    # If every zone read 0 the loop above would pass while testing nothing, so
    # the five containing zones must all carry force...
    assert_true(
        loaded == 5,
        "expected the five containing zones to carry force and the far one"
        " not to; got " + String(loaded) + " loaded. If this is 0 the"
        " comparison is vacuous; if it is 6 the 'far' zone is not actually"
        " outside the contact and the gate no longer tests exclusion.",
    )
    # ...and the deliberately out-of-reach one must read exactly zero, which
    # is what proves the ray test can still MISS. A zone test that returned
    # "hit" unconditionally would pass every row above.
    var far = touch_sphere_site[DTYPE, TMD](
        d, mf.sites.data, len(names) - 1, 1.0
    )
    assert_true(
        far == 0.0,
        "the 'z_far' capsule zone sits above the ball and must read 0.0, not "
        + String(far) + " — otherwise the zone test hits unconditionally",
    )
    print("  the far capsule zone reads 0.0: the ray test can still miss")


def test_capsule_and_cylinder_zones_no_longer_raise() raises:
    """The AUD-45 bug itself, stated as its own assertion.

    Before this fix `touch_sphere_site` RAISED on a capsule or cylinder site.
    The row above would catch a regression by value, but only while the
    fixture keeps producing a contact; this one fails the moment the type is
    rejected, contact or no contact, and says so in those terms.
    """
    print("=== capsule and cylinder zones are served, not refused ===")
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var sf = TM.make_spec_fields[DTYPE]()
    TM.init_fields[DTYPE](ctx, mf)
    TM.reset_data(sf, d)

    # No stepping: zero contacts, so every zone reads 0.0. What is under test
    # is that the CALL completes for a capsule (site 3) and a cylinder
    # (site 4) — the two types that used to raise.
    var names = _zone_names()
    for s in range(len(names)):
        var v = touch_sphere_site[DTYPE, TMD](d, mf.sites.data, s, 1.0)
        print("  ", names[s], "->", v)
        assert_true(
            v == 0.0,
            names[s] + " should read 0.0 with no contacts, got " + String(v),
        )
    print("  all six zone types callable; none raised")


def main() raises:
    var suite = TestSuite()
    suite.test[test_capsule_and_cylinder_zones_no_longer_raise]()
    suite.test[test_all_six_zone_types_match_mujoco]()
    suite^.run()
