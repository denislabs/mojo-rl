"""`<magnetometer>` — the last AUD-23 kind any model in this tree declares.

    pixi run mojo run -I . tests/physics3d/test_magnetometer_vs_mujoco.mojo

`mjSENS_MAGNETOMETER` (engine_sensor.c:538) is one line:

    mju_mulMatTVec(sensordata, d->site_xmat + 9*objid, m->opt.magnetic, 3, 3)

the world-frame field rotated INTO the site's frame. Two declarations,
`apptronik_apollo` and `agility_cassie`, and they were the last two AUD-23
lines the 83-model Menagerie sweep printed.

⚠⚠ THE DEFAULT FIELD IS (0, -0.5, 0), NOT ZERO (`engine_init.c:75-77`), AND
THAT IS THE HALF THIS FILE EXISTS FOR. Neither apollo nor cassie sets
`<option magnetic>`, so both take the default — an implementation that
treated an absent attribute as an absent field would return three zeros on
exactly the two models the sensor was built for, and three zeros are a
perfectly plausible magnetometer reading. The second fixture here declares no
`<option magnetic>` at all and requires a real value.

⚠ THE FIELD TRAVELS ON THE SENSOR ROW (`SENSOR_IDX_MAG_*`), not in
`Model.meta` where `<option wind>` lives. `m->opt.magnetic` has exactly one
consumer in the whole reference, and `mmeta` is not among the sensor stage
kernel's 25 buffers — binding it for three floats would spend a 26th against
a Metal argument table that fails with NO DIAGNOSTIC at 29.

⚠⚠ THE SITE ORIENTATION WAS CHOSEN BY SEARCH, NOT BY TASTE. With an
axis-aligned site on an untilted body `R_site` is the identity and the sensor
returns the world field verbatim, so an implementation that rotated NOTHING
would pass; with a nearly-symmetric one a TRANSPOSED rotation passes too. The
first orientation tried (`euler="15 -25 40"`) left the largest component only
0.018 from the raw field — the body tilt very nearly cancelled it, and the
"rotation matters" arm was almost vacuous. `euler="120 0 -40"` came out of a
grid scan maximising the SMALLER of two separations, and it keeps EVERY
component at least 0.45 from the raw field and 0.41 from `R_site . magnetic`
(the transpose). Both wrong answers are named and excluded below.

⚠ AIRBORNE: `ncon == 0` asserted, as in every other sensordata gate.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float64
comptime NSD = 3

# The world field this fixture sets. Off-axis in all three components so a
# dropped or swapped rotation cannot land on the right answer.
comptime MAG_X: Float64 = 0.21
comptime MAG_Y: Float64 = -0.37
comptime MAG_Z: Float64 = 0.44

# MuJoCo's own default, which the second fixture must pick up.
comptime DEF_MAG_Y: Float64 = -0.5

comptime _BODY = """
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1"/>
    <body name="torso" pos="0 0 1.2">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.1 0.08 0.05" density="700"/>
      <site name="imu" pos="0.03 0.02 0.05" euler="120 0 -40" size="0.02"/>
    </body>
  </worldbody>
  <sensor>
    <magnetometer name="mag" site="imu"/>
  </sensor>
</mujoco>
"""

comptime M_XML = (
    '<mujoco model="magnetometer">'
    + '<option timestep="0.002" gravity="0 0 -9.81"'
    + ' magnetic="0.21 -0.37 0.44"/>'
    + _BODY
)

# ⚠ A SECOND COMPTIME MODEL, NOT A RUNTIME EDIT. `ModelDefFromXML` binds its
# MJCF as a comptime parameter, so a string changed at run time reaches
# MuJoCo and never reaches OUR model. The two documents differ in ONE
# attribute and are kept adjacent so a drift between them is visible.
comptime D_XML = (
    '<mujoco model="magnetometer default">'
    + '<option timestep="0.002" gravity="0 0 -9.81"/>'
    + _BODY
)

comptime mp = parse_xml(M_XML)
comptime dp = parse_xml(D_XML)


# ⚠ TYPE ALIASES, NOT THE `def _mk() -> ModelDefFromXML[...]: return {}`
# FACTORY. `ModelDims[M]` takes a `ModelDefLike` PARAMETER; the factory hands
# back a VALUE, and the error it produces is a two-screen dump of the inlined
# XML that says nothing about the cause. Spell the type.
comptime MS = ModelDefFromXML[
    xml=M_XML, nbody=mp.NBODY, njoint=mp.NJOINT, nq=mp.NQ, nv=mp.NV,
    ngeom=mp.NGEOM, nact=mp.NACT, ntex=mp.NTEX, nmat=mp.NMAT,
    nlight=mp.NLIGHT, ncam=mp.NCAM, nsite=mp.NSITE,
    nsensor=1, nsensordata=NSD, max_tendon=mp.NTENDON,
    cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0, timestep=mp.TIMESTEP,
]

comptime MD_ = ModelDefFromXML[
    xml=D_XML, nbody=dp.NBODY, njoint=dp.NJOINT, nq=dp.NQ, nv=dp.NV,
    ngeom=dp.NGEOM, nact=dp.NACT, ntex=dp.NTEX, nmat=dp.NMAT,
    nlight=dp.NLIGHT, ncam=dp.NCAM, nsite=dp.NSITE,
    nsensor=1, nsensordata=NSD, max_tendon=dp.NTENDON,
    cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0, timestep=dp.TIMESTEP,
]


def _qpos() -> List[Float64]:
    """A tilted free-joint pose. ⚠ THE TILT IS PART OF THE TEST — at the
    identity attitude `R_body` drops out and half the composition is
    untested."""
    var q = List[Float64]()
    q.append(0.0)
    q.append(0.0)
    q.append(1.2)
    var w = 0.86
    var x = 0.21
    var y = -0.17
    var z = 0.31
    var n = sqrt(w * w + x * x + y * y + z * z)
    q.append(w / n)
    q.append(x / n)
    q.append(y / n)
    q.append(z / n)
    return q^


# ⚠ TWO CONCRETE RUNNERS, NOT ONE GENERIC ONE. `ModelDims[M]` wants a
# parameter of `ModelDefLike` type and a function generic over
# `ModelDefFromXML` does not satisfy it — the same wall
# `test_velocity_actuator_gpu_parity` works around by spelling `Dims[...]`
# out. Two fifteen-line runners over one shared body is the smaller price,
# and it keeps the two models visibly separate.
def _ours_set() raises -> List[Float64]:
    comptime MDM = ModelDims[MS]
    comptime Integ = EulerIntegrator[
        DTYPE, MDM, MS.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
    ]
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MDM]()
    MS.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MDM, 1]()
    var q = _qpos()
    for i in range(MS.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
        d.qacc.data[i] = Scalar[DTYPE](0)
        d.qacc_warmstart.data[i] = Scalar[DTYPE](0)
    for i in range(MS.NQ):
        d.qpos.data[i] = Scalar[DTYPE](q[i])
    var integ = Integ()
    integ.step["cpu"](d, mf)
    var out = List[Float64]()
    for k in range(NSD):
        out.append(Float64(d.sensordata.data[k]))
    return out^


def _ours_def() raises -> List[Float64]:
    comptime MDM = ModelDims[MD_]
    comptime Integ = EulerIntegrator[
        DTYPE, MDM, MD_.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
    ]
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MDM]()
    MD_.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MDM, 1]()
    var q = _qpos()
    for i in range(MD_.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
        d.qacc.data[i] = Scalar[DTYPE](0)
        d.qacc_warmstart.data[i] = Scalar[DTYPE](0)
    for i in range(MD_.NQ):
        d.qpos.data[i] = Scalar[DTYPE](q[i])
    var integ = Integ()
    integ.step["cpu"](d, mf)
    var out = List[Float64]()
    for k in range(NSD):
        out.append(Float64(d.sensordata.data[k]))
    return out^


def _theirs(xml: String) raises -> Tuple[List[Float64], Int, List[Float64]]:
    """MuJoCo's reading, its contact count, and — third — `R_site . magnetic`,
    the UNTRANSPOSED product, built from MuJoCo's own `site_xmat`.

    That third value is what a transposed implementation returns. Computing
    it here rather than asserting "not equal to something" keeps the wrong
    answer named and numeric.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(xml))
    var dat = mujoco.MjData(m)
    var q = _qpos()
    for i in range(len(q)):
        dat.qpos[i] = q[i]
    mujoco.mj_forward(m, dat)
    var out = List[Float64]()
    for k in range(NSD):
        out.append(Float64(py=dat.sensordata[k]))
    # R . mag, row-major site_xmat.
    var untransposed = List[Float64]()
    for r in range(3):
        var acc = Float64(0)
        for c in range(3):
            acc += (
                Float64(py=dat.site_xmat[0][r * 3 + c])
                * Float64(py=m.opt.magnetic[c])
            )
        untransposed.append(acc)
    return (out^, Int(py=dat.ncon), untransposed^)


def test_the_site_frame_is_not_the_world_frame() raises:
    """⚠ RUN FIRST. If `R_site` is the identity the sensor returns the world
    field verbatim, and an implementation that rotated NOTHING would match
    MuJoCo on every value below.

    The fixture tilts both the body and the site, so MuJoCo's own reading
    must be far from the field it was given.
    """
    print("=== the site frame is genuinely rotated ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var got = _theirs(String(M_XML))
    var raw = List[Float64]()
    raw.append(MAG_X)
    raw.append(MAG_Y)
    raw.append(MAG_Z)
    var moved = 1e9
    for k in range(NSD):
        var dd = abs(got[0][k] - raw[k])
        if dd < moved:
            moved = dd
        print("  axis", k, " world =", raw[k], " site frame =", got[0][k])
    print("  SMALLEST component change under the rotation =", moved)
    assert_true(
        moved > 0.3,
        "MuJoCo's reading differs from the raw world field by only "
        + String(moved) + " in its closest component — this fixture's site is"
        " nearly axis-aligned and cannot tell a rotated reading from an"
        " unrotated one",
    )

    # ...and the TRANSPOSE is separated too.
    var flipped = got[2].copy()
    var tmoved = 1e9
    for k in range(NSD):
        var dd = abs(got[0][k] - flipped[k])
        if dd < tmoved:
            tmoved = dd
        print("  axis", k, " R^T.mag =", got[0][k], " R.mag =", flipped[k])
    print("  SMALLEST component gap to the untransposed product =", tmoved)
    assert_true(
        tmoved > 0.3,
        "`R^T . magnetic` and `R . magnetic` agree to " + String(tmoved)
        + " in their closest component — this fixture cannot tell the"
        " rotation from its transpose",
    )


def test_magnetometer_matches_mujoco() raises:
    """The explicit-field fixture, all three values."""
    print("=== <magnetometer> vs MuJoCo, <option magnetic> set ===")
    var got = _theirs(String(M_XML))
    assert_true(
        got[1] == 0,
        "the fixture is in contact (" + String(got[1]) + " contacts); this"
        " gate must stay airborne",
    )
    var ours = _ours_set()
    var worst = 0.0
    for k in range(NSD):
        var dd = abs(ours[k] - got[0][k])
        if dd > worst:
            worst = dd
        print("  axis", k, " ours =", ours[k], " MuJoCo =", got[0][k])
        assert_true(
            dd <= 1e-13,
            "axis " + String(k) + ": ours " + String(ours[k]) + " vs MuJoCo "
            + String(got[0][k]),
        )
    print("  worst |d| =", worst)

    # ⚠ AND NOT THE TRANSPOSE. The comparison above already excludes it — the
    # two are 0.41 apart in every component — but stating it makes a failure
    # read as "the rotation is the wrong way round" instead of as nine
    # unrelated digits.
    var flipped = got[2].copy()
    var tgap = 1e9
    for k in range(NSD):
        var dd = abs(ours[k] - flipped[k])
        if dd < tgap:
            tgap = dd
    print("  smallest gap to `R . magnetic` (the transposed answer) =", tgap)
    assert_true(
        tgap > 0.3,
        "our reading is within " + String(tgap) + " of `R . magnetic` in some"
        " component — the site rotation is being applied the wrong way round",
    )


def test_an_absent_option_still_has_a_field() raises:
    """⚠⚠ THE ARM THAT NAMES THE DEFECT'S VALUE: three zeros.

    `<option magnetic>` defaults to (0, -0.5, 0), not (0, 0, 0). Neither
    apollo nor cassie — the only two models in this tree with a
    magnetometer — sets the attribute, so a parser that read an absent
    attribute as an absent field would return zeros on both, and zeros are a
    plausible magnetometer reading.

    Same pose, same site, no `<option magnetic>`: the value must be MuJoCo's
    and its magnitude must be the default field's 0.5.
    """
    print("=== an absent <option magnetic> still reads (0, -0.5, 0) ===")
    var got = _theirs(String(D_XML))
    var ours = _ours_def()
    var norm = 0.0
    for k in range(NSD):
        print("  axis", k, " ours =", ours[k], " MuJoCo =", got[0][k])
        assert_true(
            abs(ours[k] - got[0][k]) <= 1e-13,
            "axis " + String(k) + ": ours " + String(ours[k]) + " vs MuJoCo "
            + String(got[0][k]),
        )
        norm += ours[k] * ours[k]
    norm = sqrt(norm)
    print("  |reading| =", norm, " (the default field's magnitude is",
          abs(DEF_MAG_Y), ")")
    assert_true(
        abs(norm - abs(DEF_MAG_Y)) <= 1e-12,
        "the reading's magnitude is " + String(norm) + ", not "
        + String(abs(DEF_MAG_Y)) + ". A rotation preserves length, so this"
        " says the FIELD is wrong — zeros if the default was read as absent",
    )
    assert_true(
        norm > 1e-6,
        "the reading is (0, 0, 0) — the absent `<option magnetic>` was read"
        " as an absent field instead of MuJoCo's (0, -0.5, 0) default",
    )


def test_the_two_fixtures_report_different_fields() raises:
    """The set and default fixtures must not agree, or one of the two tests
    above is reading the other's model."""
    print("=== the two fixtures are two different fields ===")
    var a = _ours_set()
    var b = _ours_def()
    var moved = 0.0
    for k in range(NSD):
        var dd = abs(a[k] - b[k])
        if dd > moved:
            moved = dd
    print("  largest component gap between the fixtures =", moved)
    assert_true(
        moved > 1e-2,
        "the explicit-field and default-field models report the same reading"
        " (gap " + String(moved) + ") — `<option magnetic>` is not reaching"
        " the sensor row",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_site_frame_is_not_the_world_frame]()
    suite.test[test_magnetometer_matches_mujoco]()
    suite.test[test_an_absent_option_still_has_a_field]()
    suite.test[test_the_two_fixtures_report_different_fields]()
    suite^.run()
