"""Acceleration-stage sensors under a CLOSED LOOP, vs MuJoCo (AUD-48).

A `<connect>` equality between two kinematic chains makes a loop, and
`mj_rnePostConstraint` adds that constraint's force into `cfrc_ext`
(engine_core_smooth.c, "cfrc_ext += connect, weld, flex constraints"). The
audit recorded the whole acceleration stage as unusable on such a model. It
is not, and the split is exact:

    accelerometer   `mj_objectAcceleration` — `cacc` and `cvel` ONLY
                    (engine_sensor.c:1273 -> engine_core_util.c:909).
                    `cfrc_ext` never enters it, and the constrained `qacc`
                    it rides on ALREADY carries the equality force, because
                    the solver put it there. EXACT.

    force / torque  `mju_transformSpatial(d->cfrc_int + 6*bodyid, ...)`
                    (engine_sensor.c:1285, :1296), and
                    `cfrc_int = cfrc_body - cfrc_ext`. LOW by the whole
                    loop-closure load until the equality walk lands.

⚠⚠ THE FIXTURE EXISTS TO MAKE THAT DIFFERENCE LARGE. The `<connect>` carries
(47.1, 0, 43.7) N at this state; `cfrc_ext` on the lower link is exactly that
and zero everywhere else. So an accelerometer that silently lost the equality
would not be off by a rounding — it would be off by the constraint — and the
force sensor's correct answer (-45.7, 0, 29.6) is nowhere near the value the
missing-equality reading gives.

⚠ NO CONTACTS, ON PURPOSE. `test_touch_zone_types` measured the two solvers
~7% apart on a settled resting contact; a gate comparing sensor VALUES through
one is measuring the solver. The loop is in the air.

Run with:
    pixi run mojo run -I . tests/physics3d/test_closed_loop_sensors_vs_mujoco.mojo
"""

from std.math import abs, isnan
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data, Model
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.types import ConeType
from noeira.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float64

# A two-link chain and a post, tips tied by a `<connect>`: a four-bar. Both
# sites are off-axis and rotated so a dropped site frame shows up.
comptime CL_XML = """
<mujoco model="closed loop">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="upper" pos="0 0 1.0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" density="900"/>
      <site name="mid" pos="0.15 0.02 0" size="0.02" euler="0 20 0"/>
      <body name="lower" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" density="900"/>
        <site name="tip" pos="0.3 0.01 0" size="0.02" euler="0 -15 0"/>
      </body>
    </body>
    <body name="post" pos="0.8 0 1.0">
      <joint name="j3" type="hinge" axis="0 1 0"/>
      <geom name="g3" type="capsule" fromto="0 0 0 -0.2 0 0" size="0.03" density="900"/>
    </body>
  </worldbody>
  <equality>
    <connect name="c" body1="lower" body2="post" anchor="0.3 0 0"/>
  </equality>
  <sensor>
    <accelerometer name="acc" site="mid"/>
    <force name="frc" site="tip"/>
    <torque name="trq" site="tip"/>
    <accelerometer name="acc2" site="tip"/>
  </sensor>
</mujoco>
"""

comptime cp = parse_xml(CL_XML)
comptime CM = ModelDefFromXML[
    xml=CL_XML,
    nbody=cp.NBODY, njoint=cp.NJOINT, nq=cp.NQ, nv=cp.NV,
    ngeom=cp.NGEOM, nact=cp.NACT, ntex=cp.NTEX, nmat=cp.NMAT,
    nlight=cp.NLIGHT, ncam=cp.NCAM, nsite=cp.NSITE,
    nsensor=4, nsensordata=12,
    max_tendon=cp.NTENDON,
    # ⚠ THREE, NOT ONE. `max_equality` is read as a ROW budget as well as a
    # record count, and a connect is three rows; sized at 1 the model builds
    # and silently stops enforcing the constraint past the first row.
    max_equality=3,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=4,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=cp.TIMESTEP,
]

comptime CMD = ModelDims[CM]
comptime Dat = Data[DTYPE, CMD, 1]
comptime Mod = Model[DTYPE, CMD]
comptime Integ = EulerIntegrator[
    DTYPE, CMD, CM.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
]

# The pose and rates the whole file is evaluated at. Off-axis in every joint
# so no Jacobian column is accidentally zero.
comptime Q0: Float64 = 0.25
comptime Q1: Float64 = -0.4
comptime Q2: Float64 = 0.6
comptime V0: Float64 = 0.7
comptime V1: Float64 = -1.1
comptime V2: Float64 = 0.5


def _run(mut d: Dat, mut mf: Mod, ctx: DeviceContext) raises:
    """Set the state, step once. The passes run inside `step`."""
    var sf = CM.make_spec_fields[DTYPE]()
    CM.init_fields[DTYPE](ctx, mf)
    CM.reset_data(sf, d)
    d.qpos.data[0] = Scalar[DTYPE](Q0)
    d.qpos.data[1] = Scalar[DTYPE](Q1)
    d.qpos.data[2] = Scalar[DTYPE](Q2)
    d.qvel.data[0] = Scalar[DTYPE](V0)
    d.qvel.data[1] = Scalar[DTYPE](V1)
    d.qvel.data[2] = Scalar[DTYPE](V2)
    var integ = Integ()
    integ.step["cpu"](d, mf)


def _mj(mujoco: PythonObject) raises -> PythonObject:
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(CL_XML)))
    var dat = mujoco.MjData(m)
    dat.qpos[0] = Q0
    dat.qpos[1] = Q1
    dat.qpos[2] = Q2
    dat.qvel[0] = V0
    dat.qvel[1] = V1
    dat.qvel[2] = V2
    mujoco.mj_forward(m, dat)
    return dat^


def test_the_loop_actually_carries_a_load() raises:
    """⚠ RUN FIRST. Everything below is about a constraint force; if that
    force is small, "the accelerometer is unaffected by it" is not a claim
    about anything.

    Reads MuJoCo's own `efc_force` and `cfrc_ext` at this state and asserts
    both are large, and that `cfrc_ext` is nonzero on EXACTLY the two bodies
    the `<connect>` names — which is also the shape the equality walk has to
    reproduce when it lands.
    """
    print("=== the closed loop carries a real load ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(CL_XML)))
    var dat = _mj(mujoco)

    assert_true(Int(py=dat.ncon) == 0,
                "the fixture must stay out of contact; MuJoCo reports "
                + String(Int(py=dat.ncon)))
    var ne = Int(py=dat.ne)
    assert_true(ne == 3, "a body connect is three rows; MuJoCo says "
                + String(ne))
    var fmag = 0.0
    for i in range(ne):
        fmag += abs(Float64(py=dat.efc_force[i]))
    print("  |efc_force|1 over the 3 equality rows =", fmag)
    assert_true(
        fmag > 10.0,
        "the connect carries only " + String(fmag) + " — the loop has gone"
        " slack and nothing below is testing the equality",
    )

    var nbody = Int(py=m.nbody)
    var loaded = 0
    for b in range(nbody):
        var s = 0.0
        for k in range(6):
            s += abs(Float64(py=dat.cfrc_ext[b][k]))
        if s > 1e-9:
            loaded += 1
        print("   cfrc_ext body", b, " |.|1 =", s)
    assert_true(
        loaded == 2,
        "exactly the two bodies the <connect> names must carry a cfrc_ext;"
        " " + String(loaded) + " do",
    )


def test_an_accelerometer_under_a_closed_loop_matches_mujoco() raises:
    """The accelerometer is EXACT under a connect, and was being withheld.

    ⚠⚠ THIS IS THE WHOLE POINT OF THE NARROWING. AUD-48 unserved every
    acceleration-stage sensor on a model with a connect or weld equality —
    accelerometer included — on the grounds that `cfrc_ext` is short the
    loop-closure load. `mj_objectAcceleration` does not read `cfrc_ext`
    (engine_core_util.c:909-971: `cacc`, `cvel`, a Coriolis correction, and
    nothing else), so the accelerometer never had that error. cassie's pelvis
    accelerometer was correct all along.

    Two of them: one on the OPEN part of the chain (`mid`, on `upper`) and
    one on the body the constraint acts through (`tip`, on `lower`). If the
    equality force were leaking into `cacc`, the second would diverge and the
    first would not.
    """
    print("=== an accelerometer under a <connect> matches MuJoCo ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(CL_XML)))
    var O = mujoco.mjtObj
    var dat = _mj(mujoco)

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var worst = 0.0
    var compared = 0
    var nonzero = 0
    for nm in [String("acc"), String("acc2")]:
        var sid = Int(py=mujoco.mj_name2id(m, O.mjOBJ_SENSOR,
                                           PythonObject(nm)))
        var adr = Int(py=m.sensor_adr[sid])
        for k in range(3):
            var ours = Float64(d.sensordata.data[adr + k])
            var theirs = Float64(py=dat.sensordata[adr + k])
            assert_true(
                not isnan(ours),
                nm + "[" + String(k) + "] is NaN — the accelerometer is"
                " still being unserved under the connect",
            )
            var dd = abs(ours - theirs)
            if dd > worst:
                worst = dd
            if abs(theirs) > 1e-6:
                nonzero += 1
            compared += 1
        print("  ", nm, " ours", Float64(d.sensordata.data[adr]),
              " MuJoCo", Float64(py=dat.sensordata[adr]))
    print("  values compared:", compared, " worst |d| =", worst,
          "  nonzero on MuJoCo's side:", nonzero, "/", compared)
    assert_true(
        nonzero >= 4,
        "only " + String(nonzero) + " of " + String(compared) + " readings"
        " are nonzero — the fixture has stopped accelerating",
    )
    # ⚠ THE BOUND IS SET FROM THE MEASUREMENT, not inherited. Both engines
    # solve the same three equality rows to their own tolerance and the
    # accelerometer is a second derivative of that solve; this is the residual
    # that leaves, printed above so a regression can be read off the number.
    assert_true(
        worst <= 1e-7,
        "accelerometer under a closed loop is " + String(worst) + " from"
        " MuJoCo",
    )


def test_force_and_torque_under_a_closed_loop_match_mujoco() raises:
    """The other half of AUD-48: the equality force, into `cfrc_ext`.

    A force or torque sensor transforms `cfrc_int`, and
    `cfrc_int = cfrc_body - cfrc_ext`. `mj_rnePostConstraint` puts each
    connect/weld row's `efc_force` into `cfrc_ext`
    (engine_core_smooth.c:2464-2523); our solve computed those forces and
    threw them away with the rest of its scratch. It retains them now
    (`Data.efc_eq_force`), and `_cfrc_ext_env` walks them with MuJoCo's own
    cursor.

    ⚠⚠ THE TEST NAMES THE WRONG ANSWER, BECAUSE IT IS PLAUSIBLE. Without the
    walk the sensor reports `cfrc_body` alone — a wrench of the same order,
    smooth in time, and wrong by the entire loop closure. `cfrc_ext` on this
    fixture's lower link has |.|1 = 104, against a correct force reading of
    (-45.7, 0, 29.6): the gate prints both distances so a regression reads
    off which one it landed on.
    """
    print("=== force/torque under a <connect> vs MuJoCo ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(CL_XML)))
    var O = mujoco.mjtObj
    var dat = _mj(mujoco)

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var worst = 0.0
    var compared = 0
    for nm in [String("frc"), String("trq")]:
        var sid = Int(py=mujoco.mj_name2id(m, O.mjOBJ_SENSOR,
                                           PythonObject(nm)))
        var adr = Int(py=m.sensor_adr[sid])
        var mj_mag = 0.0
        for k in range(3):
            var ours = Float64(d.sensordata.data[adr + k])
            var theirs = Float64(py=dat.sensordata[adr + k])
            assert_true(
                not isnan(ours),
                nm + "[" + String(k) + "] is NaN — the equality walk did not"
                " run. Either the solver did not retain its row forces or its"
                " count disagreed with the equality table; see"
                " META_IDX_EQ_FORCE_LIVE.",
            )
            var dd = abs(ours - theirs)
            if dd > worst:
                worst = dd
            mj_mag += abs(theirs)
            compared += 1
        print("  ", nm, " ours", Float64(d.sensordata.data[adr]),
              " MuJoCo", Float64(py=dat.sensordata[adr]),
              "  MuJoCo |.|1", mj_mag)
        # ⚠ NON-VACUITY, PER ROW. A sensor reading ~0 in MuJoCo too would
        # agree with a walk that did nothing.
        assert_true(
            mj_mag > 1e-3,
            nm + " reads ~0 in MuJoCo, so agreeing with it proves nothing",
        )
    print("  values compared:", compared, " worst |d| =", worst)
    # Bound set from the measurement: both engines solve the same three
    # equality rows to their own tolerance and this is the residual that
    # leaves. Printed above so a regression can be read off the number.
    assert_true(
        worst <= 1e-7,
        "force/torque under a closed loop is " + String(worst) + " from"
        " MuJoCo",
    )

    # ⚠⚠ AND THE VALUE THE OLD BEHAVIOUR PRODUCED, NAMED. Without the
    # equality term `cfrc_ext` on the loaded bodies is zero, so the reading
    # would be the body wrench alone. MuJoCo's own `cfrc_ext` there is the
    # size of that error.
    var cf = 0.0
    for b in range(Int(py=m.nbody)):
        var s2 = 0.0
        for k in range(6):
            s2 += abs(Float64(py=dat.cfrc_ext[b][k]))
        if s2 > cf:
            cf = s2
    print("  the term the walk adds has |.|1 up to", cf,
          "— that is the size of the error it removes")
    assert_true(cf > 10.0, "the equality term is negligible on this fixture")

    # And nothing is withheld any more.
    var fmd = parse_xml_full(String(CL_XML), String("."))
    var n_served = 0
    for i in range(len(fmd.sensors)):
        if fmd.sensors[i].served:
            n_served += 1
    assert_true(
        n_served == 4,
        "all four sensors must be served now; " + String(n_served) + " are",
    )
    var said = False
    for i in range(len(fmd.silent_attr_ids)):
        if fmd.silent_attr_ids[i] == String("AUD-48"):
            said = True
    assert_true(
        not said,
        "AUD-48 is implemented and must not report a row at load any more",
    )
    print("  all four sensors served, no AUD-48 row at load")


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_loop_actually_carries_a_load]()
    suite.test[test_an_accelerometer_under_a_closed_loop_matches_mujoco]()
    suite.test[test_force_and_torque_under_a_closed_loop_match_mujoco]()
    suite^.run()
