"""`sensors/eval.mojo` vs MuJoCo's `d.sensordata` (AUD-23, AUD-47).

The three evaluation passes — `sensor_pos` / `sensor_vel` / `sensor_acc` —
walk the model's sensor table, dispatch each sensor to its kernel, write the
values at that sensor's own `adr`, and apply `cutoff`. This compares the whole
buffer against MuJoCo's at the same state.

⚠⚠ FREE FLIGHT, NO CONTACTS, AND THAT IS THE POINT. `test_touch_zone_types`
learned it the hard way: at a settled resting contact the two solvers disagree
by ~7% on the normal force, so any gate comparing sensor VALUES through a
contact is measuring the solver, not the sensor. With the body in the air
there is no constraint force to disagree about, and the acceleration-stage
sensors (accelerometer, force, torque) reduce to arithmetic on `cacc` and
`cfrc_int` that both engines should agree on to rounding.

⚠ THE TOUCH SENSOR IS IN THE MODEL AND READS ZERO HERE. That is not a gap in
the fixture — it is the one sensor whose value in free flight is known exactly
in both engines, so it still checks that the slot is written and that an
untouched zone does not leak a stale value. Its zone types have their own gate.

⚠ CUTOFF IS TESTED BY MUTATION, NOT BY HOPE. A cutoff wide enough to be inert
would leave `_apply_cutoff` unexercised while every row still passed, which is
this tree's default failure mode. The second test declares the SAME model with
cutoffs tight enough to bite, and asserts both that MuJoCo clamps and that we
clamp to the same value — including the REAL-vs-POSITIVE distinction that
AUD-47 recorded wrongly.

Run with:
    pixi run mojo run -I . tests/physics3d/test_sensordata_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.sensors import sensor_pos, sensor_vel, sensor_acc

comptime DTYPE = DType.float64

# A free body with a hinged child, in the air. `force`/`torque` are on the
# CHILD's site so they measure a real interaction force rather than zero, and
# the rangefinder looks down at a plane far enough below that nothing touches.
comptime SD_XML = """
<mujoco model="sensordata">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 2.0">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.12 0.1 0.08" density="700"/>
      <site name="imu" pos="0.03 0 0.05" size="0.02"/>
      <site name="down" pos="0 0 -0.08" size="0.01"/>
      <site name="pad" pos="0 0 -0.09" type="box" size="0.1 0.1 0.02"/>
      <body name="link" pos="0.2 0 0">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="gl" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
              density="900"/>
        <site name="wrist" pos="0.25 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <rangefinder name="rf" site="down"/>
    <velocimeter name="vel" site="imu"/>
    <gyro name="gyr" site="imu"/>
    <subtreelinvel name="slv" body="torso"/>
    <accelerometer name="acc" site="imu"/>
    <force name="frc" site="wrist"/>
    <torque name="trq" site="wrist"/>
    <touch name="tch" site="pad"/>
  </sensor>
</mujoco>
"""

comptime sp = parse_xml(SD_XML)
comptime SM = ModelDefFromXML[
    xml=SD_XML,
    nbody=sp.NBODY, njoint=sp.NJOINT, nq=sp.NQ, nv=sp.NV,
    ngeom=sp.NGEOM, nact=sp.NACT, ntex=sp.NTEX, nmat=sp.NMAT,
    nlight=sp.NLIGHT, ncam=sp.NCAM, nsite=sp.NSITE,
    # `parse_xml` does not count sensors — see its constructor note. Eight
    # sensors; 1+3+3+3+3+3+3+1 = 20 values.
    nsensor=8, nsensordata=20,
    max_tendon=sp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=sp.TIMESTEP,
]
# ⚠ A SECOND COMPTIME MODEL, NOT A RUNTIME `.replace()`. `ModelDefFromXML`
# binds its MJCF as a comptime parameter, so a string edited at run time can
# reach MuJoCo but never reaches OUR model — the cutoff test would then compare
# our UNCLAMPED output against MuJoCo's clamped one and fail for the wrong
# reason. The two documents are kept adjacent so a drift between them is
# visible; only three attributes differ.
comptime SD_XML_CUT = """
<mujoco model="sensordata cutoff">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 2.0">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.12 0.1 0.08" density="700"/>
      <site name="imu" pos="0.03 0 0.05" size="0.02"/>
      <site name="down" pos="0 0 -0.08" size="0.01"/>
      <site name="pad" pos="0 0 -0.09" type="box" size="0.1 0.1 0.02"/>
      <body name="link" pos="0.2 0 0">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="gl" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
              density="900"/>
        <site name="wrist" pos="0.25 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <rangefinder name="rf" site="down" cutoff="0.75"/>
    <velocimeter name="vel" site="imu" cutoff="0.3"/>
    <gyro name="gyr" site="imu" cutoff="0.5"/>
    <subtreelinvel name="slv" body="torso"/>
    <accelerometer name="acc" site="imu"/>
    <force name="frc" site="wrist"/>
    <torque name="trq" site="wrist"/>
    <touch name="tch" site="pad" cutoff="5"/>
  </sensor>
</mujoco>
"""

comptime spc = parse_xml(SD_XML_CUT)
comptime SMC = ModelDefFromXML[
    xml=SD_XML_CUT,
    nbody=spc.NBODY, njoint=spc.NJOINT, nq=spc.NQ, nv=spc.NV,
    ngeom=spc.NGEOM, nact=spc.NACT, ntex=spc.NTEX, nmat=spc.NMAT,
    nlight=spc.NLIGHT, ncam=spc.NCAM, nsite=spc.NSITE,
    nsensor=8, nsensordata=20,
    max_tendon=spc.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=spc.TIMESTEP,
]

comptime SMD = ModelDims[SM]
comptime Dat = Data[DTYPE, SMD, 1]
comptime Mod = Model[DTYPE, SMD]
comptime Integ = EulerIntegrator[
    DTYPE, SMD, SM.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
]

# The cutoff model's own aliases. ⚠ SPELLED OUT RATHER THAN MADE GENERIC:
# `ModelDefLike` as a parameter bound does not expose `make_spec_fields` /
# `init_fields` / `reset_data`, and the tree's own note on
# `test_subtree_linvel_vs_mujoco` warns that monomorphizing the init + FK chain
# through a trait pushes compile time into the minutes. Two models means two
# instantiations either way.
comptime SMCD = ModelDims[SMC]
comptime DatC = Data[DTYPE, SMCD, 1]
comptime ModC = Model[DTYPE, SMCD]
comptime IntegC = EulerIntegrator[
    DTYPE, SMCD, SMC.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
]





def _names() -> List[String]:
    return [
        String("rf"), String("vel"), String("gyr"), String("slv"),
        String("acc"), String("frc"), String("trq"), String("tch"),
    ]


def _run_plain(
    mut d: Dat, mut mf: Mod, ctx: DeviceContext
) raises -> Tuple[List[Float64], List[Float64]]:
    """Put the model in a non-trivial airborne state, step once, evaluate.

    The step is what fills `cacc` / `cfrc_int` and the `*_acc` pose snapshot;
    after it, those describe the PRE-integration state, which is the state
    MuJoCo is evaluated at below.
    """
    var sf = SM.make_spec_fields[DTYPE]()
    SM.init_fields[DTYPE](ctx, mf)
    SM.reset_data(sf, d)
    d.qpos.data[2] = Scalar[DTYPE](2.0)
    # A tilted orientation, so the site frames are not axis-aligned and a
    # dropped rotation would show. w first.
    d.qpos.data[3] = Scalar[DTYPE](0.9238795325112867)
    d.qpos.data[4] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[5] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[6] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[7] = Scalar[DTYPE](0.6)  # elbow
    # Linear + angular motion, and an elbow rate, so every velocity-stage
    # sensor has something to report.
    d.qvel.data[0] = Scalar[DTYPE](0.7)
    d.qvel.data[1] = Scalar[DTYPE](-0.4)
    d.qvel.data[2] = Scalar[DTYPE](1.1)
    d.qvel.data[3] = Scalar[DTYPE](0.9)
    d.qvel.data[4] = Scalar[DTYPE](-1.3)
    d.qvel.data[5] = Scalar[DTYPE](0.5)
    d.qvel.data[6] = Scalar[DTYPE](2.0)

    # ⚠⚠ SNAPSHOT BEFORE THE STEP, AND THIS IS THE WHOLE HARNESS.
    # `step` integrates, so afterwards `d.qpos`/`d.qvel` describe the NEXT
    # state — while `cacc`, `cfrc_int` and the `*_acc` pose snapshot describe
    # the PRE-integration one, which is what the sensors were computed from.
    # Handing MuJoCo the post-step state compares two different instants: the
    # first draft did exactly that and the velocimeter came back 0.0098 out
    # (-0.0457 vs -0.0359) while the rangefinder, which depends only on a pose
    # the step does not refresh, matched exactly. That asymmetry is the
    # fingerprint.
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(SM.NQ):
        qpos.append(Float64(d.qpos.data[i]))
    for i in range(SM.NV):
        qvel.append(Float64(d.qvel.data[i]))

    var integ = Integ()
    integ.step["cpu"](d, mf)
    sensor_pos[DTYPE, SMD](d, mf)
    sensor_vel[DTYPE, SMD](d, mf)
    sensor_acc[DTYPE, SMD](d, mf)
    return (qpos^, qvel^)


def _run_cut(
    mut d: DatC, mut mf: ModC, ctx: DeviceContext
) raises -> Tuple[List[Float64], List[Float64]]:
    """`_run_plain` for the cutoff model. Same state, same step, same passes."""
    var sf = SMC.make_spec_fields[DTYPE]()
    SMC.init_fields[DTYPE](ctx, mf)
    SMC.reset_data(sf, d)
    d.qpos.data[2] = Scalar[DTYPE](2.0)
    # A tilted orientation, so the site frames are not axis-aligned and a
    # dropped rotation would show. w first.
    d.qpos.data[3] = Scalar[DTYPE](0.9238795325112867)
    d.qpos.data[4] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[5] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[6] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[7] = Scalar[DTYPE](0.6)  # elbow
    # Linear + angular motion, and an elbow rate, so every velocity-stage
    # sensor has something to report.
    d.qvel.data[0] = Scalar[DTYPE](0.7)
    d.qvel.data[1] = Scalar[DTYPE](-0.4)
    d.qvel.data[2] = Scalar[DTYPE](1.1)
    d.qvel.data[3] = Scalar[DTYPE](0.9)
    d.qvel.data[4] = Scalar[DTYPE](-1.3)
    d.qvel.data[5] = Scalar[DTYPE](0.5)
    d.qvel.data[6] = Scalar[DTYPE](2.0)

    # ⚠⚠ SNAPSHOT BEFORE THE STEP, AND THIS IS THE WHOLE HARNESS.
    # `step` integrates, so afterwards `d.qpos`/`d.qvel` describe the NEXT
    # state — while `cacc`, `cfrc_int` and the `*_acc` pose snapshot describe
    # the PRE-integration one, which is what the sensors were computed from.
    # Handing MuJoCo the post-step state compares two different instants: the
    # first draft did exactly that and the velocimeter came back 0.0098 out
    # (-0.0457 vs -0.0359) while the rangefinder, which depends only on a pose
    # the step does not refresh, matched exactly. That asymmetry is the
    # fingerprint.
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(SMC.NQ):
        qpos.append(Float64(d.qpos.data[i]))
    for i in range(SMC.NV):
        qvel.append(Float64(d.qvel.data[i]))

    var integ = IntegC()
    integ.step["cpu"](d, mf)
    sensor_pos[DTYPE, SMCD](d, mf)
    sensor_vel[DTYPE, SMCD](d, mf)
    sensor_acc[DTYPE, SMCD](d, mf)
    return (qpos^, qvel^)


def _mj_at(
    mujoco: PythonObject, xml: String,
    qpos: List[Float64], qvel: List[Float64],
) raises -> PythonObject:
    """MuJoCo forward-evaluated at the state we just measured ours at.

    Takes plain lists rather than a `Data`, so one helper serves both models —
    `Data[…, SMD]` and `Data[…, SMCD]` are distinct types even though their
    dimensions are equal.
    """
    var m = mujoco.MjModel.from_xml_string(PythonObject(xml))
    var dat = mujoco.MjData(m)
    for i in range(len(qpos)):
        dat.qpos[i] = qpos[i]
    for i in range(len(qvel)):
        dat.qvel[i] = qvel[i]
    mujoco.mj_forward(m, dat)
    return dat^


def _compare(
    xml: String, label: String, tol: Float64,
    ours: List[Float64], qpos: List[Float64], qvel: List[Float64],
) raises -> Int:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(xml))
    var dat = _mj_at(mujoco, xml, qpos, qvel)

    assert_true(
        Int(py=dat.ncon) == 0,
        label + ": the fixture must stay AIRBORNE — MuJoCo reports "
        + String(Int(py=dat.ncon)) + " contacts, so this is measuring the"
        " solver, not the sensors",
    )

    var names = _names()
    var worst = 0.0
    var compared = 0
    var nonzero = 0
    print("  sensor  adr dim   ours[0]              MuJoCo[0]")
    for i in range(len(names)):
        var adr = Int(py=m.sensor_adr[i])
        var dim = Int(py=m.sensor_dim[i])
        for k in range(dim):
            var o = ours[adr + k]
            var t = Float64(py=dat.sensordata[adr + k])
            var diff = abs(o - t)
            if diff > worst:
                worst = diff
            if abs(t) > 1e-12:
                nonzero += 1
            assert_true(
                diff <= tol * (abs(t) + 1.0),
                label + " " + names[i] + "[" + String(k) + "] (adr "
                + String(adr + k) + "): ours " + String(o)
                + " vs MuJoCo " + String(t) + ", |d| " + String(diff),
            )
            compared += 1
        print("  ", names[i], adr, dim, ours[adr],
              Float64(py=dat.sensordata[adr]))

    print("  values compared:", compared, " differing: 0  worst |d| =", worst)
    print("  values MuJoCo reports NONZERO:", nonzero, "/", compared)
    # ⚠ NON-VACUITY. All-zero `sensordata` on both sides would pass every row
    # above while testing nothing at all.
    assert_true(
        nonzero >= 12,
        label + ": only " + String(nonzero) + " of " + String(compared)
        + " values are nonzero on MuJoCo's side — the fixture has stopped"
        " exercising the sensors and the comparison is near-vacuous",
    )
    return compared


def test_sensordata_matches_mujoco() raises:
    print("=== sensordata vs MuJoCo, all three stages, free flight ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var st = _run_plain(d, mf, ctx)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var ours = List[Float64]()
    for i in range(SM.NSENSORDATA):
        ours.append(Float64(d.sensordata.data[i]))

    var n = _compare(String(SD_XML), String("plain"), 1e-9, ours, qpos, qvel)
    assert_true(n == 20, "expected 20 values, compared " + String(n))


def test_cutoff_clamps_like_mujoco() raises:
    """`cutoff` on both datatypes, at values tight enough to bite.

    ⚠ THE MUTATION IS THE TEST. Declaring a cutoff and asserting agreement
    proves nothing if the cutoff never binds — `_apply_cutoff` would return at
    its first line and every row would still match. So the values are chosen
    BELOW the readings the first test prints, and this asserts that MuJoCo's
    own output actually changed before comparing anything.
    """
    print("=== cutoff clamps, and clamps the way MuJoCo does ===")
    var mujoco = Python.import_module("mujoco")

    var ctx = DeviceContext()
    var mf = ModC()
    var d = DatC()
    var st = _run_cut(d, mf, ctx)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var ours = List[Float64]()
    for i in range(SMC.NSENSORDATA):
        ours.append(Float64(d.sensordata.data[i]))

    # Do the cutoffs BIND on MuJoCo's side? Compare its own output with and
    # without them at the same state.
    var dat_plain = _mj_at(mujoco, String(SD_XML), qpos, qvel)
    var dat_cut = _mj_at(mujoco, String(SD_XML_CUT), qpos, qvel)
    var bound = 0
    for k in range(20):
        if abs(
            Float64(py=dat_plain.sensordata[k])
            - Float64(py=dat_cut.sensordata[k])
        ) > 1e-12:
            bound += 1
    print("  values MuJoCo's own cutoffs changed:", bound, "/ 20")
    assert_true(
        bound >= 3,
        "the declared cutoffs do not bind on MuJoCo's side (only "
        + String(bound) + " values moved), so asserting agreement would be"
        " vacuous — lower them",
    )

    var n = _compare(
        String(SD_XML_CUT), String("cutoff"), 1e-9, ours, qpos, qvel
    )
    assert_true(n == 20, "expected 20 values, compared " + String(n))
    print("  our clamped sensordata matches MuJoCo's, all 20 values")


def main() raises:
    var suite = TestSuite()
    suite.test[test_sensordata_matches_mujoco]()
    suite.test[test_cutoff_clamps_like_mujoco]()
    suite^.run()
