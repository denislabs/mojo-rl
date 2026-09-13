"""`sensors/eval.mojo` vs MuJoCo's `d.sensordata` (AUD-23, AUD-47).

Ten sensors, twenty-two values, all three stages. `jointpos`/`jointvel` joined
on 2026-09-13 (audit §6 phase 1a) and sit BEHIND A FREEJOINT deliberately —
see `test_the_joint_sensors_read_their_own_joints_address` for why a model of
hinges alone cannot tell the two address tables apart.

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

from std.math import abs, isnan
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    MODEL_SENSOR_SIZE,
    SENSOR_IDX_ADR,
    SENSOR_IDX_DIM,
    SENSOR_IDX_TYPE,
)
from mojo_rl.physics3d.constants import (
    SENS_ACCELEROMETER,
    SENS_FORCE,
    SENS_TORQUE,
)

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
    <jointpos name="jp" joint="el"/>
    <velocimeter name="vel" site="imu"/>
    <gyro name="gyr" site="imu"/>
    <jointvel name="jv" joint="el"/>
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
    # `parse_xml` does not count sensors — see its constructor note. Ten
    # sensors; 1+1+3+3+1+3+3+3+3+1 = 22 values.
    nsensor=10, nsensordata=22,
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
    <jointpos name="jp" joint="el" cutoff="0.4"/>
    <velocimeter name="vel" site="imu" cutoff="0.3"/>
    <gyro name="gyr" site="imu" cutoff="0.5"/>
    <jointvel name="jv" joint="el" cutoff="1.2"/>
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
    nsensor=10, nsensordata=22,
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
        String("rf"), String("jp"), String("vel"), String("gyr"),
        String("jv"), String("slv"),
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
    # ⚠⚠ NO EXPLICIT `sensor_*` CALLS. `EulerIntegrator.step` runs the three
    # passes itself, at MuJoCo's own stage points, and this gate exists to
    # prove that. The first version called them here as well — which is
    # idempotent and therefore would have passed even with the hook removed,
    # testing the passes while claiming to test the wiring.
    integ.step["cpu"](d, mf)
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
    # ⚠⚠ NO EXPLICIT `sensor_*` CALLS. `EulerIntegrator.step` runs the three
    # passes itself, at MuJoCo's own stage points, and this gate exists to
    # prove that. The first version called them here as well — which is
    # idempotent and therefore would have passed even with the hook removed,
    # testing the passes while claiming to test the wiring.
    integ.step["cpu"](d, mf)
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
    assert_true(n == 22, "expected 22 values, compared " + String(n))


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
    for k in range(22):
        if abs(
            Float64(py=dat_plain.sensordata[k])
            - Float64(py=dat_cut.sensordata[k])
        ) > 1e-12:
            bound += 1
    print("  values MuJoCo's own cutoffs changed:", bound, "/ 22")
    assert_true(
        bound >= 3,
        "the declared cutoffs do not bind on MuJoCo's side (only "
        + String(bound) + " values moved), so asserting agreement would be"
        " vacuous — lower them",
    )

    var n = _compare(
        String(SD_XML_CUT), String("cutoff"), 1e-9, ours, qpos, qvel
    )
    assert_true(n == 22, "expected 22 values, compared " + String(n))
    print("  our clamped sensordata matches MuJoCo's, all 22 values")


def test_a_stage_that_never_runs_is_loud() raises:
    """An acceleration-stage sensor on an integrator without `RNE_POST`.

    ⚠⚠ THE FAILURE THIS PREVENTS HAS NO FINGERPRINT IF THE BUFFER IS ZEROED.
    A model whose acceleration stage never runs would report 0.0 for its
    accelerometer, its force and torque sensors and its touch pads — every one
    of which is legitimately 0.0 in free flight. There is no value an observer
    could inspect to tell the two apart. `Data` therefore fills `sensordata`
    with NaN, and the passes overwrite only what they serve, so what survives
    to a reader is exactly what was never computed.

    ⚠ IT USED TO RAISE FROM `step`, AND THAT WAS THE WRONG PLACE. Jaco
    declares force/torque sensors and the seven manipulation envs run
    `RNE_POST=False`; the raise landed inside `custom_reset_full_cpu`, whose
    failure `Phyics3dEnv` prints rather than propagates, so their reset
    silently aborted before the TCP initializer and left the arm at qpos0.
    Loud when READ costs nothing when it is not read.
    """
    print("=== an uncomputed acceleration-stage sensor reads NaN ===")
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var sf = SM.make_spec_fields[DTYPE]()
    SM.init_fields[DTYPE](ctx, mf)
    SM.reset_data(sf, d)

    # The same model on an integrator WITHOUT the post-constraint RNE. Our
    # fixture declares accelerometer, force and torque, which read
    # `cacc`/`cfrc_int` and so cannot be computed here.
    comptime IntegNoRne = EulerIntegrator[
        DTYPE, SMD, SM.CONE_TYPE, 1, SOLVER="newton", RNE_POST=False
    ]
    var integ = IntegNoRne()
    # ⚠ IT MUST STEP. The whole point of the change is that it no longer
    # refuses.
    integ.step["cpu"](d, mf)
    print("  RNE_POST=False steps cleanly")

    # Every sensor that needs the post-constraint RNE must read back NaN, and
    # every sensor that does not must read back a real number.
    var n_nan = 0
    var n_real = 0
    for i in range(SMD.NSENSOR):
        var o = i * MODEL_SENSOR_SIZE
        var st = Int(mf.sensors.data[o + SENSOR_IDX_TYPE])
        var adr = Int(mf.sensors.data[o + SENSOR_IDX_ADR])
        var dim = Int(mf.sensors.data[o + SENSOR_IDX_DIM])
        var needs_rne = (
            st == SENS_ACCELEROMETER or st == SENS_FORCE or st == SENS_TORQUE
        )
        for k in range(dim):
            var v = Float64(d.sensordata.data[adr + k])
            if needs_rne:
                assert_true(
                    isnan(v),
                    "sensor " + String(i) + " (type " + String(st) + ") needs"
                    " the post-constraint RNE, which this integrator does not"
                    " run — its slot must read NaN, not " + String(v),
                )
                n_nan += 1
            else:
                assert_true(
                    not isnan(v),
                    "sensor " + String(i) + " (type " + String(st) + ") does"
                    " NOT need the post-constraint RNE and must have been"
                    " computed, but its slot is NaN",
                )
                n_real += 1
    print("  slots NaN (uncomputed):", n_nan,
          "  slots computed:", n_real)

    # ⚠ NON-VACUITY, BOTH WAYS. If every slot were NaN the first assertion
    # would pass while the engine computed nothing; if none were, the test
    # would be checking an empty set.
    assert_true(
        n_nan > 0,
        "no slot came back NaN — the fixture no longer declares a sensor that"
        " needs the post-constraint RNE, so this test proves nothing",
    )
    assert_true(
        n_real > 0,
        "every slot came back NaN — the passes computed nothing at all, so the"
        " NaN above is not evidence about RNE_POST",
    )

    # ⚠ AND THE SAME MODEL ON `RNE_POST=True` MUST FILL THOSE SAME SLOTS.
    # Without this the NaN could be a sensor we never serve on any integrator.
    var mf2 = Mod()
    var d2 = Dat()
    var sf2 = SM.make_spec_fields[DTYPE]()
    SM.init_fields[DTYPE](ctx, mf2)
    SM.reset_data(sf2, d2)
    var integ_ok = Integ()
    integ_ok.step["cpu"](d2, mf2)
    var n_filled = 0
    for i in range(SMD.NSENSOR):
        var o = i * MODEL_SENSOR_SIZE
        var st = Int(mf2.sensors.data[o + SENSOR_IDX_TYPE])
        if not (
            st == SENS_ACCELEROMETER or st == SENS_FORCE or st == SENS_TORQUE
        ):
            continue
        var adr = Int(mf2.sensors.data[o + SENSOR_IDX_ADR])
        var dim = Int(mf2.sensors.data[o + SENSOR_IDX_DIM])
        for k in range(dim):
            assert_true(
                not isnan(Float64(d2.sensordata.data[adr + k])),
                "sensor " + String(i) + " stayed NaN with RNE_POST=True — the"
                " NaN on the other leg was not about RNE_POST at all",
            )
            n_filled += 1
    print("  with RNE_POST=True those same", n_filled, "slots are computed")


def test_the_joint_sensors_read_their_own_joints_address() raises:
    """`jointpos`/`jointvel` read `jnt_qposadr` / `jnt_dofadr`, not each other.

    ⚠⚠ THE TWO ADDRESSES DIVERGE ONLY WHEN A MULTI-DOF JOINT COMES FIRST, and
    that is why the fixture's `el` sits behind a freejoint. With `root`
    consuming 7 qpos and 6 qvel, `el` has `qposadr == 7` and `dofadr == 6`:
    a `jointvel` that read `qposadr` would index qvel[7] — past NV — and a
    `jointpos` that read `dofadr` would return qpos[6], the quaternion's z
    component, a number of entirely plausible magnitude. On a model of hinges
    alone the two are equal and the bug is invisible, so this asserts the
    divergence BEFORE trusting the agreement above.

    It also pins the values themselves against MuJoCo's own addresses rather
    than against our table, so a disagreement about joint ORDER shows up here
    and not as a mysterious offset.
    """
    print("=== jointpos/jointvel address their own joint ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(SD_XML)))

    var O = mujoco.mjtObj
    var jid = Int(py=mujoco.mj_name2id(m, O.mjOBJ_JOINT, PythonObject("el")))
    assert_true(jid >= 0, "the fixture no longer declares a joint named `el`")
    var qadr = Int(py=m.jnt_qposadr[jid])
    var vadr = Int(py=m.jnt_dofadr[jid])
    print("  MuJoCo: joint `el` id", jid, " qposadr", qadr, " dofadr", vadr)
    # ⚠ NON-VACUITY FOR THIS TEST SPECIFICALLY.
    assert_true(
        qadr != vadr,
        "`el` has qposadr == dofadr == " + String(qadr) + " — the fixture has"
        " lost the freejoint in front of it, so reading the wrong table would"
        " give the RIGHT answer and this test proves nothing",
    )

    # And our own table agrees about which joint that is, and where it lives.
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var st = _run_plain(d, mf, ctx)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    assert_true(
        Int(mf.joints.data[jid * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
        == qadr,
        "our jnt_qposadr for `el` is "
        + String(Int(mf.joints.data[jid * MODEL_JOINT_SIZE
                                    + JOINT_IDX_QPOS_ADR]))
        + ", MuJoCo says " + String(qadr),
    )
    assert_true(
        Int(mf.joints.data[jid * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR])
        == vadr,
        "our jnt_dofadr for `el` is "
        + String(Int(mf.joints.data[jid * MODEL_JOINT_SIZE
                                    + JOINT_IDX_DOF_ADR]))
        + ", MuJoCo says " + String(vadr),
    )

    # The readings, against the state the step was evaluated at.
    var jp_adr = Int(py=m.sensor_adr[
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_SENSOR, PythonObject("jp")))])
    var jv_adr = Int(py=m.sensor_adr[
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_SENSOR, PythonObject("jv")))])
    var ours_jp = Float64(d.sensordata.data[jp_adr])
    var ours_jv = Float64(d.sensordata.data[jv_adr])
    print("  jointpos ours", ours_jp, " qpos[qposadr]", qpos[qadr])
    print("  jointvel ours", ours_jv, " qvel[dofadr] ", qvel[vadr])
    assert_true(
        abs(ours_jp - qpos[qadr]) <= 1e-15,
        "jointpos read " + String(ours_jp) + ", qpos[" + String(qadr)
        + "] is " + String(qpos[qadr]),
    )
    assert_true(
        abs(ours_jv - qvel[vadr]) <= 1e-15,
        "jointvel read " + String(ours_jv) + ", qvel[" + String(vadr)
        + "] is " + String(qvel[vadr]),
    )

    # ⚠ THE OVER-FIX CONTROL, SPELLED OUT: name the value the swap produces.
    # qpos[dofadr] is the tilt quaternion's z; a jointpos reading it would be
    # 0.22, not 0.6.
    print("  the value a qposadr/dofadr swap would have produced:",
          qpos[vadr])
    assert_true(
        abs(ours_jp - qpos[vadr]) > 1e-6,
        "jointpos returned qpos[dofadr] = " + String(qpos[vadr])
        + " — it is reading the DOF address",
    )
    print("  both joint sensors read their own address")


def test_a_joint_sensor_on_a_multi_dof_joint_refuses() raises:
    """MuJoCo's compiler refuses `jointpos` on a free or ball joint; so do we.

    `user_objects.cc:7902-7913` — "joint must be slide or hinge in sensor".
    The sensor reports ONE scalar, and there is no single qpos of a free joint
    that answers to "the joint's position". Accepting it and taking
    `qpos[qposadr]` would return the body's WORLD X, which is a number, has
    the right units for a slide joint, and is not a joint position.

    ⚠ THE ORACLE IS ASSERTED TO REFUSE TOO, in the same test. A refusal we
    invented would be a divergence from MuJoCo dressed up as strictness.
    """
    print("=== jointpos on a free joint refuses, in both engines ===")
    var bad = String(
        "<mujoco><worldbody><body name='b' pos='0 0 1'>"
        "<freejoint name='root'/>"
        "<geom type='sphere' size='0.1'/>"
        "</body></worldbody>"
        "<sensor><jointpos name='jp' joint='root'/></sensor></mujoco>"
    )

    var mujoco = Python.import_module("mujoco")
    var mj_refused = False
    try:
        _ = mujoco.MjModel.from_xml_string(PythonObject(bad))
    except:
        mj_refused = True
    assert_true(
        mj_refused,
        "MuJoCo 3.12 ACCEPTED <jointpos> on a freejoint — the premise of our"
        " refusal has moved and this loader is now stricter than the oracle",
    )
    print("  MuJoCo refuses it")

    var we_refused = False
    try:
        _ = parse_xml_full(bad, String("."))
    except:
        we_refused = True
    assert_true(
        we_refused,
        "our loader ACCEPTED <jointpos joint='root'> on a freejoint — it would"
        " report qpos[0], the body's world X, as a joint position",
    )
    print("  we refuse it")

    # ⚠ AND THE SAME DOCUMENT WITH A HINGE MUST LOAD. Without this the two
    # refusals above could both be about the fixture, not about the joint type.
    var good = String(
        "<mujoco><worldbody><body name='b' pos='0 0 1'>"
        "<joint name='h' type='hinge' axis='0 1 0'/>"
        "<geom type='sphere' size='0.1'/>"
        "</body></worldbody>"
        "<sensor><jointpos name='jp' joint='h'/></sensor></mujoco>"
    )
    _ = mujoco.MjModel.from_xml_string(PythonObject(good))
    var fmd = parse_xml_full(good, String("."))
    assert_true(
        len(fmd.sensors) == 1 and fmd.sensors[0].served,
        "the hinge form must load and be SERVED; it is not",
    )
    print("  the hinge form loads and is served, in both")


def main() raises:
    var suite = TestSuite()
    suite.test[test_sensordata_matches_mujoco]()
    suite.test[test_cutoff_clamps_like_mujoco]()
    suite.test[test_a_stage_that_never_runs_is_loud]()
    suite.test[test_the_joint_sensors_read_their_own_joints_address]()
    suite.test[test_a_joint_sensor_on_a_multi_dof_joint_refuses]()
    suite^.run()
