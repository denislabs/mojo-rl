"""`sensordata` on the BATCHED CPU leg, vs MuJoCo per env (AUD-53).

`EulerIntegrator.step` ran the three sensor passes under
`comptime if target == "cpu" and BATCH == 1`, because `sensors/eval.mojo`
took host `List`s and indexed them from zero. A batched model therefore had
`d.sensordata` allocated and left at `Data`'s NaN — safe, because nothing read
it there, and only because of that NaN.

The pass now walks the table through the `_gpu` kernels — the ones that take
`(tensor..., env)` — so one implementation serves both legs and BATCH > 1
gets filled.

⚠⚠ THE FAILURE THIS FILE EXISTS FOR IS "EVERY ENV GOT ENV 0's VALUES". A
sensor pass that dropped the env index would produce four identical, entirely
plausible rows, each matching MuJoCo for env 0. So the four envs are put at
DIFFERENT states, the test asserts MuJoCo's own readings differ between them
before comparing anything, and it compares env by env.

⚠ AIRBORNE, like every other sensordata gate: a settled contact puts a ~7%
solver disagreement into the force sensors and the comparison stops being
about sensors. `ncon == 0` is asserted per env.

Run with:
    pixi run mojo run -I . tests/physics3d/test_batched_sensordata_vs_mujoco.mojo
"""

from std.math import abs, isnan
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float64
comptime NENV = 4

# One of every served kind that does not need an actuator or a tendon: the
# three stages, both `objtype` families, and a rangefinder pointing at a plane
# far below.
comptime B_XML = """
<mujoco model="batched sensordata">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1"/>
    <body name="torso" pos="0 0 2.0">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.12 0.1 0.08" density="700"/>
      <site name="imu" pos="0.03 0.02 0.05" size="0.02" euler="0 25 40"/>
      <site name="down" pos="0 0 -0.08" size="0.01"/>
      <site name="pad" pos="0 0 -0.09" type="box" size="0.1 0.1 0.02"/>
      <body name="link" pos="0.2 0 0" euler="0 0 15">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="gl" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
              density="900"/>
        <site name="wrist" pos="0.25 0.01 0" size="0.02" euler="10 0 0"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <rangefinder name="rf" site="down"/>
    <jointpos name="jp" joint="el"/>
    <framepos name="fp" objtype="site" objname="wrist"/>
    <framequat name="fq" objtype="body" objname="link"/>
    <subtreecom name="scm" body="torso"/>
    <velocimeter name="vel" site="imu"/>
    <gyro name="gyr" site="imu"/>
    <jointvel name="jv" joint="el"/>
    <framelinvel name="flv" objtype="site" objname="wrist"/>
    <frameangvel name="fav" objtype="geom" objname="gl"/>
    <subtreelinvel name="slv" body="torso"/>
    <accelerometer name="acc" site="imu"/>
    <force name="frc" site="wrist"/>
    <torque name="trq" site="wrist"/>
    <touch name="tch" site="pad"/>
  </sensor>
</mujoco>
"""

comptime bp = parse_xml(B_XML)
comptime BM = ModelDefFromXML[
    xml=B_XML,
    nbody=bp.NBODY, njoint=bp.NJOINT, nq=bp.NQ, nv=bp.NV,
    ngeom=bp.NGEOM, nact=bp.NACT, ntex=bp.NTEX, nmat=bp.NMAT,
    nlight=bp.NLIGHT, ncam=bp.NCAM, nsite=bp.NSITE,
    # 15 sensors; 1+1+3+4+3 +3+3+1+3+3+3 +3+3+3+1 = 38 values.
    nsensor=15, nsensordata=38,
    max_tendon=bp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=bp.TIMESTEP,
]

comptime BMD = ModelDims[BM]
comptime Dat = Data[DTYPE, BMD, NENV]
comptime Mod = Model[DTYPE, BMD]
comptime Integ = EulerIntegrator[
    DTYPE, BMD, BM.CONE_TYPE, NENV, SOLVER="newton", RNE_POST=True
]


def _state(e: Int) -> Tuple[List[Float64], List[Float64]]:
    """Env `e`'s pose and rates — DIFFERENT per env, by construction.

    ⚠ THE SPREAD IS THE TEST. Envs that share a state cannot distinguish "the
    pass indexed the env" from "the pass wrote env 0 four times", and that is
    the one bug this file is for. The elbow, the height and every velocity
    component move with `e`, and the quaternion is renormalised so each env
    sits at a genuinely different attitude rather than a scaled one.
    """
    var f = Float64(e)
    var qpos = List[Float64]()
    qpos.append(0.05 * f)
    qpos.append(-0.03 * f)
    qpos.append(2.0 + 0.1 * f)
    # w, x, y, z — a different tilt per env, normalised.
    var wx = 0.9 - 0.1 * f
    var xx = 0.2 + 0.05 * f
    var yy = 0.15 + 0.04 * f
    var zz = 0.1 + 0.06 * f
    var n = (wx * wx + xx * xx + yy * yy + zz * zz) ** 0.5
    qpos.append(wx / n)
    qpos.append(xx / n)
    qpos.append(yy / n)
    qpos.append(zz / n)
    qpos.append(0.2 + 0.35 * f)  # elbow

    var qvel = List[Float64]()
    qvel.append(0.7 - 0.2 * f)
    qvel.append(-0.4 + 0.3 * f)
    qvel.append(1.1 - 0.15 * f)
    qvel.append(0.9 + 0.2 * f)
    qvel.append(-1.3 + 0.4 * f)
    qvel.append(0.5 - 0.25 * f)
    qvel.append(2.0 - 0.5 * f)
    return (qpos^, qvel^)


def _run(mut d: Dat, mut mf: Mod, ctx: DeviceContext) raises:
    """⚠ NO `reset_data` — IT IS BATCH=1 ONLY. `ModelDefFromXML.reset_data`
    takes `Data[DTYPE, D2]`, i.e. one env; there is no batched form, which is
    itself part of why the batched leg went so long without its sensors. The
    fixture writes every `qpos`/`qvel` anyway and `Data.__init__` zeroes the
    rest, so the pose here is fully determined without it.
    """
    BM.init_fields[DTYPE](ctx, mf)
    for i in range(NENV * BM.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
        d.qacc.data[i] = Scalar[DTYPE](0)
        d.qacc_warmstart.data[i] = Scalar[DTYPE](0)
    for e in range(NENV):
        var st = _state(e)
        for i in range(BM.NQ):
            d.qpos.data[e * BM.NQ + i] = Scalar[DTYPE](st[0][i])
        for i in range(BM.NV):
            d.qvel.data[e * BM.NV + i] = Scalar[DTYPE](st[1][i])
    var integ = Integ()
    integ.step["cpu"](d, mf)


def _mj_at(mujoco: PythonObject, e: Int) raises -> PythonObject:
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(B_XML)))
    var dat = mujoco.MjData(m)
    var st = _state(e)
    for i in range(len(st[0])):
        dat.qpos[i] = st[0][i]
    for i in range(len(st[1])):
        dat.qvel[i] = st[1][i]
    mujoco.mj_forward(m, dat)
    return dat^


def test_the_four_envs_are_actually_different() raises:
    """⚠ RUN FIRST. Without this the whole file passes on a pass that wrote
    env 0's values into all four rows.

    Compares MuJoCo's own `sensordata` between env 0 and each other env and
    counts how many of the 38 values move. If most of them do not, the states
    are too close and the env indexing is untested.
    """
    print("=== the four env states are distinguishable ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var d0 = _mj_at(mujoco, 0)
    for e in range(1, NENV):
        var de = _mj_at(mujoco, e)
        var moved = 0
        for k in range(38):
            if abs(
                Float64(py=d0.sensordata[k]) - Float64(py=de.sensordata[k])
            ) > 1e-6:
                moved += 1
        print("  env", e, "differs from env 0 in", moved, "of 38 values")
        assert_true(
            moved >= 30,
            "env " + String(e) + " differs from env 0 in only "
            + String(moved) + " of 38 values — the states are too close for"
            " this file to detect a dropped env index",
        )


def test_batched_sensordata_matches_mujoco_per_env() raises:
    """All 38 values, all four envs, against MuJoCo forwarded at each state."""
    print("=== batched sensordata vs MuJoCo, per env ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(B_XML)))

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var worst = 0.0
    var compared = 0
    var nonzero = 0
    for e in range(NENV):
        var dat = _mj_at(mujoco, e)
        assert_true(
            Int(py=dat.ncon) == 0,
            "env " + String(e) + " is in contact (" + String(Int(py=dat.ncon))
            + " contacts); this gate must stay airborne",
        )
        var e_worst = 0.0
        for k in range(38):
            var ours = Float64(d.sensordata.data[e * 38 + k])
            var theirs = Float64(py=dat.sensordata[k])
            assert_true(
                not isnan(ours),
                "env " + String(e) + " value " + String(k) + " is NaN — the"
                " batched leg did not fill it",
            )
            var dd = abs(ours - theirs)
            if dd > e_worst:
                e_worst = dd
            if abs(theirs) > 1e-9:
                nonzero += 1
            assert_true(
                dd <= 1e-9,
                "env " + String(e) + " value " + String(k) + ": ours "
                + String(ours) + " vs MuJoCo " + String(theirs),
            )
            compared += 1
        if e_worst > worst:
            worst = e_worst
        print("  env", e, " worst |d| =", e_worst)
    print("  values compared:", compared, " worst |d| =", worst)
    print("  values MuJoCo reports NONZERO:", nonzero, "/", compared)
    assert_true(compared == NENV * 38,
                "expected " + String(NENV * 38) + " comparisons")
    assert_true(
        nonzero >= 100,
        "only " + String(nonzero) + " of " + String(compared) + " values are"
        " nonzero on MuJoCo's side — the fixture has gone quiet",
    )


def test_no_env_is_a_copy_of_another() raises:
    """OUR rows must differ from each other too, not only MuJoCo's.

    ⚠ THE COMPARISON ABOVE ALREADY CATCHES A DROPPED ENV INDEX — env 1 would
    be compared against MuJoCo's env 1 and fail. This states the property
    directly so that a failure reads as "the pass is not per-env" rather than
    as a numerical mismatch on 30 unrelated rows, and so that it survives
    someone loosening the tolerance.
    """
    print("=== our four rows are four different readings ===")
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var pairs = 0
    for a in range(NENV):
        for b in range(a + 1, NENV):
            var moved = 0
            for k in range(38):
                if abs(
                    Float64(d.sensordata.data[a * 38 + k])
                    - Float64(d.sensordata.data[b * 38 + k])
                ) > 1e-6:
                    moved += 1
            assert_true(
                moved >= 30,
                "our env " + String(a) + " and env " + String(b) + " agree on"
                " all but " + String(38 - moved) + " values — the pass is"
                " writing one env's readings into several rows",
            )
            pairs += 1
    print("  every one of the", pairs, "env pairs differs in >= 30 of 38")


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_four_envs_are_actually_different]()
    suite.test[test_batched_sensordata_matches_mujoco_per_env]()
    suite.test[test_no_env_is_a_copy_of_another]()
    suite^.run()
