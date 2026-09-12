"""AUD-43 (ball-joint spring), AUD-08 (tendon damping), AUD-38 (tendon-limit
margin) against MuJoCo 3.12, plus the device-safe `atan2` against libm.

    pixi run mojo run -I . tests/physics3d/test_ball_spring_and_tendon_damping_vs_mujoco.mojo

AUD-43. A ball joint's spring torque is `-stiffness * subQuat(q, identity)`,
the rotation's axis-angle vector (engine_passive.c:696-707). The passive
routine used to subtract `springref` from the quaternion COMPONENTS, a
torque of `-10` on a joint at rest. `atan2_device` gives the angle without
libm so the same routine runs inside the GPU kernel; it is gated against
`std.math.atan2` here first.

AUD-08. `-damping * ten_velocity` in `qfrc_passive` (engine_passive.c:823),
`ten_velocity = sum(coef * qvel)` for a fixed tendon. It was never parsed.

AUD-38. Inside the margin band the tendon-limit row priced `-dist` where
MuJoCo prices `dist - margin`; the sign of the reference acceleration was
wrong (it pulled the tendon into the limit).
"""

from std.math import abs, atan2, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.kinematics.quat_math import atan2_device
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.types import ConeType

comptime DTYPE = DType.float64

comptime BALL_XML = """<mujoco model="ball spring">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="jb" type="ball" stiffness="10"/>
      <geom type="box" size="0.1 0.05 0.02" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime TEN_XML = """<mujoco model="tendon damping and margin">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="td" damping="3">
      <joint joint="j0" coef="1"/>
      <joint joint="j1" coef="0.5"/>
    </fixed>
    <fixed name="tl" limited="true" range="-1 0.3" margin="0.2"
           solreflimit="0.04 1" solimplimit="0.9 0.99 0.2">
      <joint joint="j1" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""

comptime bp = parse_xml(BALL_XML)


def _model_ball() -> ModelDefFromXML[
    xml=BALL_XML,
    nbody=bp.NBODY, njoint=bp.NJOINT, nq=bp.NQ, nv=bp.NV,
    ngeom=bp.NGEOM, nact=bp.NACT, ntex=bp.NTEX, nmat=bp.NMAT,
    nlight=bp.NLIGHT, ncam=bp.NCAM, nsite=bp.NSITE,
    max_tendon=bp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    max_condim=bp.MAX_CONDIM,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=bp.TIMESTEP,
]:
    return {}


comptime tp = parse_xml(TEN_XML)


def _model_ten() -> ModelDefFromXML[
    xml=TEN_XML,
    nbody=tp.NBODY, njoint=tp.NJOINT, nq=tp.NQ, nv=tp.NV,
    ngeom=tp.NGEOM, nact=tp.NACT, ntex=tp.NTEX, nmat=tp.NMAT,
    nlight=tp.NLIGHT, ncam=tp.NCAM, nsite=tp.NSITE,
    max_tendon=tp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    max_condim=tp.MAX_CONDIM,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=tp.TIMESTEP,
]:
    return {}


comptime BM = _model_ball()
comptime TM = _model_ten()


def _mj() raises -> PythonObject:
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    return Python.import_module("mujoco")


def _ours_qacc[
    M: ModelDefFromXML, CONTACTS: Bool
](qpos: List[Float64], qvel: List[Float64], timestep: Float64) raises -> List[Float64]:
    comptime MD = Dims[
        nq=M.NQ, nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM,
        nsite=M.NSITE, max_contacts=M.MAX_CONTACTS, nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON, nexclude=M.NEXCLUDE, nmesh_verts=0,
        npair=M.NPAIR, nact=M.NACT, nten=M.NTEN_F, nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    for i in range(M.NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos[i])
    for i in range(M.NV):
        d.qvel.data[i] = Scalar[DTYPE](qvel[i])
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)
    # the tendon springs/dampers live in the actuation pass, as in the env step
    var ctrl = List[Float64]()
    var act = List[Scalar[DTYPE]]()
    apply_actions_fields[DTYPE](sf, d, ctrl, act, timestep)
    var integ = EulerIntegrator[
        DTYPE, MD, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM=M.MAX_CONDIM,
        NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu", CONTACTS=CONTACTS](d, mf)
    var out = List[Float64]()
    for i in range(M.NV):
        out.append(Float64(integ.scratch.qacc_constrained.data[i]))
    return out^


def _compare(label: String, ours: List[Float64], dat: PythonObject, tol: Float64) raises:
    var worst = Float64(0)
    var mag = Float64(0)
    for i in range(len(ours)):
        var theirs = Float64(py=dat.qacc[i])
        var e = abs(ours[i] - theirs)
        print("   ", label, " dof", i, " ours", ours[i], " MuJoCo", theirs, " |d|", e)
        if e > worst:
            worst = e
        if abs(theirs) > mag:
            mag = abs(theirs)
    assert_true(mag > 1e-3, label + ": the reference acceleration is ~0, vacuous")
    assert_true(worst < tol, label + ": qacc differs from MuJoCo by " + String(worst))


def test_atan2_device_matches_libm() raises:
    print("=== atan2_device vs std.math.atan2 ===")
    var worst = Float64(0)
    var vals = List[Float64]()
    for i in range(-20, 21):
        vals.append(Float64(i) * 0.37)
    vals.append(1e-12)
    vals.append(-1e-12)
    vals.append(1e6)
    for a in range(len(vals)):
        for b in range(len(vals)):
            var y = vals[a]
            var x = vals[b]
            var expect = atan2(y, x)
            var got = Float64(atan2_device[DType.float64](Scalar[DType.float64](y), Scalar[DType.float64](x)))
            var e = abs(expect - got)
            if e > worst:
                worst = e
    print("  worst |d| =", worst, " over", len(vals) * len(vals), "pairs")
    # 4 ulp near pi: the halvings and libm round differently at the last bit
    assert_true(worst < 2e-15, "atan2_device is off libm by " + String(worst))


def test_ball_spring_matches_mujoco() raises:
    print("=== AUD-43: ball-joint spring torque ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(BALL_XML)
    var dat = mujoco.MjData(m)
    # two poses: a pure y rotation, and a general one
    var poses = List[List[Float64]]()
    var q1 = List[Float64]()
    q1.append(0.9887710779360422); q1.append(0.0); q1.append(0.14943813247359922); q1.append(0.0)
    poses.append(q1^)
    var q2 = List[Float64]()
    var n = sqrt(0.9 * 0.9 + 0.1 * 0.1 + 0.3 * 0.3 + 0.2 * 0.2)
    q2.append(0.9 / n); q2.append(0.1 / n); q2.append(0.3 / n); q2.append(-0.2 / n)
    poses.append(q2^)
    for p in range(2):
        for i in range(4):
            dat.qpos[i] = poses[p][i]
        for i in range(3):
            dat.qvel[i] = 0.0
        mujoco.mj_forward(m, dat)
        print("  pose", p, " MuJoCo qfrc_passive =", dat.qfrc_passive)
        var qv = List[Float64]()
        for _ in range(3):
            qv.append(0.0)
        _compare("pose " + String(p), _ours_qacc[M=BM, CONTACTS=False](poses[p], qv, 0.002), dat, 1e-9)


def test_tendon_damping_matches_mujoco() raises:
    print("=== AUD-08: fixed-tendon damping ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(TEN_XML)
    var dat = mujoco.MjData(m)
    dat.qpos[0] = 0.1
    dat.qpos[1] = 0.0
    dat.qvel[0] = 1.0
    dat.qvel[1] = -0.5
    mujoco.mj_forward(m, dat)
    print("  MuJoCo tendon_damping =", m.tendon_damping, " ten_velocity =", dat.ten_velocity,
          " qfrc_passive =", dat.qfrc_passive, " nefc =", Int(py=dat.nefc))
    assert_true(Int(py=dat.nefc) == 0, "the limit tendon must be out of its band here")
    assert_true(abs(Float64(py=dat.qfrc_passive[0])) > 1.0, "damping carries no force, vacuous")
    var qp = List[Float64](); qp.append(0.1); qp.append(0.0)
    var qv = List[Float64](); qv.append(1.0); qv.append(-0.5)
    _compare("damping", _ours_qacc[M=TM, CONTACTS=False](qp, qv, 0.005), dat, 1e-9)


def test_tendon_limit_margin_matches_mujoco() raises:
    print("=== AUD-38: tendon limit inside the margin band ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(TEN_XML)
    var dat = mujoco.MjData(m)
    dat.qpos[0] = 0.1
    dat.qpos[1] = 0.25
    dat.qvel[0] = 0.0
    dat.qvel[1] = 0.5
    mujoco.mj_forward(m, dat)
    var nefc = Int(py=dat.nefc)
    print("  MuJoCo nefc =", nefc, " efc_pos =", dat.efc_pos, " efc_margin =", dat.efc_margin,
          " aref =", dat.efc_aref)
    assert_true(nefc == 1 and Int(py=dat.efc_type[0]) == 4,
                "expected exactly one tendon-limit row (mjCNSTR_LIMIT_TENDON = 4)")
    assert_true(Float64(py=dat.efc_aref[0]) > 0.0,
                "inside the band the reference acceleration pushes AWAY from the limit")
    var qp = List[Float64](); qp.append(0.1); qp.append(0.25)
    var qv = List[Float64](); qv.append(0.0); qv.append(0.5)
    _compare("tendon margin", _ours_qacc[M=TM, CONTACTS=True](qp, qv, 0.005), dat, 1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
