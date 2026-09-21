"""`<joint margin>` (AUD-03) and a direct-form `solreflimit` (AUD-29) vs MuJoCo.

    pixi run mojo run -I . tests/physics3d/test_joint_margin_and_direct_solref_vs_mujoco.mojo

Two rows of `docs/PHYSICS3D_MUJOCO_312_AUDIT.md` that were wrong on models
in the tree:

AUD-03. `jnt_margin` was never parsed, so every limit row engaged at
`dist < 0` where MuJoCo engages it at `dist < margin` and prices it on
`dist - margin` (engine_core_constraint.c:1394-1425, :2109, :3255). ant,
hopper, dog and humanoid_cmu all carry a joint `margin`. The fixture below
sits INSIDE the band — 0.3 rad short of the stop with a 0.5 margin — so
MuJoCo builds one row and the old code built none: the difference is the
whole limit force, not a rounding.

AUD-29. A negative `solreflimit` is MuJoCo's direct form `(-K, -B)`.
`fields_build` kept the parsed value only `if >= 0` and the Newton/CG
builders re-applied the default `if <= 0`, so `-1000 -50` became `0.02 1`:
K 2770 instead of 1020, B 105 instead of 50.5. The fixture drives the joint
past its stop and reads MuJoCo's `efc_KBIP` to prove the reference is on the
direct form before comparing accelerations.

Three legs per fixture, because the limit row is built in FOUR places:
`constraints/limits.mojo` (CONTACTS=False), the per-env pyramidal Newton
builder in `solver/newton_solve.mojo` and `constraints/scalar_rows.mojo`
(elliptic Newton). The NVIDIA block kernel mirrors the per-env builder and
is gated by `test-physics3d-gpu` on the box.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.fields import Model, Data, Dims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.types import ConeType

comptime DTYPE = DType.float64

# j1: range +-1 rad, margin 0.5, posed at 0.7 with velocity into the stop.
# dist_hi = 0.3 < margin -> ONE row; dist_lo = 1.7 > margin -> none.
comptime MARGIN_XML = """<mujoco model="joint_margin">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" range="-1 1" margin="0.5"
               solreflimit="0.04 1" solimplimit="0.9 0.99 0.2"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# j1 past its +30 deg stop on a DIRECT-form solreflimit.
comptime DIRECT_XML = """<mujoco model="direct_solref">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" limited="true"
               range="-30 30" solreflimit="-1000 -50"
               solimplimit="0.9 0.99 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

comptime mp = parse_xml(MARGIN_XML)
comptime dp = parse_xml(DIRECT_XML)


# ⚠ VALUE idiom, not a type alias: a helper generic over `M: ModelDefFromXML`
# takes the model def as a parameter VALUE (see test_connect_equality_vs_mujoco).
def _model_margin_pyr() -> ModelDefFromXML[
    xml=MARGIN_XML,
    nbody=mp.NBODY, njoint=mp.NJOINT, nq=mp.NQ, nv=mp.NV,
    ngeom=mp.NGEOM, nact=mp.NACT, ntex=mp.NTEX, nmat=mp.NMAT,
    nlight=mp.NLIGHT, ncam=mp.NCAM, nsite=mp.NSITE,
    max_tendon=mp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    max_condim=mp.MAX_CONDIM,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=mp.TIMESTEP,
]:
    return {}


def _model_margin_ell() -> ModelDefFromXML[
    xml=MARGIN_XML,
    nbody=mp.NBODY, njoint=mp.NJOINT, nq=mp.NQ, nv=mp.NV,
    ngeom=mp.NGEOM, nact=mp.NACT, ntex=mp.NTEX, nmat=mp.NMAT,
    nlight=mp.NLIGHT, ncam=mp.NCAM, nsite=mp.NSITE,
    max_tendon=mp.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=8,
    max_condim=mp.MAX_CONDIM,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=mp.TIMESTEP,
]:
    return {}


def _model_direct() -> ModelDefFromXML[
    xml=DIRECT_XML,
    nbody=dp.NBODY, njoint=dp.NJOINT, nq=dp.NQ, nv=dp.NV,
    ngeom=dp.NGEOM, nact=dp.NACT, ntex=dp.NTEX, nmat=dp.NMAT,
    nlight=dp.NLIGHT, ncam=dp.NCAM, nsite=dp.NSITE,
    max_tendon=dp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    max_condim=dp.MAX_CONDIM,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=dp.TIMESTEP,
]:
    return {}


comptime MM = _model_margin_pyr()
comptime MME = _model_margin_ell()
comptime DM = _model_direct()
comptime NV = 2

comptime MQ0 = 0.3
comptime MQ1 = 0.7
comptime MV1 = 1.5
comptime DQ0 = 0.3
comptime DQ1 = 0.785398163397448
comptime DV1 = 1.5


def _mj_qacc(
    xml: String, q0: Float64, q1: Float64, v1: Float64
) raises -> Tuple[PythonObject, PythonObject]:
    """MuJoCo's (model, data) at the pose, forwarded, contacts off."""
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    m.opt.disableflags = (
        Int(py=m.opt.disableflags)
        | Int(py=mujoco.mjtDisableBit.mjDSBL_CONTACT)
    )
    var dat = mujoco.MjData(m)
    dat.qpos[0] = q0
    dat.qpos[1] = q1
    dat.qvel[0] = 0.0
    dat.qvel[1] = v1
    mujoco.mj_forward(m, dat)
    return (m, dat)


def _ours_qacc[
    M: ModelDefFromXML, CONTACTS: Bool
](q0: Float64, q1: Float64, v1: Float64) raises -> List[Float64]:
    comptime MD = Dims[
        nq=M.NQ,
        nv=M.NV,
        nbody=M.NBODY,
        njoint=M.NJOINT,
        ngeom=M.NGEOM,
        nsite=M.NSITE,
        max_contacts=M.MAX_CONTACTS,
        nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON,
        nexclude=M.NEXCLUDE,
        nmesh_verts=0,
        npair=M.NPAIR,
        nact=M.NACT,
        nten=M.NTEN_F,
        nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = Scalar[DTYPE](q0)
    d.qpos.data[1] = Scalar[DTYPE](q1)
    d.qvel.data[0] = Scalar[DTYPE](0)
    d.qvel.data[1] = Scalar[DTYPE](v1)
    for i in range(NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[
        DTYPE, MD, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM=M.MAX_CONDIM,
        NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu", CONTACTS=CONTACTS](d, mf)
    var out = List[Float64]()
    for i in range(NV):
        out.append(Float64(integ.scratch.qacc_constrained.data[i]))
    return out^


def _compare(
    label: String, ours: List[Float64], dat: PythonObject, tol: Float64
) raises:
    var worst = Float64(0)
    for i in range(NV):
        var theirs = Float64(py=dat.qacc[i])
        var e = abs(ours[i] - theirs)
        print("   ", label, " dof", i, " ours", ours[i], " MuJoCo", theirs, " |d|", e)
        if e > worst:
            worst = e
    assert_true(
        worst < tol,
        label + ": qacc disagrees with MuJoCo by " + String(worst),
    )


def test_joint_margin_builds_the_row_inside_the_band() raises:
    print("--- AUD-03: <joint margin> — one row 0.3 rad short of the stop ---")
    var h = _mj_qacc(String(MARGIN_XML), MQ0, MQ1, MV1)
    var m = h[0]
    var dat = h[1]
    var nefc = Int(py=dat.nefc)
    var margin = Float64(py=m.jnt_margin[1])
    print("  MuJoCo jnt_margin[1] =", margin, " nefc =", nefc,
          " efc_pos =", Float64(py=dat.efc_pos[0]),
          " efc_margin =", Float64(py=dat.efc_margin[0]))
    assert_true(
        margin == 0.5 and nefc == 1 and Int(py=dat.efc_type[0]) == 3,
        "the fixture must give MuJoCo exactly ONE joint-limit row inside the"
        " margin band (got nefc " + String(nefc) + ")",
    )
    assert_true(
        abs(Float64(py=dat.efc_pos[0]) - 0.3) < 1e-12,
        "MuJoCo's efc_pos is the raw distance (0.3), not dist - margin",
    )
    # non-vacuity: MuJoCo with `jnt_margin` zeroed at runtime must give a
    # DIFFERENT acceleration — that difference is what the old code got wrong.
    var qacc_margin = Float64(py=dat.qacc[1])
    var mujoco = Python.import_module("mujoco")
    m.jnt_margin[1] = 0.0
    mujoco.mj_forward(m, dat)
    var qacc_nomargin = Float64(py=dat.qacc[1])
    print("  MuJoCo qacc[1] with margin =", qacc_margin, " without =",
          qacc_nomargin, " nefc without =", Int(py=dat.nefc))
    assert_true(
        Int(py=dat.nefc) == 0 and abs(qacc_margin - qacc_nomargin) > 1e-6,
        "zeroing jnt_margin did not change MuJoCo's answer; the fixture"
        " does not express AUD-03",
    )
    m.jnt_margin[1] = 0.5
    mujoco.mj_forward(m, dat)
    _compare("limits.mojo (CONTACTS=False)",
             _ours_qacc[M=MM, CONTACTS=False](MQ0, MQ1, MV1), dat, 1e-9)
    _compare("newton pyramidal (CONTACTS=True)",
             _ours_qacc[M=MM, CONTACTS=True](MQ0, MQ1, MV1), dat, 1e-6)
    _compare("scalar_rows elliptic (CONTACTS=True)",
             _ours_qacc[M=MME, CONTACTS=True](MQ0, MQ1, MV1), dat, 1e-6)


def test_direct_form_solreflimit_reaches_the_row() raises:
    print("--- AUD-29: solreflimit=\"-1000 -50\" is the direct form ---")
    var h = _mj_qacc(String(DIRECT_XML), DQ0, DQ1, DV1)
    var m = h[0]
    var dat = h[1]
    var nefc = Int(py=dat.nefc)
    var K_mj = Float64(py=dat.efc_KBIP[0][0])
    var B_mj = Float64(py=dat.efc_KBIP[0][1])
    print("  MuJoCo jnt_solref[1] =", Float64(py=m.jnt_solref[1][0]),
          Float64(py=m.jnt_solref[1][1]), " nefc =", nefc,
          " efc_KBIP K =", K_mj, " B =", B_mj)
    assert_true(nefc == 1, "expected exactly one limit row")
    # K = -ref0/dmax^2 = 1000/0.99^2, B = -ref1/dmax = 50/0.99
    assert_true(
        abs(K_mj - 1000.0 / (0.99 * 0.99)) < 1e-6
        and abs(B_mj - 50.0 / 0.99) < 1e-6,
        "MuJoCo is not on the direct form (K " + String(K_mj) + ", B "
        + String(B_mj) + ") — the fixture no longer expresses AUD-29",
    )
    _compare("limits.mojo (CONTACTS=False)",
             _ours_qacc[M=DM, CONTACTS=False](DQ0, DQ1, DV1), dat, 1e-9)
    _compare("newton pyramidal (CONTACTS=True)",
             _ours_qacc[M=DM, CONTACTS=True](DQ0, DQ1, DV1), dat, 1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
