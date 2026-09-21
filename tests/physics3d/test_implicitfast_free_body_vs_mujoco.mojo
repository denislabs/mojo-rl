"""AUD-44 — `implicitfast` on a standalone free body carries the gyroscopic
derivative through a local 6x6 solve (MuJoCo 3.11, commit f0fa3d82).

    pixi run mojo run -I . tests/physics3d/test_implicitfast_free_body_vs_mujoco.mojo

A 0.1 x 0.05 x 0.02 box of mass 1 tumbling at (1, 5, 2) rad/s in vacuum,
dt 0.002, integrator implicitfast, no contacts. Before the port ours was
3.10's implicitfast (the symmetric part only): 8.2e-5 of qvel per step and
6.8e-3 of qpos after 500 steps off MuJoCo 3.12. With the block both engines
run the same unsymmetric solve.

⚠ The body must be STANDALONE — one free joint, no children — for the block
to apply; `test_implicitfast_vs_mujoco` (legged robots, whose base has
children) is why this gap was invisible.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.fields import Model, Data, Dims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.integrator.implicit import ImplicitIntegrator
from noeira.physics3d.types import ConeType

comptime DTYPE = DType.float64

comptime XML = """<mujoco model="free body implicitfast">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.05 0.02" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime fp = parse_xml(XML)


def _model() -> ModelDefFromXML[
    xml=XML,
    nbody=fp.NBODY, njoint=fp.NJOINT, nq=fp.NQ, nv=fp.NV,
    ngeom=fp.NGEOM, nact=fp.NACT, ntex=fp.NTEX, nmat=fp.NMAT,
    nlight=fp.NLIGHT, ncam=fp.NCAM, nsite=fp.NSITE,
    max_tendon=fp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    max_condim=fp.MAX_CONDIM,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=fp.TIMESTEP,
]:
    return {}


comptime FM = _model()
comptime MD = Dims[
    nq=FM.NQ, nv=FM.NV, nbody=FM.NBODY, njoint=FM.NJOINT, ngeom=FM.NGEOM,
    nsite=FM.NSITE, max_contacts=FM.MAX_CONTACTS, nequality=FM.MAX_EQUALITY,
    ntendon=FM.MAX_TENDON, nexclude=FM.NEXCLUDE, nmesh_verts=0,
    npair=FM.NPAIR, nact=FM.NACT, nten=FM.NTEN_F, nkey=FM.NKEY,
]
comptime NSTEPS = 500


def test_tumbling_free_body_matches_mujoco_implicitfast() raises:
    print("=== AUD-44: tumbling free box under implicitfast ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    assert_true(Int(py=m.opt.integrator) == 3, "fixture: MuJoCo must be on implicitfast (3)")
    var dat = mujoco.MjData(m)
    dat.qvel[3] = 1.0
    dat.qvel[4] = 5.0
    dat.qvel[5] = 2.0

    var sf = FM.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    FM.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    FM.reset_data[DTYPE](sf, d)
    d.qvel.data[3] = Scalar[DTYPE](1.0)
    d.qvel.data[4] = Scalar[DTYPE](5.0)
    d.qvel.data[5] = Scalar[DTYPE](2.0)
    for i in range(FM.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)
    var integ = ImplicitIntegrator[
        DTYPE, MD, FM.CONE_TYPE, 1, "newton", SKIP_RNE_DERIV=True,
        MAX_CONDIM=FM.MAX_CONDIM, NOSLIP_ITER=FM.NOSLIP_ITER,
    ]()

    var worst_v1 = Float64(0)
    var worst_v = Float64(0)
    var worst_q = Float64(0)
    for step in range(NSTEPS):
        mujoco.mj_step(m, dat)
        integ.step["cpu", CONTACTS=False](d, mf)
        var wv = Float64(0)
        for i in range(FM.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=dat.qvel[i]))
            if e > wv:
                wv = e
        var wq = Float64(0)
        for i in range(FM.NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=dat.qpos[i]))
            if e > wq:
                wq = e
        if step == 0:
            worst_v1 = wv
        if wv > worst_v:
            worst_v = wv
        if wq > worst_q:
            worst_q = wq
    print("  after 1 step   |d qvel| =", worst_v1)
    print("  over", NSTEPS, "steps: worst |d qvel| =", worst_v, " worst |d qpos| =", worst_q)
    var speed = sqrt(
        Float64(py=dat.qvel[3]) ** 2 + Float64(py=dat.qvel[4]) ** 2 + Float64(py=dat.qvel[5]) ** 2
    )
    assert_true(speed > 1.0, "the body stopped tumbling; the fixture is vacuous")
    assert_true(
        worst_v1 < 1e-10,
        "one step differs from MuJoCo implicitfast by " + String(worst_v1)
        + " (3.10's symmetric implicitfast gives 8.2e-5 here)",
    )
    assert_true(
        worst_v < 1e-7 and worst_q < 1e-7,
        "500 steps drift " + String(worst_v) + " / " + String(worst_q)
        + " from MuJoCo (3.10's implicitfast: 5.4e-2 / 6.8e-3)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
