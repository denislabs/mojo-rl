"""The noslip pass under `<option integrator="implicitfast">` — vs MuJoCo 3.10.0.

WHY THIS EXISTS. `3bc98c55` moved the CPU noslip off the dense `M^-1` and
onto the step's tree LDL of M (`noslip._minv_apply` solving against
`scratch.L` / `scratch.D`, as `mj_solNoSlip` solves against `qLD`). The Euler
and RK4 steps fill that factor with `ldl_factor` before their Newton, so every
gate that stepped through them stayed green. `ImplicitIntegrator.step` never
did — it LU-factors M for its own re-solve — and so, from that commit, every
implicitfast model with contacts and `NOSLIP_ITER > 0` ran its noslip against
an UNFACTORED slab. The studio steps with `NOSLIP_ITER = 1`, so on the
Menagerie board unitree_g1 went 5.9e-17 -> 1.0e-02 and kinova_gen3
2.4e-15 -> 4.5e-02 after ONE step, and g1 bounced off the floor like a
trampoline. None of the bench models says `implicitfast`; none of the noslip
gates did either. This one does.

WHAT IT GATES. The chain fixture of `test_noslip_reaches_the_runtime_path`
with the integrator switched to `implicitfast`, on BOTH cones (the pyramidal
and elliptic noslip are different routines over different row layouts), taken
through the runtime loader and the studio's own `StudioImpFast*` integrators
— the path the studio and the board use. The state is settled by MuJoCo and
shared, the slam velocity is seeded on both sides, and `qvel` after N steps
is compared.

⚠ THE OFF ARM IS ASSERTED TOO. `MIN_OFF_MISS` requires the same rollout with
the attribute deleted to land measurably elsewhere, so a green here cannot be
two engines agreeing on a pass that never ran (the shape of
`feedback_a_gate_that_shares_its_reference_implementation_is_blind`).

⚠ THE FIXTURE MUST BE IN CONTACT AT STEP 0 OR IT GATES NOTHING —
`test_the_fixtures_are_in_contact` asserts the contact count.

MEASURED (MuJoCo 3.10.0, 3 steps from the shared settled state):
    see the printed lines; the tolerance is set well above them and well
    below the 1e-2 the defect produced.

Run: pixi run mojo run -I . tests/physics3d/test_noslip_implicitfast_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.studio.stepping import (
    StudioImpFastEll, StudioImpFastPyr, studio_cone_of, studio_integrator_of,
)
from mojo_rl.physics3d.types import ConeType, IntegratorType
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS


comptime DT = DType.float64


def _chain(cone: String, noslip: Bool) -> String:
    var opt = String('<option timestep="0.002" gravity="0 0 -9.81" cone="')
    opt += cone + '" integrator="implicitfast" noslip_tolerance="0"'
    if noslip:
        opt += ' noslip_iterations="5"'
    opt += "/>"
    return String("""
<mujoco model="noslip_implicitfast">
  """) + opt + """
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .05">
      <joint type="free" name="root"/>
      <geom name="g1" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
            condim="3" friction=".7 .05 .05"/>
      <body name="l2" pos=".3 0 0">
        <joint type="hinge" name="j2" axis="0 1 0" range="-60 60"
               limited="true" frictionloss="0.05" damping="0.1"/>
        <geom name="g2" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
              condim="3" friction=".7 .05 .05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


comptime N_SETTLE: Int = 400
comptime N_STEP: Int = 3
comptime SEED_VX: Float64 = 3.0
comptime SEED_VY: Float64 = 1.0
comptime SEED_VZ: Float64 = -40.0
comptime SEED_WZ: Float64 = 3.0
comptime SEED_J2: Float64 = 4.0

# The measured agreement is rounding-level (see the printed lines); 1e-9 is
# orders above it and orders below the 1e-2 the unfactored slab produced.
comptime TOL_ON: Float64 = 1e-9
comptime MIN_OFF_MISS: Float64 = 1e-4


@fieldwise_init
struct _Ours(Movable):
    var qvel: List[Float64]
    var ncon: Int
    var cone: Int
    var integ: Int


def _seed[DTV: DType](mut d: Data[DTV, DynDims, 1], nv: Int) raises:
    for i in range(nv):
        d.qvel.data[i] = Scalar[DTV](0)
    d.qvel.data[0] = Scalar[DTV](SEED_VX)
    d.qvel.data[1] = Scalar[DTV](SEED_VY)
    d.qvel.data[2] = Scalar[DTV](SEED_VZ)
    d.qvel.data[5] = Scalar[DTV](SEED_WZ)
    d.qvel.data[6] = Scalar[DTV](SEED_J2)


def _ours(xml: String, qpos: List[Float64], nstep: Int) raises -> _Ours:
    """The studio's path: `parse_xml_full` -> `build_model_runtime` ->
    `StudioImpFastEll` / `StudioImpFastPyr` by the file's cone."""
    var fmd = parse_xml_full(xml, String("."))
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=1024)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)

    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(min(nq, len(qpos))):
        d.qpos.data[i] = Scalar[DT](qpos[i])
    _seed[DT](d, nv)

    var cone = studio_cone_of(fmd)
    var integ = studio_integrator_of(fmd)
    assert_true(
        integ == IntegratorType.IMPLICITFAST,
        "the fixture must dispatch to the implicit integrator, or this gates"
        " the Euler path a second time",
    )
    if cone == ConeType.ELLIPTIC:
        var ell = StudioImpFastEll(dims)
        for _ in range(nstep):
            ell.step["cpu"](d, m)
    else:
        var pyr = StudioImpFastPyr(dims)
        for _ in range(nstep):
            pyr.step["cpu"](d, m)

    var qv = List[Float64]()
    for i in range(nv):
        qv.append(Float64(d.qvel.data[i]))
    return _Ours(
        qvel=qv^,
        ncon=Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS])),
        cone=cone,
        integ=integ,
    )


def _mj(
    xml: String, qpos: List[Float64], nstep: Int, nv: Int
) raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for i in range(len(qpos)):
        d.qpos[i] = qpos[i]
    d.qvel[0] = SEED_VX
    d.qvel[1] = SEED_VY
    d.qvel[2] = SEED_VZ
    d.qvel[5] = SEED_WZ
    d.qvel[6] = SEED_J2
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(py=qv[i]))
    return out^


def _settled(xml: String) raises -> List[Float64]:
    """MuJoCo drops the chain from `qpos0` and lets it rest; shared by every
    rollout of that cone, and settled from the OFF text so the pass under
    test has no hand in the state it is measured on."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, d)
    var qp = d.qpos.flatten().tolist()
    var out = List[Float64]()
    for i in range(len(qp)):
        out.append(Float64(py=qp[i]))
    return out^


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(min(len(a), len(b))):
        var e = abs(a[i] - b[i])
        if e > w:
            w = e
    return w


def _gate(cone: String) raises:
    var on = _chain(cone, True)
    var off = _chain(cone, False)
    var qpos = _settled(off)

    var ours_on = _ours(on, qpos, N_STEP)
    var nv = len(ours_on.qvel)
    var mj_on = _mj(on, qpos, N_STEP, nv)
    var mj_off = _mj(off, qpos, N_STEP, nv)
    var ours_off = _ours(off, qpos, N_STEP)

    var d_on = _maxdiff(ours_on.qvel, mj_on)
    var d_off = _maxdiff(ours_off.qvel, mj_off)
    var pass_worth = _maxdiff(mj_on, mj_off)
    print("  cone", cone, " ncon", ours_on.ncon,
          " integrator", ours_on.integ)
    print("    ours ON  vs MuJoCo ON   |d(qvel)| =", d_on)
    print("    ours OFF vs MuJoCo OFF  |d(qvel)| =", d_off)
    print("    MuJoCo ON vs MuJoCo OFF |d(qvel)| =", pass_worth,
          " (what the pass is worth here)")

    assert_true(
        ours_on.ncon > 0,
        "the fixture is not in contact at the measured steps — nothing gated",
    )
    assert_true(
        pass_worth > MIN_OFF_MISS,
        "MuJoCo's own noslip changes nothing on this fixture — the seeded"
        " slam no longer exercises the pass",
    )
    assert_true(
        d_off <= TOL_ON,
        "implicitfast WITHOUT noslip disagrees with MuJoCo by "
        + String(d_off) + " — the defect is upstream of the pass",
    )
    assert_true(
        d_on <= TOL_ON,
        "implicitfast WITH noslip disagrees with MuJoCo by " + String(d_on)
        + " (tol " + String(TOL_ON) + ") — the pass is reading a factor the"
        " implicit step did not build (3bc98c55's shape), or has drifted",
    )


def test_the_fixtures_are_in_contact() raises:
    print("=== implicitfast noslip: the fixtures touch the floor ===")
    var ours = _ours(_chain("elliptic", True), _settled(_chain("elliptic", False)), 1)
    print("  elliptic ncon after 1 step:", ours.ncon)
    assert_true(ours.ncon > 0, "the settled chain is not on the floor")
    print("  PASS")


def test_elliptic_noslip_under_implicitfast() raises:
    print("=== implicitfast + elliptic noslip vs MuJoCo ===")
    _gate("elliptic")
    print("  PASS")


def test_pyramidal_noslip_under_implicitfast() raises:
    print("=== implicitfast + pyramidal noslip vs MuJoCo ===")
    _gate("pyramidal")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
