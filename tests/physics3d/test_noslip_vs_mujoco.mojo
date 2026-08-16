"""`mj_solNoSlip` vs MuJoCo — a sliding, spinning ball with noslip enabled.

The pass is a friction-only Gauss-Seidel sweep run AFTER the primal solve, with
the normal forces held fixed. dm_control's dog is the only suite model that
asks for it, and this file is the cheap gate that proves the implementation
before dog's ~30-minute build does.

⚠ WHAT THIS FILE CAN AND CANNOT PROVE — read before trusting it.

Measured, MuJoCo against MuJoCo with only `noslip_iterations` changed on this
model: `max|d(qvel)|` is **8.9e-16**. The pass is INERT here, and that is not a
bad fixture, it is the physics: with one contact and a solver that converges to
machine precision there is no residual slip left to remove. Making it bite on a
model this small needs `iterations=2`, which sends the box flying (contact
drops to 5 of 40 steps) — a degenerate rollout that would gate nothing either.

So this file gates the half it can gate, and says so:

  * the option is PARSED and reaches the model def;
  * the pass COMPILES and RUNS through the real `Phyics3dEnv`/integrator path;
  * and — a real invariant, not a consolation prize — running it does NOT
    perturb an already-converged solve. A sweep that corrupted the solution
    would show up here immediately as a divergence from MuJoCo.

THE OTHER HALF — that the sweep computes the RIGHT thing when it actually acts
— is gated on dm_control's dog, where it moves `qvel` by 2.9e-2 on the first
contacting step. There is no small model in this repo that exercises it
honestly, and inventing one by detuning the solver would test a configuration
nothing runs.

WHAT THE POSE IS FOR

noslip only ever moves the FRICTION rows, so the pose has to load them:

  * `qvel[0] = 1.2` — a linear slide, so both slide rows are live and away
    from the cone's apex.
  * `qvel[5] = 4.0` — spin about the contact normal (the torsional row).
  * `qvel[3] = 2.0` — roll about a tangent (a rolling row).

and `condim="6"` on the ball with `priority="1"` makes all five friction
dimensions real rather than clipped to the floor's condim 3.

TOLERANCE. Both sides are float64 running the same arithmetic on the same
state, so the budget is round-off over the rollout, not a solver allowance.
Set from the measured value with three orders of headroom, per the standing
rule that an inherited tolerance is a placeholder.

Run with:
    pixi run mojo run -I . tests/physics3d/test_noslip_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from max.gpu.host import DeviceContext


comptime DTYPE = DType.float64

# `noslip_iterations="4"` is dog's own setting, so this exercises the same
# iteration count the real consumer uses.
comptime SLIP_XML = """
<mujoco model="slipball">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"
          noslip_iterations="4"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3" friction="1 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.15">
      <joint type="free" name="ball_root"/>
      <geom name="ball" size=".15" condim="6" priority="1" friction=".7 .05 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime pp = parse_xml(SLIP_XML)
comptime M = ModelDefFromXML[
    xml=SLIP_XML,
    nbody=pp.NBODY, njoint=pp.NJOINT, nq=pp.NQ, nv=pp.NV,
    ngeom=pp.NGEOM, nact=pp.NACT, ntex=pp.NTEX, nmat=pp.NMAT,
    nlight=pp.NLIGHT, ncam=pp.NCAM, nsite=pp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp.TIMESTEP,
    max_condim=pp.MAX_CONDIM,
    noslip_iter=pp.NOSLIP_ITER,
]

comptime N_SETTLE: Int = 300
comptime N_STEPS: Int = 40
comptime TOL: Float64 = 1e-11


def _mj(noslip: Int = -1) raises -> PythonObject:
    """MuJoCo from the same XML. `noslip >= 0` overrides the iteration count."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[SLIP_XML]())
    if noslip >= 0:
        m.opt.noslip_iterations = noslip
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _settle(mujoco: PythonObject, m: PythonObject, md: PythonObject) raises:
    """Drop the ball and let MuJoCo settle it, then load the friction rows.

    Starting from a settled pose rather than a fresh `Data` avoids the
    degenerate `(0,0,0,0)` free-joint quaternion a zeroed `Data` carries.
    """
    mujoco.mj_resetData(m, md)
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, md)
    md.qvel[0] = 1.2
    md.qvel[3] = 2.0
    md.qvel[5] = 4.0


def test_noslip_is_parsed() raises:
    """`<option noslip_iterations="4">` reaches the model def."""
    print("--- noslip: the option is parsed ---")
    print("  NOSLIP_ITER =", M.NOSLIP_ITER, " MAX_CONDIM =", M.MAX_CONDIM)
    assert_true(
        M.NOSLIP_ITER == 4,
        "parse_xml did not pick up noslip_iterations=4 — the pass would be"
        " compiled out entirely and every comparison below would be a"
        " no-op-vs-no-op",
    )
    assert_true(M.MAX_CONDIM == 6, "condim=6 did not reach the model def")


def test_noslip_is_inert_on_a_converged_solve() raises:
    """MuJoCo vs MuJoCo, only the option changed — here they must AGREE.

    This pins the scope of the file. It is a measurement of MuJoCo, and it says
    the pass has nothing to do on this model, which is why
    `test_noslip_matches_mujoco` below cannot distinguish a correct sweep from
    an empty one. If this number ever grows, this fixture HAS become
    discriminating and the assertion should be inverted — but until then,
    claiming otherwise would be the same self-deception as a parity gate whose
    tolerance is nine orders too loose.
    """
    print("--- noslip: scope check (a fact about MuJoCo) ---")
    var np = Python.import_module("numpy")
    var on = _mj(4)
    var off = _mj(0)
    var mujoco = on[0]
    _settle(mujoco, on[1], on[2])
    _settle(mujoco, off[1], off[2])
    var d_on = on[2]
    var d_off = off[2]

    var worst = 0.0
    for _ in range(N_STEPS):
        mujoco.mj_step(on[1], d_on)
        mujoco.mj_step(off[1], d_off)
        var e = Float64(py=np.abs(np.subtract(d_on.qvel, d_off.qvel)).max())
        if e > worst:
            worst = e
    print("  max |d(qvel)| between noslip 4 and 0 =", worst,
          " (expected ~1e-15: nothing to remove on a converged solve)")
    assert_true(
        worst < 1e-9,
        "noslip now CHANGES MuJoCo's answer on this model — this fixture has"
        " become discriminating, so `test_noslip_matches_mujoco` below is now"
        " a real parity gate and this assertion should be inverted to"
        " `worst > 1e-4`",
    )


def test_noslip_matches_mujoco() raises:
    """Our rollout against MuJoCo's, both with the pass on."""
    var sf = M.make_spec_fields[DTYPE]()
    print("--- noslip: ours vs MuJoCo ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    _settle(mujoco, m, md)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](sf, d)

    # Start from MuJoCo's settled state so the comparison is about the sweep,
    # not about how each side reached the pose.
    var sq = md.qpos.flatten().tolist()
    var sv = md.qvel.flatten().tolist()
    for i in range(M.NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=sq[i]))
    for i in range(M.NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=sv[i]))
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()

    var worst_q = 0.0
    var worst_v = 0.0
    var contact_steps = 0
    for _s in range(N_STEPS):
        for i in range(M.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(m, md)
        if Int(py=md.ncon) > 0:
            contact_steps += 1
        var mq = md.qpos.flatten().tolist()
        var mv = md.qvel.flatten().tolist()
        for i in range(M.NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=mq[i]))
            if e > worst_q:
                worst_q = e
        for i in range(M.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv[i]))
            if e > worst_v:
                worst_v = e

    print("  contact on", contact_steps, "/", N_STEPS, "steps")
    print("  worst |d(qpos)| =", worst_q, "  worst |d(qvel)| =", worst_v)
    # The ball bounces, so it is airborne for part of the rollout. What
    # matters is that contact happens AT ALL, since the airborne steps gate
    # the smooth dynamics rather than the solver.
    assert_true(
        contact_steps > 5,
        "the ball never really contacts the floor — the rollout exercises no"
        " contact rows at all",
    )
    assert_true(worst_q <= TOL, "qpos diverged from MuJoCo under noslip")
    assert_true(worst_v <= TOL, "qvel diverged from MuJoCo under noslip")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
