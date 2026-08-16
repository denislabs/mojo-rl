"""A condim-1 contact under the PYRAMIDAL cone, against MuJoCo.

WHY THIS FILE EXISTS

MuJoCo emits ONE row for a frictionless contact — `mjCNSTR_CONTACT_FRICTIONLESS`
(`efc_type == 5`), the pure normal row — and `2*(dim-1)` pyramid rows for a
frictional one. Our pyramidal builder emits `2*(dim-1)` rows unconditionally,
and at `dim == 1` that arithmetic is **zero rows**: the contact is detected,
recorded, and then contributes no constraint at all. The two bodies pass
through each other while `ncon` cheerfully reports the contact.

Measured on dm_control's dog at a settled pose: MuJoCo reports
`efc_type {6: 40, 5: 3, 3: 2}` — three of its thirteen contacts are
frictionless (`collision_primitive` sets `condim="1"`, and dog has 81 such
geoms), so three real constraints were missing from our solve.

⚠ WHY THE FIXTURE NEEDS `priority`, NOT JUST `condim="1"`

MuJoCo mixes condim as `max(condim1, condim2)` at equal priority. A condim-1
ball on the default condim-3 floor is a condim-3 contact and would gate
nothing. `priority="1"` makes the ball dictate condim wholesale, which is the
only way to get a frictionless contact against an ordinary floor — and it is
exactly how dog gets its own (there, both sides are condim-1 primitives).

THE MODEL CARRIES BOTH KINDS ON PURPOSE. One frictionless ball and one
ordinary condim-3 ball, so the fix is gated for what it adds AND for not
disturbing the fixed-stride edge layout the frictional contacts live in. A
model with only the new case would let a stride bug through.

NON-VACUITY. The test asserts MuJoCo really produces one row of each type
before comparing anything. A fixture where both balls came out condim 3 would
post a perfect number and prove nothing.

Run with:
    pixi run mojo run -I . tests/physics3d/test_frictionless_contact_pyramidal.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
)
from max.gpu.host import DeviceContext


comptime DTYPE = DType.float64

comptime FL_XML = """
<mujoco model="frictionless">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3" friction="1 0.005 0.0001"/>
    <body name="slick" pos="-0.4 0 0.14">
      <joint type="free" name="slick_root"/>
      <geom name="slick" size=".15" condim="1" priority="1"/>
    </body>
    <body name="grippy" pos="0.4 0 0.14">
      <joint type="free" name="grippy_root"/>
      <geom name="grippy" size=".15" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime pp = parse_xml(FL_XML)
comptime M = ModelDefFromXML[
    xml=FL_XML,
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

comptime N_SETTLE: Int = 200
comptime N_STEPS: Int = 60
# Both sides run the same float64 arithmetic from the same state, so the
# budget is round-off over the rollout, not a solver allowance. MEASURED with
# the fix in place: `|d(qpos)| = 4.4e-16`, `|d(qvel)| = 3.6e-15` over 60
# contacting steps. Budgeted three orders above that, which still fails the
# pre-fix behaviour (`0.072` / `1.18`) by eleven orders.
comptime TOL: Float64 = 1e-12


def _mj() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[FL_XML]())
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _settle(mujoco: PythonObject, m: PythonObject, md: PythonObject) raises:
    """Drop both balls and let MuJoCo settle them, then push them sideways.

    The sideways push is what separates the two contacts: a frictionless ball
    keeps sliding, a frictional one is arrested. Without it both would sit
    still and the condim would not matter.
    """
    mujoco.mj_resetData(m, md)
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, md)
    md.qvel[0] = 0.9
    md.qvel[7] = 0.9


def test_mujoco_really_emits_one_row_of_each_kind() raises:
    """Non-vacuity: the fixture must carry a frictionless AND a pyramidal
    contact, or everything below is a comparison of two empty sets."""
    print("--- frictionless: fixture check (a fact about MuJoCo) ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    _settle(mujoco, m, md)
    mujoco.mj_forward(m, md)

    var n_fl = 0
    var n_pyr = 0
    var nefc = Int(py=md.nefc)
    for i in range(nefc):
        var t = Int(py=md.efc_type[i])
        if t == 5:  # mjCNSTR_CONTACT_FRICTIONLESS
            n_fl += 1
        elif t == 6:  # mjCNSTR_CONTACT_PYRAMIDAL
            n_pyr += 1
    print("  ncon", Int(py=md.ncon), " nefc", nefc,
          " frictionless rows", n_fl, " pyramidal rows", n_pyr)
    assert_true(
        n_fl >= 1,
        "MuJoCo emitted no frictionless row — `priority` did not take, so the"
        " slick ball is a condim-3 contact and this file gates nothing",
    )
    assert_true(
        n_pyr >= 4,
        "MuJoCo emitted no pyramidal rows — the frictional ball is not"
        " touching, so the fixed-stride edge layout is untested",
    )


def test_frictionless_contact_matches_mujoco() raises:
    """Our rollout against MuJoCo's, with both contact kinds live."""
    var sf = M.make_spec_fields[DTYPE]()
    print("--- frictionless: ours vs MuJoCo ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    _settle(mujoco, m, md)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, Dims[nq=M.NQ, nv=M.NV, nbody=M.NBODY, max_contacts=M.MAX_CONTACTS, nsite=M.NSITE], 1]()
    M.reset_data[DTYPE](sf, d)

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

    # The slick ball slides for the whole rollout; anything less means it
    # never touched and the frictionless row was never built.
    assert_true(
        contact_steps > N_STEPS - 5,
        "the balls are not resting on the floor for the rollout — no contact"
        " rows of either kind were exercised",
    )
    assert_true(
        worst_q <= TOL,
        "qpos diverged — a frictionless (condim-1) contact under the pyramidal"
        " cone does not match MuJoCo",
    )
    assert_true(
        worst_v <= TOL,
        "qvel diverged — a frictionless (condim-1) contact under the pyramidal"
        " cone does not match MuJoCo",
    )


def test_frictionless_contact_records_no_tangential_force() raises:
    """The contact RECORD, which the rollout above cannot see.

    ⚠ THIS IS A SEPARATE CLAIM FROM THE ROLLOUT, and the rollout passing is no
    evidence for it. The frictionless row's Jacobian is the pure normal, so the
    SOLVE is frictionless no matter what the record says — `qacc`, `qpos` and
    `qvel` all stay exact while `contact.force` reads a friction that does not
    exist. Its consumers are `cfrc_ext` (hence any contact-cost reward term)
    and the force/touch sensors, and this repo has three prior instances of
    exactly that failure mode.

    The decode is `ft1 = (f_e0 - f_e1) * mu`, which cannot tell a frictionless
    contact's single live row from a pyramid pair whose negative edge happens
    to be zero. Measured on dm_control's dog before the guard: `ft1/f_n =
    0.9002` on all three of its frictionless contacts — precisely the model's
    default `friction="0.9"` — where MuJoCo reports exactly 0.
    """
    var sf = M.make_spec_fields[DTYPE]()
    print("--- frictionless: the contact record ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    var np = Python.import_module("numpy")
    _settle(mujoco, m, md)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, Dims[nq=M.NQ, nv=M.NV, nbody=M.NBODY, max_contacts=M.MAX_CONTACTS, nsite=M.NSITE], 1]()
    M.reset_data[DTYPE](sf, d)
    var sq = md.qpos.flatten().tolist()
    var sv = md.qvel.flatten().tolist()
    for i in range(M.NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=sq[i]))
    for i in range(M.NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=sv[i]))
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu"](d, mf)
    mujoco.mj_forward(m, md)

    var buf = np.zeros(6)
    var n_frictionless = 0
    var worst_tan = 0.0
    var worst_fn = 0.0
    var nc = Int(py=md.ncon)
    for c in range(nc):
        mujoco.mj_contactForce(m, md, c, buf)
        var o = c * CONTACT_SIZE
        var dim = Int(Float64(d.contacts.data[o + CONTACT_IDX_CONDIM]))
        var f_n = Float64(d.contacts.data[o + CONTACT_IDX_FORCE_N])
        var t1 = abs(Float64(d.contacts.data[o + CONTACT_IDX_FORCE_T1]))
        var t2 = abs(Float64(d.contacts.data[o + CONTACT_IDX_FORCE_T2]))
        print("   c", c, " dim", dim, " ours fn", f_n, " |t1|", t1, " |t2|", t2,
              "  MuJoCo fn", Float64(py=buf[0]),
              " t1", Float64(py=buf[1]), " t2", Float64(py=buf[2]))
        if dim == 1:
            n_frictionless += 1
            if t1 > worst_tan:
                worst_tan = t1
            if t2 > worst_tan:
                worst_tan = t2
            var dfn = abs(f_n - Float64(py=buf[0]))
            if dfn > worst_fn:
                worst_fn = dfn

    print("  frictionless contacts:", n_frictionless,
          " worst |tangential| =", worst_tan, " worst |d(fn)| =", worst_fn)

    # NON-VACUITY: if the slick ball is not touching, every number above is 0
    # and the assertions are satisfied by an empty set.
    assert_true(
        n_frictionless >= 1,
        "no condim-1 contact in the record — the slick ball is not touching,"
        " so the tangential assertion below is vacuous",
    )
    assert_true(
        worst_tan == 0.0,
        "a frictionless contact recorded a TANGENTIAL force — the pyramid"
        " decode `(f_e0 - f_e1) * mu` is being applied to a single normal row."
        " `qacc` is unaffected, so only the sensors and cfrc_ext are wrong,"
        " which is why a passing rollout says nothing about this",
    )
    assert_true(
        worst_fn <= TOL,
        "the frictionless NORMAL force disagrees with MuJoCo",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
