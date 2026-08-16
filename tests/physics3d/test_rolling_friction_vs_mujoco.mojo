"""condim=6 rolling friction vs MuJoCo — a spinning, rolling ball on a plane.

WHY THIS EXISTS. `condim=6` adds THREE constraint rows beyond the usual
normal + two slide: one TORSIONAL row about the contact normal, and two ROLLING
rows about the tangents t1 and t2. dm_control's quadruped `fetch` is the first
model in the tree to ask for 6, on its ball; everything else runs 1, 3 or 4.

⚠ THE CAPABILITY WAS TWICE REPORTED ABSENT during this arc, on the strength of
`CNSTR_FRICTION_ROLL1/2` appearing only in the constraint-type enum. Those
names are unused plumbing and the rows really are built — but building them was
not the same as USING them, which is what this file established:

  * the PGS env body in `constraints/contact_solve.mojo` did generalise over
    condim (its `ws_df` direction array carries 5 directions), yet
  * the PYRAMIDAL EDGE BUILDER that CG and Newton actually consume hardcoded
    FOUR edges per contact and only ever multiplied by `mu_slide`. It never
    read `condim`, `friction[2]` or `friction[3..4]` at all.

So under the production solver the torsional and rolling rows were built into a
workspace nothing downstream ever looked at. Measured here before the fix, over
40 steps from a 5.0 rad/s spin: MuJoCo shed 53% of it, we shed 4%. `condim=3`
agreed to 2e-15 the whole time, which is exactly why nothing caught it.

MuJoCo emits one OPPOSING PAIR per friction dimension — `2*(dim-1)` rows, with
row k built as `J_n ± friction[k-1] * J_k` (engine_core_constraint.c:1383).
Rows 3.. use the ANGULAR Jacobian rotated by the same contact frame, so
row 3 is torsion about n, rows 4/5 roll about t1/t2. All `2*(dim-1)` rows share
ONE regulariser `Rpy = 2*mu^2*R` (line 1899); the per-direction
`R[j] = R[1]*friction[0]^2/friction[j]^2` rescaling immediately above it is the
ELLIPTIC branch only, and applying it to the pyramid would make the spin row
(0.7/0.05)^2 = 196x too soft.

THE POSE loads all three angular rows at once:

  * `qvel[5] = 5.0`  — spin about world z, i.e. about the CONTACT NORMAL: the
    torsional row, the one condim=4 already covers.
  * `qvel[3] = 3.0`  — roll about x, a TANGENT direction: a rolling row.
  * `qvel[0] = 0.6`  — linear slide, so the two slide rows are live and the
    friction cone is not degenerate.

A ball on a plane touches at ONE point, so all of its resistance to spinning
comes from these rows: with them inert the ball spins essentially forever,
which is precisely what `fetch` would have exhibited.

⚠ `priority="1"` ON THE BALL IS LOAD-BEARING, not decoration. The floor
declares condim 3; without priority MuJoCo's equal-priority rule takes the MAX,
which is still 6, so the test would pass either way and silently stop covering
the priority path. With it, the ball's condim, friction AND solref are taken
wholesale — the configuration quadruped's ball actually uses.

⚠ THE BALL USES `<joint type="free">`, NOT `<freejoint>`. The `<freejoint>`
alias is normalised to that form by `merge_mjcf`, and this file builds straight
from `parse_xml` — so a `<freejoint>` here parses to NO JOINT AT ALL, giving
nq = nv = 0. It fails silently and late: every `for i in range(M.NV)` simply
does nothing and the first symptom is an empty list being indexed somewhere
unrelated, which reads like a lifetime bug. Any test that skips `merge_mjcf`
must spell the free joint out.

⚠ THE BALL IS SETTLED BY MuJoCo FIRST and BOTH engines start from the settled
qpos. Spawning at a hand-picked depth does not work: at 0.5 mm MuJoCo pushes
the ball out within ~20 steps, `ncon` drops to 0 and the rollout stops
exercising the rows under test while still reporting a small, reassuring error.

Run: pixi run mojo run -I . tests/physics3d/test_rolling_friction_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator


comptime DTYPE = DType.float64

comptime ROLL_XML = """
<mujoco model="rollball">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3" friction="1 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.15">
      <joint type="free" name="ball_root"/>
      <geom name="ball" size=".15" condim="6" priority="1" friction=".7 .05 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime pp = parse_xml(ROLL_XML)
comptime M = ModelDefFromXML[
    xml=ROLL_XML,
    nbody=pp.NBODY, njoint=pp.NJOINT, nq=pp.NQ, nv=pp.NV,
    ngeom=pp.NGEOM, nact=pp.NACT, ntex=pp.NTEX, nmat=pp.NMAT,
    nlight=pp.NLIGHT, ncam=pp.NCAM, nsite=pp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp.TIMESTEP,
    max_condim=pp.MAX_CONDIM,
]

comptime N_SETTLE: Int = 300
comptime N_STEPS: Int = 40
# Measured worst |d(qvel)| is 2.4e-15 — this is float64 round-off over 40 steps
# of a 10-row contact, not a solver budget. Left three orders loose so the gate
# reports a real regression rather than a recompilation.
comptime TOL: Float64 = 1e-12


def _mj() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[ROLL_XML]())
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _settled(mut mujoco: PythonObject, m: PythonObject, md: PythonObject)
        raises -> PythonObject:
    """Drop the ball under gravity so the contact carries real load."""
    md.qpos[2] = 0.15
    md.qpos[3] = 1.0  # free-joint qpos is [x,y,z, qw,qx,qy,qz] — w FIRST, and
    # a fresh MjData/Data is all zeros, i.e. a DEGENERATE (0,0,0,0) quaternion.
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, md)
    return md.qpos.flatten().tolist()


def test_condim6_contact_has_six_rows() raises:
    """MuJoCo must give this contact SIX dimensions, or the file gates nothing.

    Asserted before any comparison: if the priority rule or the condim parse
    regressed, the contact would silently drop to 3 or 4 dimensions and every
    number below would still agree — on a configuration that no longer
    exercises rolling friction at all.
    """
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    _ = _settled(mujoco, m, md)
    md.qvel[0] = 0.6
    md.qvel[3] = 3.0
    md.qvel[5] = 5.0
    mujoco.mj_forward(m, md)
    print("--- condim6: MuJoCo ncon", Int(py=md.ncon), " nefc", Int(py=md.nefc),
          " contact.dim", Int(py=md.contact[0].dim))
    assert_true(Int(py=md.ncon) == 1, "expected a single ball/plane contact")
    assert_true(
        Int(py=md.contact[0].dim) == 6,
        "the contact is not condim 6 — the ball's priority=1 or its condim"
        " stopped taking effect, and this file would gate nothing",
    )
    # ⚠ PYRAMIDAL, SO THE ROW COUNT IS 2*(dim-1) = 10, NOT 6. The elliptic
    # cone would give 6. Asserting 6 here is what the first draft did, and it
    # would have failed for a reason that has nothing to do with rolling
    # friction — `<option cone="pyramidal">` is set at the top of the model.
    assert_true(
        Int(py=md.nefc) == 10,
        "expected 2*(dim-1) = 10 pyramidal rows for one condim-6 contact,"
        " got " + String(Int(py=md.nefc)),
    )
    # The spin friction must actually reach the contact. friction is
    # [slide1, slide2, spin, roll1, roll2]; the ball's `.7 .05 .05` expands to
    # 0.7 0.7 0.05 0.05 0.05 and the floor's `1 0.005 0.0001` must lose.
    var fr = md.contact[0].friction.tolist()
    assert_true(
        abs(Float64(py=fr[2]) - 0.05) < 1e-15
        and abs(Float64(py=fr[3]) - 0.05) < 1e-15,
        "the ball's spin/roll friction did not win the priority mix",
    )


def test_rolling_friction_matches_mujoco() raises:
    """Forty steps of a spinning, rolling, sliding ball against MuJoCo.

    The ball's angular velocity is the quantity of interest: with the torsional
    and rolling rows inert it decays far too slowly, and `qvel[3..5]` is where
    that shows up first.
    """
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    var settled = _settled(mujoco, m, md)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, Dims[nq=M.NQ, nv=M.NV, nbody=M.NBODY, max_contacts=M.MAX_CONTACTS, nsite=M.NSITE], 1]()

    for i in range(M.NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=settled[i]))
    var v0 = [0.6, 0.0, 0.0, 3.0, 0.0, 5.0]
    for i in range(M.NV):
        d.qvel.data[i] = Scalar[DTYPE](v0[i])
        md.qvel[i] = v0[i]
    forward_kinematics["cpu"](d, mf)

    # ⚠ MAX_CONDIM IS WHAT SIZES THE PYRAMID, and it comes from `parse_xml`
    # scanning the XML rather than from a literal here — passing the wrong
    # number by hand IS the original defect, and a gate that hardcodes 6
    # would not notice the scanner regressing. It is the model's WORST
    # condim, not every contact's: the builder zeroes the tail per contact,
    # so the floor/ball pair spans all 10 edges while a condim-3 pair in the
    # same model would still span exactly 4. At the default 3 the extra rows
    # are built and silently dropped.
    assert_true(
        M.MAX_CONDIM == 6,
        "parse_xml did not pick up condim=6 from the XML (got "
        + String(M.MAX_CONDIM) + ") — the pyramid would be sized for 4 edges",
    )
    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM=M.MAX_CONDIM,
    ]()

    var worst_q = Float64(0)
    var worst_v = Float64(0)
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

    # ⚠ HOIST THE LIST. Indexing `md.qvel.flatten().tolist()[k]` inline reads
    # a temporary that is already gone by the subscript — it fails as an
    # out-of-bounds on an EMPTY list, which reads like a dimension bug rather
    # than a lifetime one. The loop above hoists it for the same reason.
    var mv_end = md.qvel.flatten().tolist()
    var spin_mj = Float64(py=mv_end[5])
    print("  contact on", contact_steps, "/", N_STEPS)
    print("  worst |d(qpos)| =", worst_q, "  worst |d(qvel)| =", worst_v)
    print("  spin qvel[5]:  ours", Float64(d.qvel.data[5]), " MuJoCo", spin_mj,
          "  (started at 5.0)")
    print("  roll qvel[3]:  ours", Float64(d.qvel.data[3]), " MuJoCo",
          Float64(py=mv_end[3]), "  (started at 3.0)")

    # NON-VACUITY 1: the rows must be LIVE for a decent part of the rollout.
    # Sliding and spinning make the ball climb out of its resting penetration
    # partway through, so this cannot demand all 40 — but a handful of steps
    # would let the whole comparison pass on ballistic flight, where the two
    # engines agree trivially.
    assert_true(
        contact_steps >= 12,
        "the ball spent only " + String(contact_steps) + " of "
        + String(N_STEPS) + " steps in contact — the rollout has stopped"
        " exercising the contact rows",
    )
    # NON-VACUITY 2: the angular rows must actually DO something. With them
    # inert the spin is essentially undamped (measured: 5.0 -> 4.81), so
    # requiring MuJoCo itself to have shed most of it is what makes the
    # agreement below meaningful.
    assert_true(
        abs(spin_mj) < 3.5,
        "MuJoCo barely damped the spin over this rollout, so agreeing with it"
        " says nothing about the torsional row — lengthen the rollout or raise"
        " the spin friction",
    )
    assert_true(
        worst_v <= TOL,
        "rolling/torsional friction diverges from MuJoCo: worst |d(qvel)| = "
        + String(worst_v),
    )
    assert_true(
        worst_q <= TOL,
        "qpos diverges from MuJoCo: worst |d(qpos)| = " + String(worst_q),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
