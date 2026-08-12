"""Narrow-phase branch coverage gate — every primitive pair type, BOTH
geom-index orderings, against MuJoCo.

WHY THIS EXISTS. The engine picks its narrow-phase branch from the ORDER the
two geoms arrive in. The REVERSED-ORDER branches call a primitive written for
the other operand order and negate its normal to compensate; until 2026-08-01
twelve of them ALSO swapped `body_a`/`body_b`, and that double flip left them
emitting `normal = body_b -> body_a` where the canonical-order branches emit
`body_a -> body_b` (bug 35, commit b2904d14).

A coverage audit of the whole suite afterwards found that of the six reversed
pair types the fix touched, only ONE (sphere/capsule) was exercised by any test
and MuJoCo-anchored, one (box/cylinder) was exercised by sawyer with nothing
but self-goldens frozen from the legacy kernels, and FOUR were never executed
at all — while remaining reachable by already-ported domains. This file closes
that gap permanently instead of auditing it once.

THE MODEL is synthetic on purpose: twelve isolated groups, one per (pair type,
ordering), spaced 1 m apart so no group can touch another, each pair set to a
5 mm penetration. Every body carries a slide joint — without one every body
would be welded to the world and MuJoCo would exclude every pair.

WHAT IS ASSERTED
  * NON-VACUITY: every group produces at least one contact, i.e. the branch
    actually fired. A silent zero here would make the rest of the file
    meaningless, which is how the phantom mesh goldens survived.
  * GEOMETRY: `dist` against MuJoCo per group.
  * THE DIRECTION INVARIANT, which is the whole point. MuJoCo's normal points
    `geom1 -> geom2`; ours points `body_b -> body_a`. So our normal must equal
    MuJoCo's when our `(body_a, body_b)` is `(mj_body2, mj_body1)` and its
    negation when it is `(mj_body1, mj_body2)`. Checking the SIGN against the
    BODY LABELS is what bug 35 violated — comparing normals alone would have
    passed.

  * PER-GROUP CONTACT COUNTS against MuJoCo with `mjDSBL_MULTICCD`, printed
    beside MuJoCo's default count so the manifold rows we do not produce stay
    visible rather than silently tolerated.
  * THE DEVICE LEGS (`test_narrow_phase_pairs_gpu_matches_cpu`): O(N^2)-GPU and
    SAP-GPU against the O(N^2)-CPU leg, which the asserts above have already
    anchored to MuJoCo.

⚠ THE DEVICE LEGS WERE MISSING UNTIL 2026-08-09, and that gap cost a week.
Every assertion here ran `detect_contacts["cpu"]` only, and the other detection
golden pins walker2d, whose contacts are all PLANE pairs — so the GPU O(N^2)
non-plane path was gated by NOTHING. When it stopped emitting contacts entirely
(defect 26, a compiler bug fixed by Mojo 1.0.0rc2), the only symptom anywhere in
the suite was a mesh gate reporting "vacuous", which was read as a stale-pose
problem. A CPU-only gate over a CPU/GPU dual-path engine is half a gate.

Run: pixi run mojo run -I . tests/physics3d/test_narrow_phase_pairs.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.collision.multi_ccd import (
    MULTICCD_PERTURBATION_ANGLE,
)
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
)


comptime DTYPE = DType.float64

comptime PAIRS_XML = """
<mujoco model="pairs">
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <geom friction="1 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>
    <body name="g0a" pos="0.0 0 0.5">
      <joint name="j0a" type="slide" axis="1 0 0"/>
      <geom name="c0a" type="sphere" size=".05"/>
    </body>
    <body name="g0b" pos="0.08499999999999999 0 0.5">
      <joint name="j0b" type="slide" axis="1 0 0"/>
      <geom name="c0b" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g1a" pos="1.0 0 0.5">
      <joint name="j1a" type="slide" axis="1 0 0"/>
      <geom name="c1a" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g1b" pos="1.085 0 0.5">
      <joint name="j1b" type="slide" axis="1 0 0"/>
      <geom name="c1b" type="sphere" size=".05"/>
    </body>
    <body name="g2a" pos="2.0 0 0.5">
      <joint name="j2a" type="slide" axis="1 0 0"/>
      <geom name="c2a" type="sphere" size=".05"/>
    </body>
    <body name="g2b" pos="2.095 0 0.5">
      <joint name="j2b" type="slide" axis="1 0 0"/>
      <geom name="c2b" type="box" size=".05 .05 .05"/>
    </body>
    <body name="g3a" pos="3.0 0 0.5">
      <joint name="j3a" type="slide" axis="1 0 0"/>
      <geom name="c3a" type="box" size=".05 .05 .05"/>
    </body>
    <body name="g3b" pos="3.095 0 0.5">
      <joint name="j3b" type="slide" axis="1 0 0"/>
      <geom name="c3b" type="sphere" size=".05"/>
    </body>
    <body name="g4a" pos="4.0 0 0.5">
      <joint name="j4a" type="slide" axis="1 0 0"/>
      <geom name="c4a" type="sphere" size=".05"/>
    </body>
    <body name="g4b" pos="4.095 0 0.5">
      <joint name="j4b" type="slide" axis="1 0 0"/>
      <geom name="c4b" type="cylinder" size=".05 .05"/>
    </body>
    <body name="g5a" pos="5.0 0 0.5">
      <joint name="j5a" type="slide" axis="1 0 0"/>
      <geom name="c5a" type="cylinder" size=".05 .05"/>
    </body>
    <body name="g5b" pos="5.095 0 0.5">
      <joint name="j5b" type="slide" axis="1 0 0"/>
      <geom name="c5b" type="sphere" size=".05"/>
    </body>
    <body name="g6a" pos="6.0 0 0.5">
      <joint name="j6a" type="slide" axis="1 0 0"/>
      <geom name="c6a" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g6b" pos="6.085 0 0.5">
      <joint name="j6b" type="slide" axis="1 0 0"/>
      <geom name="c6b" type="box" size=".05 .05 .05"/>
    </body>
    <body name="g7a" pos="7.0 0 0.5">
      <joint name="j7a" type="slide" axis="1 0 0"/>
      <geom name="c7a" type="box" size=".05 .05 .05"/>
    </body>
    <body name="g7b" pos="7.085 0 0.5">
      <joint name="j7b" type="slide" axis="1 0 0"/>
      <geom name="c7b" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g8a" pos="8.0 0 0.5">
      <joint name="j8a" type="slide" axis="1 0 0"/>
      <geom name="c8a" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g8b" pos="8.085 0 0.5">
      <joint name="j8b" type="slide" axis="1 0 0"/>
      <geom name="c8b" type="cylinder" size=".05 .05"/>
    </body>
    <body name="g9a" pos="9.0 0 0.5">
      <joint name="j9a" type="slide" axis="1 0 0"/>
      <geom name="c9a" type="cylinder" size=".05 .05"/>
    </body>
    <body name="g9b" pos="9.085 0 0.5">
      <joint name="j9b" type="slide" axis="1 0 0"/>
      <geom name="c9b" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g10a" pos="10.0 0 0.5">
      <joint name="j10a" type="slide" axis="1 0 0"/>
      <geom name="c10a" type="box" size=".05 .05 .05"/>
    </body>
    <body name="g10b" pos="10.095 0 0.5">
      <joint name="j10b" type="slide" axis="1 0 0"/>
      <geom name="c10b" type="cylinder" size=".05 .05"/>
    </body>
    <body name="g11a" pos="11.0 0 0.5">
      <joint name="j11a" type="slide" axis="1 0 0"/>
      <geom name="c11a" type="cylinder" size=".05 .05"/>
    </body>
    <body name="g11b" pos="11.095 0 0.5">
      <joint name="j11b" type="slide" axis="1 0 0"/>
      <geom name="c11b" type="box" size=".05 .05 .05"/>
    </body>
    <geom name="w12" type="sphere" size=".05" pos="12.0 0 0.5"/>
    <body name="g12b" pos="12.085 0 0.5">
      <joint name="j12b" type="slide" axis="1 0 0"/>
      <geom name="c12b" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g13a" pos="13.0 0 0.5">
      <joint name="j13a" type="slide" axis="1 0 0"/>
      <geom name="c13a" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <geom name="w13" type="sphere" size=".05" pos="13.085 0 0.5"/>
  </worldbody>
</mujoco>
"""

comptime pp = parse_xml(PAIRS_XML)
comptime PM = ModelDefFromXML[
    xml=PAIRS_XML,
    nbody=pp.NBODY, njoint=pp.NJOINT, nq=pp.NQ, nv=pp.NV,
    ngeom=pp.NGEOM, nact=pp.NACT, ntex=pp.NTEX, nmat=pp.NMAT,
    nlight=pp.NLIGHT, ncam=pp.NCAM, nsite=pp.NSITE,
    max_tendon=pp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp.TIMESTEP,
]

comptime NGEOM = PM.NGEOM
comptime NBODY = PM.NBODY
comptime MC = PM.MAX_CONTACTS
comptime NGROUPS = 14

# Gates set from the measured worst case with headroom. Measured 2026-08-01:
#   direction  <= 2.9e-12 for every pair EXCEPT box/cylinder
#   direction  == 2.18e-6 and dist == 1.09e-7 for box/cylinder, which is the
#              DOCUMENTED capsule-approximation of the cylinder in
#              `cylinder_box` (it reuses `box_capsule`, so the virtual capsule's
#              hemispherical caps extend slightly past the flat ones). That is a
#              modelling approximation, not a direction error — a direction
#              error shows up as ~2.0 on a unit vector, which is exactly what
#              this file measured before `cylinder_box`'s missing negation was
#              fixed.
# The split keeps the exact pairs exact: a regression on them cannot hide
# behind the approximate one.
comptime TOL_DIR: Float64 = 1e-10
comptime TOL_DIST: Float64 = 1e-12
comptime TOL_DIR_APPROX: Float64 = 1e-5
comptime TOL_DIST_APPROX: Float64 = 1e-6
# Defect 27 is FIXED (2026-08-12) and this is no longer a defect ratchet — it
# is an ordinary float32 CPU-vs-GPU tolerance. It was 0.2775, pinned to the
# divergence on the box/capsule SECOND manifold point; the measurement is now
# 1.943426397588155e-08, so this is set at the float32 rounding scale (~1 ulp
# at unit magnitude) rather than at that number, to leave room for ordinary
# device noise without ever readmitting a branch-level divergence.
comptime TOL_GPU_MANIFOLD: Float64 = 1e-7

comptime Dat = Data[DTYPE, PM.NQ, PM.NV, NBODY, MC, PM.NSITE, 1]
comptime Mod = Model[
    DTYPE, PM.NV, NBODY, PM.NJOINT, NGEOM, PM.MAX_EQUALITY, PM.MAX_TENDON,
    PM.NSITE, PM.NEXCLUDE, 0,
]

# The device legs need their OWN float32 instantiation: this fixture is float64
# for the MuJoCo comparison, and Metal rejects f64 outright — `air.sin.f64`,
# `air.sqrt.f64` and friends are "Metal-unsupported instructions", so a f64 GPU
# kernel does not fail at runtime, it fails to BUILD. Comparing f32-CPU against
# f32-GPU is the right comparison anyway: identical source and identical dtype,
# so the two are required to be bit-exact, and any difference is the device
# path. The MuJoCo anchoring stays on the f64 leg above.
comptime DTYPE32 = DType.float32
comptime Dat32 = Data[DTYPE32, PM.NQ, PM.NV, NBODY, MC, PM.NSITE, 1]
comptime Mod32 = Model[
    DTYPE32, PM.NV, NBODY, PM.NJOINT, NGEOM, PM.MAX_EQUALITY, PM.MAX_TENDON,
    PM.NSITE, PM.NEXCLUDE, 0,
]


def _group_names() -> List[String]:
    return [
        String("sphere/capsule"), String("capsule/sphere"),
        String("sphere/box"), String("box/sphere"),
        String("sphere/cylinder"), String("cylinder/sphere"),
        String("capsule/box"), String("box/capsule"),
        String("capsule/cylinder"), String("cylinder/capsule"),
        String("box/cylinder"), String("cylinder/box"),
        String("WORLD-first sphere/capsule"),
        String("WORLD-second capsule/sphere"),
    ]


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    PM.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_narrow_phase_pairs_vs_mujoco() raises:
    """Direction invariant + geometry, every pair type in both orderings."""
    print("--- narrow-phase pair coverage: NGEOM =", NGEOM, " NBODY =", NBODY)
    var ctx = DeviceContext()
    var mf = _build()
    var d = Dat()
    PM.reset_data(d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)

    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/physics3d")
    var pr = Python.import_module("narrow_phase_pairs_ref")
    var mujoco = Python.import_module("mujoco")
    var m = pr.model()
    var dat = mujoco.MjData(m)
    mujoco.mj_forward(m, dat)

    var names = _group_names()
    var n_ours = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var n_mj = Int(py=dat.ncon)
    print("  contacts: ours", n_ours, " MuJoCo", n_mj)

    # per-group bookkeeping: group g owns bodies 2g+1 and 2g+2
    var seen = List[Int]()
    for _ in range(NGROUPS):
        seen.append(0)

    var worst_dist = Float64(0)
    var worst_dir = Float64(0)
    for c in range(n_ours):
        var b = c * CONTACT_SIZE
        var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])
        var px = Float64(d.contacts.data[b + CONTACT_IDX_POS_X])
        var g = Int(px + 0.5)
        assert_true(
            g >= 0 and g < NGROUPS and abs(px - Float64(g)) < 0.3,
            String("contact ") + String(c) + " at x=" + String(px)
            + " is not inside any group — the groups are 1 m apart, so this is"
            " a broadphase or FK error, not a narrow-phase one",
        )
        seen[g] = seen[g] + 1

        var nx = Float64(d.contacts.data[b + CONTACT_IDX_NX])
        var ny = Float64(d.contacts.data[b + CONTACT_IDX_NY])
        var nz = Float64(d.contacts.data[b + CONTACT_IDX_NZ])
        var dist = Float64(d.contacts.data[b + CONTACT_IDX_DIST])

        # MuJoCo's contact for the same body pair, matched by POSITION.
        #
        # ⚠ THIS USED TO TAKE THE FIRST MATCH, on the reasoning that "the
        # manifold points agree in direction to ~1e-3". That held only while we
        # emitted a single point per pair. Multi-point convex contact makes both
        # sides emit up to five, and the perturbed rows carry DELIBERATELY
        # tilted normals — MuJoCo's own read [-1, 0, +-0.001] and [-1, +-0.001,
        # 0], which is the +-1e-3 rad perturbation showing through. Comparing
        # our second row against MuJoCo's first then fails by exactly 1e-3,
        # which is the perturbation, not an error.
        #
        # So every one of our rows is still compared against MuJoCo's FIRST row
        # for the pair — its PRIMARY, untilted contact — and the tolerance is
        # widened by the perturbation angle for our rows AFTER the first.
        #
        # ⚠ TWO PAIRINGS THAT LOOK BETTER AND ARE NOT, both tried and measured:
        #   * NEAREST BY POSITION. A cylinder/capsule manifold's rows sit ~1e-7
        #     apart (the tilt barely moves the point) so the choice among them
        #     is arbitrary; a box/cylinder manifold's rows are spread ACROSS THE
        #     FACE, centimetres apart, so position dominates any combined score.
        #     Our single box/cylinder row then paired with an off-centre tilted
        #     row and reported 9.96e-4 — the perturbation, not a defect.
        #   * BEST-AGREEING ROW (minimise the direction error itself). That is
        #     self-fulfilling: "find the row that agrees best, then assert it
        #     agrees" cannot fail while any row is close, so it would mask a
        #     real error smaller than the perturbation.
        # Comparing against the primary keeps the ORIGINAL strong gate on the
        # contact that carries the physics, and asks of the extra rows only
        # that they be within a perturbation of it — which is exactly what
        # MuJoCo's own rows are.
        var mj_n0 = Float64(0)
        var mj_n1 = Float64(0)
        var mj_n2 = Float64(0)
        var mj_d = Float64(0)
        var mj_b1 = -1
        var mj_b2 = -1
        for k in range(n_mj):
            var cc = dat.contact[k]
            var q1 = Int(py=m.geom_bodyid[cc.geom1])
            var q2 = Int(py=m.geom_bodyid[cc.geom2])
            if (q1 == ba and q2 == bb) or (q1 == bb and q2 == ba):
                mj_b1 = q1
                mj_b2 = q2
                mj_n0 = Float64(py=cc.frame[0])
                mj_n1 = Float64(py=cc.frame[1])
                mj_n2 = Float64(py=cc.frame[2])
                mj_d = Float64(py=cc.dist)
                break
        # `seen[g]` was incremented above, so 1 means this is our primary row.
        var is_primary = (seen[g] == 1)
        assert_true(
            mj_b1 >= 0,
            String("group ") + names[g] + ": we emit a contact for bodies "
            + String(ba) + "," + String(bb) + " and MuJoCo emits none",
        )

        var dd = abs(dist - mj_d)
        if dd > worst_dist:
            worst_dist = dd

        # THE INVARIANT. Ours points body_b -> body_a; MuJoCo's geom1 -> geom2.
        var expect_same = (mj_b1 == bb and mj_b2 == ba)
        var sgn = Float64(1.0) if expect_same else Float64(-1.0)
        var ex = abs(nx - sgn * mj_n0)
        var ey = abs(ny - sgn * mj_n1)
        var ez = abs(nz - sgn * mj_n2)
        var e = max(ex, max(ey, ez))
        if e > worst_dir:
            worst_dir = e
        # box/cylinder reduces the cylinder to a capsule (documented in
        # `cylinder_box`), so it gets the looser pair; everything else is exact.
        var is_world = (ba == 0 or bb == 0)
        if is_world:
            print("      WORLD group: ours a", ba, "b", bb,
                  " our n [", nx, ny, nz, "]  mj n [", mj_n0, mj_n1, mj_n2,
                  "]  mj bodies", mj_b1, mj_b2)
        # ⚠ GROUPS 10/11 KEEP A LOOSER BOUND, BUT THE REASON CHANGED. It used
        # to be that `cylinder_box` APPROXIMATED the cylinder as a capsule.
        # That is gone — they now run GJK+EPA like MuJoCo's own dispatch — and
        # I tightened them to the closed-form bound on that basis, which
        # FAILED: measured distance error 1.09e-07 against TOL_DIST = 1e-12.
        #
        # The right reason is that these two are now ITERATIVE rather than
        # closed-form. Every other group is solved analytically and lands at
        # machine precision; EPA converges to `EPA_TOLERANCE` (1e-8), so its
        # residual is bounded by the algorithm's own stopping rule, not by
        # float64. 1e-6 sits two orders above the measured 1.09e-07 and six
        # orders below the millimetre scale this file is testing.
        var iterative = (g == 10 or g == 11)
        var approx = iterative
        var tol_d = TOL_DIST_APPROX if approx else TOL_DIST
        var tol_n = TOL_DIR_APPROX if approx else TOL_DIR
        # A perturbed manifold row is tilted off the primary normal BY
        # CONSTRUCTION — MuJoCo's own extra rows read [+-1, 0, +-0.001]. So a
        # row after the first is allowed that much and no more; 1.5x leaves
        # room for the tilt landing on two axes at once without letting a real
        # error through, since the next thing bigger than the perturbation is a
        # gross one. The primary row keeps the original tolerance untouched.
        if not is_primary:
            tol_n += 1.5 * MULTICCD_PERTURBATION_ANGLE
        print("   ", names[g], " bodies", ba, bb, " mj", mj_b1, mj_b2,
              " expect_same", expect_same, " dist err", dd, " dir err", e)
        assert_true(
            dd <= tol_d,
            String("group ") + names[g] + ": dist diverges from MuJoCo",
        )
        assert_true(
            e <= tol_n,
            String("group ") + names[g] + ": CONTACT DIRECTION diverges from"
            " MuJoCo. An error near 2.0 on a unit vector is a full reversal —"
            " check that the narrow-phase branch negates XOR swaps the bodies"
            " (never both), and that any primitive which delegates with"
            " SWAPPED operands negates the normal it forwards.",
        )

    print("  worst dist err =", worst_dist, "  worst direction err =", worst_dir)
    assert_true(
        n_ours >= NGROUPS,
        "fewer contacts than groups — some pair type produced none",
    )
    for g in range(NGROUPS):
        assert_true(
            seen[g] > 0,
            String("group ") + names[g] + " produced NO contact — that branch"
            " is not being exercised, so this file gates nothing for it",
        )

    # ── MANIFOLD COUNTS — defect 21 ──────────────────────────────────────
    #
    # This file used to print the two totals and say "counts are not
    # compared". That was a real blind spot: MuJoCo routes every
    # cylinder/ellipsoid/mesh pair to `mjc_Convex`, which returns a MULTI-POINT
    # manifold, and we dispatch those to single-point primitives. The whole
    # difference was invisible to CI, and it is what makes a dog resting a
    # forelimb on fetch's `target` CYLINDER sink to 1.5e-2 m of penetration
    # where MuJoCo holds 3.5e-4.
    #
    # ⚠ MIND THE FLAG'S POLARITY, IT INVERTED. The 3.6.0 tree has
    # `mjENBL_MULTICCD` (1<<4), OPT-IN. The 3.10.0 runtime has no such enable
    # bit — it has `mjDSBL_MULTICCD` (1<<19), a DISABLE bit, so the behaviour
    # is default-ON and you opt out. Searching only `mjtEnableBit`, finding
    # nothing and concluding "unconditional" is the mistake here; check BOTH
    # enums. `feedback_reference_tree_version_drift`.
    #
    # ⚠⚠ AND THE GATE IS AN EQUALITY, NOT A TOLERANCE, because the reference is
    # not a hand-recorded number that rots — it is MuJoCo itself, run twice.
    # `mj_multi` is DEFAULT MuJoCo, which is what the runtime does and what we
    # are now supposed to match. `mj_nomulti` is MuJoCo with `mjDSBL_MULTICCD`
    # set, which is what this engine emitted BEFORE multi-point convex contact
    # landed — kept because it is the exact statement of what that feature is
    # worth, and because a regression that silently disabled the manifold would
    # land right back on it.
    var mujoco_mod = Python.import_module("mujoco")
    var m_no = pr.model()
    m_no.opt.disableflags = mujoco_mod.mjtDisableBit.mjDSBL_MULTICCD
    var dat_no = mujoco_mod.MjData(m_no)
    mujoco_mod.mj_forward(m_no, dat_no)

    var mj_multi = List[Int]()
    var mj_nomulti = List[Int]()
    for _ in range(NGROUPS):
        mj_multi.append(0)
        mj_nomulti.append(0)
    # Groups are 1 m apart in x, so a contact's own x names its group — the
    # same mapping the ours-side loop above uses.
    for k in range(n_mj):
        var gx = Int(Float64(py=dat.contact[k].pos[0]) + 0.5)
        if gx >= 0 and gx < NGROUPS:
            mj_multi[gx] = mj_multi[gx] + 1
    var n_mj_no = Int(py=dat_no.ncon)
    for k in range(n_mj_no):
        var gx = Int(Float64(py=dat_no.contact[k].pos[0]) + 0.5)
        if gx >= 0 and gx < NGROUPS:
            mj_nomulti[gx] = mj_nomulti[gx] + 1

    print(
        "  manifold counts — ours", n_ours, " MuJoCo(multiCCD off)", n_mj_no,
        " MuJoCo(default)", n_mj,
    )
    var deficit = 0
    for g in range(NGROUPS):
        deficit += mj_multi[g] - seen[g]
        # Groups 10 and 11 are box/cylinder and cylinder/box — the pair whose
        # primitive reduces the cylinder to a capsule. Same two groups the
        # tolerance block above calls `approx`.
        # Label kept for the two GJK/EPA groups, but they are no longer
        # APPROXIMATE — they are ITERATIVE. See the tolerance note above.
        var approx_group = (g == 10 or g == 11)
        print(
            "   ", names[g], "  ours", seen[g],
            " mj_nomulti", mj_nomulti[g], " mj_default", mj_multi[g],
            "  (iterative)" if approx_group else "",
        )
        # THE CONTRACT: our narrow phase must equal DEFAULT MuJoCo, group by
        # group — except on the two groups whose PRIMITIVE is an approximation,
        # which are the same two this file already gives a looser distance and
        # direction tolerance.
        #
        # ⚠ THIS EXCEPTION IS RETIRED, AND IT RETIRED ITSELF. It used to read:
        # "box/cylinder and cylinder/box reach 4 of MuJoCo's 5 ... recording 4
        # rather than widening the count to '4 or 5': the missing row is the
        # approximation's residue, and IT SHOULD FAIL HERE THE DAY
        # `cylinder_box` BECOMES EXACT." That day arrived — cylinder/box now
        # goes through `mjc_Convex`'s GJK+EPA like MuJoCo's own dispatch, the
        # capsule reduction is gone, and both groups emit 5 of 5. The assert
        # fired exactly as designed, on an improvement.
        #
        # Every group now equals DEFAULT MuJoCo with no exceptions.
        var want = mj_multi[g]
        assert_true(
            seen[g] == want,
            String("group ") + names[g] + ": we emit " + String(seen[g])
            + " contacts, expected " + String(want) + " (default MuJoCo "
            + String(mj_multi[g]) + ", MuJoCo with mjDSBL_MULTICCD "
            + String(mj_nomulti[g]) + "). If ours matches the mjDSBL_MULTICCD"
            " number, multi-point convex contact is not running for this pair —"
            " check `multi_ccd_pair_supported`. If it is one BELOW default on a"
            " non-approximate pair, a perturbed re-query is being rejected as"
            " non-distinct. See defect 21.",
        )
    print("  manifold rows we do not produce =", deficit, " (defect 21)")

    # NON-VACUITY: the manifold has to be doing something, or every assert
    # above would be satisfied by the old single-point engine.
    var multi_rows = 0
    for g in range(NGROUPS):
        multi_rows += mj_multi[g] - mj_nomulti[g]
    assert_true(
        multi_rows > 0,
        "no group in this fixture has a multi-point manifold, so nothing here"
        " gates multi-point convex contact at all",
    )


def test_narrow_phase_pairs_gpu_matches_cpu() raises:
    """The SAME fixture through the GPU kernels. This is the leg that was
    missing, and its absence cost a week.

    ⚠ EVERY MuJoCo-anchored narrow-phase assertion in this file ran
    `detect_contacts["cpu"]` ONLY, and the other detection golden
    (`test_contact_detection_fields`) pins walker2d, whose contacts are all
    PLANE pairs. So the GPU O(N^2) non-plane path was gated by nothing at all —
    and when it stopped emitting contacts entirely (defect 26: the kernel
    silently stopped writing once `3dbc4c33` grew it, a compiler bug fixed by
    Mojo 1.0.0rc2), the only symptom anywhere in the suite was
    `test_mesh_detection_fields` reporting "gate is vacuous", which was read as
    a stale-pose problem for a week.

    Both device legs are compared against the CPU leg rather than against
    MuJoCo again: the CPU leg is already MuJoCo-anchored above, so agreement
    with it inherits that anchor, and a device path that disagrees with its own
    CPU source is a port bug regardless of what MuJoCo says.

      * O(N^2) GPU vs O(N^2) CPU — same source, same emission order, so this is
        BIT-EXACT on the count and on every populated record column.
      * SAP GPU vs O(N^2) CPU — as contact SETS. SAP sweeps in aabb_min_x
        order, so record indices do not correspond. This fixture has no plane,
        so the BODY_B world convention (0 vs -1) does not enter.
    """
    print("--- narrow-phase pair coverage: GPU legs vs the CPU leg")
    var ctx = DeviceContext()
    var mf = Mod32()
    PM.init_fields[DTYPE32, 0](ctx, mf)

    var dc = Dat32()
    PM.reset_data(dc)
    forward_kinematics["cpu"](dc, mf)
    detect_contacts["cpu"](dc, mf)
    var n_cpu = Int(dc.meta.data[META_IDX_NUM_CONTACTS])

    # NON-VACUITY FIRST. Everything below is trivially satisfied by 0 == 0.
    assert_true(
        n_cpu > 0,
        "CPU leg produced no contacts, so the GPU comparison is vacuous",
    )

    var dg = Dat32()
    PM.reset_data(dg)
    dg.upload_all(ctx)
    forward_kinematics["gpu"](dg, mf, ctx)
    detect_contacts["gpu"](dg, mf, ctx)
    dg.contacts.download(ctx)
    dg.meta.download(ctx)
    var n_gpu = Int(dg.meta.data[META_IDX_NUM_CONTACTS])
    print("  O(N^2): CPU", n_cpu, " GPU", n_gpu)
    assert_true(
        n_cpu == n_gpu,
        "O(N^2) GPU emitted "
        + String(n_gpu)
        + " contacts where CPU emitted "
        + String(n_cpu)
        + " — same `_detect_contacts_env` source, so this is a device-path bug",
    )

    var worst = Float64(0)
    var worst_c = -1
    var worst_k = -1
    for c in range(n_cpu):
        for k in range(CONTACT_SIZE):
            var e = abs(
                Float64(dc.contacts.data[c * CONTACT_SIZE + k])
                - Float64(dg.contacts.data[c * CONTACT_SIZE + k])
            )
            if e > worst:
                worst = e
                worst_c = c
                worst_k = k
    print("  O(N^2) worst CPU-vs-GPU record delta:", worst)
    # ⚠ OPEN DEFECT 27, pinned rather than hidden. The FIRST contact of every
    # pair agrees bit-exactly; what diverges is the SECOND manifold point of
    # the box/capsule group (bodies 15/16), measured 2026-08-09:
    #     column 6 (NY)   CPU -1.0643675096844163e-07   GPU -0.2747207283973694
    #     column 8 (DIST) CPU -0.004999961704015732     GPU -0.0035994164645671844
    # ✅ DEFECT 27 — FIXED 2026-08-12. It was a METAL PER-THREAD ARRAY
    # MISCOMPUTE, not a collision-algorithm bug. `_capsule_box_second_pos`
    # held `s`/`hax`/`pos`/`axis` as `InlineArray[Scalar, 3]` and indexed them
    # by a runtime axis; on Metal the value arrived correctly and read back
    # WRONG. Measured from the live GPU run with the parameter and the array
    # element smuggled out side by side through this very record:
    #
    #     hax_y (param)   CPU 0.059999994933605194   GPU 0.059999994933605194
    #     hax[1] (array)  CPU 0.059999994933605194   GPU -0.0
    #
    # `e1 = 2*s/|hax|` then became nan, BOTH `if e1 < secondpos` clamps
    # silently failed, and `secondpos` kept its initial `1 - bestsegmentpos`
    # (1.8333 instead of 1.6667) — the 0.27 swing in a unit normal. Fixed by
    # reading components inline via `_sel3`; delta 0.2747206219606184 ->
    # 1.943426397588155e-08, contact COUNT unchanged at 31, MuJoCo-anchored
    # numbers unmoved.
    #
    # ⚠⚠ WHAT IT COST, because the pattern is worth more than the fix. THREE
    # fixes were aimed at INFERRED locations and all three changed nothing:
    # an `axisdir` sign tie-break; a relative tie margin on the edge
    # comparison (that one is a real observation — `d2 < best_d2 - 1e-15` IS a
    # no-op in float32 — but not this defect); and reasoning from a
    # reconstruction of the call. A probe that called the same function in
    # ISOLATION with reconstructed inputs AGREED, a FALSE NEGATIVE, because
    # the reconstruction's `clcorner` was 0 where the real call's is 1.
    # Only instrumenting the LIVE call found it. `print` is unavailable inside
    # a Metal kernel, but a kernel already writes a downloadable contact
    # record — so temporarily overwriting the second point's pos/normal
    # columns smuggles internals out.
    #
    # The tolerance below is set FROM THAT MEASUREMENT so this is a RATCHET: it
    # passes today and fails the moment the divergence grows or spreads to
    # another column. It is deliberately NOT loose enough to hide the class of
    # bug this test was added for — a device path that emits the wrong NUMBER
    # of contacts, or none at all, is caught by the count assert above, which
    # is exactly what defect 26 would have tripped.
    if worst > 0.0:
        # WHICH column moved is the whole diagnosis — a body id, a normal
        # component and a depth fail for completely different reasons.
        print(
            "    at contact", worst_c, "column", worst_k,
            " CPU", Float64(dc.contacts.data[worst_c * CONTACT_SIZE + worst_k]),
            " GPU", Float64(dg.contacts.data[worst_c * CONTACT_SIZE + worst_k]),
        )
        for c in range(n_cpu):
            var bc = c * CONTACT_SIZE
            print(
                "    c", c,
                " CPU bodies(", Int(dc.contacts.data[bc + CONTACT_IDX_BODY_A]),
                ",", Int(dc.contacts.data[bc + CONTACT_IDX_BODY_B]), ")",
                " GPU bodies(", Int(dg.contacts.data[bc + CONTACT_IDX_BODY_A]),
                ",", Int(dg.contacts.data[bc + CONTACT_IDX_BODY_B]), ")",
                " dist CPU", Float64(dc.contacts.data[bc + CONTACT_IDX_DIST]),
                " GPU", Float64(dg.contacts.data[bc + CONTACT_IDX_DIST]),
            )
    # First contact of each body pair must be BIT-EXACT — that is the part with
    # no known divergence, and keeping it exact is what makes the ratchet mean
    # something.
    var seen_pair = List[Int]()
    var worst_first = Float64(0)
    for c in range(n_cpu):
        var bc = c * CONTACT_SIZE
        var ba = Int(dc.contacts.data[bc + CONTACT_IDX_BODY_A])
        var bb = Int(dc.contacts.data[bc + CONTACT_IDX_BODY_B])
        var key = ba * 1000 + bb
        var first = True
        for s in seen_pair:
            if s == key:
                first = False
        if not first:
            continue
        seen_pair.append(key)
        for k in range(CONTACT_SIZE):
            var e = abs(
                Float64(dc.contacts.data[bc + k])
                - Float64(dg.contacts.data[bc + k])
            )
            if e > worst_first:
                worst_first = e
    print("  O(N^2) worst delta on FIRST contact of each pair:", worst_first)
    assert_true(
        worst_first == 0.0,
        "the first contact of a pair differs between CPU and GPU by "
        + String(worst_first)
        + " — identical source and dtype, so this must be bit-exact",
    )
    assert_true(
        worst <= TOL_GPU_MANIFOLD,
        "O(N^2) GPU records differ from CPU by " + String(worst)
        + ", above the pinned defect-27 ratchet " + String(TOL_GPU_MANIFOLD)
        + " — the divergence has grown or spread",
    )

    # ---- SAP GPU, matched as SETS by unordered body pair ----
    var ds = Dat32()
    PM.reset_data(ds)
    ds.upload_all(ctx)
    forward_kinematics["gpu"](ds, mf, ctx)
    detect_contacts_sap["gpu"](ds, mf, ctx)
    ds.contacts.download(ctx)
    ds.meta.download(ctx)
    var n_sap = Int(ds.meta.data[META_IDX_NUM_CONTACTS])
    print("  SAP GPU contacts:", n_sap)
    assert_true(
        n_sap == n_cpu,
        "SAP GPU emitted " + String(n_sap) + " contacts where the O(N^2) CPU"
        " leg emitted " + String(n_cpu),
    )

    var worst_sap = Float64(0)
    for c in range(n_cpu):
        var bc = c * CONTACT_SIZE
        var ba = Int(dc.contacts.data[bc + CONTACT_IDX_BODY_A])
        var bb = Int(dc.contacts.data[bc + CONTACT_IDX_BODY_B])
        var matched = False
        for s in range(n_sap):
            var bs = s * CONTACT_SIZE
            var sa = Int(ds.contacts.data[bs + CONTACT_IDX_BODY_A])
            var sb = Int(ds.contacts.data[bs + CONTACT_IDX_BODY_B])
            if not ((sa == ba and sb == bb) or (sa == bb and sb == ba)):
                continue
            matched = True
            var e = abs(
                Float64(dc.contacts.data[bc + CONTACT_IDX_DIST])
                - Float64(ds.contacts.data[bs + CONTACT_IDX_DIST])
            )
            if e > worst_sap:
                worst_sap = e
            break
        assert_true(
            matched,
            "SAP has no contact for body pair (" + String(ba) + ", "
            + String(bb) + ") that the O(N^2) leg emitted",
        )
    print("  SAP vs O(N^2) worst dist delta:", worst_sap)
    assert_true(
        worst_sap < TOL_DIST_APPROX,
        "SAP dist differs from O(N^2) by " + String(worst_sap),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
