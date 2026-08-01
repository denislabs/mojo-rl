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

⚠ Contact COUNTS are not compared. MuJoCo emits multi-point manifolds for
capsule/cylinder and box/cylinder (5 points) and box/capsule (2); our narrow
phase is single-point there. That is a known scope difference, not a direction
bug, so this file gates direction and geometry and leaves manifold richness
alone.

Run: pixi run mojo run -I . tests/physics3d/test_narrow_phase_pairs.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
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
comptime NGROUPS = 12

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

comptime Dat = Data[DTYPE, PM.NQ, PM.NV, NBODY, MC, PM.NSITE, 1]
comptime Mod = Model[
    DTYPE, PM.NV, NBODY, PM.NJOINT, NGEOM, PM.MAX_EQUALITY, PM.MAX_TENDON,
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
    print("  contacts: ours", n_ours, " MuJoCo", n_mj,
          " (MuJoCo emits multi-point manifolds; counts are not compared)")

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
        var g = (ba - 1) // 2
        assert_true(
            g >= 0 and g < NGROUPS and (bb - 1) // 2 == g,
            String("contact ") + String(c) + " spans groups (bodies "
            + String(ba) + "," + String(bb) + ") — the groups are 1 m apart,"
            " so this is a broadphase or FK error, not a narrow-phase one",
        )
        seen[g] = seen[g] + 1

        var nx = Float64(d.contacts.data[b + CONTACT_IDX_NX])
        var ny = Float64(d.contacts.data[b + CONTACT_IDX_NY])
        var nz = Float64(d.contacts.data[b + CONTACT_IDX_NZ])
        var dist = Float64(d.contacts.data[b + CONTACT_IDX_DIST])

        # MuJoCo's contact for the same body pair (first is enough — the
        # manifold points agree in direction to ~1e-3).
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
        var approx = (g == 10 or g == 11)
        var tol_d = TOL_DIST_APPROX if approx else TOL_DIST
        var tol_n = TOL_DIR_APPROX if approx else TOL_DIR
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
