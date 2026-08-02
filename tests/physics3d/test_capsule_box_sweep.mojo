"""capsule/box narrow phase, swept over many poses (task #45).

WHY A SWEEP AND NOT ANOTHER SINGLE POSE. `test_narrow_phase_pairs.mojo` gates
capsule/box to 1e-10 on direction and 1e-12 on distance — at ONE pose, which it
passes. stacker then found the same primitive wrong by 1.4e-3..3.9e-3 in depth
and by a whole face-vs-edge normal at other poses. One pose per pair type
answers "is the DIRECTION convention right"; it cannot answer "is the GEOMETRY
right", because a single sample of a piecewise function only ever exercises one
piece.

WHAT IS BEING TESTED. `collision_primitives.box_capsule` reduces capsule/box to
box/sphere at one point on the capsule segment. Which point it picks is the
whole algorithm:

    MuJoCo (`mjraw_CapsuleBox`)  the point closest to the box SURFACE, found by
                                testing both endpoints against the faces and
                                all 12 edges against the segment
    ours, before this landed     the point closest to the box CENTRE

The two agree only when the box's nearest feature happens to be the face the
centre projection points at, which is why a face-on contact passed and every
edge or corner contact did not.

The sweep drives poses through a fixed LCG so a failure is reproducible by
index, and it reports the WORST case rather than the first — a primitive that is
right in 90% of poses is still broken, and the first failure tells you nothing
about how bad it gets.

⚠ `test_capsule_box_sweep_vs_mujoco` COMPARES ONLY MuJoCo'S `contact[0]`, and
that is deliberate: it was written while we emitted one point, and it stays that
way so it keeps measuring the PRIMARY point in isolation. It compares against
`contact[0]`, the point at `bestsegmentpos` — NOT against MuJoCo's deepest,
because `bestsegmentpos` minimises distance to the box SURFACE, so for a
penetrating capsule the second point is often deeper, and selecting by depth
compares our primary against its manifold extra.

The SECOND point is now emitted too, and
`test_capsule_box_manifold_vs_mujoco` at the bottom of this file gates the whole
record set. Note the random sweep is a weak gate for it on its own — a uniformly
random orientation is almost never parallel to a face, so only 7 of the 400
poses get two points. That test adds fixed lying-along-a-face poses for the
configuration the second point actually exists for.

All FOUR parts of the record are compared: depth, normal AND position. An
earlier version checked only the first two and passed while the contact POINT
was misplaced; see `TOL_POS`.

Run: pixi run mojo run -I . tests/physics3d/test_capsule_box_sweep.mojo
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
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
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

# A DELIBERATELY UNEQUAL box: .03 x .02 x .05. A cube cannot distinguish an axis
# mix-up from a correct answer, and the failing stacker case was a cube.
comptime CB_XML = """
<mujoco model="capsule box sweep">
  <option timestep="0.002"/>
  <worldbody>
    <body name="boxb" pos="0 0 0">
      <geom name="bx" type="box" size=".03 .02 .05"/>
    </body>
    <body name="capb" pos="0 0 0">
      <joint name="capjnt" type="free"/>
      <geom name="cp" type="capsule" size=".008 .04"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime cb = parse_xml(CB_XML)
comptime CBM = ModelDefFromXML[
    xml=CB_XML,
    nbody=cb.NBODY, njoint=cb.NJOINT, nq=cb.NQ, nv=cb.NV,
    ngeom=cb.NGEOM, nact=cb.NACT, ntex=cb.NTEX, nmat=cb.NMAT,
    nlight=cb.NLIGHT, ncam=cb.NCAM, nsite=cb.NSITE,
    max_tendon=cb.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=cb.TIMESTEP,
]

comptime NQ: Int = CBM.NQ
comptime NV: Int = CBM.NV
comptime NBODY: Int = CBM.NBODY
comptime NGEOM: Int = CBM.NGEOM
comptime MC: Int = CBM.MAX_CONTACTS

comptime Dat = Data[DTYPE, NQ, NV, NBODY, MC, CBM.NSITE, 1]
comptime Mod = Model[
    DTYPE, NV, NBODY, CBM.NJOINT, NGEOM, CBM.MAX_EQUALITY, CBM.MAX_TENDON,
    CBM.NSITE, CBM.NEXCLUDE, 0,
]

# Number of swept poses. Large enough that the face / edge / corner branches are
# all hit many times; the coverage assertion below checks that rather than
# assuming it.
comptime NPOSE: Int = 400

# Sampling half-width per axis. The capsule's bounding radius is .048 and the
# box's is .062, so +/-.085 straddles "clear", "touching" and "well inside".
comptime SPAN: Float64 = 0.085

# Gates. Both engines do the same box/sphere reduction in float64, so a correct
# `bestsegmentpos` should agree to round-off; these sit a few decades above it.
comptime TOL_DIST: Float64 = 1e-11
comptime TOL_DIR: Float64 = 1e-9
# ⚠ POSITION IS GATED TOO, AND IT HAS TO BE. An earlier version of this file
# compared only depth and normal, and passed at 5.9e-12 / 2.5e-10 while
# `box_sphere` was placing the contact POINT `face_gap` away from MuJoCo's
# whenever the sphere centre landed inside the box. Depth and normal are
# invariant to that error; only the point moves — and the point is what sets the
# contact Jacobian's moment arm, so it cost stacker a 15-41% qacc error with
# every depth and normal matching. A contact record has four parts and a gate
# that checks three of them is a gate with a known blind spot.
comptime TOL_POS: Float64 = 1e-11


struct Lcg(Copyable, Movable):
    """A fixed 64-bit LCG, so a failing pose index is reproducible."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0x1FFFFFFFFFFFFF) / 9007199254740992.0

    def sym(mut self, span: Float64) -> Float64:
        return (self.next() * 2.0 - 1.0) * span


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    CBM.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_capsule_box_sweep_vs_mujoco() raises:
    """Depth and normal over NPOSE capsule poses against the same box."""
    print("--- capsule/box sweep:", NPOSE, "poses")
    var mf = _build()
    var d = Dat()

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(CB_XML))
    var dat = mujoco.MjData(m)

    var rng = Lcg(0x9E3779B97F4A7C15)

    var n_touch = 0
    var n_ours = 0
    var n_both = 0
    var worst_dist = Float64(0)
    var worst_dir = Float64(0)
    var worst_pos = Float64(0)
    var worst_dist_pose = -1
    var worst_dir_pose = -1
    var worst_pos_pose = -1
    var n_bad_dist = 0
    var n_bad_dir = 0
    var n_bad_pos = 0

    for p in range(NPOSE):
        var px = rng.sym(SPAN)
        var py = rng.sym(SPAN)
        var pz = rng.sym(SPAN)
        # A random orientation: a normalised gaussian-ish quaternion is not
        # needed, a normalised uniform one covers the sphere well enough and
        # keeps the sweep reproducible without a gaussian sampler.
        var qx = rng.sym(1.0)
        var qy = rng.sym(1.0)
        var qz = rng.sym(1.0)
        var qw = rng.sym(1.0)
        var qn = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if qn < 1e-6:
            qx = 0.0
            qy = 0.0
            qz = 0.0
            qw = 1.0
            qn = 1.0
        qx /= qn
        qy /= qn
        qz /= qn
        qw /= qn

        # OURS. ⚠ `d.qpos` for a FREE JOINT is (x, y, z, qw, qx, qy, qz) — the
        # same order MuJoCo uses, NOT the (x, y, z, w) order `d.xquat` and the
        # contact records use. Writing the xquat order here poses the capsule
        # differently in the two engines and turns this whole file into a
        # measurement of the harness: it reported 24 mm depth errors and fully
        # reversed normals that had nothing to do with the primitive, and the
        # tell was that fixing a real defect in the primitive changed almost
        # nothing.
        CBM.reset_data(d)
        d.qpos.data[0] = Scalar[DTYPE](px)
        d.qpos.data[1] = Scalar[DTYPE](py)
        d.qpos.data[2] = Scalar[DTYPE](pz)
        d.qpos.data[3] = Scalar[DTYPE](qw)
        d.qpos.data[4] = Scalar[DTYPE](qx)
        d.qpos.data[5] = Scalar[DTYPE](qy)
        d.qpos.data[6] = Scalar[DTYPE](qz)
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])

        # MUJOCO. `qpos` for a free joint is (x, y, z, qw, qx, qy, qz).
        dat.qpos[0] = px
        dat.qpos[1] = py
        dat.qpos[2] = pz
        dat.qpos[3] = qw
        dat.qpos[4] = qx
        dat.qpos[5] = qy
        dat.qpos[6] = qz
        for i in range(NV):
            dat.qvel[i] = 0.0
        mujoco.mj_forward(m, dat)
        var mjn = Int(py=dat.ncon)

        if mjn > 0:
            n_touch += 1
        if nc > 0:
            n_ours += 1
        if mjn == 0 or nc == 0:
            continue
        n_both += 1

        # MuJoCo's PRIMARY contact for the pair is `contact[0]`: `mjraw_
        # CapsuleBox` emits the point at `bestsegmentpos` first and appends any
        # second manifold point after it. This model has exactly one geom pair,
        # so index 0 is that primary.
        #
        # ⚠ DO NOT PICK THE DEEPEST. `bestsegmentpos` minimises distance to the
        # box SURFACE, which for a PENETRATING capsule is not the deepest point
        # — MuJoCo's second point is deeper at 7 of the 88 contacting poses
        # here. Selecting by depth silently compares our primary against
        # MuJoCo's manifold extra and reports the missing second point
        # (task #42) as a depth error in this primitive. That cost one full
        # debugging cycle.
        var con = dat.contact[0]
        var bestd = Float64(py=con.dist)

        var b = 0
        var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])
        var od = Float64(d.contacts.data[b + CONTACT_IDX_DIST])
        var nx = Float64(d.contacts.data[b + CONTACT_IDX_NX])
        var ny = Float64(d.contacts.data[b + CONTACT_IDX_NY])
        var nz = Float64(d.contacts.data[b + CONTACT_IDX_NZ])

        var dd = abs(od - bestd)
        if dd > worst_dist:
            worst_dist = dd
            worst_dist_pose = p
        if dd > TOL_DIST:
            n_bad_dist += 1

        var pe = max(
            abs(Float64(d.contacts.data[b + CONTACT_IDX_POS_X])
                - Float64(py=con.pos[0])),
            max(
                abs(Float64(d.contacts.data[b + CONTACT_IDX_POS_Y])
                    - Float64(py=con.pos[1])),
                abs(Float64(d.contacts.data[b + CONTACT_IDX_POS_Z])
                    - Float64(py=con.pos[2])),
            ),
        )
        if pe > worst_pos:
            worst_pos = pe
            worst_pos_pose = p
        if pe > TOL_POS:
            n_bad_pos += 1

        # THE DIRECTION INVARIANT. Ours points body_b -> body_a, MuJoCo's
        # geom1 -> geom2, so the expected sign comes from the BODY LABELS and
        # not from whichever normal looks closer.
        var mb1 = Int(py=m.geom_bodyid[con.geom1])
        var mb2 = Int(py=m.geom_bodyid[con.geom2])
        var sgn = Float64(1.0) if (mb1 == bb and mb2 == ba) else Float64(-1.0)
        var e = max(
            abs(nx - sgn * Float64(py=con.frame[0])),
            max(
                abs(ny - sgn * Float64(py=con.frame[1])),
                abs(nz - sgn * Float64(py=con.frame[2])),
            ),
        )
        if e > worst_dir:
            worst_dir = e
            worst_dir_pose = p
        if e > TOL_DIR:
            n_bad_dir += 1

    print("  poses with a MuJoCo contact =", n_touch,
          " with ours =", n_ours, " with both =", n_both)
    print("  worst |d dist| =", worst_dist, " at pose", worst_dist_pose,
          " (", n_bad_dist, "poses over tol )")
    print("  worst |d normal| =", worst_dir, " at pose", worst_dir_pose,
          " (", n_bad_dir, "poses over tol )")
    print("  worst |d pos| =", worst_pos, " at pose", worst_pos_pose,
          " (", n_bad_pos, "poses over tol )")

    # Coverage first: a sweep that never touches gates nothing, and one where
    # the two engines disagree about WHETHER they touch is a different bug from
    # the one this file measures.
    assert_true(
        n_both >= NPOSE / 8,
        String("only ") + String(n_both) + " of " + String(NPOSE)
        + " poses produced a contact in both engines — the sampling box no"
        " longer straddles the geoms and this sweep gates almost nothing",
    )
    assert_true(
        abs(n_touch - n_ours) <= NPOSE / 50,
        String("the two engines disagree about WHETHER there is a contact on ")
        + String(abs(n_touch - n_ours)) + " poses (MuJoCo " + String(n_touch)
        + ", ours " + String(n_ours) + ") — that is a detection-threshold"
        " difference, not the depth/normal defect this file measures",
    )

    assert_true(
        worst_dist <= TOL_DIST,
        String("capsule/box DEPTH diverges from MuJoCo by ")
        + String(worst_dist) + " at pose " + String(worst_dist_pose)
        + " (" + String(n_bad_dist) + " poses over tolerance)",
    )
    assert_true(
        worst_pos <= TOL_POS,
        String("capsule/box contact POSITION diverges from MuJoCo by ")
        + String(worst_pos) + " at pose " + String(worst_pos_pose)
        + " (" + String(n_bad_pos) + " poses over tolerance). Depth and normal"
        " can both be exact while this is wrong — it is the moment arm.",
    )
    assert_true(
        worst_dir <= TOL_DIR,
        String("capsule/box NORMAL diverges from MuJoCo by ")
        + String(worst_dir) + " at pose " + String(worst_dir_pose)
        + " (" + String(n_bad_dir) + " poses over tolerance)",
    )


# Poses that DELIBERATELY lay the capsule along a box face. The random sweep
# above produces a second point at only 7 of its 400 poses, because a uniformly
# random orientation is almost never near-parallel to a face — so a gate built
# on the sweep alone would barely exercise the manifold it is meant to check.
# These are the configuration the second point exists for: a limb resting on a
# surface.
#
# ⚠ EVERY ONE OF THEM IS TILTED A FEW DEGREES, AND THAT IS NOT COSMETIC. A
# capsule lying EXACTLY flat and overhanging leaves the two opposite box edges
# exactly equidistant, and which one `mjraw_CapsuleBox` picks is then decided by
# the last bits of the rotation matrix. The first version of this table used
# perfectly flat poses; MuJoCo and this engine disagreed on two of them — one in
# contact COUNT and one in contact ORDER — purely because `forward_kinematics`
# and `mj_forward` round the quaternion differently. Neither engine was wrong.
# Each pose below was screened by re-running MuJoCo with the position AND the
# quaternion nudged by 1e-7 and requiring the same two contacts in the same
# order; screening on position alone passes the degenerate poses.
#
# (x, y, z, qx, qy, qz, qw); q rotates the capsule's local +z into its axis.
comptime NFIXED: Int = 6


def _fixed_poses() -> List[Float64]:
    """The NFIXED poses above, flattened; see the comment block."""
    return [
        # A: along the +x face, tilted 2 deg, capsule shorter than the face.
        0.035999999999999997, 0.0, 0.0,
        0.0, 0.017452406437283512, 0.0, 0.99984769515639127,
        # B: along the +x face, tilted 5 deg, shifted along its axis, deeper.
        0.034000000000000002, 0.0, 0.0060000000000000001,
        0.0, 0.043619387365336, 0.0, 0.9990482215818578,
        # C: along the +y face, tilted 2 deg the other way.
        0.0, 0.026000000000000002, -0.0089999999999999993,
        0.017452406437283512, 0.0, 0.0, 0.99984769515639127,
        # D: along the +z face, capsule OVERHANGS both ends of the face.
        0.0060000000000000001, 0.0, 0.056000000000000001,
        0.0, 0.71933980033865108, 0.0, 0.69465837045899737,
        # E: along the -z face, so the gate is not all on +ve faces.
        -0.0089999999999999993, 0.0, -0.054000000000000006,
        0.0, 0.73727733681012397, 0.0, 0.67559020761566024,
        # F: axis +z, capsule centre INSIDE the box — the one face case with
        # no second point (`clface == -1`), so the gate sees that branch too.
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]


def test_capsule_box_manifold_vs_mujoco() raises:
    """The WHOLE capsule/box manifold — count and every point — vs MuJoCo.

    `test_capsule_box_sweep_vs_mujoco` above deliberately compares only
    MuJoCo's `contact[0]`, because it was written while we emitted one point.
    This one compares the record SET, which is what the solver actually sees.

    Run through the engine (`detect_contacts`), not the primitive, so it gates
    `_capsule_box_contacts`'s wiring and its normal sign as well as
    `box_capsule_manifold` itself. The record order is MuJoCo's: the point at
    `bestsegmentpos` first, the second point after it.
    """
    print("--- capsule/box manifold:", NFIXED, "fixed +", NPOSE, "random")
    var mf = _build()
    var d = Dat()

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(CB_XML))
    var dat = mujoco.MjData(m)

    var rng = Lcg(0x9E3779B97F4A7C15)
    var fixed = _fixed_poses()

    var n_both = 0
    var n_two = 0
    var n_two_fixed = 0
    var mj_points = 0
    var our_points = 0
    var n_count_bad = 0
    var first_bad = -1
    var worst_dist = Float64(0)
    var worst_pos = Float64(0)
    var worst_dir = Float64(0)
    var worst_pose = -1

    for p in range(NFIXED + NPOSE):
        var px = Float64(0)
        var py = Float64(0)
        var pz = Float64(0)
        var qx = Float64(0)
        var qy = Float64(0)
        var qz = Float64(0)
        var qw = Float64(1)
        if p < NFIXED:
            px = fixed[7 * p + 0]
            py = fixed[7 * p + 1]
            pz = fixed[7 * p + 2]
            qx = fixed[7 * p + 3]
            qy = fixed[7 * p + 4]
            qz = fixed[7 * p + 5]
            qw = fixed[7 * p + 6]
        else:
            px = rng.sym(SPAN)
            py = rng.sym(SPAN)
            pz = rng.sym(SPAN)
            qx = rng.sym(1.0)
            qy = rng.sym(1.0)
            qz = rng.sym(1.0)
            qw = rng.sym(1.0)
            var qn = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if qn < 1e-6:
                qx = 0.0
                qy = 0.0
                qz = 0.0
                qw = 1.0
                qn = 1.0
            qx /= qn
            qy /= qn
            qz /= qn
            qw /= qn

        CBM.reset_data(d)
        d.qpos.data[0] = Scalar[DTYPE](px)
        d.qpos.data[1] = Scalar[DTYPE](py)
        d.qpos.data[2] = Scalar[DTYPE](pz)
        d.qpos.data[3] = Scalar[DTYPE](qw)
        d.qpos.data[4] = Scalar[DTYPE](qx)
        d.qpos.data[5] = Scalar[DTYPE](qy)
        d.qpos.data[6] = Scalar[DTYPE](qz)
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])

        dat.qpos[0] = px
        dat.qpos[1] = py
        dat.qpos[2] = pz
        dat.qpos[3] = qw
        dat.qpos[4] = qx
        dat.qpos[5] = qy
        dat.qpos[6] = qz
        for i in range(NV):
            dat.qvel[i] = 0.0
        mujoco.mj_forward(m, dat)
        var mjn = Int(py=dat.ncon)

        if mjn == 0 and nc == 0:
            continue
        n_both += 1
        mj_points += mjn
        our_points += nc
        if mjn >= 2:
            n_two += 1
            if p < NFIXED:
                n_two_fixed += 1

        if nc != mjn:
            n_count_bad += 1
            if first_bad < 0:
                first_bad = p
            continue

        for i in range(mjn):
            var con = dat.contact[i]
            var b = i * CONTACT_SIZE
            var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
            var bbi = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])

            var dd = abs(
                Float64(d.contacts.data[b + CONTACT_IDX_DIST])
                - Float64(py=con.dist)
            )
            if dd > worst_dist:
                worst_dist = dd
                worst_pose = p

            for c in range(3):
                var dp = abs(
                    Float64(d.contacts.data[b + CONTACT_IDX_POS_X + c])
                    - Float64(py=con.pos[c])
                )
                if dp > worst_pos:
                    worst_pos = dp

            # Same direction invariant as the sweep above: the expected sign
            # comes from the BODY LABELS, not from whichever normal is closer.
            var mb1 = Int(py=m.geom_bodyid[con.geom1])
            var mb2 = Int(py=m.geom_bodyid[con.geom2])
            var sgn = Float64(1.0) if (
                mb1 == bbi and mb2 == ba
            ) else Float64(-1.0)
            for c in range(3):
                var dn = abs(
                    Float64(d.contacts.data[b + CONTACT_IDX_NX + c])
                    - sgn * Float64(py=con.frame[c])
                )
                if dn > worst_dir:
                    worst_dir = dn

    print("  contacting poses =", n_both, " of which MuJoCo gives 2 points on",
          n_two, "(", n_two_fixed, "of the", NFIXED, "fixed )")
    print("  manifold points: MuJoCo", mj_points, " ours", our_points)
    print("  count mismatches =", n_count_bad, " first at pose", first_bad)
    print("  worst |d dist| =", worst_dist, " at pose", worst_pose)
    print("  worst |d pos|  =", worst_pos)
    print("  worst |d n|    =", worst_dir)

    # Coverage: the fixed poses exist so the second point is exercised on
    # purpose rather than by luck. If MuJoCo stops emitting two points on them
    # the poses have drifted off the faces and this gate is vacuous.
    assert_true(
        n_two_fixed == NFIXED - 1,
        String("only ") + String(n_two_fixed) + " of the " + String(NFIXED)
        + " FIXED poses gave MuJoCo two contacts; " + String(NFIXED - 1)
        + " should (all but the capsule-inside-the-box one). The poses no"
        " longer lie along the faces and this gate is vacuous",
    )
    assert_true(
        n_count_bad == 0,
        String("capsule/box manifold has a different POINT COUNT from MuJoCo"
        " on ") + String(n_count_bad) + " poses, first at pose "
        + String(first_bad) + " (ours " + String(our_points) + " points,"
        " MuJoCo " + String(mj_points) + " over " + String(n_both) + " poses)",
    )
    assert_true(
        worst_dist <= TOL_DIST,
        String("capsule/box manifold DEPTH diverges by ") + String(worst_dist)
        + " at pose " + String(worst_pose),
    )
    assert_true(
        worst_pos <= TOL_POS,
        String("capsule/box manifold POSITION diverges by ")
        + String(worst_pos),
    )
    assert_true(
        worst_dir <= TOL_DIR,
        String("capsule/box manifold NORMAL diverges by ") + String(worst_dir),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
