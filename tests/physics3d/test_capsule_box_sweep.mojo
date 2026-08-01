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

⚠ CONTACT COUNTS ARE NOT COMPARED. MuJoCo emits a SECOND point for a capsule
lying along a box face; we emit one (task #42, open). This file compares our
single point against MuJoCo's `contact[0]`, the primary at `bestsegmentpos`.
NOT against its deepest — `bestsegmentpos` minimises distance to the box
SURFACE, so for a penetrating capsule MuJoCo's second point is often deeper, and
selecting by depth compares our primary against its manifold extra and reports
the missing second point as a defect in this primitive.

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
