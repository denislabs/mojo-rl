"""box/box narrow phase, swept over many poses (task #42).

WHAT THIS MEASURES, AND WHY IT COMES BEFORE THE FIX. `mjc_BoxBox` does two
things: it picks a separating axis (its `code`, from 6 face axes on each box
plus 9 edge-edge cross products), and it then builds a CONTACT MANIFOLD of up
to four points on that axis. Our `box_box` does the first half by SAT and emits
a single point.

So there are two independent questions, and the size of the remaining work
depends entirely on which of them is already right:

    normal   does our separating AXIS agree with MuJoCo's `code`?
    depth    does our penetration agree with MuJoCo's deepest point?
    count    how many points does MuJoCo put on that axis? (we emit 1)

If the normal already matches, the remaining work is manifold generation alone.
If it does not, the axis selection has to be ported too — and a manifold built
on the wrong axis would be worse than one point on the right one.

The first run answered it: the axis and the depth ALREADY matched to 1e-10, so
only manifold generation was missing. `test_box_box_manifold_vs_mujoco` below
now gates that manifold, point for point, on BOTH kinds of axis.

Both halves are ported. Of the 217 contacting poses here, 90 take the face path
(210 points) and 127 take edge-edge (361 points); the engine emits all 571,
matching MuJoCo exactly. An earlier version of this comment claimed edge-edge
emitted one point in MuJoCo too; that was wrong, read off the `n = 1; ... n = 2`
prologue of `mjc_BoxBox`'s edge branch without following it to the clipping
that comes after — it emits 1 to 6.

⚠ The edge branch has a quirk that has to be reproduced, not cleaned up:
MuJoCo overwrites its barycentric determinant `c1` inside its own
reference-corner loop, so every corner after the first divides by a squared
distance. Without that we emit 368 edge points where the runtime emits 361, and
the extra ones look perfectly reasonable.

⚠ COMPARED AGAINST MuJoCo'S DEEPEST POINT, and that IS the right choice here,
unlike capsule/box: on a face manifold every point shares one normal, and the
deepest is the one a single-point reduction should represent. Stating it because
picking the wrong reference point cost a debugging cycle on the capsule/box
sweep.

Run: pixi run mojo run -I . tests/physics3d/test_box_box_sweep.mojo
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
from mojo_rl.physics3d.collision.collision_primitives import (
    box_box_manifold,
    BB_MAX_POINTS,
)
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
)


comptime DTYPE = DType.float64

# Two DELIBERATELY UNEQUAL boxes with different shapes. Two cubes cannot
# distinguish an axis mix-up from a correct answer.
comptime BB_XML = """
<mujoco model="box box sweep">
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 0">
      <geom name="g1" type="box" size=".03 .02 .05"/>
    </body>
    <body name="b2" pos="0 0 0">
      <joint name="j2" type="free"/>
      <geom name="g2" type="box" size=".025 .04 .015"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime bb = parse_xml(BB_XML)
comptime BBM = ModelDefFromXML[
    xml=BB_XML,
    nbody=bb.NBODY, njoint=bb.NJOINT, nq=bb.NQ, nv=bb.NV,
    ngeom=bb.NGEOM, nact=bb.NACT, ntex=bb.NTEX, nmat=bb.NMAT,
    nlight=bb.NLIGHT, ncam=bb.NCAM, nsite=bb.NSITE,
    max_tendon=bb.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=bb.TIMESTEP,
]

comptime NQ: Int = BBM.NQ
comptime NV: Int = BBM.NV
comptime NBODY: Int = BBM.NBODY
comptime MC: Int = BBM.MAX_CONTACTS

comptime Dat = Data[DTYPE, NQ, NV, NBODY, MC, BBM.NSITE, 1]
comptime Mod = Model[
    DTYPE, NV, NBODY, BBM.NJOINT, BBM.NGEOM, BBM.MAX_EQUALITY, BBM.MAX_TENDON,
    BBM.NSITE, BBM.NEXCLUDE, 0,
]

comptime NPOSE: Int = 400
comptime SPAN: Float64 = 0.075

comptime TOL_DIST: Float64 = 1e-11
comptime TOL_DIR: Float64 = 1e-9


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
    BBM.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_box_box_sweep_vs_mujoco() raises:
    """Separating axis and penetration depth over NPOSE box poses."""
    print("--- box/box sweep:", NPOSE, "poses")
    var mf = _build()
    var d = Dat()

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(BB_XML))
    var dat = mujoco.MjData(m)

    var rng = Lcg(0x9E3779B97F4A7C15)

    var n_touch = 0
    var n_ours = 0
    var n_both = 0
    var worst_dist = Float64(0)
    var worst_dir = Float64(0)
    var worst_dist_pose = -1
    var worst_dir_pose = -1
    var n_bad_dist = 0
    var n_bad_dir = 0
    # How many points MuJoCo puts on the manifold, so the size of the missing
    # half is reported rather than assumed.
    var mj_points = 0
    var mj_max_points = 0
    var our_points = 0

    for p in range(NPOSE):
        var px = rng.sym(SPAN)
        var py = rng.sym(SPAN)
        var pz = rng.sym(SPAN)
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

        # ⚠ Free-joint qpos is (x, y, z, qw, qx, qy, qz) — MuJoCo's order, NOT
        # the (x, y, z, w) order the contact records use.
        BBM.reset_data(d)
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

        if mjn > 0:
            n_touch += 1
            mj_points += mjn
            if mjn > mj_max_points:
                mj_max_points = mjn
        if nc > 0:
            n_ours += 1
            our_points += nc
        if mjn == 0 or nc == 0:
            continue
        n_both += 1

        # MuJoCo's DEEPEST point on the manifold.
        var best = 0
        var bestd = Float64(py=dat.contact[0].dist)
        for k in range(1, mjn):
            var dk = Float64(py=dat.contact[k].dist)
            if dk < bestd:
                bestd = dk
                best = k
        var con = dat.contact[best]

        # ...against OUR deepest. Before the face manifold landed we emitted
        # exactly one contact and index 0 was it; now a face pose emits the
        # whole manifold in MuJoCo's own order, whose first point is not
        # generally the deepest.
        var obest = 0
        var od = Float64(d.contacts.data[CONTACT_IDX_DIST])
        for k in range(1, nc):
            var dk = Float64(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_DIST])
            if dk < od:
                od = dk
                obest = k
        var o_off = obest * CONTACT_SIZE
        var ba = Int(d.contacts.data[o_off + CONTACT_IDX_BODY_A])
        var bb2 = Int(d.contacts.data[o_off + CONTACT_IDX_BODY_B])

        var dd = abs(od - bestd)
        if dd > worst_dist:
            worst_dist = dd
            worst_dist_pose = p
        if dd > TOL_DIST:
            n_bad_dist += 1

        var mb1 = Int(py=m.geom_bodyid[con.geom1])
        var mb2 = Int(py=m.geom_bodyid[con.geom2])
        var sgn = Float64(1.0) if (mb1 == bb2 and mb2 == ba) else Float64(-1.0)
        var e = max(
            abs(Float64(d.contacts.data[o_off + CONTACT_IDX_NX])
                - sgn * Float64(py=con.frame[0])),
            max(
                abs(Float64(d.contacts.data[o_off + CONTACT_IDX_NY])
                    - sgn * Float64(py=con.frame[1])),
                abs(Float64(d.contacts.data[o_off + CONTACT_IDX_NZ])
                    - sgn * Float64(py=con.frame[2])),
            ),
        )
        if e > worst_dir:
            worst_dir = e
            worst_dir_pose = p
        if e > TOL_DIR:
            n_bad_dir += 1

    print("  poses with a MuJoCo contact =", n_touch,
          " with ours =", n_ours, " with both =", n_both)
    print("  manifold points: MuJoCo", mj_points, " ours", our_points,
          " (MuJoCo max on one pose", mj_max_points, ")")
    print("  worst |d dist| =", worst_dist, " at pose", worst_dist_pose,
          " (", n_bad_dist, "poses over tol )")
    print("  worst |d normal| =", worst_dir, " at pose", worst_dir_pose,
          " (", n_bad_dir, "poses over tol )")

    assert_true(
        n_both >= NPOSE / 8,
        String("only ") + String(n_both) + " of " + String(NPOSE)
        + " poses produced a contact in both engines — this sweep gates almost"
        " nothing",
    )
    assert_true(
        worst_dir <= TOL_DIR,
        String("box/box SEPARATING AXIS diverges from MuJoCo by ")
        + String(worst_dir) + " at pose " + String(worst_dir_pose)
        + " (" + String(n_bad_dir) + " poses over tolerance) — the manifold"
        " port needs MuJoCo's axis selection too, not just its clipping",
    )
    assert_true(
        worst_dist <= TOL_DIST,
        String("box/box PENETRATION diverges from MuJoCo by ")
        + String(worst_dist) + " at pose " + String(worst_dist_pose)
        + " (" + String(n_bad_dist) + " poses over tolerance)",
    )
    # The manifold has to reach the CONTACT RECORDS, not just the primitive:
    # `_box_box_contacts` is wired into two separate narrow phases and this is
    # the one that runs `detect_contacts`. `MAX_CONTACTS` is 8 here and MuJoCo's
    # worst pose has 6, so nothing is being clipped by the cap.
    assert_true(
        our_points == mj_points,
        String("the narrow phase emitted ") + String(our_points)
        + " contacts where MuJoCo emitted " + String(mj_points)
        + " over the same " + String(n_ours) + " contacting poses — the"
        " manifold is not reaching the records intact even though"
        " `box_box_manifold` builds it",
    )


def test_box_box_manifold_vs_mujoco() raises:
    """Every point of the manifold, in MuJoCo's own order, on BOTH axis kinds.

    Calls `box_box_manifold` directly rather than going through the engine, so
    a failure says which of the two is wrong: the primitive, or the wiring in
    `_box_box_contacts`. The engine path is gated by
    `test_box_box_sweep_vs_mujoco` above and by the stacker qacc buckets.

    Face and edge poses are counted separately so a regression in one path
    cannot hide behind the other's totals — they are different code with
    different failure modes (the face path picks a reference FACE and its
    normal is that face's; the edge path's normal is a cross product and its
    reference frame is chosen from the leading corner).

    ⚠ Both geoms sit at their body origin with no local offset in `BB_XML`, so
    the geom world pose IS the body pose and the sweep's own `(p, q)` can be
    passed straight in. Add a `pos=`/`quat=` to either `<geom>` and this stops
    being true.
    """
    print("--- box/box manifold:", NPOSE, "poses")

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(BB_XML))
    var dat = mujoco.MjData(m)

    var rng = Lcg(0x9E3779B97F4A7C15)

    var n_face = 0
    var n_edge = 0
    var n_sep = 0
    var n_face_points = 0
    var n_edge_points = 0
    var n_count_bad = 0
    var first_bad = -1
    var worst_dist = Float64(0)
    var worst_pos = Float64(0)
    var worst_dir = Float64(0)
    var worst_pose = -1

    for p in range(NPOSE):
        var px = rng.sym(SPAN)
        var py = rng.sym(SPAN)
        var pz = rng.sym(SPAN)
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

        var n_bb = 0
        var bb_dist = InlineArray[Scalar[DTYPE], BB_MAX_POINTS](
            fill=Scalar[DTYPE](0)
        )
        var bb_pos = InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS](
            fill=Scalar[DTYPE](0)
        )
        var bb_n = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        var code = box_box_manifold[DTYPE](
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](1),
            Scalar[DTYPE](0.03), Scalar[DTYPE](0.02), Scalar[DTYPE](0.05),
            Scalar[DTYPE](px), Scalar[DTYPE](py), Scalar[DTYPE](pz),
            Scalar[DTYPE](qx), Scalar[DTYPE](qy), Scalar[DTYPE](qz),
            Scalar[DTYPE](qw),
            Scalar[DTYPE](0.025), Scalar[DTYPE](0.04), Scalar[DTYPE](0.015),
            Scalar[DTYPE](0),
            n_bb,
            bb_dist,
            bb_pos,
            bb_n,
        )

        if code < 0:
            n_sep += 1
            if mjn != 0 and first_bad < 0:
                first_bad = p
            if mjn != 0:
                n_count_bad += 1
            continue
        if code >= 12:
            n_edge += 1
            n_edge_points += n_bb
        else:
            n_face += 1
            n_face_points += n_bb

        if n_bb != mjn:
            n_count_bad += 1
            if first_bad < 0:
                first_bad = p
            continue

        for i in range(mjn):
            var con = dat.contact[i]
            var dd = abs(Float64(bb_dist[i]) - Float64(py=con.dist))
            if dd > worst_dist:
                worst_dist = dd
                worst_pose = p
            for c in range(3):
                var dp = abs(
                    Float64(bb_pos[3 * i + c]) - Float64(py=con.pos[c])
                )
                if dp > worst_pos:
                    worst_pos = dp
                var dn = abs(Float64(bb_n[c]) - Float64(py=con.frame[c]))
                if dn > worst_dir:
                    worst_dir = dn

    print("  poses: face =", n_face, " edge-edge =", n_edge,
          " separated =", n_sep)
    print("  face manifold points =", n_face_points,
          " (one point per pose would be", n_face, ")")
    print("  edge manifold points =", n_edge_points,
          " (one point per pose would be", n_edge, ")")
    print("  count mismatches =", n_count_bad, " first at pose", first_bad)
    print("  worst |d dist| =", worst_dist, " at pose", worst_pose)
    print("  worst |d pos|  =", worst_pos)
    print("  worst |d n|    =", worst_dir)

    assert_true(
        n_face >= 40,
        String("only ") + String(n_face) + " poses took the FACE path — this"
        " test gates almost nothing",
    )
    assert_true(
        n_edge >= 40,
        String("only ") + String(n_edge) + " poses took the EDGE-EDGE path —"
        " this test gates almost nothing",
    )
    assert_true(
        n_face_points > n_face,
        String("the face path emitted ") + String(n_face_points)
        + " points over " + String(n_face) + " poses, i.e. no manifold at all",
    )
    assert_true(
        n_edge_points > n_edge,
        String("the edge-edge path emitted ") + String(n_edge_points)
        + " points over " + String(n_edge) + " poses, i.e. no manifold at all",
    )
    assert_true(
        n_count_bad == 0,
        String("box/box manifold has a different POINT COUNT from MuJoCo"
        " on ") + String(n_count_bad) + " poses, first at pose "
        + String(first_bad),
    )
    assert_true(
        worst_dist <= TOL_DIST,
        String("manifold DEPTH diverges by ") + String(worst_dist)
        + " at pose " + String(worst_pose),
    )
    assert_true(
        worst_pos <= TOL_DIST,
        String("manifold POSITION diverges by ") + String(worst_pos),
    )
    assert_true(
        worst_dir <= TOL_DIR,
        String("manifold NORMAL diverges by ") + String(worst_dir),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
