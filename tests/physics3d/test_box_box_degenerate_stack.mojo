"""Two boxes stacked face to face at zero separation — the pose a brick tower
is MADE of, and the one `test_box_box_sweep` cannot reach.

That sweep draws a UNIFORM random quaternion and a position over +/- 75 mm on
boxes 30-50 mm across, so a pose with the two faces parallel to a microradian
AND separated by nanometres has probability ~0 in it. It was green throughout
the defect below.

`reassemble_5` is made of nothing else. A Duplo base is a box of half-extent
(0.0159, 0.0318, 0.0096) and the bricks stack at EXACTLY 0.0192, so every
adjacent pair in the tower sits face to face at `dist = -2.8e-09` with
IDENTICAL quaternions and zero lateral offset. Measured there: MuJoCo emits
four face points and we emitted FIVE, the extra one a DUPLICATE of another to
within an ulp — and a duplicated contact is a duplicated ROW in the constraint
Jacobian, i.e. an exactly rank-deficient Hessian handed to the solver.

WHY A DUPLICATE SURVIVED

`_bb_post_filter` drops duplicate manifold points with `pos[i] == pos[j]`,
exactly as `engine_collision_box.c:1394` does. It is a faithful port and it was
INERT: MuJoCo's construction produces BIT-IDENTICAL values for a coincident
point, and ours were one ulp apart, so the comparison never fired.

⚠⚠ THE ULP DID NOT COME FROM box/box. Called directly on MuJoCo's own
`geom_xpos`/`geom_xmat`, `box_box_manifold` emits 8 pre-filter points, dedups
exactly 4 away and matches MuJoCo point for point. It came from QUATERNION
NORMALISATION: we renormalised a quaternion that was ALREADY unit to rounding
and moved it by an ulp, where MuJoCo deliberately does not
(`mju_normalize4`, `engine_util_blas.c:258`:
`else if (mju_abs(norm - 1) > mjMINVAL)`; `mjuu_normvec`,
`user_util.cc:162`: "don't normalize if nrm is within mjEPS of 1"). Hence
`test_quat_normalize_leaves_a_unit_quaternion_alone` below, which gates the
CAUSE, while the sweep gates the consequence.

Run:
    pixi run mojo run -I . tests/physics3d/test_box_box_degenerate_stack.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.kinematics.quat_math import (
    quat_normalize, gpu_quat_normalize,
)
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.fields import SpecFields
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE, META_IDX_NUM_CONTACTS, CONTACT_IDX_DIST,
    CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
)

comptime DTYPE = DType.float64

# The tower's own numbers, read off MuJoCo at a real `reassemble_5` state:
# both base boxes carry the SAME quaternion (yaw 70.68 deg) and their centres
# differ by (0, 0, 0.019199997186660767), giving `dist = -2.8133e-09`.
comptime STACK: Float64 = 0.019199997186660767
comptime YW: Float64 = 0.8158341149610219
comptime YZ: Float64 = 0.5782859991956973

# ⚠ `b1` CARRIES THE YAW AND THAT IS PART OF THE FIXTURE, not decoration. With
# both boxes axis-aligned, `rot = mat1^T mat2` is the identity EXACTLY and every
# clip denominator is a clean zero or one; at 70.68 degrees it is the identity
# only to ~1e-16, which is the asymmetry the manifold has to survive. It is also
# what makes `b1`'s quaternion go through the parser while `b2`'s goes through
# `qpos` — two different routes to the same number, which is exactly how a
# renormalisation that moves an already-unit quaternion becomes visible.
comptime BB_XML = """
<mujoco model="degenerate stacked duplo bases">
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 0" quat="0.8158341149610219 0 0 0.5782859991956973">
      <geom name="g1" type="box" size="0.0159 0.0318 0.0096"/>
    </body>
    <body name="b2" pos="0 0 0">
      <joint name="j2" type="free"/>
      <geom name="g2" type="box" size="0.0159 0.0318 0.0096"/>
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
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=bb.TIMESTEP,
]
comptime NV: Int = BBM.NV
comptime MD = ModelDims[BBM]
comptime Dat = Data[DTYPE, MD, 1]
comptime Mod = Model[DTYPE, MD]

comptime NPER: Int = 60
comptime TOL_POS: Float64 = 1e-9
comptime TOL_DIST: Float64 = 1e-9
# Two contact points closer together than this are the SAME corner reported
# twice — near-duplicate rows in the constraint Jacobian. The corners of this
# box are 31.8 mm and 63.6 mm apart, so 1e-6 cannot flag a legitimate pair.
comptime DUP_TOL: Float64 = 1e-6
# A MISSED manifold corner is a whole corner away (31.8 mm on this box); the
# axis tie displaces one by ~1e-07. This sits between them, far from both.
comptime TOL_MISS: Float64 = 1e-5


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


def test_quat_normalize_leaves_a_unit_quaternion_alone() raises:
    """An already-unit quaternion must come back BIT-IDENTICAL.

    `mju_normalize4` (`engine_util_blas.c:250`) normalises only when
    `mju_abs(norm - 1) > mjMINVAL`, and `mjuu_normvec` (`user_util.cc:149`)
    only when `std::abs(nrm - 1) > mjEPS`, with the comment "don't normalize if
    nrm is within mjEPS of 1". Both guards exist for the same reason: a
    quaternion written to full double precision is generally NOT exactly unit
    — the one below has `w^2 + z^2 = 0.9999999999999999` — and renormalising it
    MOVES it by an ulp for no gain.

    ⚠ AN ULP IS NOT A ROUNDING DETAIL WHEN A DOWNSTREAM TEST IS `==`. This one
    made `_bb_post_filter`'s duplicate removal inert and put a duplicated
    contact row into the constraint Hessian. See the module docstring.

    The second half asserts the guard did not turn into "never normalise": a
    quaternion genuinely off unit (dm_control's humanoid writes
    `quat="1.000 0 -.002 0"` on `lower_waist`) must still be repaired.
    """
    print("--- quat_normalize on an already-unit quaternion")
    var qx = Scalar[DTYPE](0)
    var qy = Scalar[DTYPE](0)
    var qz = Scalar[DTYPE](YZ)
    var qw = Scalar[DTYPE](YW)
    var nsq = qx * qx + qy * qy + qz * qz + qw * qw
    print("   input  norm^2 =", Float64(nsq), " (exactly 1:", nsq == Scalar[DTYPE](1), ")")

    var r = quat_normalize[DTYPE](qx, qy, qz, qw)
    print("   quat_normalize     ->", Float64(r[0]), Float64(r[1]),
          Float64(r[2]), Float64(r[3]))
    var g = gpu_quat_normalize[DTYPE](qx, qy, qz, qw)
    print("   gpu_quat_normalize ->", Float64(g[0]), Float64(g[1]),
          Float64(g[2]), Float64(g[3]))

    assert_true(
        r[0] == qx and r[1] == qy and r[2] == qz and r[3] == qw,
        String(
            "quat_normalize MOVED an already-unit quaternion: z "
        )
        + String(Float64(qz)) + " -> " + String(Float64(r[2]))
        + ", w " + String(Float64(qw)) + " -> " + String(Float64(r[3]))
        + ". MuJoCo does not (`mju_normalize4`, engine_util_blas.c:258:"
        " `else if (mju_abs(norm - 1) > mjMINVAL)`). An ulp here is not"
        " cosmetic — `_bb_post_filter` compares contact positions with `==`,"
        " so it turns a duplicate manifold point into a duplicated constraint"
        " row.",
    )
    assert_true(
        g[0] == qx and g[1] == qy and g[2] == qz and g[3] == qw,
        String(
            "gpu_quat_normalize MOVED an already-unit quaternion: z "
        )
        + String(Float64(qz)) + " -> " + String(Float64(g[2]))
        + ". The CPU and GPU normalisers must carry the same rule or the two"
        " legs disagree about every body pose in the model.",
    )

    # ── and the guard must not have become "never normalise" ──────────────
    var ux = Scalar[DTYPE](0)
    var uy = Scalar[DTYPE](-0.002)
    var uz = Scalar[DTYPE](0)
    var uw = Scalar[DTYPE](1.000)
    var u = quat_normalize[DTYPE](ux, uy, uz, uw)
    var un = sqrt(
        u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
    )
    print("   off-unit input (norm 1.000002) -> norm", Float64(un))
    assert_true(
        abs(Float64(un) - 1.0) < 1e-15,
        String(
            "a genuinely off-unit quaternion was NOT repaired: norm came back "
        )
        + String(Float64(un))
        + ". The near-unit guard is `|norm - 1| > mjMINVAL`, not `never`.",
    )


def _pose_manifold(
    mut d: Dat, mut mf: Mod, sf: SpecFields[DTYPE, MD],
    dat: PythonObject, mujoco: PythonObject, m: PythonObject,
    px: Float64, py: Float64, pz: Float64,
    qw: Float64, qx: Float64, qy: Float64, qz: Float64,
    leg: Int,
    mut mjn: Int, mut nc: Int,
    mut worst_ours_to_mj: Float64,
    mut worst_mj_to_ours: Float64,
    mut worst_dist: Float64,
    mut closest_pair: Float64,
) raises:
    """Run ONE pose through both engines and accumulate the four distances.

    Both directions are measured on purpose. An EXTRA point of ours is a
    different failure from a MISSING one — an extra at a corner MuJoCo already
    covers is a duplicated Jacobian row, while a missing one is a contact the
    solver never sees — and a single symmetric "worst error" would blur them.
    """
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
    mjn = Int(py=dat.ncon)

    BBM.reset_data(sf, d)
    d.qpos.data[0] = Scalar[DTYPE](px)
    d.qpos.data[1] = Scalar[DTYPE](py)
    d.qpos.data[2] = Scalar[DTYPE](pz)
    d.qpos.data[3] = Scalar[DTYPE](qw)
    d.qpos.data[4] = Scalar[DTYPE](qx)
    d.qpos.data[5] = Scalar[DTYPE](qy)
    d.qpos.data[6] = Scalar[DTYPE](qz)
    forward_kinematics["cpu"](d, mf)
    if leg == 0:
        detect_contacts["cpu"](d, mf)
    else:
        detect_contacts_sap["cpu"](d, mf)
    nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])

    for k in range(nc):
        var ok = k * CONTACT_SIZE
        var od = Float64(d.contacts.data[ok + CONTACT_IDX_DIST])
        var ox = Float64(d.contacts.data[ok + CONTACT_IDX_POS_X])
        var oy = Float64(d.contacts.data[ok + CONTACT_IDX_POS_Y])
        var oz = Float64(d.contacts.data[ok + CONTACT_IDX_POS_Z])
        var bp = 1e30
        var bd = 1e30
        for j in range(mjn):
            var c = dat.contact[j]
            var ex = ox - Float64(py=c.pos[0])
            var ey = oy - Float64(py=c.pos[1])
            var ez = oz - Float64(py=c.pos[2])
            var rr = sqrt(ex * ex + ey * ey + ez * ez)
            if rr < bp:
                bp = rr
                bd = abs(od - Float64(py=c.dist))
        if mjn > 0:
            if bp > worst_ours_to_mj:
                worst_ours_to_mj = bp
            if bd > worst_dist:
                worst_dist = bd
        # The closest pair among OUR OWN points: two contacts at one corner are
        # two identical rows in the constraint Jacobian.
        for k2 in range(k + 1, nc):
            var kb = k2 * CONTACT_SIZE
            var ax0 = ox - Float64(d.contacts.data[kb + CONTACT_IDX_POS_X])
            var ay0 = oy - Float64(d.contacts.data[kb + CONTACT_IDX_POS_Y])
            var az0 = oz - Float64(d.contacts.data[kb + CONTACT_IDX_POS_Z])
            var sep = sqrt(ax0 * ax0 + ay0 * ay0 + az0 * az0)
            if sep < closest_pair:
                closest_pair = sep

    for j in range(mjn):
        var c = dat.contact[j]
        var bp2 = 1e30
        for k in range(nc):
            var ok2 = k * CONTACT_SIZE
            var ex = Float64(py=c.pos[0]) - Float64(d.contacts.data[ok2 + CONTACT_IDX_POS_X])
            var ey = Float64(py=c.pos[1]) - Float64(d.contacts.data[ok2 + CONTACT_IDX_POS_Y])
            var ez = Float64(py=c.pos[2]) - Float64(d.contacts.data[ok2 + CONTACT_IDX_POS_Z])
            var rr = sqrt(ex * ex + ey * ey + ez * ez)
            if rr < bp2:
                bp2 = rr
        if nc > 0 and bp2 > worst_mj_to_ours:
            worst_mj_to_ours = bp2


def test_box_box_at_the_exactly_degenerate_pose() raises:
    """The pose the tower actually sits in — no perturbation, no rounding to
    disagree about, both engines handed bit-identical geometry.

    THIS is the assertion that regressed and the one worth defending. The
    swept version below cannot assert a point count (see its docstring), but
    here there is nothing to tie-break: if the manifolds differ, the manifold
    is wrong.
    """
    print("--- box/box at the EXACTLY degenerate stacked pose")
    var ctx = DeviceContext()
    var mf = Mod()
    BBM.init_fields[DTYPE](ctx, mf)
    var sf = BBM.make_spec_fields[DTYPE]()
    var d = Dat()
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(BB_XML))
    var dat = mujoco.MjData(m)

    for leg in range(2):
        var mjn = 0
        var nc = 0
        var w_om = 0.0
        var w_mo = 0.0
        var w_d = 0.0
        var cp = 1e30
        _pose_manifold(
            d, mf, sf, dat, mujoco, m,
            0.0, 0.0, STACK, YW, 0.0, 0.0, YZ, leg,
            mjn, nc, w_om, w_mo, w_d, cp,
        )
        var name = String("O(N^2)") if leg == 0 else String("SAP")
        print("  ", name, ": MuJoCo", mjn, "points, ours", nc,
              " ours->mj", w_om, " mj->ours", w_mo, " |d dist|", w_d,
              " closest pair", cp)

        assert_true(
            mjn >= 4,
            String("VACUOUS: MuJoCo found only ") + String(mjn)
            + " contacts at the stacked pose — the fixture is not in contact,"
            " so nothing below tests anything.",
        )
        assert_true(
            nc == mjn,
            String("the ") + name + " leg emits " + String(nc)
            + " manifold points where MuJoCo emits " + String(mjn)
            + ". Both engines see bit-identical geometry here, so this is the"
            " box/box manifold disagreeing outright. The known cause is an ulp"
            " introduced UPSTREAM of box/box — renormalising an already-unit"
            " quaternion — which makes `_bb_post_filter`'s `==` duplicate"
            " removal inert. See"
            " `test_quat_normalize_leaves_a_unit_quaternion_alone`.",
        )
        assert_true(
            w_om <= TOL_POS and w_mo <= TOL_POS,
            String("manifold POINTS differ: ours->MuJoCo ") + String(w_om)
            + ", MuJoCo->ours " + String(w_mo) + ", over " + String(TOL_POS),
        )
        assert_true(
            w_d <= TOL_DIST,
            String("manifold DEPTH differs by ") + String(w_d),
        )
        assert_true(
            cp >= DUP_TOL,
            String("two of our manifold points are ") + String(cp)
            + " apart — the SAME corner reported twice, which hands the solver"
            " two identical constraint rows and an exactly rank-deficient"
            " Hessian. This is what the `reassemble_5` tower blew up on.",
        )
    print("  PASS")


def test_box_box_near_degenerate_sweep_vs_mujoco() raises:
    """The neighbourhood of that pose, swept by PERTURBATION SCALE.

    ⚠⚠ THE POINT COUNT IS DELIBERATELY NOT ASSERTED HERE, AND THAT IS A
    MEASURED DECISION RATHER THAN A FUDGE. On two identical boxes face to face
    the separating-axis loop is an EXACT TIE: `c1 = -|pos21[i]| + size1[i] +
    plen2[i]` and its `c2` counterpart are the same number in real arithmetic,
    because equal boxes at equal orientations give `|pos21| == |pos12|` and
    `plen1 == plen2`. The loop breaks the tie with a strict `<`, so the LAST
    BIT of forward kinematics decides it: MuJoCo takes `code = 11`, box 2's -z
    face, and we take `code = 2`, box 1's +z face. Both describe the same plane
    at the same depth; they enumerate the four corners through different loops,
    and one enumeration puts a reference-face corner and an incident-face
    corner on the same corner while the other does not. Verified with a
    line-for-line transcription of `_boxbox`'s face path
    (`docs/reassemble5_harnesses/bbref.py`), which reproduces MuJoCo's four
    points exactly and reports `code 11`.

    Asserting equal counts would therefore be asserting that our forward
    kinematics rounds like MuJoCo's. It does not, it cannot in general, and
    that is not what this file is about.

    WHAT *IS* ASSERTED holds whichever way the tie falls:
      * every point MuJoCo reports has one of ours on it — we never MISS a
        contact, which is the dangerous direction;
      * the depth of every point we report matches MuJoCo's.

    The counts and the closest-pair distance are PRINTED so a regression in
    them is visible even though it is not fatal.
    """
    print("--- box/box swept by perturbation scale around that pose")
    var ctx = DeviceContext()
    var mf = Mod()
    BBM.init_fields[DTYPE](ctx, mf)
    var sf = BBM.make_spec_fields[DTYPE]()
    var d = Dat()
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(BB_XML))
    var dat = mujoco.MjData(m)

    var scales = List[Float64]()
    scales.append(1e-12)
    scales.append(1e-10)
    scales.append(1e-8)
    scales.append(1e-6)

    var tested = 0
    var w_om = 0.0
    var w_mo = 0.0
    var w_d = 0.0
    var cp = 1e30
    var w_om_all = 0.0
    var w_mo_all = 0.0
    var agreed = 0
    var n_count_bad = 0

    for si in range(len(scales)):
        var sc = scales[si]
        var rng = Lcg(0x9E3779B97F4A7C15 + UInt64(si) * 7919)
        var cbad = 0
        for p in range(NPER):
            var px = rng.sym(sc)
            var py = rng.sym(sc)
            var pz = STACK + rng.sym(sc)
            var ax = rng.sym(sc)
            var ay = rng.sym(sc)
            var az = rng.sym(sc)
            var qw = YW - 0.5 * (YZ * az)
            var qx = 0.5 * (YW * ax - YZ * ay)
            var qy = 0.5 * (YW * ay + YZ * ax)
            var qz = YZ + 0.5 * (YW * az)
            # ⚠ `mju_normalize4`'s near-unit rule, and here it is load-bearing
            # for the FIXTURE. `(YW, 0, 0, YZ)` has norm 0.9999999999999999, so
            # an unconditional divide moves `b2`'s quaternion off `b1`'s by an
            # ulp and the pose stops being the one under test. This was wrong
            # in the first draft and cost a full sweep to notice.
            var qn = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if abs(qn - 1.0) > 1e-15:
                qw /= qn
                qx /= qn
                qy /= qn
                qz /= qn

            for leg in range(2):
                var mjn = 0
                var nc = 0
                # Per-pose accumulators, merged by the caller — because WHERE
                # they are merged depends on whether the tie fell the same way,
                # and that is not known until the counts come back.
                var p_om = 0.0
                var p_mo = 0.0
                var p_d = 0.0
                var p_cp = 1e30
                _pose_manifold(
                    d, mf, sf, dat, mujoco, m,
                    px, py, pz, qw, qx, qy, qz, leg,
                    mjn, nc, p_om, p_mo, p_d, p_cp,
                )
                if mjn == 0 and nc == 0:
                    continue
                if leg == 0:
                    tested += 1
                # Depth does not depend on which face won the tie, so it is
                # accumulated over EVERY pose.
                if p_d > w_d:
                    w_d = p_d
                if p_om > w_om_all:
                    w_om_all = p_om
                if p_mo > w_mo_all:
                    w_mo_all = p_mo
                if mjn != nc:
                    cbad += 1
                    continue
                # ⚠ POSITIONS ARE COMPARED ONLY WHERE THE TIE FELL THE SAME
                # WAY. With a different reference face the four corners are
                # projected onto the other plane, so they legitimately sit
                # O(tilt x box size) apart — 2.4e-07 at a 1e-6 tilt on a 31.8 mm
                # box, which is the measured worst and is geometry, not error.
                # Comparing across the tie would be comparing two different
                # (correct) answers.
                agreed += 1
                if p_om > w_om:
                    w_om = p_om
                if p_mo > w_mo:
                    w_mo = p_mo
                if p_cp < cp:
                    cp = p_cp
        n_count_bad += cbad
        print("   scale", sc, ": poses", NPER, " count mismatches", cbad,
              "(the axis TIE — see the docstring)")

    print("  poses compared:", tested, "  of which the tie fell the same way:",
          agreed)
    print("  SAME-AXIS poses  — worst MuJoCo->ours:", w_mo,
          "   ours->MuJoCo:", w_om, "   closest pair among ours:", cp)
    print("  ALL poses        — worst MuJoCo->ours:", w_mo_all,
          "   ours->MuJoCo:", w_om_all,
          "  (inflated by the axis tie; not asserted)")
    print("  worst |d dist| over ALL poses:", w_d)

    assert_true(
        tested >= 100,
        String("only ") + String(tested)
        + " poses produced a contact — this sweep gates almost nothing.",
    )
    # ⚠ NON-VACUITY OF THE RESTRICTION. The position assertions below look only
    # at poses where both engines chose the same separating axis; if that set
    # were empty or tiny they would pass while testing nothing, and the tie is
    # common enough at the smallest scales for that to be a real risk.
    # ⚠⚠ WHAT IS *NOT* ASSERTED, AND WHY. Away from the exact pose every
    # position statistic is dominated by the separating-axis TIE, and the tie
    # is not observable from here: `mjn == nc` is NOT a proxy for "same axis",
    # since both faces yield four points. Two engines that pick different
    # reference faces project the same four corners onto different planes and
    # land O(perturbation x box extent) apart — 2.4e-07 measured, against a
    # bound of ~7e-08 from a 1e-6 tilt on the 63.6 mm diagonal. Near-duplicate
    # pairs are the same story: at these poses MuJoCo produces them too, so
    # "no near-duplicates" is a property of the EXACT pose (asserted in
    # `test_box_box_at_the_exactly_degenerate_pose`, where the margin is
    # 0.0318 against a 1e-6 threshold) and NOT of this neighbourhood.
    #
    # The two things below survive the tie.
    assert_true(
        w_d <= TOL_DIST,
        String("a manifold DEPTH is ") + String(w_d)
        + " from MuJoCo's, over " + String(TOL_DIST)
        + ". Depth is the distance between two parallel planes and does not"
        " depend on which of them was chosen as the reference, so unlike the"
        " point positions this is tie-independent and a real disagreement.",
    )
    # A genuinely MISSED contact is a whole corner away — 31.8 mm on this box —
    # so this catches the failure it exists for by three orders of magnitude
    # while leaving room for the tie's O(1e-7) corner displacement.
    assert_true(
        w_mo_all <= TOL_MISS,
        String("a contact MuJoCo reports has NO point of ours within ")
        + String(w_mo_all) + ", over " + String(TOL_MISS)
        + " — at that distance it is not the axis tie (which moves a corner by"
        " ~1e-07 here) but a corner of the manifold we are not emitting at"
        " all, leaving the solver blind to it.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
