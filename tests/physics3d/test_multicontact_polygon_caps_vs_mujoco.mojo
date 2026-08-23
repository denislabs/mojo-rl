"""A face polygon wider than the cap silently lost its whole manifold.

    pixi run mojo run -I . tests/physics3d/test_multicontact_polygon_caps_vs_mujoco.mojo

WHAT WAS THERE. `native_multicontact` sizes its working polygons with two
comptime constants, and `_mesh_face` opens with

    if num > MC_MAX_POLYVERT:
        return 0

A return of 0 is the routine's own "these features do not line up" answer, so
the caller did exactly what the reference does when no faces align: it emitted
the single EPA point. The cap therefore did not overflow, crash or warn — it
DOWNGRADED the pair, and the file's own comment claimed the opposite ("THESE
CAPS ARE CHECKED AT MODEL BUILD, NOT SILENTLY OBEYED HERE").

⚠⚠ THE OLD VALUES WERE 16 AND 16, AND MENAGERIE IS NOT NEAR THEM. Measured on
the 3.10.0 runtime over the 59 mesh-bearing scenes, from `mesh_polyvertnum`
and `mesh_polymapnum`:

    widest face polygon         144 vertices (robotiq_2f85)   47/59 over 16
    most polygons at one vertex  47          (flexiv_rizon4)  39/59 over 16

MuJoCo has no such cap at all: `npolygonmax` / `nmeshdegmax` are runtime model
fields, sized per model. Ours must be comptime — a Metal kernel cannot size a
local array from a model field.

⚠⚠ METAL USED TO SET THE CEILING, AND NO LONGER DOES. These arrays were
function-local, so they were per-thread stack in the collision kernel.
Measured by bisection on this machine, `test_plane_mesh_fields` compiled at
(56, 48) and failed at (64, 48) and at (56, 64) with

    Failed to create compute pipeline state: Compute function exceeds
    available stack space

so 56 and 48 were the largest values that kept the GPU path alive, and 21
scenes carried a polygon wider than 56 — they kept the single point on their
widest faces and said so at build. The `MC_MAX_POLYVERT`-sized buffers now
live in the CCD workspace row (`collision/ccd_workspace.mojo`), which is where
MuJoCo has always kept its equivalents, so the width cap is Menagerie's 144
and the degree cap stays 48 (the tree's worst is 47, so that axis never needed
unlocking).

MEASURED CONSEQUENCE. kinova_gen3 ships `home` with its base and shoulder
hulls 12 mm interpenetrated across two faces of 31 and 29 vertices. MuJoCo
clips them to a four-point manifold; we returned one point.

    kinova_gen3   4.351e-02 -> 5.709e-12   (worst |d(qpos)|, one step)

⚠ EXACTLY TWO SCENES IN THE SWEEP MOVED, and that understates it. The cap only
bites on a pair that is actually touching, and at step one out of a keyframe
most of those 47 scenes are not touching on their widest face. The other one
that moved is `hello_robot_stretch_3`, 1.733e-02 -> 5.626e-02: its manifold is
now GENERATED and lands on a different face pair from MuJoCo's, a separate
narrow-phase defect this cap was masking behind a single point.

⚠⚠ THE SAME BLINDNESS IS WHY THE 56 -> 144 RAISE NEEDED ITS OWN FIXTURE.
NEITHER sweep column moved between 56 and 144 — not `csweep.py`, not
`sweepN.py` — because the tree's wide faces are not in contact at their own
keyframes. Both columns are demonstrably sensitive to the cap (dropping it to
4 takes csweep from 86/93 clean to 84 and the board from 77/85 to 76), so that
is a real negative, not a blind measurement. `test_a_face_wider_than_the_old_cap_still_clips`
is the row that can see it: a synthetic 100-gon prism with a box on its cap,
where 56 gives ONE point and 144 gives MuJoCo's four.

⚠ THE COST IS THE WORK, NOT THE ARRAYS. Interleaved, min of 3 rounds, 200
steps: barkour 106.1 -> 105.7 us/step and spot 115.7 -> 115.5 — unchanged,
because a pair that never reached the manifold path never touches the bigger
buffers. kinova pays 55.4 -> 70.8 us/step (+28%), which is the price of
clipping a manifold and solving four contact rows instead of one.
"""

from std.math import abs, sqrt
from std.python import Python
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.integrator.implicit import ImplicitIntegrator
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.collision.ccd_workspace import (
    MC_MAX_POLYVERT, MC_MAX_DEG,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_MESH_META_SIZE, MESH_META_IDX_POLYADR, MESH_META_IDX_POLYNUM,
    MODEL_MESH_POLY_SIZE, MESH_POLY_IDX_VERTNUM,
)
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE,
    CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX, CONTACT_IDX_NY, CONTACT_IDX_NZ, CONTACT_IDX_DIST,
)

comptime DT = DType.float64

comptime KINOVA = String(
    "references/mujoco_menagerie-main/kinova_gen3/scene.xml"
)

# A 100-gon prism with a box resting flat on its top cap — the WIDTH axis on
# its own. Menagerie's wide faces (robotiq_2f85's 144-vertex base_mount, g1's
# 117) are not touching anything at their own keyframes, so no sweep row moves
# when the cap is raised; this fixture puts one in contact deliberately.
#
# ⚠ SYNTHETIC AND CONVEX BY CONSTRUCTION, deliberately. The first attempt used
# robotiq_2f85's real base_mount against a box and could not be used: our
# surface sits ~17 mm from MuJoCo's on that mesh, so the fixture measured a
# hull/frame divergence rather than the cap. A prism whose hull IS its vertex
# set removes that variable — both engines agree on 200 verts, 102 polygons
# and a widest face of 100.
comptime NGON = String(
    """
<mujoco model="wide face manifold">
  <option timestep="0.002"/>
  <asset>
    <mesh name="disc" file="tests/physics3d/assets/ngon100_prism.stl"/>
  </asset>
  <worldbody>
    <geom name="disc" type="mesh" mesh="disc"/>
    <body name="b" pos="0 0 0.0295"><freejoint/>
      <geom name="blk" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""
)
# MuJoCo 3.10.0 on that string: the box face clipped against the 100-gon cap,
# four points at the box's own corners.
comptime NGON_MJ_NCON = 4
comptime NGON_MJ_DIST = -0.0005
# The cap this fixture needs. Below it the manifold collapses to ONE point and
# the box tips: |qvel|max after one step goes 0.00212 -> 0.1221, a factor of 58.
comptime NGON_POLYVERT = 100

# MuJoCo 3.10.0 at kinova's keyframe 0, `mj_forward`. Four contacts, all on
# one plane, all at one depth, normal +z (geom1 -> geom2; our record stores
# `body_b -> body_a`, which is the opposite sign).
comptime MJ_NCON = 4
comptime MJ_DIST = -0.012044999767961329
comptime MJ_PLANE_Z = 0.164827498456591
# The centroid of MuJoCo's four points, and the radii they sit at around it.
comptime MJ_CX = -0.0014811771189564577
comptime MJ_CY = 0.0008911941017506531
comptime MJ_RMIN = 0.042
comptime MJ_RMAX = 0.046

# ⚠ THESE ARE MENAGERIE'S OWN WORST CASE, NOT A MACHINE LIMIT ANY MORE. They
# used to be 56 and 48 — the largest pair that still compiled a Metal
# collision kernel, found by bisection, because the buffers were per-thread
# stack. They now live in the CCD workspace row (`ccd_workspace.mojo`), so the
# requirement is the tree's and not the compiler's.
comptime REQUIRED_POLYVERT = 144
comptime REQUIRED_DEG = 50
# Menagerie's worst, measured per SCENE over collision meshes only — all 96
# `scene*.xml` in the tree.
comptime MENAGERIE_MAX_POLYVERT = 144  # robotiq_2f85
comptime MENAGERIE_MAX_DEG = 47        # flexiv_rizon4
# ⚠⚠ AND THE OTHER HALF OF THE CORPUS. `MC_MAX_DEG` sat at 48 — Menagerie's
# worst plus one — for as long as the census only ever ran on the reference
# tree, while a model this repo SHIPS needs 50. Same method, all 57 in-repo
# `*.xml`: `envs/robots/assets/so_arm101.xml` peaks at 80 / 50, its STS3215
# servo hulls each carrying a vertex with 50 incident polygons and its mirror
# with 49. Everything else in the repo is at or under 50 / 34.
comptime REPO_MAX_POLYVERT = 80        # so_arm101 (82 before the hull
                                       # vertex reduction narrowed its faces)
comptime REPO_MAX_DEG = 50             # so_arm101, sts3215_03a_v1

comptime _IMPFAST = ImplicitIntegrator[
    DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", SKIP_RNE_DERIV=True,
    MAX_CONDIM=6,
]


def test_caps_cover_menagerie() raises:
    """The constants are a REQUIREMENT, not a convenience.

    ⚠ THIS IS THE ROW THAT STOPS THEM SHRINKING BACK. The kinova assertion
    below needs only 38 and 21 and the n-gon one only 100; every other scene
    that would lose a manifold is invisible to both, because a cap only bites
    on a pair that is actually touching on its widest face. Stating the tree's
    own worst here is what keeps the rest of Menagerie in the gate.

    ⚠ IT USED TO ASSERT A METAL CEILING AND NOW ASSERTS A MODEL REQUIREMENT.
    The old floor was 56 — the largest width that still compiled a collision
    kernel, found by bisection, because the buffers were per-thread stack.
    They live in the CCD workspace row now, so the number that matters is the
    tree's, not the compiler's.
    """
    print("=== the caps against Menagerie's worst ===")
    print("  MC_MAX_POLYVERT", MC_MAX_POLYVERT, " required",
          REQUIRED_POLYVERT, " Menagerie's worst", MENAGERIE_MAX_POLYVERT)
    print("  MC_MAX_DEG     ", MC_MAX_DEG, " required", REQUIRED_DEG,
          " Menagerie's worst", MENAGERIE_MAX_DEG)
    assert_true(
        MC_MAX_POLYVERT >= MENAGERIE_MAX_POLYVERT,
        "MC_MAX_POLYVERT is " + String(MC_MAX_POLYVERT) + " but robotiq_2f85"
        " has a " + String(MENAGERIE_MAX_POLYVERT) + "-vertex face. Every"
        " pair touching on a wider face falls back to a SINGLE point where"
        " MuJoCo clips a four-point manifold — and a single point cannot hold"
        " a flat contact, it lets the body rotate about it.",
    )
    assert_true(
        MC_MAX_DEG >= MENAGERIE_MAX_DEG,
        "MC_MAX_DEG is " + String(MC_MAX_DEG) + " but flexiv_rizon4 has a"
        " vertex where " + String(MENAGERIE_MAX_DEG) + " polygons meet."
        " `_mesh_normals` stops collecting candidate face normals at the cap,"
        " so the matching face may never be offered to `alignedFaces`.",
    )
    # ⚠⚠ THE ROW THAT WOULD HAVE CAUGHT THE 48. Asserting only against the
    # reference tree is what let a cap that a SHIPPED model exceeds stand: the
    # census ran on Menagerie, the number read as "every model", and
    # so_arm101's 50 was invisible to every gate in the suite.
    print("  REPO worst  pv", REPO_MAX_POLYVERT, " deg", REPO_MAX_DEG,
          " (so_arm101)")
    assert_true(
        MC_MAX_POLYVERT >= REPO_MAX_POLYVERT and MC_MAX_DEG >= REPO_MAX_DEG,
        "the caps are " + String(MC_MAX_POLYVERT) + " / " + String(MC_MAX_DEG)
        + " but this repo's own so_arm101 needs " + String(REPO_MAX_POLYVERT)
        + " / " + String(REPO_MAX_DEG) + ". A cap measured only against"
        " `references/` is a cap for someone else's models.",
    )
    print("  PASS")


def test_a_face_wider_than_the_old_cap_still_clips() raises:
    """A 100-vertex face against a box — the width axis, isolated.

    ⚠ THIS ROW EXISTS BECAUSE NO SWEEP COLUMN COULD SHOW THE RAISE. Both
    `csweep.py` and `sweepN.py` are demonstrably sensitive to this cap —
    dropping it to 4 takes csweep from 86/93 clean to 84 and the board from
    77/85 to 76, with kinova going 6.7e-12 to 5.8e-03 — and NEITHER moved
    between 56 and 144, because the tree's wide faces (robotiq_2f85's 144,
    unitree_g1's 117, spot's 110) are not touching anything at their own
    keyframes. A raise nothing can see is a raise nobody can defend, so this
    fixture puts a wide face in contact on purpose.

    ⚠ MEASURED AT BOTH CAPS, on this exact fixture:

        MC_MAX_POLYVERT   contacts   |qvel|max after one step
        56                1          0.1221
        144               4          0.00212

    The single point is the reference's own no-manifold fallback, and the box
    tips about it — a factor of 58 in the angular response. That is what a
    lost manifold costs, and it is why this is a fidelity fix and not a
    capacity tidy-up.
    """
    print("=== a 100-vertex face clips to MuJoCo's four points ===")
    var src = expand_mjcf(NGON, String("."))
    var fmd = parse_xml_full(src, String("."))
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=4096)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]

    # ⚠ NON-VACUITY FIRST: the fixture is only a test of the cap if its widest
    # polygon is actually wider than the cap used to be. Read it off the model
    # rather than trusting the .stl.
    var pa = Int(m.mesh_meta.data[MESH_META_IDX_POLYADR])
    var pn = Int(m.mesh_meta.data[MESH_META_IDX_POLYNUM])
    var widest = 0
    for k in range(pn):
        var nv = Int(
            m.mesh_polys.data[(pa + k) * MODEL_MESH_POLY_SIZE
                              + MESH_POLY_IDX_VERTNUM]
        )
        if nv > widest:
            widest = nv
    print("  fixture polygons", pn, " widest", widest)
    assert_true(
        widest == NGON_POLYVERT,
        "the fixture's widest polygon is " + String(widest) + ", not "
        + String(NGON_POLYVERT) + " — it no longer exercises the cap",
    )
    assert_true(
        widest > 56,
        "the fixture's widest polygon fits the OLD cap of 56, so this row"
        " would pass with the raise reverted",
    )

    var imp = _IMPFAST(dims)
    imp.step["cpu"](d, m)
    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])

    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_string(String(NGON))
    var md = mujoco.MjData(mm)
    mujoco.mj_forward(mm, md)
    print("  ncon ours", ncon, " MuJoCo", Int(py=md.ncon))
    assert_true(
        Int(py=md.ncon) == NGON_MJ_NCON,
        "the reference moved: MuJoCo now reports " + String(Int(py=md.ncon))
        + " on this fixture, not " + String(NGON_MJ_NCON),
    )
    assert_true(
        ncon == NGON_MJ_NCON,
        "our narrow phase reports " + String(ncon) + " contacts on a "
        + String(widest) + "-vertex face where MuJoCo clips "
        + String(NGON_MJ_NCON) + ". ONE means the manifold was lost to"
        " MC_MAX_POLYVERT — `_mesh_face` returns 0 past the cap, which is the"
        " routine's own no-aligned-faces answer.",
    )

    # Every MuJoCo point must have one of ours at the same place and depth.
    var worst_pos = Float64(0)
    var worst_dist = Float64(0)
    var used = List[Int]()
    for _k in range(ncon):
        used.append(0)
    for i in range(NGON_MJ_NCON):
        var rx = Float64(py=md.contact[i].pos[0])
        var ry = Float64(py=md.contact[i].pos[1])
        var rz = Float64(py=md.contact[i].pos[2])
        var best = -1
        var bd = Float64(1e30)
        for k in range(ncon):
            if used[k] == 1:
                continue
            var o = k * CONTACT_SIZE
            var e = (
                abs(Float64(d.contacts.data[o + CONTACT_IDX_POS_X]) - rx)
                + abs(Float64(d.contacts.data[o + CONTACT_IDX_POS_Y]) - ry)
                + abs(Float64(d.contacts.data[o + CONTACT_IDX_POS_Z]) - rz)
            )
            if e < bd:
                bd = e
                best = k
        assert_true(best >= 0, "ran out of contacts to match")
        used[best] = 1
        var dd = abs(
            Float64(d.contacts.data[best * CONTACT_SIZE + CONTACT_IDX_DIST])
            - Float64(py=md.contact[i].dist)
        )
        if bd > worst_pos:
            worst_pos = bd
        if dd > worst_dist:
            worst_dist = dd
    print("  worst |d pos|", worst_pos, " worst |d dist|", worst_dist)
    # Both engines clip the same box face against the same convex cap, so this
    # is an equality check, not a tolerance negotiation.
    assert_true(
        worst_pos < 1e-9 and worst_dist < 1e-8,
        "the four points differ from MuJoCo's by " + String(worst_pos)
        + " / " + String(worst_dist),
    )
    _ = d^
    _ = m^
    print("  PASS")


def test_kinova_manifold_is_four_points() raises:
    """The model it was found on, against MuJoCo's own four.

    ⚠ THE POSITIONS ARE NOT PINNED, ON PURPOSE. MuJoCo keeps a different four
    of the same clipped ring (a pruner tie-break documented in
    `native_multicontact`'s `MC_DEBUG_RING`), so pinning them would gate a
    known-open residual instead of the cap. What IS pinned is everything the
    manifold must agree on: how many points, their common depth, the plane
    they lie in, the normal, and the ring radius they sit at around MuJoCo's
    centroid — which together say "we clipped the same two faces".
    """
    print("=== kinova_gen3 base/shoulder manifold ===")
    var src = read_model_source(KINOVA)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    assert_true(
        dims.get_nkey() > 0,
        "kinova_gen3 must carry a keyframe — qpos0 does not interpenetrate"
        " and this gate would measure nothing",
    )
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var integ = _IMPFAST(dims)
    integ.step["cpu"](d, m)

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    print("  ncon", nc, "  (MuJoCo", MJ_NCON, ")")
    for k in range(nc):
        var o = k * CONTACT_SIZE
        print(
            "   ", k, " bodies",
            Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A])),
            Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B])),
            " pos", Float64(d.contacts.data[o + CONTACT_IDX_POS_X]),
            Float64(d.contacts.data[o + CONTACT_IDX_POS_Y]),
            Float64(d.contacts.data[o + CONTACT_IDX_POS_Z]),
            " dist", Float64(d.contacts.data[o + CONTACT_IDX_DIST]),
        )
    assert_true(
        nc == MJ_NCON,
        "ncon is " + String(nc) + " where MuJoCo has " + String(MJ_NCON)
        + ". A 1 here is the single EPA point, i.e. the manifold was dropped;"
        " anything above 4 means the clip is emitting points MuJoCo prunes.",
    )
    var worst_d = 0.0
    var worst_z = 0.0
    var worst_n = 0.0
    var rmin = 1e30
    var rmax = -1e30
    for k in range(nc):
        var o = k * CONTACT_SIZE
        var px = Float64(d.contacts.data[o + CONTACT_IDX_POS_X])
        var py = Float64(d.contacts.data[o + CONTACT_IDX_POS_Y])
        var pz = Float64(d.contacts.data[o + CONTACT_IDX_POS_Z])
        var ed = abs(Float64(d.contacts.data[o + CONTACT_IDX_DIST]) - MJ_DIST)
        if ed > worst_d:
            worst_d = ed
        var ez = abs(pz - MJ_PLANE_Z)
        if ez > worst_z:
            worst_z = ez
        # The record stores `body_b -> body_a`, the opposite of MuJoCo's
        # frame, so only the AXIS is compared: |nz| must be 1 and nx, ny 0.
        var en = abs(abs(Float64(d.contacts.data[o + CONTACT_IDX_NZ])) - 1.0)
        var ex = abs(Float64(d.contacts.data[o + CONTACT_IDX_NX]))
        var ey = abs(Float64(d.contacts.data[o + CONTACT_IDX_NY]))
        if ex > en:
            en = ex
        if ey > en:
            en = ey
        if en > worst_n:
            worst_n = en
        var dx = px - MJ_CX
        var dy = py - MJ_CY
        var r = sqrt(dx * dx + dy * dy)
        if r < rmin:
            rmin = r
        if r > rmax:
            rmax = r
    print("  worst |d dist|", worst_d, " |d plane z|", worst_z,
          " |d normal|", worst_n)
    print("  ring radius about MuJoCo's centroid: [", rmin, ",", rmax, "]")
    assert_true(
        worst_d < 1e-8,
        "the four points must share MuJoCo's depth " + String(MJ_DIST)
        + "; worst |d| = " + String(worst_d),
    )
    assert_true(
        worst_z < 1e-8,
        "the four points must lie in MuJoCo's contact plane z = "
        + String(MJ_PLANE_Z) + "; worst |d| = " + String(worst_z),
    )
    assert_true(
        worst_n < 1e-12,
        "the manifold normal must be the face normal, +-z here; worst"
        " deviation " + String(worst_n),
    )
    # ⚠ THE ROW THAT SAYS "THE SAME TWO FACES". Four points at the right depth
    # could still come from the wrong feature; MuJoCo's sit on a ring of
    # radius 0.0427-0.0451 about its centroid, and a manifold clipped from a
    # different face pair does not land there.
    assert_true(
        rmin > MJ_RMIN and rmax < MJ_RMAX,
        "the four points sit at radius [" + String(rmin) + ", "
        + String(rmax) + "] about MuJoCo's centroid, outside its own ["
        + String(MJ_RMIN) + ", " + String(MJ_RMAX) + "] — the clip found a"
        " different face pair, not a different four points of the same one.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
