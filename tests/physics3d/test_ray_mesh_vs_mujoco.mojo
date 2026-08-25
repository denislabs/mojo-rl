"""`ray/mesh.mojo` vs `mj_rayMesh` — rays at a NON-CONVEX mesh.

    pixi run mojo run -I . tests/physics3d/test_ray_mesh_vs_mujoco.mojo

⚠⚠ THE FIXTURE HAS TO BE NON-CONVEX OR THIS FILE PROVES NOTHING. The triangle
store (`Model.mesh_tris`) exists only because `Model.mesh_verts` is the convex
HULL and a ray aimed into a cutout must find the hole. On a convex mesh the
hull IS the mesh, so a `ray_mesh` that quietly walked hull triangles — or a
`ray_mesh` on a model whose soup was never populated but whose hull was —
would agree with MuJoCo on every ray. Every other STL fixture in
`tests/physics3d/assets/` is convex (a cube, a hex prism, an n-gon prism), so
`notch.stl` was added for this: a box with a slot cut in its top face.

`test_the_fixture_is_not_convex` is the standing guard, and it is not a
formality — it asserts the two rays that separate the surfaces:

    straight down the SLOT  ->  z = -0.010   (the slot floor)
    straight down the LID   ->  z = +0.040   (the top face)

A hull would answer +0.040 for BOTH. If those ever agree, the fixture has
stopped being non-convex and the sweep below is measuring nothing.

⚠ `half_extents` IS AN OPEN QUESTION THIS FILE ANSWERS, and the answer is that
ours is a PLACEHOLDER. `mj_rayMesh` rejects on `geom_size` before touching a
triangle; MuJoCo stores the mesh's AABB half-extents there ([0.0423, 0.05,
0.05] for this notch — not symmetric, because a mesh is recentred on its centre
of mass) and our parser stores 0.5. Inert today, since
`broadphase_sap._aabb_half_extents` sends a mesh to `rbound` instead, and safe
for `ray_mesh` because too-LARGE only costs triangles walked. See
`test_our_mesh_box_over_approximates_mujocos_aabb`, which asserts the asymmetry
that actually matters rather than an equality our tables do not hold.

WHAT THIS GATE WAS PROVEN ABLE TO FAIL
======================================
    injected defect                            caught by       |dt|
    ----------------------------------------   -------------   ---------
    negative triangle distances accepted        hits 454->103   0.21
    normal not rotated out of the local frame   |dnormal| 1.81  UNCHANGED
    ---
    the bounding-box reject removed             NOTHING         UNCHANGED

⚠ THE LAST ROW IS A CONFIRMED PREDICTION, NOT A HOLE. Removing the reject
cannot change an answer — it only skips a rejection — and since our box
over-approximates, it never rejected a real hit in the first place. Identical
output is what should happen; it is recorded so nobody reads the reject as
untested and tightens it without re-measuring.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_MESH_ID,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRIADR,
    MESH_META_IDX_TRINUM,
)
from mojo_rl.physics3d.ray import ray_mesh

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime Quat = QuatGeneric[DT]

# Moved and turned, for the reason the hfield gate spells out: at the origin
# with no rotation `ray_map` is the identity and every frame error reads exact.
comptime MESH_XML = String(
    """
<mujoco model="notch ray gate">
  <asset>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"/>
  </asset>
  <worldbody>
    <geom name="g" type="mesh" mesh="notch" pos="0.09 -0.04 0.03" euler="15 -25 40"/>
  </worldbody>
</mujoco>
"""
)

comptime NCASE = 500


struct Lcg(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u01(mut self) -> Float64:
        self.s = self.s * 1664525 + 1013904223
        return Float64((self.s >> 16) & 0xFFFFFFF) / Float64(0x10000000)

    def sym(mut self, a: Float64) -> Float64:
        return (self.u01() * 2.0 - 1.0) * a


struct Built(Movable):
    var m: Model[DT, DynDims]

    def __init__(out self) raises:
        var fmd = parse_xml_full(MESH_XML, String("."))
        # ⚠ `nmesh_tri` is what turns the soup ON. Left at its default 0 the
        # model carries no triangles and every ray reports NO HIT — which is
        # what `test_the_soup_is_actually_carried` exists to catch.
        var dims = dims_from_flat(
            fmd, max_contacts=8, nmesh_verts=256, nmesh_tri=64
        )
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        _ = spec_fields_runtime[DT](fmd, dims, m)
        self.m = m^


def _geom_pose(b: Built) -> Tuple[Vec3, Quat, Vec3]:
    return (
        Vec3(
            Float64(b.m.geoms.data[GEOM_IDX_POS_X]),
            Float64(b.m.geoms.data[GEOM_IDX_POS_Y]),
            Float64(b.m.geoms.data[GEOM_IDX_POS_Z]),
        ),
        Quat(
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_W]),
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_X]),
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_Y]),
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_Z]),
        ),
        Vec3(
            Float64(b.m.geoms.data[GEOM_IDX_HALF_X]),
            Float64(b.m.geoms.data[GEOM_IDX_HALF_Y]),
            Float64(b.m.geoms.data[GEOM_IDX_HALF_Z]),
        ),
    )


def _tri_window(b: Built) -> Tuple[Int, Int]:
    var mid = Int(Float64(b.m.geoms.data[GEOM_IDX_MESH_ID]))
    var base = mid * MODEL_MESH_META_SIZE
    return (
        Int(Float64(b.m.mesh_meta.data[base + MESH_META_IDX_TRIADR])),
        Int(Float64(b.m.mesh_meta.data[base + MESH_META_IDX_TRINUM])),
    )


def test_the_fixture_is_not_convex() raises:
    """A hull answers the same distance for both rays; the mesh does not."""
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(
        String(
            """
<mujoco model="axis aligned notch">
  <asset><mesh name="notch" file="tests/physics3d/assets/notch.stl"/></asset>
  <worldbody><geom name="g" type="mesh" mesh="notch" pos="0 0 0"/></worldbody>
</mujoco>
"""
        )
    )
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)
    var down = np.zeros(3)
    down[2] = -1.0
    var nrm = np.zeros(3)

    var slot = np.zeros(3)
    slot[2] = 1.0
    var t_slot = Float64(py=mujoco.mj_rayMesh(m, d, 0, slot, down, nrm))
    var lid = np.zeros(3)
    lid[0] = 0.035
    lid[2] = 1.0
    var t_lid = Float64(py=mujoco.mj_rayMesh(m, d, 0, lid, down, nrm))

    var z_slot = 1.0 - t_slot
    var z_lid = 1.0 - t_lid
    print("  down the SLOT z =", z_slot, "  down the LID z =", z_lid)
    assert_true(
        abs(z_slot + 0.01) < 1e-6,
        "the slot floor is at z=" + String(z_slot) + ", expected -0.01 —"
        " notch.stl is not the shape this gate assumes",
    )
    assert_true(
        abs(z_lid - 0.04) < 1e-6,
        "the lid is at z=" + String(z_lid) + ", expected +0.04",
    )
    assert_true(
        abs(z_slot - z_lid) > 1e-3,
        "THE FIXTURE IS CONVEX: both rays land at the same depth, so this"
        " file cannot tell the triangle store from the hull",
    )


def test_the_soup_is_actually_carried() raises:
    """`nmesh_tri` off is silent, so the count is asserted before any ray."""
    var b = Built()
    var w = _tri_window(b)
    print("  triadr", w[0], " trinum", w[1])
    assert_true(
        w[1] == 28,
        "the mesh carries " + String(w[1]) + " triangles, expected 28 —"
        " a soup of 0 is `nmesh_tri` left at its default, and every ray below"
        " would report NO HIT while looking like a clean pass",
    )


def test_our_mesh_box_over_approximates_mujocos_aabb() raises:
    """The box `mj_rayMesh` rejects on: ours must never be SMALLER.

    ⚠⚠ MEASURED, AND OURS IS NOT THE AABB. MuJoCo puts the mesh's axis-aligned
    bounding half-extents in `geom_size` — for this notch, and note it is not
    symmetric because a mesh is recentred on its centre of mass:

        MuJoCo   [0.0423, 0.05, 0.05]
        ours     [0.5,    0.5,  0.5 ]   <- the parser's placeholder

    That is INERT rather than wrong today: `broadphase_sap._aabb_half_extents`
    dispatches on geom type and a mesh falls through to `rbound`, its
    bounding-SPHERE radius, so nothing in collision reads these three for a
    mesh. `ray_mesh` is the first consumer, and a too-LARGE box only costs
    triangles walked while a too-SMALL one silently loses hits — which is the
    asymmetry this asserts, rather than an equality our tables do not hold.

    ⚠ If our `geom_size` is ever made tight, this test should tighten with it;
    it is written as an inequality because that is the property `ray_mesh`
    actually needs, not because equality would be nice to have.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(MESH_XML))
    var b = Built()
    var g = _geom_pose(b)
    var half = g[2]
    var slack = 0.0
    for k in range(3):
        var ours = Float64(half.x if k == 0 else (half.y if k == 1 else half.z))
        var theirs = Float64(py=m.geom_size[0][k])
        assert_true(
            ours >= theirs - 1e-12,
            "axis " + String(k) + ": our box half-extent " + String(ours)
            + " is SMALLER than MuJoCo's AABB " + String(theirs)
            + " — `ray_mesh`'s bounding reject would drop rays the reference"
            " reports as hits, and the sweep's `split` count is where that"
            " would show",
        )
        slack = max(slack, ours - theirs)
    print("  ours [", half.x, half.y, half.z, "]  slack over the AABB", slack)


def test_ray_mesh_vs_mujoco() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(String(MESH_XML))
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)

    var b = Built()
    var g = _geom_pose(b)
    var pos = g[0]
    var quat = g[1]
    var half = g[2]
    var w = _tri_window(b)
    var triadr = w[0]
    var ntri = w[1]

    var a_pnt = np.zeros(3)
    var a_vec = np.zeros(3)
    var a_nrm = np.zeros(3)

    var rng = Lcg(0x5EED11)
    var hits = 0
    var split = 0
    var worst_t = 0.0
    var worst_n = 0.0

    var lx = quat.rotate_vec(Vec3(1.0, 0.0, 0.0))
    var ly = quat.rotate_vec(Vec3(0.0, 1.0, 0.0))
    var lz = quat.rotate_vec(Vec3(0.0, 0.0, 1.0))

    for _ in range(NCASE):
        var pick = rng.u01()
        var eye: Vec3
        var aim: Vec3
        if pick < 0.35:
            # Down the slot's axis, jittered across it — the rays whose answer
            # the hull would get wrong.
            eye = pos + lz * (0.3 + rng.u01() * 0.5) + lx * rng.sym(0.03) + ly * rng.sym(0.05)
            aim = pos + lx * rng.sym(0.03) + ly * rng.sym(0.05)
        elif pick < 0.6:
            # Origin INSIDE the solid — the family that caught the capsule
            # defect in the `mju_rayGeom` sweep.
            eye = pos + lx * rng.sym(0.04) + ly * rng.sym(0.04) + lz * rng.sym(0.03)
            aim = eye + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0))
        elif pick < 0.85:
            # General, aimed at the body.
            eye = pos + Vec3(rng.sym(0.6), rng.sym(0.6), rng.sym(0.6))
            aim = pos + lx * rng.sym(0.06) + ly * rng.sym(0.06) + lz * rng.sym(0.05)
        else:
            # Grazing the silhouette.
            eye = pos + Vec3(rng.sym(0.8), rng.sym(0.8), rng.sym(0.8))
            aim = pos + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)).normalized() * 0.055

        var vec = aim - eye
        var ours = ray_mesh[DT](
            pos, quat, half, b.m.mesh_tris.data, triadr, ntri, eye, vec
        )

        a_pnt[0] = eye.x
        a_pnt[1] = eye.y
        a_pnt[2] = eye.z
        a_vec[0] = vec.x
        a_vec[1] = vec.y
        a_vec[2] = vec.z
        var t_mj = Float64(py=mujoco.mj_rayMesh(m, d, 0, a_pnt, a_vec, a_nrm))

        var t_ours = Float64(ours[0])
        if (t_ours >= 0.0) != (t_mj >= 0.0):
            split += 1
            continue
        if t_mj < 0.0:
            continue

        hits += 1
        worst_t = max(worst_t, abs(t_ours - t_mj))
        var n = ours[1]
        worst_n = max(worst_n, abs(Float64(n.x) - Float64(py=a_nrm[0])))
        worst_n = max(worst_n, abs(Float64(n.y) - Float64(py=a_nrm[1])))
        worst_n = max(worst_n, abs(Float64(n.z) - Float64(py=a_nrm[2])))

    print("  hits", hits, "/", NCASE, " splits", split)
    print("  worst |dt|      ", worst_t)
    print("  worst |dnormal| ", worst_n)
    assert_true(
        hits > NCASE // 4,
        "only " + String(hits) + " hits — the sweep is vacuous",
    )
    assert_true(split == 0, String(split) + " hit/miss disagreements")
    # float32-rounded vertices on both sides, so this is the last-bit fold of
    # the plane intersection and not a tolerance chosen to pass.
    assert_true(worst_t < 1e-9, "worst |dt| " + String(worst_t))
    assert_true(worst_n < 1e-9, "worst |dnormal| " + String(worst_n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
