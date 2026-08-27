"""The convex hull must be the SAME at float32 and float64.

    pixi run mojo run -I . tests/physics3d/test_convex_hull_dtype_invariance.mojo

⚠ RUN FROM THE REPO ROOT — the fixture meshes are addressed by repo-root
relative path.

WHAT THIS GATES, AND WHY IT IS A CORRECTNESS TEST NOT A PERFORMANCE ONE.
`compute_convex_hull` used to run in `DTYPE`, and float32 built a DIFFERENT
HULL from float64 on the same mesh:

    SO-ARM100, ten collision meshes    float64  2 551 vertices
                                       float32  2 636 vertices

So a float32 env and a float64 env of the same model carried different
collision geometry. ⚠⚠ AND NOTHING COULD SEE IT: every mesh gate in the tree
builds at float64, while the float32 path is what the RENDERER and the GPU
batch use. The hull is now built in float64 always and converted on the way
out, so the two agree by construction — this file is what keeps that true.

The symptom that led here was speed, and it is worth recording that it was only
a symptom: float32 was 6x slower on SO-ARM100 and >15x on SO-ARM101 (>280 s vs
19 s) because it was doing MORE WORK, not slower arithmetic. This is plain CPU
code. Both now run in the same time.

⚠ THE FIXTURE IS A REAL MESH, NOT A SYNTHETIC CLOUD. The divergence lived in
near-coplanar decisions on scanned geometry; a tidy ellipsoid of points does
not reproduce it, and a gate that cannot reproduce the bug is decoration.

⚠ HULL VERTEX COUNT IS THE ASSERTION, and it is the right one: a hull that
loses vertices is a strictly SMALLER SOLID, which is the failure mode the
whole exact-hull path exists to prevent — GJK/EPA then sees a shrunken shape
and drops shallow contacts, with one sign, silently.
"""

from std.math import abs
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.collision.convex_hull import load_mesh_hull
from mojo_rl.physics3d.model.mesh_inertia import MeshInertia


def _hull_size[DTYPE: DType](path: String) raises -> Int:
    """Hull vertex count for one STL, built at `DTYPE`."""
    var mesh_vert = List[Scalar[DTYPE]]()
    var mesh_vertadr = List[Int]()
    var mesh_vertnum = List[Int]()
    var num_meshes = 0
    var mesh_polyadr = List[Int]()
    var mesh_polynum = List[Int]()
    var poly_vert = List[Int]()
    var poly_vertadr = List[Int]()
    var poly_vertnum = List[Int]()
    var poly_normal = List[Scalar[DTYPE]]()
    var polymap = List[Int]()
    var polymap_adr = List[Int]()
    var polymap_num = List[Int]()
    var edge_adr = List[Int]()
    var edge_list = List[Int]()
    # ⚠ THE TRIANGLE SOUP, added when `mj_rayMesh` needed the mesh's ORIGINAL
    # faces (the hull has no holes). This file did not compile from that change
    # until now — a gate that cannot parse is not a gate, and this is the one
    # watching for a dtype-dependent hull.
    var mesh_tri = List[Scalar[DTYPE]]()
    var mesh_triadr = List[Int]()
    var mesh_trinum = List[Int]()
    var mi = MeshInertia[DTYPE]()
    _ = load_mesh_hull[DTYPE](
        path, mesh_vert, mesh_vertadr, mesh_vertnum, num_meshes,
        mesh_polyadr, mesh_polynum, poly_vert, poly_vertadr, poly_vertnum,
        poly_normal, polymap, polymap_adr, polymap_num, edge_adr, edge_list,
        mesh_tri, mesh_triadr, mesh_trinum,
        mi,
    )
    return mesh_vertnum[0]


def _both(name: String, path: String) raises:
    var a = _hull_size[DType.float64](path)
    var b = _hull_size[DType.float32](path)
    print("  ", name, " f64 hull", a, " f32 hull", b)
    assert_true(
        a == b,
        name + ": float64 gives " + String(a) + " hull vertices, float32 gives "
        + String(b) + " — the hull must not depend on the runtime dtype; it is"
        " build-time geometry and is constructed in float64 for exactly this"
        " reason",
    )
    assert_true(a > 0, name + ": empty hull — did the mesh load at all?")


def test_small_collision_mesh() raises:
    """A hand-authored collision mesh — few vertices, many coplanar faces.

    Coplanarity is where a tolerance-sensitive hull diverges, so the SMALL
    fixture is not the easy case here; it is the sharp one.
    """
    _both(
        "Fixed_Jaw_Collision_2",
        "mojo_rl/envs/robots/assets/so_arm100/Fixed_Jaw_Collision_2.stl",
    )


def test_scanned_visual_mesh() raises:
    """A full scanned part — thousands of vertices, the case that diverged."""
    _both(
        "Wrist_Pitch_Roll",
        "mojo_rl/envs/robots/assets/so_arm100/Wrist_Pitch_Roll.stl",
    )


def test_large_scanned_mesh() raises:
    """SO-ARM101's largest, and the one whose float32 build took minutes.

    ⚠ Kept despite the cost (~2 s) because it is the only fixture at the size
    where the old float32 path became unusable rather than merely different.
    """
    _both(
        "wrist_roll_pitch_so101_v2",
        "mojo_rl/envs/robots/assets/so_arm101/wrist_roll_pitch_so101_v2.stl",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
