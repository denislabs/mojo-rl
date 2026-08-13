"""The hill-climb support point must EQUAL the linear scan's, on real hulls.

    pixi run mojo run -I . tests/physics3d/test_gjk_hillclimb_support.mojo

⚠ RUN FROM THE REPO ROOT — the fixture meshes are addressed by repo-root
relative path.

WHY THIS EXISTS. `_support_mesh` is the hottest function in mesh collision —
GJK and EPA call it 10-30 times per geom pair per step — and it used to scan
every hull vertex. It now walks the hull's edge graph instead, greedily
stepping to whichever neighbour scores higher on `dir`. On a CONVEX hull the
dot product has exactly one local maximum, so the walk is EXACT rather than
approximate, and this file is what holds that claim to account:

    SO-ARM101, 33 076 hull verts    76.03 ms/step -> 13.11 ms   (13 -> 76 Hz)
    SO-ARM100,  2 551 hull verts    11.95 ms/step ->  5.46 ms   (84 -> 183 Hz)

⚠⚠ A WRONG SUPPORT POINT DOES NOT CRASH. It yields a contact in a plausible
but wrong place, or a slightly shallow penetration, and everything downstream
keeps running. Convergence checks cannot see it either — GJK converges happily
against a support function that is consistently wrong. Only comparing against
the exhaustive answer catches it, which is what this does.

HOW THE TWO PATHS ARE SELECTED WITHOUT A FLAG. `_support_mesh` falls back to
the scan when `mesh_vert_edgeadr[vert_adr] < 0`, which is how `fields/model.
mojo` marks a mesh whose graph was never built. Filling that tensor with -1
therefore exercises the scan on the SAME hull and the SAME directions, with no
test-only branch in the shipping code — a flag would have been a second thing
to keep honest.

⚠ THE FIXTURES ARE REAL SCANNED HULLS, NOT A CUBE. Hill climbing is trivially
correct on a cube and on any tidy convex cloud; where it can go wrong is
near-coplanar facets and long thin geometry, which is what CAD exports are made
of. `test_mesh_support.mojo` already covers the cube.

⚠ THE DIRECTIONS ARE DETERMINISTIC AND COVER THE SPHERE. A handful of axis
directions would miss exactly the wedges where a greedy walk could stall; the
sweep below is a Fibonacci lattice, which spreads without needing an RNG.
"""

from std.math import sqrt, cos, sin, pi, acos
from layout import Layout, LayoutTensor
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.collision.convex_hull import load_mesh_hull
from mojo_rl.physics3d.collision.gjk import _support_mesh
from mojo_rl.physics3d.gpu.constants import mesh_max_edge
from mojo_rl.physics3d.model.mesh_inertia import MeshInertia

comptime D = DType.float64
# Big enough for the largest fixture below; the tensors are comptime-shaped.
comptime NV: Int = 8192
comptime NE: Int = mesh_max_edge(NV)
comptime NDIR: Int = 512


def _check(name: String, path: String) raises:
    """Build one mesh's hull, then compare both support paths over the sphere."""
    var mesh_vert = List[Scalar[D]]()
    var mesh_vertadr = List[Int]()
    var mesh_vertnum = List[Int]()
    var num_meshes = 0
    var mesh_polyadr = List[Int]()
    var mesh_polynum = List[Int]()
    var poly_vert = List[Int]()
    var poly_vertadr = List[Int]()
    var poly_vertnum = List[Int]()
    var poly_normal = List[Scalar[D]]()
    var polymap = List[Int]()
    var polymap_adr = List[Int]()
    var polymap_num = List[Int]()
    var edge_adr = List[Int]()
    var edge_list = List[Int]()
    var mi = MeshInertia[D]()
    _ = load_mesh_hull[D](
        path, mesh_vert, mesh_vertadr, mesh_vertnum, num_meshes,
        mesh_polyadr, mesh_polynum, poly_vert, poly_vertadr, poly_vertnum,
        poly_normal, polymap, polymap_adr, polymap_num, edge_adr, edge_list,
        mi,
    )
    var nverts = mesh_vertnum[0]
    assert_true(
        nverts <= NV,
        name + ": hull has " + String(nverts) + " vertices, over the "
        + String(NV) + " the fixture tensors are sized for",
    )
    assert_true(
        len(edge_list) <= NE,
        name + ": edge graph has " + String(len(edge_list)) + " slots, over "
        + String(NE),
    )

    # Pack into the tensors `_support_mesh` reads.
    # ⚠ HEAP, NOT `InlineArray`. NV*3 float64 is ~200 KB and the edge buffer
    # more; on the stack that is an overflow, not a slow test.
    # ⚠⚠ THE LISTS MUST OUTLIVE THE TENSORS — Mojo destroys at LAST USE, so the
    # `_ = ...` at the end of this function is load-bearing, not tidy-up.
    var vbuf = List[Scalar[D]](length=NV * 3, fill=Scalar[D](0))
    var verts = LayoutTensor[D, Layout.row_major(NV, 3), MutAnyOrigin](
        vbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    for i in range(nverts):
        verts[i, 0] = mesh_vert[i * 3 + 0]
        verts[i, 1] = mesh_vert[i * 3 + 1]
        verts[i, 2] = mesh_vert[i * 3 + 2]

    var abuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var eadr = LayoutTensor[D, Layout.row_major(NV), MutAnyOrigin](
        abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    for v in range(len(edge_adr)):
        eadr[v] = Scalar[D](edge_adr[v])

    var ebuf = List[Scalar[D]](length=NE, fill=Scalar[D](-1))
    var edges = LayoutTensor[D, Layout.row_major(NE), MutAnyOrigin](
        ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    for k in range(len(edge_list)):
        edges[k] = Scalar[D](edge_list[k])

    # ⚠ The scan arm: an all -1 `edgeadr` is exactly how an unbuilt graph is
    # marked, so this takes the fallback WITHOUT a test-only branch.
    var nbuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var no_graph = LayoutTensor[D, Layout.row_major(NV), MutAnyOrigin](
        nbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )

    # A non-trivial pose, so the quaternion rotation is exercised too rather
    # than cancelling out of both arms identically.
    var qx = Scalar[D](0.1830127)
    var qy = Scalar[D](0.3535534)
    var qz = Scalar[D](0.1830127)
    var qw = Scalar[D](0.9014649)
    var px = Scalar[D](0.31)
    var py = Scalar[D](-0.17)
    var pz = Scalar[D](0.44)

    var worst = Scalar[D](0)
    for k in range(NDIR):
        # Fibonacci lattice on the sphere — even coverage, no RNG.
        var t = (Scalar[D](k) + Scalar[D](0.5)) / Scalar[D](NDIR)
        var z = Scalar[D](1) - Scalar[D](2) * t
        var r = sqrt(Scalar[D](1) - z * z)
        var phi = Scalar[D](k) * Scalar[D](2.399963229728653)
        var dx = r * cos(phi)
        var dy = r * sin(phi)

        var hc = _support_mesh[D, NV](
            dx, dy, z, px, py, pz, qx, qy, qz, qw,
            verts, eadr, edges, 0, nverts,
        )
        var sc = _support_mesh[D, NV](
            dx, dy, z, px, py, pz, qx, qy, qz, qw,
            verts, no_graph, edges, 0, nverts,
        )
        var e0 = hc[0] - sc[0]
        var e1 = hc[1] - sc[1]
        var e2 = hc[2] - sc[2]
        var err = sqrt(e0 * e0 + e1 * e1 + e2 * e2)
        if err > worst:
            worst = err
        assert_true(
            err == Scalar[D](0),
            name + ": direction " + String(k) + " (" + String(dx) + ", "
            + String(dy) + ", " + String(z) + ") — hill climb returned ("
            + String(hc[0]) + ", " + String(hc[1]) + ", " + String(hc[2])
            + ") but the exhaustive scan returns (" + String(sc[0]) + ", "
            + String(sc[1]) + ", " + String(sc[2]) + "). The walk stalled at a"
            " vertex that is not the extreme one; on a convex hull that can"
            " only mean the edge graph is not the hull's full adjacency",
        )
    print("  ", name, " hull", nverts, " edges", len(edge_list),
          " directions", NDIR, " worst error", worst)
    # Keep the backing storage alive past the last tensor read — see above.
    _ = vbuf^
    _ = abuf^
    _ = ebuf^
    _ = nbuf^


def test_small_collision_hull() raises:
    """A hand-authored collision mesh — few vertices, many coplanar facets.

    Coplanarity is where a greedy walk can find a plateau, so the SMALL fixture
    is the sharp one here, not the easy one.
    """
    _check(
        "Fixed_Jaw_Collision_2",
        "mojo_rl/envs/robots/assets/so_arm100/Fixed_Jaw_Collision_2.stl",
    )


def test_scanned_hull() raises:
    """A full scanned part — the shape GJK actually walks in the viewer."""
    _check(
        "Wrist_Pitch_Roll  ",
        "mojo_rl/envs/robots/assets/so_arm100/Wrist_Pitch_Roll.stl",
    )


def test_long_thin_hull() raises:
    """SO-ARM101's under-arm: long and thin, so the graph diameter is large and
    the walk takes many steps. If a step budget were ever too tight, this is
    the fixture that would expose it."""
    _check(
        "under_arm_so101_v1",
        "mojo_rl/envs/robots/assets/so_arm101/under_arm_so101_v1.stl",
    )


def test_below_hillclimb_min_uses_the_scan() raises:
    """Under `_HILLCLIMB_MIN = 10` vertices MuJoCo keeps the scan, and so do
    we. Pins that the threshold branch does not change the answer — a mesh in
    that range must give the same point either way."""
    # ⚠ HEAP, NOT `InlineArray`. NV*3 float64 is ~200 KB and the edge buffer
    # more; on the stack that is an overflow, not a slow test.
    # ⚠⚠ THE LISTS MUST OUTLIVE THE TENSORS — Mojo destroys at LAST USE, so the
    # `_ = ...` at the end of this function is load-bearing, not tidy-up.
    var vbuf = List[Scalar[D]](length=NV * 3, fill=Scalar[D](0))
    var verts = LayoutTensor[D, Layout.row_major(NV, 3), MutAnyOrigin](
        vbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    var n = 0
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                verts[n, 0] = Scalar[D](sx) - Scalar[D](0.5)
                verts[n, 1] = Scalar[D](sy) - Scalar[D](0.5)
                verts[n, 2] = Scalar[D](sz) - Scalar[D](0.5)
                n += 1
    var abuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var eadr = LayoutTensor[D, Layout.row_major(NV), MutAnyOrigin](
        abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    var ebuf = List[Scalar[D]](length=NE, fill=Scalar[D](-1))
    var edges = LayoutTensor[D, Layout.row_major(NE), MutAnyOrigin](
        ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    var s = _support_mesh[D, NV](
        Scalar[D](0), Scalar[D](0), Scalar[D](1),
        Scalar[D](0), Scalar[D](0), Scalar[D](0),
        Scalar[D](0), Scalar[D](0), Scalar[D](0), Scalar[D](1),
        verts, eadr, edges, 0, 8,
    )
    print("   cube (8 verts, below the threshold) +z support z =", s[2])
    assert_true(
        s[2] == Scalar[D](0.5),
        "an 8-vertex cube's +z support should be z = 0.5, got " + String(s[2]),
    )
    _ = vbuf^
    _ = abuf^
    _ = ebuf^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
