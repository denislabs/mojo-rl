"""The qhull shim returns `MakeGraph`'s face list, in qhull's order.

    pixi run build-qhull && pixi run mojo run -I . tests/physics3d/test_qhull_shim.mojo

WHY THE SHIM EXISTS. `mjCMesh::MakeGraph` does not implement a convex hull: it
calls qhull and walks `FORALLfacets`. That order is what `MakePolygons` inserts
faces in, which is where `MeshPolygon::Paths()` starts each vertex cycle, which
is which three vertices `MakePolygonNormals` reads. Our own exact hull matches
qhull's vertex SET but not its facet order, and the polygon normals that fall
out land up to 0.26 deg from `mesh_polynormal` — against the 0.09167 deg
tolerance `alignedFaces` / `alignedFaceEdge` test with.

⚠ THIS FILE GATES THE SHIM, NOT THE PIPELINE. It asserts the C boundary is
sound — counts, orientation, error codes — on fixtures whose answer is fixed by
geometry rather than by qhull's internals. The claim that this reproduces
MuJoCo's polygon paths was measured separately against a real mesh
(shadow_dexee's `Asm-MRH-F-Mid-Visual,00+MagTac,00.stl`): the shim's face list,
run through a transcription of `MakePolygons` / `InsertFace` / `Paths`,
reproduces all 2580 polygon paths of the compiled model exactly, INCLUDING the
cycle start, with 0 wrong starts and 0 missing vertex sets.

⚠ IT SKIPS, LOUDLY, WHEN THE SHIM IS ABSENT. The dylib is a build artifact and
is not tracked; a red test here for a missing `pixi run build-qhull` would be
noise. A skip that says which command to run is not.
"""

from std.math import abs

from mojo_rl.physics3d.collision.qhull_native import (
    qhull_faces, qhull_max_faces, qhull_shim_available,
)


def _hull(
    verts: List[Float64], nvert: Int
) raises -> Tuple[List[Int32], Int]:
    var cap = qhull_max_faces(nvert)
    var f = List[Int32](length=cap * 3 if cap > 0 else 3, fill=Int32(0))
    var n = qhull_faces(verts.unsafe_ptr(), nvert, -1, f.unsafe_ptr(), cap)
    return (f^, n)


def _check_closed(f: List[Int32], nf: Int) raises:
    """Every directed edge appears exactly once — a closed, consistently
    oriented triangulated surface. This is what `toporient` is for, and it is
    the one property of the face list the caller actually depends on."""
    var seen = List[Int](capacity=nf * 6)
    for i in range(nf):
        for k in range(3):
            var a = Int(f[i * 3 + k])
            var b = Int(f[i * 3 + (k + 1) % 3])
            seen.append(a * 1000000 + b)
    for i in range(len(seen)):
        var fwd = 0
        var rev = 0
        var a = seen[i] // 1000000
        var b = seen[i] % 1000000
        for j in range(len(seen)):
            if seen[j] == a * 1000000 + b:
                fwd += 1
            if seen[j] == b * 1000000 + a:
                rev += 1
        if fwd != 1 or rev != 1:
            raise Error(
                "edge (" + String(a) + "," + String(b) + ") appears "
                + String(fwd) + " times forward and " + String(rev)
                + " reversed — the face list is not a closed oriented surface"
            )


def main() raises:
    print("--- qhull shim gate ---")
    if not qhull_shim_available():
        print("  SKIP: shim not built. Run `pixi run build-qhull`.")
        return

    # ---- a unit cube: 8 points, 12 triangles, Euler 2V-4 = 12 ------------
    var cube = List[Float64](length=24, fill=0.0)
    var c = 0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                cube[c * 3 + 0] = Float64(x)
                cube[c * 3 + 1] = Float64(y)
                cube[c * 3 + 2] = Float64(z)
                c += 1
    var r = _hull(cube, 8)
    var cf = r[0].copy()
    var cn = r[1]
    print("  cube faces:", cn)
    if cn != 12:
        raise Error("cube: expected 12 triangles, got " + String(cn))
    _check_closed(cf, cn)
    print("  PASS: cube is 12 triangles, closed and consistently oriented")

    # ---- INTERIOR POINTS MUST NOT APPEAR. A point inside the hull is not a
    # vertex of it, and a face list that named one would be a collision shape
    # with a spurious feature. 8 corners + 1 centre.
    var cube9 = List[Float64](length=27, fill=0.0)
    for i in range(24):
        cube9[i] = cube[i]
    cube9[24] = 0.5
    cube9[25] = 0.5
    cube9[26] = 0.5
    var r9 = _hull(cube9, 9)
    var f9 = r9[0].copy()
    var n9 = r9[1]
    if n9 != 12:
        raise Error("cube+centre: expected 12 triangles, got " + String(n9))
    for i in range(n9 * 3):
        if Int(f9[i]) == 8:
            raise Error("the interior point appears in the face list")
    print("  PASS: an interior point is excluded (", n9, "faces )")

    # ---- a tetrahedron: the minimum, 4 faces --------------------------------
    var tet = List[Float64](length=12, fill=0.0)
    tet[3] = 1.0
    tet[7] = 1.0
    tet[11] = 1.0
    var rt = _hull(tet, 4)
    var tn = rt[1]
    if tn != 4:
        raise Error("tetrahedron: expected 4 triangles, got " + String(tn))
    _check_closed(rt[0].copy(), tn)
    print("  PASS: tetrahedron is 4 triangles, closed")

    # ---- THE BUFFER GUARD IS REAL, not decorative. `mrl_qhull_faces` returns
    # -2 rather than writing past the end, and the Mojo wrapper turns that into
    # a raise. A silent 0 here would be a mesh with NO collision geometry.
    var small = List[Int32](length=3, fill=Int32(0))
    var raised = False
    try:
        _ = qhull_faces(cube.unsafe_ptr(), 8, -1, small.unsafe_ptr(), 1)
    except e:
        raised = True
        if String(e).find("face buffer too small") == -1:
            raise Error("wrong error for an undersized buffer: " + String(e))
    if not raised:
        raise Error("an undersized face buffer did not raise")
    print("  PASS: an undersized buffer raises instead of overrunning")

    _ = cube^
    _ = cube9^
    _ = tet^
    _ = small^
    print("test_qhull_shim: ALL PASS")
