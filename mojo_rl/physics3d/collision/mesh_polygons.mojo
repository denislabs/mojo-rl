"""Merge a convex hull's triangles into POLYGONS — MuJoCo's `mjCMesh::MakePolygons`.

WHY THIS EXISTS. `mjc_Convex`'s native multi-contact path (`multicontact()`,
`engine_collision_gjk.c:2111`) does not perturb and requery the way
`collision/multi_ccd.mojo` does. It reads the EPA witness face, asks each geom
which of ITS faces that witness came from, and clips one face polygon against
the other. For a box the face is recovered analytically from a corner bitmask;
for a mesh it has to come from stored topology. That topology is what this
builds: `mesh_polyvert` / `mesh_polynormal` / `mesh_polymap`, the same three
arrays MuJoCo's compiler produces.

A TRIANGULATION IS NOT ENOUGH, which is the whole point. The hull of a cube is
12 triangles; clipping a TRIANGLE against the opposing face gives a three-point
manifold on a square face, and the box it rests on tips about the missing
corner. MuJoCo merges the two coplanar triangles into one quad first. So the
merge is not a compression of the same information — it is the information.

Ported from `references/mujoco-3.11.0/src/user/user_mesh.cc:2904`
(`MakePolygons`), with `MeshPolygon::InsertFace` (:2768) and
`MeshPolygon::Paths` (:2840) inlined. The grouping key is
`MeshPolygonKey` (:2702).

⚠ GROUPING IS BY QUANTISED NORMAL DIRECTION, NOT BY ADJACENCY. Every triangle
whose normal rounds to the same (theta, phi) bucket at `kAngleTol = 0.01` rad
joins one group, even if it sits on the far side of the mesh. MuJoCo then splits
the group into ISLANDS by connectivity while inserting, and emits one polygon
per island. Merging by "coplanar AND edge-adjacent" instead would agree on
almost every mesh and then disagree on the ones with two parallel faces sharing
no edge — the common case, e.g. the top and bottom of any prism.

⚠ MuJoCo'S POLYGON ORDER IS AN `unordered_map` ITERATION ORDER, so it is a hash
artefact and cannot be reproduced by any port. `test_mesh_polygons_vs_mujoco`
therefore compares polygon SETS, matched by normal, not index by index. Where a
tie-break in `multicontact()` reads polygon order (two coplanar candidate faces
for one edge), the two engines may pick differently; that is a property of the
reference, not a defect here.

⚠ THE PATH WINDING IS CCW AS SEEN FROM OUTSIDE — `cross(p1-p0, p2-p0)` points
ALONG the stored normal. Both `meshFace` and `boxFace` in the reference hand
`polygonClip` the REVERSED order, so consumers must reverse; the storage
convention here matches `m.mesh_polyvert` exactly and is verified against it.
"""

from std.math import sqrt, atan2, acos, cos, sin, abs, round

from ..constants import MESH_POLY_ANGLE_TOL


@always_inline
def _normalize3[
    DTYPE: DType
](
    x: Scalar[DTYPE], y: Scalar[DTYPE], z: Scalar[DTYPE]
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Returns (nx, ny, nz, norm); the vector is untouched when norm is 0."""
    var n = sqrt(x * x + y * y + z * z)
    if n <= Scalar[DTYPE](0):
        return (x, y, z, n)
    return (x / n, y / n, z / n, n)


def _polygon_key[
    DTYPE: DType
](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v3x: Scalar[DTYPE], v3y: Scalar[DTYPE], v3z: Scalar[DTYPE],
) -> Tuple[Bool, Float64, Float64]:
    """`MeshPolygonKey` — the rounded (theta, phi) bucket of a face normal.

    Returns `(ok, rtheta, rphi)`; `ok` is False for a degenerate triangle,
    which MuJoCo skips entirely rather than assigning to a bucket.

    ⚠ THE POLE IS A SPECIAL CASE AND IT IS NOT AN OPTIMISATION. Near
    `|n_z| = 1`, `atan2(n_y, n_x)` is numerically meaningless — the azimuth of a
    vector that has essentially no horizontal part — so two faces of the SAME
    flat top would land in different buckets and never merge. MuJoCo collapses
    the whole cap to `theta = 0` with `phi` either 0 or `round(pi/tol)`. The
    cube's top and bottom faces both take this branch.
    """
    var d12x = v2x - v1x
    var d12y = v2y - v1y
    var d12z = v2z - v1z
    var d13x = v3x - v1x
    var d13y = v3y - v1y
    var d13z = v3z - v1z
    var nx = d12y * d13z - d12z * d13y
    var ny = d12z * d13x - d12x * d13z
    var nz = d12x * d13y - d12y * d13x
    var nrm = sqrt(nx * nx + ny * ny + nz * nz)
    # `mjMINVAL`, as MuJoCo's own guard writes it.
    if nrm < Scalar[DTYPE](1e-15):
        return (False, Float64(0), Float64(0))
    var ux = Float64(nx / nrm)
    var uy = Float64(ny / nrm)
    var uz = Float64(nz / nrm)

    var tol = MESH_POLY_ANGLE_TOL
    if abs(uz) > 1.0 - 1e-7:
        var rphi = Float64(0)
        if uz < 0.0:
            rphi = round(3.141592653589793 / tol)
        return (True, Float64(0), rphi)
    return (True, round(atan2(uy, ux) / tol), round(acos(uz) / tol))


struct _PolyGroup(Copyable, Movable):
    """One quantised-normal bucket: its boundary edges and their islands.

    `edges` is a flat list of (from, to) pairs. `InsertFace` cancels an edge
    against its REVERSE — that is what deletes the shared edge between two
    coplanar triangles and leaves the outline of their union.
    """

    var rtheta: Float64
    var rphi: Float64
    var edges: List[Int]  # flat pairs: [from0, to0, from1, to1, ...]
    var islands: List[Int]  # one per edge
    var nisland: Int

    def __init__(out self, rtheta: Float64, rphi: Float64):
        self.rtheta = rtheta
        self.rphi = rphi
        self.edges = List[Int]()
        self.islands = List[Int]()
        self.nisland = 0

    def seed(mut self, v1: Int, v2: Int, v3: Int):
        self.edges = List[Int]()
        self.edges.append(v1)
        self.edges.append(v2)
        self.edges.append(v2)
        self.edges.append(v3)
        self.edges.append(v3)
        self.edges.append(v1)
        self.islands = List[Int]()
        self.islands.append(0)
        self.islands.append(0)
        self.islands.append(0)
        self.nisland = 1

    def _erase(mut self, i: Int):
        var ne = len(self.islands)
        for k in range(i, ne - 1):
            self.edges[k * 2 + 0] = self.edges[(k + 1) * 2 + 0]
            self.edges[k * 2 + 1] = self.edges[(k + 1) * 2 + 1]
            self.islands[k] = self.islands[k + 1]
        _ = self.edges.pop()
        _ = self.edges.pop()
        _ = self.islands.pop()

    def _combine(mut self, mut island1: Int, mut island2: Int):
        """`CombineIslands` — merge, then RENUMBER every island above the loser.

        The renumbering is what keeps island ids a dense 0..nisland-1 range, so
        `Paths` can loop over `i in range(nisland)` and find each one.
        """
        if island2 < island1:
            var tmp = island1
            island1 = island2
            island2 = tmp
        for k in range(len(self.islands)):
            if self.islands[k] == island2:
                self.islands[k] = island1
            elif self.islands[k] > island2:
                self.islands[k] -= 1

    def _find_reverse(self, a: Int, b: Int) -> Int:
        """Index of the edge (a -> b), or -1. Callers pass the REVERSE pair."""
        for i in range(len(self.islands)):
            if self.edges[i * 2 + 0] == a and self.edges[i * 2 + 1] == b:
                return i
        return -1

    def insert_face(mut self, v1: Int, v2: Int, v3: Int):
        """`MeshPolygon::InsertFace`, edge for edge."""
        var add1 = True
        var add2 = True
        var add3 = True
        var island = -1

        var i1 = self._find_reverse(v2, v1)
        if i1 >= 0:
            add1 = False
            island = self.islands[i1]
            self._erase(i1)

        var i2 = self._find_reverse(v3, v2)
        if i2 >= 0:
            var island2 = self.islands[i2]
            if island == -1:
                island = island2
            elif island2 != island:
                self.nisland -= 1
                self._combine(island, island2)
            add2 = False
            self._erase(i2)

        var i3 = self._find_reverse(v1, v3)
        if i3 >= 0:
            var island3 = self.islands[i3]
            if island == -1:
                island = island3
            elif island3 != island:
                self.nisland -= 1
                self._combine(island, island3)
            add3 = False
            self._erase(i3)

        if island == -1:
            island = self.nisland
            self.nisland += 1

        if add1:
            self.edges.append(v1)
            self.edges.append(v2)
            self.islands.append(island)
        if add2:
            self.edges.append(v2)
            self.edges.append(v3)
            self.islands.append(island)
        if add3:
            self.edges.append(v3)
            self.edges.append(v1)
            self.islands.append(island)

    def paths(self) -> List[List[Int]]:
        """`MeshPolygon::Paths` — trace each island's boundary into a cycle."""
        var out = List[List[Int]]()
        var nedge = len(self.islands)

        # A group that never grew past its seed triangle IS the triangle.
        if nedge == 3:
            var tri = List[Int]()
            tri.append(self.edges[0])
            tri.append(self.edges[2])
            tri.append(self.edges[4])
            out.append(tri^)
            return out^

        for i in range(self.nisland):
            var path = List[Int]()
            for j in range(nedge):
                if self.islands[j] == i:
                    path.append(self.edges[j * 2 + 0])
                    path.append(self.edges[j * 2 + 1])
                    break
            if len(path) == 0:
                continue

            var next = path[len(path) - 1]
            for _l in range(nedge):
                var finished = False
                # ⚠ `k` STARTS AT 1, as the reference writes it: edge 0 is the
                # one the path was seeded from, and revisiting it would close
                # the loop after a single step.
                for k in range(1, nedge):
                    if self.islands[k] == i and self.edges[k * 2 + 0] == next:
                        next = self.edges[k * 2 + 1]
                        if next == path[0]:
                            out.append(path.copy())
                            finished = True
                            break
                        path.append(next)
                        break
                if finished:
                    break
        return out^


def build_mesh_polygons[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    vert_float_offset: Int,
    num_verts: Int,
    faces: List[Int],
    mut poly_vert: List[Int],
    mut poly_vertadr: List[Int],
    mut poly_vertnum: List[Int],
    mut poly_normal: List[Scalar[DTYPE]],
    mut polymap: List[Int],
    mut polymap_adr: List[Int],
    mut polymap_num: List[Int],
) -> Int:
    """Build one mesh's polygons. Returns the polygon count.

    `verts` is the FLAT vertex list this mesh's block starts at
    `vert_float_offset` within; `faces` holds triangles as indices LOCAL to the
    mesh (0 .. num_verts-1), and every index written to `poly_vert` / `polymap`
    is local too, matching MuJoCo's `mesh_polyvert` which is relative to
    `mesh_vertadr`.

    `poly_vertadr` / `polymap_adr` are absolute offsets into the caller's
    running `poly_vert` / `polymap` lists, so several meshes accumulate into
    one pair of arrays exactly as `mesh_polyvert` does.
    """
    var groups = List[_PolyGroup]()
    var nface = len(faces) // 3

    for f in range(nface):
        var i1 = faces[f * 3 + 0]
        var i2 = faces[f * 3 + 1]
        var i3 = faces[f * 3 + 2]
        var o1 = vert_float_offset + i1 * 3
        var o2 = vert_float_offset + i2 * 3
        var o3 = vert_float_offset + i3 * 3
        var key = _polygon_key[DTYPE](
            verts[o1 + 0], verts[o1 + 1], verts[o1 + 2],
            verts[o2 + 0], verts[o2 + 1], verts[o2 + 2],
            verts[o3 + 0], verts[o3 + 1], verts[o3 + 2],
        )
        if not key[0]:
            continue
        var rtheta = key[1]
        var rphi = key[2]

        var found = -1
        for g in range(len(groups)):
            if groups[g].rtheta == rtheta and groups[g].rphi == rphi:
                found = g
                break
        if found < 0:
            var grp = _PolyGroup(rtheta, rphi)
            grp.seed(i1, i2, i3)
            groups.append(grp^)
        else:
            groups[found].insert_face(i1, i2, i3)

    var npoly = 0
    for g in range(len(groups)):
        var paths = groups[g].paths()
        for p in range(len(paths)):
            var path = paths[p].copy()
            if len(path) < 3:
                continue
            poly_vertadr.append(len(poly_vert))
            poly_vertnum.append(len(path))
            for k in range(len(path)):
                poly_vert.append(path[k])
            # `MakePolygonNormals`: from the first THREE path vertices, not the
            # bucket's quantised direction. The bucket is only a grouping key —
            # rounding it to 0.01 rad would put a visible error into every
            # non-axis-aligned face normal. Measured on the hex fixture: the
            # exact side normal is (0.9333, 0.3592, 0) and the quantised one is
            # (0.9323, 0.3616, 0).
            var a = vert_float_offset + path[0] * 3
            var b = vert_float_offset + path[1] * 3
            var c = vert_float_offset + path[2] * 3
            var ux = verts[b + 0] - verts[a + 0]
            var uy = verts[b + 1] - verts[a + 1]
            var uz = verts[b + 2] - verts[a + 2]
            var vx = verts[c + 0] - verts[a + 0]
            var vy = verts[c + 1] - verts[a + 1]
            var vz = verts[c + 2] - verts[a + 2]
            var n = _normalize3[DTYPE](
                uy * vz - uz * vy,
                uz * vx - ux * vz,
                ux * vy - uy * vx,
            )
            poly_normal.append(n[0])
            poly_normal.append(n[1])
            poly_normal.append(n[2])
            npoly += 1

    # Vertex -> polygon map. Built by scanning polygons in order, so each
    # vertex's list comes out ASCENDING, which is what `m.mesh_polymap` holds
    # and what `intersect` in `multicontact` walks.
    var map_base = len(polymap)
    for v in range(num_verts):
        polymap_adr.append(len(polymap))
        var cnt = 0
        for p in range(npoly):
            var adr = poly_vertadr[len(poly_vertadr) - npoly + p]
            var num = poly_vertnum[len(poly_vertnum) - npoly + p]
            for k in range(num):
                if poly_vert[adr + k] == v:
                    polymap.append(p)
                    cnt += 1
                    break
        polymap_num.append(cnt)
    _ = map_base

    return npoly
