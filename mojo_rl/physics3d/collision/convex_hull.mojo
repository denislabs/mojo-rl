"""Convex hull computation for mesh collision.

Loads STL mesh vertices, deduplicates them, computes the 3D convex hull,
and stores hull vertices for GJK/EPA collision detection.

Algorithm: Incremental convex hull (add points one by one, remove visible
faces, add new faces from horizon edges). O(n*h) where h is hull size.
Runs once at model load time.
"""

from std.math import sqrt, abs

from .mesh_polygons import build_mesh_polygons
from ..model.mesh_inertia import MeshInertia, transform_verts_to_principal_frame


def deduplicate_vertices[
    DTYPE: DType,
](
    raw_verts: List[Scalar[DTYPE]],
    num_raw: Int,
    mut out_verts: List[Scalar[DTYPE]],
) -> Int:
    """Deduplicate vertices from a flat [x0,y0,z0, x1,y1,z1, ...] array.

    Appends unique vertices to out_verts, returns num_unique.
    """
    comptime EPS_SQ: Scalar[DTYPE] = 1e-12
    var start = len(out_verts)
    var num_unique = 0

    for i in range(num_raw):
        var vx = raw_verts[i * 3 + 0]
        var vy = raw_verts[i * 3 + 1]
        var vz = raw_verts[i * 3 + 2]

        var found = False
        for j in range(num_unique):
            var off = start + j * 3
            var dx = vx - out_verts[off + 0]
            var dy = vy - out_verts[off + 1]
            var dz = vz - out_verts[off + 2]
            if dx * dx + dy * dy + dz * dz < EPS_SQ:
                found = True
                break

        if not found:
            out_verts.append(vx)
            out_verts.append(vy)
            out_verts.append(vz)
            num_unique += 1

    return num_unique


def compute_mesh_rbound_at[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    vert_offset: Int,
    num_verts: Int,
) -> Scalar[
    DTYPE
]:
    """`geom_rbound` for a mesh geom — `mjCGeom::GetRBound`'s mesh case.

    MuJoCo takes the mesh's axis-aligned bounds `aamm`, folds them about the
    FRAME ORIGIN (`haabb[k] = max(|min_k|, |max_k|)`) and returns the AABB
    corner radius `sqrt(sum haabb^2)`. That is the radius of a sphere centred
    on the geom frame enclosing the box enclosing the mesh — deliberately
    LOOSER than the tightest enclosing sphere.

    ⚠⚠ WAS `compute_bounding_radius_at`, WHICH MEASURED FROM THE VERTEX
    CENTROID — renamed as well as rewritten, so that anything still calling
    the old name fails to compile rather than silently changing meaning.
    Three different quantities are easy to confuse here, and only one is
    MuJoCo's:

        max |v - centroid(V)|   <- what this used to return
        max |v|                 <- tightest sphere about the FRAME ORIGIN
        sqrt(sum max(|min|,|max|)^2)  <- MuJoCo, the AABB corner

    The centroid version is not even ordered against the other two: on Jaco's
    mesh 3 it returned 0.1586 against MuJoCo's 0.1364, EXCEEDING a bound that
    max |v| cannot exceed, because it is measured from a different centre.
    Measured on all nine Jaco meshes, ours spanned 0.72x to 1.16x of MuJoCo's.

    This is not cosmetic. `rbound` sets the plane-mesh spread filter
    (`0.3 * rbound` in `_plane_mesh_contacts`), so it decides how many contacts
    a plane-mesh pair emits; it also feeds broadphase culling, where MuJoCo's
    looser value can only widen the candidate set.

    Taken over HULL vertices rather than all vertices, which is the same
    answer: an axis extreme is always a hull vertex.
    """
    if num_verts == 0:
        return Scalar[DTYPE](0)
    var lo_x = verts[vert_offset + 0]
    var lo_y = verts[vert_offset + 1]
    var lo_z = verts[vert_offset + 2]
    var hi_x = lo_x
    var hi_y = lo_y
    var hi_z = lo_z
    for i in range(1, num_verts):
        var vx = verts[vert_offset + i * 3 + 0]
        var vy = verts[vert_offset + i * 3 + 1]
        var vz = verts[vert_offset + i * 3 + 2]
        if vx < lo_x:
            lo_x = vx
        if vy < lo_y:
            lo_y = vy
        if vz < lo_z:
            lo_z = vz
        if vx > hi_x:
            hi_x = vx
        if vy > hi_y:
            hi_y = vy
        if vz > hi_z:
            hi_z = vz
    var hx = max(abs(lo_x), abs(hi_x))
    var hy = max(abs(lo_y), abs(hi_y))
    var hz = max(abs(lo_z), abs(hi_z))
    return sqrt(hx * hx + hy * hy + hz * hz)


# =============================================================================
# 3D Incremental Convex Hull
# =============================================================================


@always_inline
def _hull_cross[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


@always_inline
def _hull_dot[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


def compute_convex_hull[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    num_verts: Int,
    mut hull_verts: List[Scalar[DTYPE]],
    mut hull_faces: List[Int],
) -> Int:
    """EXACT 3D convex hull by incremental insertion.

    Replaces support-point SAMPLING, which could only ever return a SUBSET of
    the hull vertices. That subset is a strictly smaller solid, so GJK/EPA saw
    a shrunken shape and lost shallow contacts — an error with ONE SIGN that no
    gate could catch, because every mesh gate in the suite was frozen from an
    implementation that had it. On eGripperBase the sampler kept 81 of 882 hull
    vertices; raising its direction count was measured and did NOT help
    (`3b1b19db`), because a subset of extreme points is the wrong object no
    matter how it is chosen.

    The hull vertex set is exactly what a support query needs, so this gives
    SHAPE PARITY with MuJoCo, which collides the convex hull of `mesh_vert`.

    `hull_faces` receives the surviving triangles as vertex triples in
    COMPACTED numbering, wound OUTWARD — see the emit loop for why the winding
    has to be established there. A degenerate input (fewer than four points, or
    points that are collinear or coplanar) takes one of the early returns and
    leaves `hull_faces` EMPTY; that is the same state MuJoCo represents with
    `mesh_polynum == 0`, and `multicontact` skips such a mesh rather than
    guessing a face for it.

    Algorithm, matching the module docstring for the first time: seed a
    tetrahedron from four non-degenerate extremes, then insert each remaining
    point — delete the faces it can see, and stitch it to the horizon. The
    horizon is the set of edges belonging to exactly ONE visible face
    (UNDIRECTED, so it does not depend on consistent winding — the same rule
    `gjk.mojo`'s EPA uses, and for the same reason).

    O(n*h). Runs once per mesh at model build.
    """
    if num_verts < 4:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    # Tolerance scaled to the mesh, not absolute: these are metres and a fixed
    # epsilon would be meaningless across a 3 cm gripper and a 1 m table.
    var scale = Scalar[DTYPE](0)
    for i in range(num_verts * 3):
        var av = abs(verts[i])
        if av > scale:
            scale = av
    if scale <= Scalar[DTYPE](0):
        scale = Scalar[DTYPE](1)
    # ⚠⚠ THE RELATIVE FACTOR IS DTYPE-DEPENDENT, AND 1e-9 HUNG float32.
    #
    # Scaling to the mesh (above) is right and stays. What was wrong is that
    # `1e-9` is a FLOAT64 relative tolerance: float32's machine epsilon is
    # ~1.19e-7, so `scale * 1e-9` sits about two orders of magnitude BELOW the
    # noise floor and `side > eps` at the expansion loop becomes a coin flip.
    # A point then reads as outside a face it is actually on, the horizon is
    # rebuilt, and the hull never converges.
    #
    # MEASURED on SO-ARM100's collision meshes (10 hulls, 2 551 vertices),
    # `ModelDefFromXML.init_fields` end to end:
    #
    #     float64   0.55 s
    #     float32   >180 s, still spinning, 99% of one core
    #
    # It is not a slowdown, it is non-termination: `sample` put 2 456 of 2 459
    # stack samples inside `compute_convex_hull` recursing on itself.
    #
    # ⚠ WHY NOTHING CAUGHT IT. Every mesh gate builds at float64, and every
    # dm_control model in the viewer leaves `NMESH_VERTS` at 0 — which skips
    # the mesh branch entirely. The SO-ARM ports are the first COLLIDABLE
    # meshes to reach the float32 renderer path, so they are the first model
    # that could hang, and it hangs before the window is ever created.
    #
    # 1e-5 is ~100x float32's epsilon, the usual margin for a geometric
    # predicate. float64 keeps 1e-9 so no existing hull moves.
    comptime REL: Float64 = 1e-9 if DTYPE == DType.float64 else 1e-5
    var eps = scale * Scalar[DTYPE](REL)

    # ---- seed tetrahedron ------------------------------------------------
    var i0 = 0
    var i1 = 0
    var lo = verts[0]
    var hi = verts[0]
    for i in range(num_verts):
        var x = verts[i * 3 + 0]
        if x < lo:
            lo = x
            i0 = i
        if x > hi:
            hi = x
            i1 = i
    if i0 == i1:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    var dx = verts[i1 * 3 + 0] - verts[i0 * 3 + 0]
    var dy = verts[i1 * 3 + 1] - verts[i0 * 3 + 1]
    var dz = verts[i1 * 3 + 2] - verts[i0 * 3 + 2]

    var i2 = -1
    var best = Scalar[DTYPE](0)
    for i in range(num_verts):
        var ax = verts[i * 3 + 0] - verts[i0 * 3 + 0]
        var ay = verts[i * 3 + 1] - verts[i0 * 3 + 1]
        var az = verts[i * 3 + 2] - verts[i0 * 3 + 2]
        var cx = ay * dz - az * dy
        var cy = az * dx - ax * dz
        var cz = ax * dy - ay * dx
        var l = sqrt(cx * cx + cy * cy + cz * cz)
        if l > best:
            best = l
            i2 = i
    if i2 < 0 or best <= eps:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    var e1x = verts[i1 * 3 + 0] - verts[i0 * 3 + 0]
    var e1y = verts[i1 * 3 + 1] - verts[i0 * 3 + 1]
    var e1z = verts[i1 * 3 + 2] - verts[i0 * 3 + 2]
    var e2x = verts[i2 * 3 + 0] - verts[i0 * 3 + 0]
    var e2y = verts[i2 * 3 + 1] - verts[i0 * 3 + 1]
    var e2z = verts[i2 * 3 + 2] - verts[i0 * 3 + 2]
    var nx = e1y * e2z - e1z * e2y
    var ny = e1z * e2x - e1x * e2z
    var nz = e1x * e2y - e1y * e2x
    var nl = sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nl
    ny /= nl
    nz /= nl

    var i3 = -1
    var bestd = Scalar[DTYPE](0)
    for i in range(num_verts):
        var ax = verts[i * 3 + 0] - verts[i0 * 3 + 0]
        var ay = verts[i * 3 + 1] - verts[i0 * 3 + 1]
        var az = verts[i * 3 + 2] - verts[i0 * 3 + 2]
        var d = abs(ax * nx + ay * ny + az * nz)
        if d > bestd:
            bestd = d
            i3 = i
    if i3 < 0 or bestd <= eps:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    # interior reference point: the seed centroid stays inside the hull for the
    # whole construction, so it fixes every face's outward orientation.
    var rx = (
        verts[i0 * 3 + 0] + verts[i1 * 3 + 0] + verts[i2 * 3 + 0]
        + verts[i3 * 3 + 0]
    ) * Scalar[DTYPE](0.25)
    var ry = (
        verts[i0 * 3 + 1] + verts[i1 * 3 + 1] + verts[i2 * 3 + 1]
        + verts[i3 * 3 + 1]
    ) * Scalar[DTYPE](0.25)
    var rz = (
        verts[i0 * 3 + 2] + verts[i1 * 3 + 2] + verts[i2 * 3 + 2]
        + verts[i3 * 3 + 2]
    ) * Scalar[DTYPE](0.25)

    var faces = List[Int]()
    var seed: InlineArray[Int, 12] = [
        i0, i1, i2, i0, i1, i3, i0, i2, i3, i1, i2, i3
    ]
    for f in range(4):
        faces.append(seed[f * 3 + 0])
        faces.append(seed[f * 3 + 1])
        faces.append(seed[f * 3 + 2])

    var on_hull = List[Bool]()
    for _ in range(num_verts):
        on_hull.append(False)
    on_hull[i0] = True
    on_hull[i1] = True
    on_hull[i2] = True
    on_hull[i3] = True

    # ---- insertion ORDER: extremes first ----------------------------------
    # ⚠ ORDER IS A PERFORMANCE PROPERTY, NOT A COSMETIC ONE. Inserting points in
    # array order makes the early insertions see most of the polytope, and both
    # the horizon count (quadratic in the visible-edge count) and the face-list
    # rebuild are paid at that size, for thousands of points. Measured: sawyer's
    # model build did not finish in TEN MINUTES.
    #
    # Front-loading support points along a spread of directions gets the hull
    # almost to its final shape within the first few dozen insertions, after
    # which nearly every remaining point is interior, sees no face, and costs
    # one visibility scan. Same hull either way — support points are hull
    # vertices by definition, so this changes only the path, never the result.
    var order = List[Int]()
    var queued = List[Bool]()
    for _ in range(num_verts):
        queued.append(False)
    for sx in range(-1, 2):
        for sy in range(-1, 2):
            for sz in range(-1, 2):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                var ddx = Scalar[DTYPE](sx)
                var ddy = Scalar[DTYPE](sy)
                var ddz = Scalar[DTYPE](sz)
                var bi = 0
                var bd = Scalar[DTYPE](-1e30)
                for i in range(num_verts):
                    var dd = (
                        ddx * verts[i * 3 + 0]
                        + ddy * verts[i * 3 + 1]
                        + ddz * verts[i * 3 + 2]
                    )
                    if dd > bd:
                        bd = dd
                        bi = i
                if not queued[bi]:
                    queued[bi] = True
                    order.append(bi)
    for i in range(num_verts):
        if not queued[i]:
            order.append(i)

    # ---- insert the remaining points -------------------------------------
    var vis = List[Bool]()
    var edges = List[Int]()
    for oi in range(len(order)):
        var p = order[oi]
        if on_hull[p]:
            continue
        var px = verts[p * 3 + 0]
        var py = verts[p * 3 + 1]
        var pz = verts[p * 3 + 2]

        var nf = len(faces) // 3
        vis.clear()
        var nvis = 0
        for f in range(nf):
            var a = faces[f * 3 + 0]
            var b = faces[f * 3 + 1]
            var c = faces[f * 3 + 2]
            var ux = verts[b * 3 + 0] - verts[a * 3 + 0]
            var uy = verts[b * 3 + 1] - verts[a * 3 + 1]
            var uz = verts[b * 3 + 2] - verts[a * 3 + 2]
            var vx = verts[c * 3 + 0] - verts[a * 3 + 0]
            var vy = verts[c * 3 + 1] - verts[a * 3 + 1]
            var vz = verts[c * 3 + 2] - verts[a * 3 + 2]
            var fx = uy * vz - uz * vy
            var fy = uz * vx - ux * vz
            var fz = ux * vy - uy * vx
            var fl = sqrt(fx * fx + fy * fy + fz * fz)
            var seen = False
            if fl > Scalar[DTYPE](0):
                fx /= fl
                fy /= fl
                fz /= fl
                # orient outward using the interior reference point
                var inward = (
                    fx * (rx - verts[a * 3 + 0])
                    + fy * (ry - verts[a * 3 + 1])
                    + fz * (rz - verts[a * 3 + 2])
                )
                if inward > Scalar[DTYPE](0):
                    fx = -fx
                    fy = -fy
                    fz = -fz
                var side = (
                    fx * (px - verts[a * 3 + 0])
                    + fy * (py - verts[a * 3 + 1])
                    + fz * (pz - verts[a * 3 + 2])
                )
                if side > eps:
                    seen = True
            vis.append(seen)
            if seen:
                nvis += 1
        if nvis == 0:
            continue

        # horizon: edges of visible faces that exactly ONE visible face owns
        edges.clear()
        for f in range(nf):
            if not vis[f]:
                continue
            for e in range(3):
                var a0 = faces[f * 3 + e]
                var a1 = faces[f * 3 + (e + 1) % 3]
                edges.append(a0 if a0 < a1 else a1)
                edges.append(a1 if a0 < a1 else a0)
        var ne = len(edges) // 2

        var kept = List[Int]()
        for f in range(nf):
            if vis[f]:
                continue
            kept.append(faces[f * 3 + 0])
            kept.append(faces[f * 3 + 1])
            kept.append(faces[f * 3 + 2])
        for e in range(ne):
            var lo_e = edges[e * 2 + 0]
            var hi_e = edges[e * 2 + 1]
            var count = 0
            for g in range(ne):
                if edges[g * 2 + 0] == lo_e and edges[g * 2 + 1] == hi_e:
                    count += 1
            if count == 1:
                kept.append(lo_e)
                kept.append(hi_e)
                kept.append(p)
        faces = kept^
        on_hull[p] = True

    # ---- collect the vertices the surviving faces actually use ------------
    for i in range(num_verts):
        on_hull[i] = False
    for i in range(len(faces)):
        on_hull[faces[i]] = True

    # `remap[i]` is vertex i's index in the COMPACTED hull, or -1. The faces
    # are emitted in compacted numbering because that is what every consumer
    # sees: `mesh_vertadr` addresses the compacted block, and MuJoCo's
    # `mesh_polyvert` is likewise relative to `mesh_vertadr`.
    var remap = List[Int](length=num_verts, fill=-1)
    var num_hull = 0
    for i in range(num_verts):
        if on_hull[i]:
            hull_verts.append(verts[i * 3 + 0])
            hull_verts.append(verts[i * 3 + 1])
            hull_verts.append(verts[i * 3 + 2])
            remap[i] = num_hull
            num_hull += 1

    # ---- emit the faces, WOUND OUTWARD ------------------------------------
    #
    # ⚠ THE WINDING IS NOT COSMETIC AND THIS LOOP IS WHERE IT IS ESTABLISHED.
    # The construction above never fixes an orientation: the visibility test
    # re-derives each face's outward normal from the interior reference point
    # every time it is used, and the horizon stitch appends `(lo, hi, p)` in
    # whatever order the edge happened to be stored. That is fine for a hull
    # whose faces are only ever tested one at a time, and it is FATAL to
    # `mesh_polygons.build_mesh_polygons`, which merges two triangles by
    # cancelling a shared edge traversed in OPPOSITE directions. With mixed
    # winding the cancellation silently fails, no edges are removed, and every
    # face stays its own polygon — a cube would come back as 12 triangles
    # instead of 6 quads, which is exactly the bug this whole path exists to
    # avoid. So orient here, once, against the same interior point.
    for f in range(len(faces) // 3):
        var a = faces[f * 3 + 0]
        var b = faces[f * 3 + 1]
        var c = faces[f * 3 + 2]
        var ux = verts[b * 3 + 0] - verts[a * 3 + 0]
        var uy = verts[b * 3 + 1] - verts[a * 3 + 1]
        var uz = verts[b * 3 + 2] - verts[a * 3 + 2]
        var vx = verts[c * 3 + 0] - verts[a * 3 + 0]
        var vy = verts[c * 3 + 1] - verts[a * 3 + 1]
        var vz = verts[c * 3 + 2] - verts[a * 3 + 2]
        var fx = uy * vz - uz * vy
        var fy = uz * vx - ux * vz
        var fz = ux * vy - uy * vx
        # Outward means pointing AWAY from the interior reference point.
        var outward = (
            fx * (verts[a * 3 + 0] - rx)
            + fy * (verts[a * 3 + 1] - ry)
            + fz * (verts[a * 3 + 2] - rz)
        )
        hull_faces.append(remap[a])
        if outward >= Scalar[DTYPE](0):
            hull_faces.append(remap[b])
            hull_faces.append(remap[c])
        else:
            hull_faces.append(remap[c])
            hull_faces.append(remap[b])

    return num_hull


# =============================================================================
# Mesh loading pipeline
# =============================================================================


def build_hull_edge_graph(
    num_hull: Int,
    hull_faces: List[Int],
    vert_base: Int,
    mut edge_adr: List[Int],
    mut edge_list: List[Int],
):
    """The hull's VERTEX ADJACENCY, in the form `mjc_PlaneConvex` consumes.

    MuJoCo keeps this per mesh in `mesh_graph` and builds it in
    `mjCMesh::MakeGraph`: run qhull, `qh_triangulate`, `qh_vertexneighbors`,
    then for each vertex walk its neighbouring FACETS and collect the other two
    vertices of each, deduplicated, terminated by -1.

    ⚠ THE ADJACENCY IS THE TRIANGULATED HULL'S, NOT THE MERGED-POLYGON HULL'S.
    `qh_triangulate` splits coplanar facets, so the diagonals it introduces
    across a flat face ARE edges here. That is not a detail: those diagonals
    are the LONG edges, and `mjc_PlaneConvex` only accepts a neighbour that
    lies at least `0.3 * rbound` from the first contact. Deriving neighbours
    from `mesh_polygons.mojo`'s MERGED polygons instead would drop exactly the
    edges most likely to pass that filter, and a flat face resting on the
    plane would yield one contact where MuJoCo yields three.

    Confirmed against the 3.6.0 tree and the 3.10.0 runtime: for all nine Jaco
    meshes `mesh_graph[graphadr+1] == 2 * numvert - 4`, which is Euler's
    relation for a fully triangulated polytope — so qhull's stored graph really
    is simplicial, and `compute_convex_hull`'s triangles are the same object.

    ⚠ NEIGHBOUR ORDER IS NOT MUJOCO'S AND CANNOT BE. MuJoCo's per-vertex order
    follows qhull's internal facet set; ours follows `hull_faces`. Order only
    decides WHICH neighbours are taken when more than two pass the filter, so
    contact COUNTS still agree — see `_plane_mesh_contacts` for the measured
    consequence.

    `edge_adr` gets one entry per hull vertex, appended in the packed vertex
    order, so it is indexed by the same GLOBAL vertex index as `mesh_verts`;
    neighbours in `edge_list` are stored as global indices too (`vert_base` is
    this mesh's first vertex). MuJoCo needs its `vert_globalid` indirection
    because `mesh_vert` holds ALL vertices and only some are on the hull; ours
    holds hull vertices only, so local and global numbering coincide.
    """
    var nface = len(hull_faces) // 3

    # Vertex -> incident faces, CSR. The obvious alternative — rescanning
    # every face for every vertex — is O(V*F), which is ~5M comparisons on
    # sawyer's largest hull and grows quadratically with mesh size.
    var inc_count = List[Int](length=num_hull, fill=0)
    for k in range(nface * 3):
        inc_count[hull_faces[k]] += 1
    var inc_adr = List[Int](length=num_hull, fill=0)
    var run = 0
    for v in range(num_hull):
        inc_adr[v] = run
        run += inc_count[v]
    var filled = List[Int](length=num_hull, fill=0)
    var inc = List[Int](length=run, fill=0)
    for f in range(nface):
        for j in range(3):
            var v = hull_faces[f * 3 + j]
            inc[inc_adr[v] + filled[v]] = f
            filled[v] += 1

    for v in range(num_hull):
        edge_adr.append(len(edge_list))
        var start = len(edge_list)
        for t in range(inc_adr[v], inc_adr[v] + filled[v]):
            var f = inc[t]
            for j in range(3):
                var w = hull_faces[f * 3 + j]
                if w == v:
                    continue
                var g = vert_base + w
                var found = False
                for e in range(start, len(edge_list)):
                    if edge_list[e] == g:
                        found = True
                        break
                if not found:
                    edge_list.append(g)
        # The -1 separator is MuJoCo's own terminator, and the consumer walks
        # to it rather than to a stored degree.
        edge_list.append(-1)


def load_mesh_hull[
    DTYPE: DType,
](
    mesh_filename: String,
    mut mesh_vert: List[Scalar[DTYPE]],
    mut mesh_vertadr: List[Int],
    mut mesh_vertnum: List[Int],
    mut num_meshes: Int,
    mut mesh_polyadr: List[Int],
    mut mesh_polynum: List[Int],
    mut poly_vert: List[Int],
    mut poly_vertadr: List[Int],
    mut poly_vertnum: List[Int],
    mut poly_normal: List[Scalar[DTYPE]],
    mut polymap: List[Int],
    mut polymap_adr: List[Int],
    mut polymap_num: List[Int],
    mut edge_adr: List[Int],
    mut edge_list: List[Int],
    mi: MeshInertia[DTYPE],
) raises -> Tuple[Int, Scalar[DTYPE]]:
    """Load STL mesh, deduplicate, compute convex hull, store in model arrays.

    Returns (mesh_id, rbound) for this mesh.

    ⚠ VERTICES ARE STORED IN THE MESH'S PRINCIPAL FRAME, not the STL's frame.
    `mi` carries the centre of mass and principal-axis rotation MuJoCo bakes
    into every mesh (`mesh_pos` / `mesh_quat`), and they are applied here —
    BEFORE the hull, the polygon normals and `rbound`, so all three are built
    in the same frame MuJoCo uses. The caller must compose the same `mi` into
    the geom's `pos`/`quat`, or the mesh will collide in the wrong place: the
    two changes are only equivalent TOGETHER.

    The `poly_*` / `polymap*` lists accumulate the hull's POLYGON topology
    across every mesh, exactly as `mesh_polyvert` and `mesh_polymap` do in
    `mjModel`; `mesh_polyadr` / `mesh_polynum` index into them per mesh. See
    `collision/mesh_polygons.mojo` for what they are for.
    """
    from mojo_rl.render.stl_loader import load_stl

    var mesh_data = load_stl(mesh_filename)

    # Extract positions from GPUVertex structs into flat array
    var raw = List[Scalar[DTYPE]]()
    var num_raw = len(mesh_data.vertices)
    for i in range(num_raw):
        raw.append(Scalar[DTYPE](mesh_data.vertices[i].px))
        raw.append(Scalar[DTYPE](mesh_data.vertices[i].py))
        raw.append(Scalar[DTYPE](mesh_data.vertices[i].pz))

    # Deduplicate into temp buffer
    var unique = List[Scalar[DTYPE]]()
    var num_unique = deduplicate_vertices[DTYPE](raw, num_raw, unique)

    # Into the principal frame, exactly where MuJoCo does it (`mjCMesh::
    # Compute` translates by -CoM then `Rotate`s by the conjugate, then records
    # the pair as mesh_pos/mesh_quat). Everything below — hull, polygons,
    # rbound — is therefore computed in MuJoCo's frame rather than the STL's.
    transform_verts_to_principal_frame[DTYPE](unique, num_unique, mi)

    # Compute convex hull
    var mesh_id = num_meshes
    # ⚠ TWO UNITS, ONE NAME. `mesh_vert` is a FLAT scalar list, so
    # `len(mesh_vert)` is an offset in FLOATS — but every collision consumer
    # indexes the packed `mesh_verts` tensor as `[vertex, component]`:
    # `mesh_verts[vert_adr + i, k]` in `gjk._support_mesh`, in
    # `_plane_mesh_contacts` and in the SAP plane-mesh branch. Storing the
    # float offset made `mesh_vertadr` 3x too large for all three.
    #
    # Measured on sawyer: mesh 11 (eGripperBase) had vertadr 1701, which as a
    # VERTEX index points past the 648 vertices actually loaded, so every read
    # returned ZERO — the gripper hull collided as an empty shape. Meshes 1..10
    # were worse than useless rather than empty: their float offsets land
    # inside the populated region, so they collided against SOME OTHER MESH'S
    # vertices. Only mesh 0, at offset 0, was ever right.
    #
    # `mesh_vertadr` is now a VERTEX index, which is also MuJoCo's convention
    # for `mesh_vertadr`. `compute_mesh_rbound_at` walks the flat list and
    # still wants FLOATS, so it is given the float offset explicitly.
    var vert_float_offset = len(mesh_vert)
    mesh_vertadr.append(vert_float_offset // 3)
    var hull_faces = List[Int]()
    var num_hull = compute_convex_hull[DTYPE](
        unique, num_unique, mesh_vert, hull_faces
    )
    mesh_vertnum.append(num_hull)
    num_meshes += 1

    # Polygons for the native multi-contact path. `polymap_adr` / `polymap_num`
    # are appended one entry per HULL VERTEX, so they stay parallel to the
    # vertex block this mesh just wrote.
    mesh_polyadr.append(len(poly_vertadr))
    var npoly = build_mesh_polygons[DTYPE](
        mesh_vert,
        vert_float_offset,
        num_hull,
        hull_faces,
        poly_vert,
        poly_vertadr,
        poly_vertnum,
        poly_normal,
        polymap,
        polymap_adr,
        polymap_num,
    )
    mesh_polynum.append(npoly)

    # Vertex adjacency for the plane-mesh path. Built from the SAME triangles
    # the polygons were merged from, and indexed by global vertex id, so it
    # stays parallel to the vertex block written above.
    build_hull_edge_graph(
        num_hull, hull_faces, vert_float_offset // 3, edge_adr, edge_list
    )

    # Compute bounding radius from hull vertices
    var rbound = compute_mesh_rbound_at[DTYPE](
        mesh_vert, vert_float_offset, num_hull
    )

    return (mesh_id, rbound)
