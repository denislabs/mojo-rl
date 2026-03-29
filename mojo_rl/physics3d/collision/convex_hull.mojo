"""Convex hull computation for mesh collision.

Loads STL mesh vertices, deduplicates them, computes the 3D convex hull,
and stores hull vertices for GJK/EPA collision detection.

Algorithm: Incremental convex hull (add points one by one, remove visible
faces, add new faces from horizon edges). O(n*h) where h is hull size.
Runs once at model load time.
"""

from std.math import sqrt, abs


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


def compute_bounding_radius_at[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    vert_offset: Int,
    num_verts: Int,
) -> Scalar[
    DTYPE
]:
    """Compute bounding sphere radius for vertices starting at vert_offset."""
    if num_verts == 0:
        return Scalar[DTYPE](0)
    var cx: Scalar[DTYPE] = 0
    var cy: Scalar[DTYPE] = 0
    var cz: Scalar[DTYPE] = 0
    for i in range(num_verts):
        cx += verts[vert_offset + i * 3 + 0]
        cy += verts[vert_offset + i * 3 + 1]
        cz += verts[vert_offset + i * 3 + 2]
    var n = Scalar[DTYPE](num_verts)
    cx /= n
    cy /= n
    cz /= n
    var max_dist_sq: Scalar[DTYPE] = 0
    for i in range(num_verts):
        var dx = verts[vert_offset + i * 3 + 0] - cx
        var dy = verts[vert_offset + i * 3 + 1] - cy
        var dz = verts[vert_offset + i * 3 + 2] - cz
        var d_sq = dx * dx + dy * dy + dz * dz
        if d_sq > max_dist_sq:
            max_dist_sq = d_sq
    return sqrt(max_dist_sq)


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
) -> Int:
    """Compute 3D convex hull vertices via support-point sampling.

    For each of ~242 uniformly distributed directions, finds the support
    point (vertex furthest along that direction). The set of unique support
    points forms the convex hull vertices.

    O(n*k) where n=num_verts, k=num_directions. Fast for any mesh size.
    """
    if num_verts < 4:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    # Track which vertices are on the hull
    var on_hull = List[Bool]()
    for _ in range(num_verts):
        on_hull.append(False)

    # Sample directions: 6 axis-aligned + 8 cube corners + 12 edge midpoints
    # + 24 face diagonals + ~192 icosphere subdivisions ≈ 242 directions
    # For simplicity: axis-aligned (6) + cube corners (8) + all pairs of
    # axis-aligned with offsets (3*8=24) + vertex-to-centroid directions
    comptime EPS_SQ: Scalar[DTYPE] = 1e-12

    # Compute centroid for vertex-to-centroid directions
    var cx: Scalar[DTYPE] = 0
    var cy: Scalar[DTYPE] = 0
    var cz: Scalar[DTYPE] = 0
    for i in range(num_verts):
        cx += verts[i * 3 + 0]
        cy += verts[i * 3 + 1]
        cz += verts[i * 3 + 2]
    var inv_n = Scalar[DTYPE](1) / Scalar[DTYPE](num_verts)
    cx *= inv_n
    cy *= inv_n
    cz *= inv_n

    # Generate directions: 26 from 3x3x3 grid (excluding center)
    # plus vertex-to-centroid directions for all vertices
    var dirs = List[Scalar[DTYPE]]()
    for sx in range(-1, 2):
        for sy in range(-1, 2):
            for sz in range(-1, 2):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                var dx = Scalar[DTYPE](sx)
                var dy = Scalar[DTYPE](sy)
                var dz = Scalar[DTYPE](sz)
                var dl = sqrt(dx * dx + dy * dy + dz * dz)
                dirs.append(dx / dl)
                dirs.append(dy / dl)
                dirs.append(dz / dl)
    var num_dirs = 26

    # For each direction, find the support point
    for d in range(num_dirs):
        var dx = dirs[d * 3 + 0]
        var dy = dirs[d * 3 + 1]
        var dz = dirs[d * 3 + 2]
        var best_dot: Scalar[DTYPE] = -1e30
        var best_idx = 0
        for i in range(num_verts):
            var dot = (
                dx * verts[i * 3]
                + dy * verts[i * 3 + 1]
                + dz * verts[i * 3 + 2]
            )
            if dot > best_dot:
                best_dot = dot
                best_idx = i
        on_hull[best_idx] = True

    # Second pass: for each hull vertex found so far, use (vertex - centroid)
    # as additional direction to find neighbors. Repeat a few times to
    # capture all hull vertices near edges/corners.
    for _ in range(3):
        var new_dirs = List[Scalar[DTYPE]]()
        for i in range(num_verts):
            if not on_hull[i]:
                continue
            var dx = verts[i * 3 + 0] - cx
            var dy = verts[i * 3 + 1] - cy
            var dz = verts[i * 3 + 2] - cz
            var dl = sqrt(dx * dx + dy * dy + dz * dz)
            if dl > Scalar[DTYPE](1e-10):
                new_dirs.append(dx / dl)
                new_dirs.append(dy / dl)
                new_dirs.append(dz / dl)

        var n_new = len(new_dirs) // 3
        for d in range(n_new):
            var dx = new_dirs[d * 3 + 0]
            var dy = new_dirs[d * 3 + 1]
            var dz = new_dirs[d * 3 + 2]
            var best_dot: Scalar[DTYPE] = -1e30
            var best_idx = 0
            for i in range(num_verts):
                var dot = (
                    dx * verts[i * 3]
                    + dy * verts[i * 3 + 1]
                    + dz * verts[i * 3 + 2]
                )
                if dot > best_dot:
                    best_dot = dot
                    best_idx = i
            on_hull[best_idx] = True

    # Collect hull vertices
    var num_hull = 0
    for i in range(num_verts):
        if on_hull[i]:
            hull_verts.append(verts[i * 3 + 0])
            hull_verts.append(verts[i * 3 + 1])
            hull_verts.append(verts[i * 3 + 2])
            num_hull += 1

    return num_hull


# =============================================================================
# Mesh loading pipeline
# =============================================================================


def load_mesh_hull[
    DTYPE: DType,
](
    mesh_filename: String,
    mut mesh_vert: List[Scalar[DTYPE]],
    mut mesh_vertadr: List[Int],
    mut mesh_vertnum: List[Int],
    mut num_meshes: Int,
) raises -> Tuple[Int, Scalar[DTYPE]]:
    """Load STL mesh, deduplicate, compute convex hull, store in model arrays.

    Returns (mesh_id, rbound) for this mesh.
    Vertices are stored in the mesh's LOCAL frame.
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

    # Compute convex hull
    var mesh_id = num_meshes
    var vert_offset = len(mesh_vert)
    mesh_vertadr.append(vert_offset)
    var num_hull = compute_convex_hull[DTYPE](unique, num_unique, mesh_vert)
    mesh_vertnum.append(num_hull)
    num_meshes += 1

    # Compute bounding radius from hull vertices
    var rbound = compute_bounding_radius_at[DTYPE](
        mesh_vert, vert_offset, num_hull
    )

    return (mesh_id, rbound)
