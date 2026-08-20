"""STL binary file loader.

Parses binary STL files and returns MeshData (vertices + indices) for GPU upload.
Each triangle gets its face normal. Vertices are NOT deduplicated — each triangle
has 3 unique vertices (simple approach suitable for meshes up to ~1000 triangles).

Auto-generates UV coordinates via box projection (largest face of bounding box).
"""

from std.memory import Pointer
from std.math import abs as math_abs, sqrt
from .gpu_types import GPUVertex, MeshData
from .obj_loader import load_obj


def _is_obj(path: String) -> Bool:
    var n = path.byte_length()
    if n < 4:
        return False
    var ext = String(path[byte = n - 4 : n])
    return ext == ".obj" or ext == ".OBJ"


def _scale_mesh(
    mut mesh: MeshData, sx: Float64, sy: Float64, sz: Float64
):
    """Apply `<mesh scale>` to an already-loaded mesh — the OBJ path.

    ⚠ THE STL PATH SCALES AS IT READS instead, because it is building the
    bounding box in the same pass and the UV box-projection picks the two
    LARGEST axes — under a non-uniform scale those can be different axes. Same
    arithmetic, applied at the only point each path can apply it.
    """
    if sx == 1.0 and sy == 1.0 and sz == 1.0:
        return
    var fx = Float32(sx)
    var fy = Float32(sy)
    var fz = Float32(sz)
    var mirrored = (sx * sy * sz) < 0.0
    for i in range(len(mesh.vertices)):
        ref v = mesh.vertices[i]
        v.px = v.px * fx
        v.py = v.py * fy
        v.pz = v.pz * fz
        var nx = v.nx / fx
        var ny = v.ny / fy
        var nz = v.nz / fz
        if mirrored:
            nx = -nx
            ny = -ny
            nz = -nz
        var nl = sqrt(nx * nx + ny * ny + nz * nz)
        if nl > Float32(1e-20):
            nx = nx / nl
            ny = ny / nl
            nz = nz / nl
        v.nx = nx
        v.ny = ny
        v.nz = nz
    if mirrored:
        # Reverse each triangle's winding so the front face still faces out.
        var ntri = len(mesh.indices) // 3
        for t in range(ntri):
            var a = mesh.indices[t * 3 + 1]
            mesh.indices[t * 3 + 1] = mesh.indices[t * 3 + 2]
            mesh.indices[t * 3 + 2] = a


def load_stl(
    path: String,
    sx: Float64 = 1.0,
    sy: Float64 = 1.0,
    sz: Float64 = 1.0,
) raises -> MeshData:
    """Load a mesh — binary STL, or Wavefront OBJ by extension.

    `sx`/`sy`/`sz` are MJCF's `<mesh scale>`, applied HERE for the same reason
    the OBJ dispatch is here: the three callers (the renderer's `draw_mesh`,
    `mesh_inertia`, `convex_hull`) must not each carry their own copy, and a
    caller that forgot would be silently wrong rather than broken. 19
    Menagerie robots set it — 38 declarations are `0.001 0.001 0.001`, i.e.
    the STL is in MILLIMETRES.

    ⚠ A NEGATIVE COMPONENT IS A MIRROR, NOT A ROTATION, and 44 Menagerie
    declarations use one (`1 -1 1` and friends) to build a left part and a
    right part from a single file. Mirroring reverses triangle winding, so
    the winding is flipped and the face normal recomputed whenever the scale's
    determinant is negative; without that the mirrored copy renders inside-out.
    Both are skipped entirely at scale 1, so the default path is byte-identical
    to what it was.

    ⚠⚠ THE OBJ DISPATCH IS HERE, NOT AT THE CALL SITES, and the name stayed
    `load_stl` for the same reason: there are three callers (the renderer's
    `draw_mesh`, `mesh_inertia`, `convex_hull`) and adding the check to each
    would eventually miss one. A missed one is silent — the mesh simply does
    not load.

    Before this, an `.obj` was read as a binary STL: bytes 80-84 of its TEXT
    became a triangle count, and the failure was "STL file too small: expected
    46324738584 bytes, got 2193823". A wrong diagnosis pointing at the right
    file, which sends you to check the file's size. Menagerie ships 1184 `.obj`
    against 1129 `.stl`, so about half of it was unreadable.

    UVs are generated via box projection: the two axes with the largest
    bounding box extent are mapped to [0, 1] UV range.
    """
    if _is_obj(path):
        var om = load_obj(path)
        _scale_mesh(om, sx, sy, sz)
        return om^
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()

    var raw_ptr = content.unsafe_ptr()

    var fx = Float32(sx)
    var fy = Float32(sy)
    var fz = Float32(sz)
    var scaled = sx != 1.0 or sy != 1.0 or sz != 1.0
    var mirrored = (sx * sy * sz) < 0.0

    # Parse number of triangles at offset 80
    var num_triangles = Int((raw_ptr.unsafe_offset(80)).unsafe_bitcast[UInt32]()[])

    # Validate file size
    var expected_size = 84 + 50 * num_triangles
    if len(content) < expected_size:
        raise Error(
            "STL file too small: expected "
            + String(expected_size)
            + " bytes, got "
            + String(len(content))
        )

    var mesh = MeshData()
    var num_vertices = num_triangles * 3
    mesh.vertices.reserve(num_vertices)
    mesh.indices.reserve(num_vertices)

    # First pass: load vertices + find bounding box
    var min_x = Float32(1e10)
    var min_y = Float32(1e10)
    var min_z = Float32(1e10)
    var max_x = Float32(-1e10)
    var max_y = Float32(-1e10)
    var max_z = Float32(-1e10)

    var offset = 84
    for tri in range(num_triangles):
        var np = (raw_ptr.unsafe_offset(offset)).unsafe_bitcast[Float32]()
        var nx = np[unsafe_offset=0]
        var ny = np[unsafe_offset=1]
        var nz = np[unsafe_offset=2]
        if scaled:
            # ⚠ A NORMAL DOES NOT SCALE LIKE A POSITION. Under a non-uniform
            # scale it transforms by the INVERSE TRANSPOSE, i.e. n_i / s_i,
            # renormalised; scaling it like a point tilts every normal and the
            # shading goes with it. Uniform scales (the 0.001 case) come out
            # unchanged after renormalising, which is why this was invisible
            # until a `0.9 1 1` model showed up.
            nx = nx / fx
            ny = ny / fy
            nz = nz / fz
            if mirrored:
                nx = -nx
                ny = -ny
                nz = -nz
            var nl = sqrt(nx * nx + ny * ny + nz * nz)
            if nl > Float32(1e-20):
                nx = nx / nl
                ny = ny / nl
                nz = nz / nl

        for v_ in range(3):
            # ⚠ THE WINDING FLIP IS AN INDEX SWAP, not a second pass: for a
            # mirrored scale the triangle is emitted 0,2,1 so its front face
            # still points outward.
            var v = v_
            if mirrored:
                v = 0 if v_ == 0 else (3 - v_)
            var vp = (raw_ptr.unsafe_offset(offset + 12 + v * 12)).unsafe_bitcast[Float32]()
            var px = vp[unsafe_offset=0] * fx
            var py = vp[unsafe_offset=1] * fy
            var pz = vp[unsafe_offset=2] * fz

            if px < min_x:
                min_x = px
            if px > max_x:
                max_x = px
            if py < min_y:
                min_y = py
            if py > max_y:
                max_y = py
            if pz < min_z:
                min_z = pz
            if pz > max_z:
                max_z = pz

            mesh.vertices.append(
                GPUVertex(px=px, py=py, pz=pz, nx=nx, ny=ny, nz=nz)
            )
            mesh.indices.append(UInt32(tri * 3 + v))

        offset += 50

    # Second pass: assign UVs via box projection
    # Pick the two largest axes for UV mapping
    var dx = max_x - min_x
    var dy = max_y - min_y
    var dz = max_z - min_z
    if dx < Float32(1e-6):
        dx = Float32(1.0)
    if dy < Float32(1e-6):
        dy = Float32(1.0)
    if dz < Float32(1e-6):
        dz = Float32(1.0)

    # Determine which axis is smallest (project onto the other two)
    var use_xy: Bool
    var use_xz = False
    # var use_yz = False
    if dz >= dx and dz >= dy:
        use_xy = True  # Z is largest → project onto XY
    elif dy >= dx and dy >= dz:
        use_xz = True  # Y is largest → project onto XZ
        use_xy = False
    else:
        # use_yz = True  # X is largest → project onto YZ
        use_xy = False

    for i in range(num_vertices):
        var vert = mesh.vertices[i]
        var u: Float32
        var v: Float32
        if use_xy:
            u = (vert.px - min_x) / dx
            v = (vert.py - min_y) / dy
        elif use_xz:
            u = (vert.px - min_x) / dx
            v = (vert.pz - min_z) / dz
        else:
            u = (vert.py - min_y) / dy
            v = (vert.pz - min_z) / dz
        mesh.vertices[i] = GPUVertex(
            px=vert.px,
            py=vert.py,
            pz=vert.pz,
            nx=vert.nx,
            ny=vert.ny,
            nz=vert.nz,
            u=u,
            v=v,
        )

    return mesh^
