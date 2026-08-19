"""STL binary file loader.

Parses binary STL files and returns MeshData (vertices + indices) for GPU upload.
Each triangle gets its face normal. Vertices are NOT deduplicated — each triangle
has 3 unique vertices (simple approach suitable for meshes up to ~1000 triangles).

Auto-generates UV coordinates via box projection (largest face of bounding box).
"""

from std.memory import Pointer
from std.math import abs as math_abs
from .gpu_types import GPUVertex, MeshData
from .obj_loader import load_obj


def _is_obj(path: String) -> Bool:
    var n = path.byte_length()
    if n < 4:
        return False
    var ext = String(path[byte = n - 4 : n])
    return ext == ".obj" or ext == ".OBJ"


def load_stl(path: String) raises -> MeshData:
    """Load a mesh — binary STL, or Wavefront OBJ by extension.

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
        return load_obj(path)
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()

    var raw_ptr = content.unsafe_ptr()

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

        for v in range(3):
            var vp = (raw_ptr.unsafe_offset(offset + 12 + v * 12)).unsafe_bitcast[Float32]()
            var px = vp[unsafe_offset=0]
            var py = vp[unsafe_offset=1]
            var pz = vp[unsafe_offset=2]

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
