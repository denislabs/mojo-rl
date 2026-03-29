"""STL binary file loader.

Parses binary STL files and returns MeshData (vertices + indices) for GPU upload.
Each triangle gets its face normal. Vertices are NOT deduplicated — each triangle
has 3 unique vertices (simple approach suitable for meshes up to ~1000 triangles).

Auto-generates UV coordinates via box projection (largest face of bounding box).
"""

from std.memory import UnsafePointer
from std.math import abs as math_abs
from .gpu_types import GPUVertex, MeshData


def load_stl(path: String) raises -> MeshData:
    """Load a binary STL file and return MeshData with auto-generated UVs.

    UVs are generated via box projection: the two axes with the largest
    bounding box extent are mapped to [0, 1] UV range.
    """
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()

    var raw_ptr = content.unsafe_ptr()

    # Parse number of triangles at offset 80
    var num_triangles = Int((raw_ptr + 80).bitcast[UInt32]()[])

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
        var np = (raw_ptr + offset).bitcast[Float32]()
        var nx = np[0]
        var ny = np[1]
        var nz = np[2]

        for v in range(3):
            var vp = (raw_ptr + offset + 12 + v * 12).bitcast[Float32]()
            var px = vp[0]
            var py = vp[1]
            var pz = vp[2]

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
            mesh.indices.append(UInt16(tri * 3 + v))

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
