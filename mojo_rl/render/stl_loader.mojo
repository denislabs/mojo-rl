"""STL binary file loader.

Parses binary STL files and returns MeshData (vertices + indices) for GPU upload.
Each triangle gets its face normal. Vertices are NOT deduplicated — each triangle
has 3 unique vertices (simple approach suitable for meshes up to ~1000 triangles).
"""

from std.memory import UnsafePointer
from .gpu_types import GPUVertex, MeshData


def load_stl(path: String) raises -> MeshData:
    """Load a binary STL file and return MeshData (vertices + indices).

    Binary STL format:
      - 80 bytes: header (ignored)
      - 4 bytes: uint32 num_triangles
      - Per triangle (50 bytes):
        - 12 bytes: normal (float32 x3)
        - 36 bytes: 3 vertices (float32 x3 each)
        - 2 bytes: attribute byte count (ignored)

    Args:
        path: Path to the binary STL file.

    Returns:
        MeshData with sequential indices (no vertex deduplication).
    """
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()

    var raw_ptr = content.unsafe_ptr()

    # Parse number of triangles at offset 80
    var num_triangles = Int((raw_ptr + 80).bitcast[UInt32]()[])

    # Validate file size: 80 (header) + 4 (count) + 50 * num_triangles
    var expected_size = 84 + 50 * num_triangles
    if len(content) < expected_size:
        raise Error(
            "STL file too small: expected "
            + String(expected_size)
            + " bytes, got "
            + String(len(content))
        )

    var mesh = MeshData()

    # Pre-allocate
    var num_vertices = num_triangles * 3
    mesh.vertices.reserve(num_vertices)
    mesh.indices.reserve(num_vertices)

    var offset = 84  # Start of triangle data

    for tri in range(num_triangles):
        # Read face normal (3 x float32)
        var np = (raw_ptr + offset).bitcast[Float32]()
        var nx = np[0]
        var ny = np[1]
        var nz = np[2]

        # Read 3 vertices (each 3 x float32)
        for v in range(3):
            var vp = (raw_ptr + offset + 12 + v * 12).bitcast[Float32]()
            mesh.vertices.append(
                GPUVertex(
                    px=vp[0],
                    py=vp[1],
                    pz=vp[2],
                    nx=nx,
                    ny=ny,
                    nz=nz,
                )
            )
            mesh.indices.append(UInt16(tri * 3 + v))

        offset += 50  # 12 (normal) + 36 (3 vertices) + 2 (attribute)

    return mesh^
