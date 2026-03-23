"""GPU Mesh Generation.

Generates indexed triangle list meshes for common 3D shapes.
All meshes are returned as MeshData (vertices + UInt16 indices)
and are meant to be uploaded once at init time.
"""

from std.math import sin, cos, sqrt
from .gpu_types import GPUVertex, MeshData


def generate_sphere(segments: Int = 16, rings: Int = 12) -> MeshData:
    """Generate a unit sphere centered at origin.

    UV sphere with normals = position (unit sphere).

    Args:
        segments: Number of longitude segments.
        rings: Number of latitude rings.

    Returns:
        MeshData with vertices and indices.
    """
    var mesh = MeshData()
    var pi = Float32(3.14159265358979)

    # Generate vertices
    for j in range(rings + 1):
        var phi = Float32(j) * pi / Float32(rings)
        var sin_phi = sin(phi)
        var cos_phi = cos(phi)

        for i in range(segments + 1):
            var theta = Float32(i) * 2.0 * pi / Float32(segments)
            var sin_theta = sin(theta)
            var cos_theta = cos(theta)

            var x = sin_phi * cos_theta
            var y = sin_phi * sin_theta
            var z = cos_phi

            var u = Float32(i) / Float32(segments)
            var v = Float32(j) / Float32(rings)

            mesh.vertices.append(GPUVertex(x, y, z, x, y, z, u, v))

    # Generate indices
    for j in range(rings):
        for i in range(segments):
            var current = j * (segments + 1) + i
            var next_ring = (j + 1) * (segments + 1) + i

            # Triangle 1
            mesh.indices.append(UInt16(current))
            mesh.indices.append(UInt16(next_ring))
            mesh.indices.append(UInt16(current + 1))

            # Triangle 2
            mesh.indices.append(UInt16(current + 1))
            mesh.indices.append(UInt16(next_ring))
            mesh.indices.append(UInt16(next_ring + 1))

    return mesh^


def generate_box() -> MeshData:
    """Generate a unit box from [-0.5, 0.5] per axis with face normals.

    Each face has 4 unique vertices (for flat normals), 6 indices.

    Returns:
        MeshData with 24 vertices and 36 indices.
    """
    var mesh = MeshData()

    # Face data: (normal, then 4 corners as offsets)
    # +Z face (top)
    _add_box_face(
        mesh,
        nx=0,
        ny=0,
        nz=1,
        p0=(-0.5, -0.5, 0.5),
        p1=(0.5, -0.5, 0.5),
        p2=(0.5, 0.5, 0.5),
        p3=(-0.5, 0.5, 0.5),
    )
    # -Z face (bottom)
    _add_box_face(
        mesh,
        nx=0,
        ny=0,
        nz=-1,
        p0=(-0.5, 0.5, -0.5),
        p1=(0.5, 0.5, -0.5),
        p2=(0.5, -0.5, -0.5),
        p3=(-0.5, -0.5, -0.5),
    )
    # +X face
    _add_box_face(
        mesh,
        nx=1,
        ny=0,
        nz=0,
        p0=(0.5, -0.5, -0.5),
        p1=(0.5, 0.5, -0.5),
        p2=(0.5, 0.5, 0.5),
        p3=(0.5, -0.5, 0.5),
    )
    # -X face
    _add_box_face(
        mesh,
        nx=-1,
        ny=0,
        nz=0,
        p0=(-0.5, 0.5, -0.5),
        p1=(-0.5, -0.5, -0.5),
        p2=(-0.5, -0.5, 0.5),
        p3=(-0.5, 0.5, 0.5),
    )
    # +Y face
    _add_box_face(
        mesh,
        nx=0,
        ny=1,
        nz=0,
        p0=(0.5, 0.5, -0.5),
        p1=(-0.5, 0.5, -0.5),
        p2=(-0.5, 0.5, 0.5),
        p3=(0.5, 0.5, 0.5),
    )
    # -Y face
    _add_box_face(
        mesh,
        nx=0,
        ny=-1,
        nz=0,
        p0=(-0.5, -0.5, -0.5),
        p1=(0.5, -0.5, -0.5),
        p2=(0.5, -0.5, 0.5),
        p3=(-0.5, -0.5, 0.5),
    )

    return mesh^


def _add_box_face(
    mut mesh: MeshData,
    nx: Float32,
    ny: Float32,
    nz: Float32,
    p0: Tuple[Float64, Float64, Float64],
    p1: Tuple[Float64, Float64, Float64],
    p2: Tuple[Float64, Float64, Float64],
    p3: Tuple[Float64, Float64, Float64],
):
    """Add a single box face (4 vertices, 2 triangles) to mesh."""
    var base = UInt16(len(mesh.vertices))

    mesh.vertices.append(
        GPUVertex(
            Float32(p0[0]), Float32(p0[1]), Float32(p0[2]), nx, ny, nz, 0, 0
        )
    )
    mesh.vertices.append(
        GPUVertex(
            Float32(p1[0]), Float32(p1[1]), Float32(p1[2]), nx, ny, nz, 1, 0
        )
    )
    mesh.vertices.append(
        GPUVertex(
            Float32(p2[0]), Float32(p2[1]), Float32(p2[2]), nx, ny, nz, 1, 1
        )
    )
    mesh.vertices.append(
        GPUVertex(
            Float32(p3[0]), Float32(p3[1]), Float32(p3[2]), nx, ny, nz, 0, 1
        )
    )

    mesh.indices.append(base + 0)
    mesh.indices.append(base + 1)
    mesh.indices.append(base + 2)
    mesh.indices.append(base + 0)
    mesh.indices.append(base + 2)
    mesh.indices.append(base + 3)


def generate_capsule(
    radius: Float32,
    half_height: Float32,
    segments: Int = 16,
    hemi_rings: Int = 6,
) -> MeshData:
    """Generate a capsule mesh along the Z-axis.

    Cylinder from z=-half_height to z=+half_height with hemispherical caps.
    Cannot use unit mesh + scale because hemispheres would get squashed.

    Args:
        radius: Capsule radius.
        half_height: Half-height of the cylindrical section.
        segments: Number of circular segments.
        hemi_rings: Number of rings per hemisphere.

    Returns:
        MeshData with vertices and indices.
    """
    var mesh = MeshData()
    var pi = Float32(3.14159265358979)

    # --- Top hemisphere ---
    # Apex vertex
    mesh.vertices.append(GPUVertex(0, 0, half_height + radius, 0, 0, 1, 0.5, 0))

    for j in range(1, hemi_rings + 1):
        var phi = Float32(j) * (pi * 0.5) / Float32(hemi_rings)
        var sin_phi = sin(phi)
        var cos_phi = cos(phi)

        for i in range(segments + 1):
            var theta = Float32(i) * 2.0 * pi / Float32(segments)
            var sin_theta = sin(theta)
            var cos_theta = cos(theta)

            var nx = sin_phi * cos_theta
            var ny = sin_phi * sin_theta
            var nz = cos_phi

            var px = radius * nx
            var py = radius * ny
            var pz = half_height + radius * nz

            var u = Float32(i) / Float32(segments)
            var v = Float32(j) / Float32(hemi_rings * 2 + 1)

            mesh.vertices.append(GPUVertex(px, py, pz, nx, ny, nz, u, v))

    # Top hemisphere indices - fan from apex
    for i in range(segments):
        mesh.indices.append(UInt16(0))  # apex
        mesh.indices.append(UInt16(1 + i))
        mesh.indices.append(UInt16(1 + i + 1))

    # Top hemisphere indices - strips
    for j in range(1, hemi_rings):
        for i in range(segments):
            var curr = 1 + (j - 1) * (segments + 1) + i
            var next_r = 1 + j * (segments + 1) + i

            mesh.indices.append(UInt16(curr))
            mesh.indices.append(UInt16(next_r))
            mesh.indices.append(UInt16(curr + 1))

            mesh.indices.append(UInt16(curr + 1))
            mesh.indices.append(UInt16(next_r))
            mesh.indices.append(UInt16(next_r + 1))

    # --- Cylinder ---
    var cyl_base_top = len(mesh.vertices)

    # Top ring of cylinder
    for i in range(segments + 1):
        var theta = Float32(i) * 2.0 * pi / Float32(segments)
        var ct = cos(theta)
        var st = sin(theta)
        var u = Float32(i) / Float32(segments)

        mesh.vertices.append(
            GPUVertex(
                radius * ct,
                radius * st,
                half_height,
                ct,
                st,
                0,
                u,
                0.5
                - Float32(half_height) / Float32(2.0 * (half_height + radius)),
            )
        )

    var cyl_base_bot = len(mesh.vertices)

    # Bottom ring of cylinder
    for i in range(segments + 1):
        var theta = Float32(i) * 2.0 * pi / Float32(segments)
        var ct = cos(theta)
        var st = sin(theta)
        var u = Float32(i) / Float32(segments)

        mesh.vertices.append(
            GPUVertex(
                radius * ct,
                radius * st,
                -half_height,
                ct,
                st,
                0,
                u,
                0.5
                + Float32(half_height) / Float32(2.0 * (half_height + radius)),
            )
        )

    # Cylinder indices
    for i in range(segments):
        var t = cyl_base_top + i
        var b = cyl_base_bot + i

        mesh.indices.append(UInt16(t))
        mesh.indices.append(UInt16(b))
        mesh.indices.append(UInt16(t + 1))

        mesh.indices.append(UInt16(t + 1))
        mesh.indices.append(UInt16(b))
        mesh.indices.append(UInt16(b + 1))

    # --- Bottom hemisphere ---
    var bot_base = len(mesh.vertices)

    for j in range(hemi_rings):
        var phi = (pi * 0.5) + Float32(j) * (pi * 0.5) / Float32(hemi_rings)
        var sin_phi = sin(phi)
        var cos_phi = cos(phi)

        for i in range(segments + 1):
            var theta = Float32(i) * 2.0 * pi / Float32(segments)
            var sin_theta = sin(theta)
            var cos_theta = cos(theta)

            var nx = sin_phi * cos_theta
            var ny = sin_phi * sin_theta
            var nz = cos_phi

            var px = radius * nx
            var py = radius * ny
            var pz = -half_height + radius * nz

            var u = Float32(i) / Float32(segments)
            var v = Float32(hemi_rings + 1 + j) / Float32(hemi_rings * 2 + 1)

            mesh.vertices.append(GPUVertex(px, py, pz, nx, ny, nz, u, v))

    # Bottom apex
    var bot_apex = len(mesh.vertices)
    mesh.vertices.append(
        GPUVertex(0, 0, -half_height - radius, 0, 0, -1, 0.5, 1)
    )

    # Bottom hemisphere indices - strips
    for j in range(hemi_rings - 1):
        for i in range(segments):
            var curr = bot_base + j * (segments + 1) + i
            var next_r = bot_base + (j + 1) * (segments + 1) + i

            mesh.indices.append(UInt16(curr))
            mesh.indices.append(UInt16(next_r))
            mesh.indices.append(UInt16(curr + 1))

            mesh.indices.append(UInt16(curr + 1))
            mesh.indices.append(UInt16(next_r))
            mesh.indices.append(UInt16(next_r + 1))

    # Bottom hemisphere indices - fan to apex
    var last_ring_base = bot_base + (hemi_rings - 1) * (segments + 1)
    for i in range(segments):
        mesh.indices.append(UInt16(last_ring_base + i))
        mesh.indices.append(UInt16(bot_apex))
        mesh.indices.append(UInt16(last_ring_base + i + 1))

    return mesh^


def generate_ground(size: Float32 = 10.0) -> MeshData:
    """Generate a large ground quad at Z=0.

    The quad lies in the XY plane with normal pointing up (+Z).
    Checkerboard pattern is handled in the shader.

    Args:
        size: Half-size of the ground quad.

    Returns:
        MeshData with 4 vertices and 6 indices.
    """
    var mesh = MeshData()

    # Vertices: corners of the ground plane
    mesh.vertices.append(GPUVertex(-size, -size, 0, 0, 0, 1, 0, 0))
    mesh.vertices.append(GPUVertex(size, -size, 0, 0, 0, 1, 1, 0))
    mesh.vertices.append(GPUVertex(size, size, 0, 0, 0, 1, 1, 1))
    mesh.vertices.append(GPUVertex(-size, size, 0, 0, 0, 1, 0, 1))

    # Two triangles
    mesh.indices.append(UInt16(0))
    mesh.indices.append(UInt16(1))
    mesh.indices.append(UInt16(2))
    mesh.indices.append(UInt16(0))
    mesh.indices.append(UInt16(2))
    mesh.indices.append(UInt16(3))

    return mesh^
