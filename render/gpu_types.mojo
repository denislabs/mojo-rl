"""GPU Types and Helpers for 3D Rendering.

Vertex structs, uniform buffer layouts, mesh handles, and conversion helpers
for the SDL3 GPU-accelerated renderer.
"""

from memory import UnsafePointer, memcpy
from math import sqrt, tan, sin, cos
from math3d import Vec3 as Vec3Generic, Mat4 as Mat4Generic, Quat as QuatGeneric
from .sdl import Ptr, AnyOrigin, GPUBuffer
from .types import Color

comptime Vec3 = Vec3Generic[DType.float64]
comptime Mat4 = Mat4Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]

# --- Vertex format ---


struct GPUVertex(TrivialRegisterPassable):
    """32-byte vertex: position (3), normal (3), uv (2)."""

    var px: Float32
    var py: Float32
    var pz: Float32
    var nx: Float32
    var ny: Float32
    var nz: Float32
    var u: Float32
    var v: Float32

    fn __init__(
        out self,
        px: Float32,
        py: Float32,
        pz: Float32,
        nx: Float32 = 0.0,
        ny: Float32 = 0.0,
        nz: Float32 = 0.0,
        u: Float32 = 0.0,
        v: Float32 = 0.0,
    ):
        self.px = px
        self.py = py
        self.pz = pz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.u = u
        self.v = v


# --- Uniform structs (std140-aligned, all Float32) ---


struct SceneUniforms(ImplicitlyCopyable, Movable):
    """Scene-wide uniforms: 128 bytes.

    Layout (std140):
      view_proj: mat4 (64 bytes)
      camera_pos: vec4 (16 bytes) - w unused
      light_dir: vec4 (16 bytes) - w unused
      light_color: vec4 (16 bytes) - w = ambient intensity
      padding: vec4 (16 bytes)
    """

    var view_proj: InlineArray[Float32, 16]
    var camera_pos: InlineArray[Float32, 4]
    var light_dir: InlineArray[Float32, 4]
    var light_color: InlineArray[Float32, 4]
    var padding: InlineArray[Float32, 4]

    fn __init__(out self):
        self.view_proj = InlineArray[Float32, 16](fill=Float32(0))
        self.camera_pos = InlineArray[Float32, 4](fill=Float32(0))
        self.light_dir = InlineArray[Float32, 4](fill=Float32(0))
        self.light_color = InlineArray[Float32, 4](fill=Float32(0))
        self.padding = InlineArray[Float32, 4](fill=Float32(0))

    fn __copyinit__(out self, read other: Self):
        self.view_proj = other.view_proj.copy()
        self.camera_pos = other.camera_pos.copy()
        self.light_dir = other.light_dir.copy()
        self.light_color = other.light_color.copy()
        self.padding = other.padding.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.view_proj = other.view_proj^
        self.camera_pos = other.camera_pos^
        self.light_dir = other.light_dir^
        self.light_color = other.light_color^
        self.padding = other.padding^


struct ObjectUniforms(ImplicitlyCopyable, Movable):
    """Per-object uniforms: 96 bytes.

    Layout (std140):
      model: mat4 (64 bytes)
      color: vec4 (16 bytes)
      material: vec4 (16 bytes) — x=shininess, y=specular, z=reflectance, w=emission
    """

    var model: InlineArray[Float32, 16]
    var color: InlineArray[Float32, 4]
    var material: InlineArray[Float32, 4]

    fn __init__(out self):
        self.model = InlineArray[Float32, 16](fill=Float32(0))
        self.color = InlineArray[Float32, 4](fill=Float32(0))
        self.material = InlineArray[Float32, 4](fill=Float32(0))
        # Defaults: shininess=0.5, specular=0.5, reflectance=0.0, emission=0.0
        self.material[0] = 0.5
        self.material[1] = 0.5

    fn __copyinit__(out self, read other: Self):
        self.model = other.model.copy()
        self.color = other.color.copy()
        self.material = other.material.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.model = other.model^
        self.color = other.color^
        self.material = other.material^


struct SkyboxUniforms(ImplicitlyCopyable, Movable):
    """Skybox uniforms: 32 bytes.

    Layout (std140):
      top_color: vec4 (16 bytes) - gradient top color
      bottom_color: vec4 (16 bytes) - gradient bottom color
    """

    var top_color: InlineArray[Float32, 4]
    var bottom_color: InlineArray[Float32, 4]

    fn __init__(out self):
        self.top_color = InlineArray[Float32, 4](fill=Float32(0))
        self.bottom_color = InlineArray[Float32, 4](fill=Float32(0))
        # Default: white top, dark blue bottom
        self.top_color[0] = 0.8
        self.top_color[1] = 0.85
        self.top_color[2] = 0.95
        self.top_color[3] = 1.0
        self.bottom_color[0] = 0.3
        self.bottom_color[1] = 0.35
        self.bottom_color[2] = 0.5
        self.bottom_color[3] = 1.0

    fn __copyinit__(out self, read other: Self):
        self.top_color = other.top_color.copy()
        self.bottom_color = other.bottom_color.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.top_color = other.top_color^
        self.bottom_color = other.bottom_color^


struct LineUniforms(ImplicitlyCopyable, Movable):
    """Line uniforms: 80 bytes.

    Layout (std140):
      view_proj: mat4 (64 bytes)
      color: vec4 (16 bytes)
    """

    var view_proj: InlineArray[Float32, 16]
    var color: InlineArray[Float32, 4]

    fn __init__(out self):
        self.view_proj = InlineArray[Float32, 16](fill=Float32(0))
        self.color = InlineArray[Float32, 4](fill=Float32(0))

    fn __copyinit__(out self, read other: Self):
        self.view_proj = other.view_proj.copy()
        self.color = other.color.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.view_proj = other.view_proj^
        self.color = other.color^


struct ShadowUniforms(ImplicitlyCopyable, Movable):
    """Shadow mapping uniforms: 80 bytes.

    Layout (std140):
      light_view_proj: mat4 (64 bytes) - light's orthographic VP matrix
      params: vec4 (16 bytes) - x=shadow_intensity, y=bias, z=unused, w=unused
    """

    var light_view_proj: InlineArray[Float32, 16]
    var params: InlineArray[Float32, 4]

    fn __init__(out self):
        self.light_view_proj = InlineArray[Float32, 16](fill=Float32(0))
        self.params = InlineArray[Float32, 4](fill=Float32(0))
        # Defaults: intensity=0.5, bias=0.005
        self.params[0] = 0.5
        self.params[1] = 0.005

    fn __copyinit__(out self, read other: Self):
        self.light_view_proj = other.light_view_proj.copy()
        self.params = other.params.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.light_view_proj = other.light_view_proj^
        self.params = other.params^


# --- Mesh data structures ---


struct MeshData(Movable):
    """CPU-side mesh data for upload to GPU."""

    var vertices: List[GPUVertex]
    var indices: List[UInt16]

    fn __init__(out self):
        self.vertices = List[GPUVertex]()
        self.indices = List[UInt16]()

    fn __moveinit__(out self, deinit other: Self):
        self.vertices = other.vertices^
        self.indices = other.indices^

    fn vertex_byte_size(self) -> Int:
        return len(self.vertices) * 32  # sizeof(GPUVertex)

    fn index_byte_size(self) -> Int:
        return len(self.indices) * 2  # sizeof(UInt16)


struct MeshHandle(Copyable, Movable):
    """GPU-side mesh reference."""

    var vertex_buffer: Ptr[GPUBuffer, AnyOrigin[True]]
    var index_buffer: Ptr[GPUBuffer, AnyOrigin[True]]
    var num_indices: UInt32
    var num_vertices: UInt32

    fn __init__(out self):
        self.vertex_buffer = Ptr[GPUBuffer, AnyOrigin[True]]()
        self.index_buffer = Ptr[GPUBuffer, AnyOrigin[True]]()
        self.num_indices = 0
        self.num_vertices = 0

    fn __init__(
        out self,
        vertex_buffer: Ptr[GPUBuffer, AnyOrigin[True]],
        index_buffer: Ptr[GPUBuffer, AnyOrigin[True]],
        num_indices: UInt32,
        num_vertices: UInt32,
    ):
        self.vertex_buffer = vertex_buffer
        self.index_buffer = index_buffer
        self.num_indices = num_indices
        self.num_vertices = num_vertices

    fn __copyinit__(out self, read other: Self):
        self.vertex_buffer = other.vertex_buffer
        self.index_buffer = other.index_buffer
        self.num_indices = other.num_indices
        self.num_vertices = other.num_vertices

    fn __moveinit__(out self, deinit other: Self):
        self.vertex_buffer = other.vertex_buffer
        self.index_buffer = other.index_buffer
        self.num_indices = other.num_indices
        self.num_vertices = other.num_vertices


struct CapsuleCacheEntry(Copyable, Movable):
    """Cached capsule mesh keyed by approximate (radius, half_height)."""

    var radius: Float32
    var half_height: Float32
    var mesh: MeshHandle

    fn __init__(
        out self,
        radius: Float32,
        half_height: Float32,
        mesh: MeshHandle,
    ):
        self.radius = radius
        self.half_height = half_height
        self.mesh = MeshHandle(
            mesh.vertex_buffer,
            mesh.index_buffer,
            mesh.num_indices,
            mesh.num_vertices,
        )

    fn __copyinit__(out self, read other: Self):
        self.radius = other.radius
        self.half_height = other.half_height
        self.mesh = MeshHandle(
            other.mesh.vertex_buffer,
            other.mesh.index_buffer,
            other.mesh.num_indices,
            other.mesh.num_vertices,
        )

    fn __moveinit__(out self, deinit other: Self):
        self.radius = other.radius
        self.half_height = other.half_height
        self.mesh = other.mesh^

    fn matches(self, radius: Float32, half_height: Float32) -> Bool:
        """Check if this entry matches the given dimensions (within tolerance).
        """
        var eps = Float32(0.001)
        return (
            abs(self.radius - radius) < eps
            and abs(self.half_height - half_height) < eps
        )


struct SolidDrawCommand(ImplicitlyCopyable, Movable):
    """Deferred draw command for solid objects."""

    var mesh_idx: Int  # 0=sphere, 1=box, or capsule cache index + 100
    var uniforms: ObjectUniforms
    var is_capsule: Bool
    var capsule_cache_idx: Int

    fn __init__(
        out self,
        mesh_idx: Int,
        uniforms: ObjectUniforms,
        is_capsule: Bool = False,
        capsule_cache_idx: Int = 0,
    ):
        self.mesh_idx = mesh_idx
        self.uniforms = uniforms
        self.is_capsule = is_capsule
        self.capsule_cache_idx = capsule_cache_idx

    fn __copyinit__(out self, read other: Self):
        self.mesh_idx = other.mesh_idx
        self.uniforms = other.uniforms
        self.is_capsule = other.is_capsule
        self.capsule_cache_idx = other.capsule_cache_idx

    fn __moveinit__(out self, deinit other: Self):
        self.mesh_idx = other.mesh_idx
        self.uniforms = other.uniforms
        self.is_capsule = other.is_capsule
        self.capsule_cache_idx = other.capsule_cache_idx


# --- Helper functions ---


fn mat4_to_gpu_f32(m: Mat4) -> InlineArray[Float32, 16]:
    """Convert row-major Mat4[float64] to column-major Float32 array for GPU.

    Mat4 is row-major, but Metal/GLSL expect column-major. We transpose during
    the conversion.

    Args:
        m: Row-major 4x4 matrix.

    Returns:
        Column-major Float32 array (16 elements).
    """
    var out = InlineArray[Float32, 16](fill=Float32(0))
    # Transpose: out[col*4 + row] = m.row_col
    # Column 0
    out[0] = Float32(m.m00)
    out[1] = Float32(m.m10)
    out[2] = Float32(m.m20)
    out[3] = Float32(m.m30)
    # Column 1
    out[4] = Float32(m.m01)
    out[5] = Float32(m.m11)
    out[6] = Float32(m.m21)
    out[7] = Float32(m.m31)
    # Column 2
    out[8] = Float32(m.m02)
    out[9] = Float32(m.m12)
    out[10] = Float32(m.m22)
    out[11] = Float32(m.m32)
    # Column 3
    out[12] = Float32(m.m03)
    out[13] = Float32(m.m13)
    out[14] = Float32(m.m23)
    out[15] = Float32(m.m33)
    return out^


fn perspective_metal(
    fov_y: Float64, aspect: Float64, near: Float64, far: Float64
) -> Mat4:
    """Metal-compatible perspective projection with Z in [0, 1].

    Unlike OpenGL's [-1, 1], Metal clip space uses Z in [0, 1].

    Args:
        fov_y: Vertical field of view in radians.
        aspect: Aspect ratio (width / height).
        near: Near clipping plane.
        far: Far clipping plane.

    Returns:
        4x4 perspective projection matrix (row-major).
    """
    var f = 1.0 / tan(fov_y * 0.5)
    var nf = 1.0 / (near - far)

    # Row-major perspective matrix with Metal Z range [0, 1]
    var m = Mat4.identity()
    m.m00 = f / aspect
    m.m01 = 0.0
    m.m02 = 0.0
    m.m03 = 0.0

    m.m10 = 0.0
    m.m11 = f
    m.m12 = 0.0
    m.m13 = 0.0

    m.m20 = 0.0
    m.m21 = 0.0
    m.m22 = far * nf  # Maps to [0, 1] instead of [-1, 1]
    m.m23 = near * far * nf

    m.m30 = 0.0
    m.m31 = 0.0
    m.m32 = -1.0
    m.m33 = 0.0

    return m


fn color_to_vec4(color: Color) -> InlineArray[Float32, 4]:
    """Convert Color to normalized Float32 RGBA.

    Args:
        color: RGBA color (0-255 per component).

    Returns:
        Float32 RGBA array with alpha from color.a.
    """
    var out = InlineArray[Float32, 4](fill=Float32(0))
    out[0] = Float32(color.r) / 255.0
    out[1] = Float32(color.g) / 255.0
    out[2] = Float32(color.b) / 255.0
    out[3] = Float32(color.a) / 255.0
    return out^


fn color_to_vec4(r: UInt8, g: UInt8, b: UInt8) -> InlineArray[Float32, 4]:
    """Convert UInt8 RGB to normalized Float32 RGBA.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        Float32 RGBA array with alpha = 1.0.
    """
    var out = InlineArray[Float32, 4](fill=Float32(0))
    out[0] = Float32(r) / 255.0
    out[1] = Float32(g) / 255.0
    out[2] = Float32(b) / 255.0
    out[3] = 1.0
    return out^


fn make_identity_f32() -> InlineArray[Float32, 16]:
    """Create a Float32 identity matrix in column-major order."""
    var out = InlineArray[Float32, 16](fill=Float32(0))
    out[0] = 1.0
    out[5] = 1.0
    out[10] = 1.0
    out[15] = 1.0
    return out^


fn ortho_metal(
    left: Float64,
    right: Float64,
    bottom: Float64,
    top: Float64,
    near: Float64,
    far: Float64,
) -> Mat4:
    """Metal-compatible orthographic projection with Z in [0, 1].

    Args:
        left: Left clipping plane.
        right: Right clipping plane.
        bottom: Bottom clipping plane.
        top: Top clipping plane.
        near: Near clipping plane.
        far: Far clipping plane.

    Returns:
        4x4 orthographic projection matrix (row-major).
    """
    var m = Mat4.identity()
    m.m00 = 2.0 / (right - left)
    m.m01 = 0.0
    m.m02 = 0.0
    m.m03 = -(right + left) / (right - left)

    m.m10 = 0.0
    m.m11 = 2.0 / (top - bottom)
    m.m12 = 0.0
    m.m13 = -(top + bottom) / (top - bottom)

    m.m20 = 0.0
    m.m21 = 0.0
    m.m22 = -1.0 / (far - near)  # Metal Z range [0, 1]
    m.m23 = -near / (far - near)

    m.m30 = 0.0
    m.m31 = 0.0
    m.m32 = 0.0
    m.m33 = 1.0

    return m
