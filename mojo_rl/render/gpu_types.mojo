"""GPU Types and Helpers for 3D Rendering.

Vertex structs, uniform buffer layouts, mesh handles, and conversion helpers
for the SDL3 GPU-accelerated renderer.
"""

from std.memory import Pointer, unsafe_memcpy
from std.math import sqrt, tan, sin, cos
from mojo_rl.math3d import (
    Vec3 as Vec3Generic,
    Mat4 as Mat4Generic,
    Quat as QuatGeneric,
)
from .sdl import Ptr, AnyOrigin, GPUBuffer, GPUTexture, GPUSampler, untracked
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

    def __init__(
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
    """Scene-wide uniforms: 240 bytes.

    Layout (std140):
      view_proj:      mat4  (64 bytes)
      camera_pos:     vec4  (16 bytes) - w = num_active_lights (float)
      light0_dir:     vec4  (16 bytes) - w = ambient0
      light0_color:   vec4  (16 bytes) - w = cast_shadow (0/1)
      light1_dir:     vec4  (16 bytes) - w = ambient1
      light1_color:   vec4  (16 bytes) - w = cast_shadow1
      light2_dir:     vec4  (16 bytes) - w = ambient2
      light2_color:   vec4  (16 bytes) - w = cast_shadow2
      light3_dir:     vec4  (16 bytes) - w = ambient3
      light3_color:   vec4  (16 bytes) - w = cast_shadow3
      ground_params:  vec4  (16 bytes) - xyz = checker_color2, w = ground_z
      fog_params:     vec4  (16 bytes) - x = fogstart, y = fogend, z = unused, w = unused
    """

    var view_proj: InlineArray[Float32, 16]
    var camera_pos: InlineArray[Float32, 4]
    var light0_dir: InlineArray[Float32, 4]
    var light0_color: InlineArray[Float32, 4]
    var light1_dir: InlineArray[Float32, 4]
    var light1_color: InlineArray[Float32, 4]
    var light2_dir: InlineArray[Float32, 4]
    var light2_color: InlineArray[Float32, 4]
    var light3_dir: InlineArray[Float32, 4]
    var light3_color: InlineArray[Float32, 4]
    var ground_params: InlineArray[Float32, 4]
    var fog_params: InlineArray[Float32, 4]

    def __init__(out self):
        self.view_proj = InlineArray[Float32, 16](fill=Float32(0))
        self.camera_pos = InlineArray[Float32, 4](fill=Float32(0))
        self.light0_dir = InlineArray[Float32, 4](fill=Float32(0))
        self.light0_color = InlineArray[Float32, 4](fill=Float32(0))
        self.light1_dir = InlineArray[Float32, 4](fill=Float32(0))
        self.light1_color = InlineArray[Float32, 4](fill=Float32(0))
        self.light2_dir = InlineArray[Float32, 4](fill=Float32(0))
        self.light2_color = InlineArray[Float32, 4](fill=Float32(0))
        self.light3_dir = InlineArray[Float32, 4](fill=Float32(0))
        self.light3_color = InlineArray[Float32, 4](fill=Float32(0))
        self.ground_params = InlineArray[Float32, 4](fill=Float32(0))
        self.fog_params = InlineArray[Float32, 4](fill=Float32(0))

    def __init__(out self, *, copy: Self):
        self.view_proj = copy.view_proj.copy()
        self.camera_pos = copy.camera_pos.copy()
        self.light0_dir = copy.light0_dir.copy()
        self.light0_color = copy.light0_color.copy()
        self.light1_dir = copy.light1_dir.copy()
        self.light1_color = copy.light1_color.copy()
        self.light2_dir = copy.light2_dir.copy()
        self.light2_color = copy.light2_color.copy()
        self.light3_dir = copy.light3_dir.copy()
        self.light3_color = copy.light3_color.copy()
        self.ground_params = copy.ground_params.copy()
        self.fog_params = copy.fog_params.copy()

    def __init__(out self, *, deinit move: Self):
        self.view_proj = move.view_proj^
        self.camera_pos = move.camera_pos^
        self.light0_dir = move.light0_dir^
        self.light0_color = move.light0_color^
        self.light1_dir = move.light1_dir^
        self.light1_color = move.light1_color^
        self.light2_dir = move.light2_dir^
        self.light2_color = move.light2_color^
        self.light3_dir = move.light3_dir^
        self.light3_color = move.light3_color^
        self.ground_params = move.ground_params^
        self.fog_params = move.fog_params^


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

    def __init__(out self):
        self.model = InlineArray[Float32, 16](fill=Float32(0))
        self.color = InlineArray[Float32, 4](fill=Float32(0))
        self.material = InlineArray[Float32, 4](fill=Float32(0))
        # Defaults: shininess=0.5, specular=0.5, reflectance=0.0, emission=0.0
        self.material[0] = 0.5
        self.material[1] = 0.5

    def __init__(out self, *, copy: Self):
        self.model = copy.model.copy()
        self.color = copy.color.copy()
        self.material = copy.material.copy()

    def __init__(out self, *, deinit move: Self):
        self.model = move.model^
        self.color = move.color^
        self.material = move.material^


struct SkyboxUniforms(ImplicitlyCopyable, Movable):
    """Skybox uniforms: 96 bytes.

    Layout (std140):
      top_color: vec4 (16 bytes) - gradient top color
      bottom_color: vec4 (16 bytes) - gradient bottom color
      mark_color: vec4 (16 bytes) - starfield rgb + density in .w (0 = off)
      cam_right: vec4 (16 bytes) - camera right basis, .w = tan(fovy/2)
      cam_up: vec4 (16 bytes) - camera up basis, .w = aspect
      cam_fwd: vec4 (16 bytes) - camera forward basis, .w unused

    ⚠ THE CAMERA BASIS IS HERE FOR THE STARS, AND ONLY FOR THEM. The gradient
    needs nothing but a screen-space y, but stars have to be fixed to the WORLD
    or they swim across the sky as the camera turns — which reads as a bug in
    the camera, not in the sky. Three basis vectors plus tan(fovy/2) and the
    aspect are enough for the fragment to rebuild the view ray per pixel, and
    are cheaper than shipping and inverting a view-projection matrix.

    ⚠ MUST MATCH `SkyboxUniforms` in gpu_shaders.mojo field for field. Metal
    reads this buffer positionally; a field added on one side only is not a
    compile error anywhere, it is garbage in the shader.
    """

    var top_color: InlineArray[Float32, 4]
    var bottom_color: InlineArray[Float32, 4]
    var mark_color: InlineArray[Float32, 4]
    var cam_right: InlineArray[Float32, 4]
    var cam_up: InlineArray[Float32, 4]
    var cam_fwd: InlineArray[Float32, 4]

    def __init__(out self):
        self.top_color = InlineArray[Float32, 4](fill=Float32(0))
        self.bottom_color = InlineArray[Float32, 4](fill=Float32(0))
        self.mark_color = InlineArray[Float32, 4](fill=Float32(0))
        self.cam_right = InlineArray[Float32, 4](fill=Float32(0))
        self.cam_up = InlineArray[Float32, 4](fill=Float32(0))
        self.cam_fwd = InlineArray[Float32, 4](fill=Float32(0))
        # Default: white top, dark blue bottom
        self.top_color[0] = 0.8
        self.top_color[1] = 0.85
        self.top_color[2] = 0.95
        self.top_color[3] = 1.0
        self.bottom_color[0] = 0.3
        self.bottom_color[1] = 0.35
        self.bottom_color[2] = 0.5
        self.bottom_color[3] = 1.0

    def __init__(out self, *, copy: Self):
        self.top_color = copy.top_color.copy()
        self.bottom_color = copy.bottom_color.copy()
        self.mark_color = copy.mark_color.copy()
        self.cam_right = copy.cam_right.copy()
        self.cam_up = copy.cam_up.copy()
        self.cam_fwd = copy.cam_fwd.copy()

    def __init__(out self, *, deinit move: Self):
        self.top_color = move.top_color^
        self.bottom_color = move.bottom_color^
        self.mark_color = move.mark_color^
        self.cam_right = move.cam_right^
        self.cam_up = move.cam_up^
        self.cam_fwd = move.cam_fwd^


struct LineUniforms(ImplicitlyCopyable, Movable):
    """Line uniforms: 80 bytes.

    Layout (std140):
      view_proj: mat4 (64 bytes)
      color: vec4 (16 bytes)
    """

    var view_proj: InlineArray[Float32, 16]
    var color: InlineArray[Float32, 4]

    def __init__(out self):
        self.view_proj = InlineArray[Float32, 16](fill=Float32(0))
        self.color = InlineArray[Float32, 4](fill=Float32(0))

    def __init__(out self, *, copy: Self):
        self.view_proj = copy.view_proj.copy()
        self.color = copy.color.copy()

    def __init__(out self, *, deinit move: Self):
        self.view_proj = move.view_proj^
        self.color = move.color^


struct ShadowUniforms(ImplicitlyCopyable, Movable):
    """Shadow mapping uniforms: 80 bytes.

    Layout (std140):
      light_view_proj: mat4 (64 bytes) - light's orthographic VP matrix
      params: vec4 (16 bytes) - x=shadow_intensity, y=bias, z=unused, w=unused
    """

    var light_view_proj: InlineArray[Float32, 16]
    var params: InlineArray[Float32, 4]

    def __init__(out self):
        self.light_view_proj = InlineArray[Float32, 16](fill=Float32(0))
        self.params = InlineArray[Float32, 4](fill=Float32(0))
        # Defaults: intensity=0.5, bias=0.005
        self.params[0] = 0.5
        self.params[1] = 0.005

    def __init__(out self, *, copy: Self):
        self.light_view_proj = copy.light_view_proj.copy()
        self.params = copy.params.copy()

    def __init__(out self, *, deinit move: Self):
        self.light_view_proj = move.light_view_proj^
        self.params = move.params^


# --- Mesh data structures ---


struct MeshData(Movable):
    """CPU-side mesh data for upload to GPU."""

    var vertices: List[GPUVertex]
    var indices: List[UInt16]

    def __init__(out self):
        self.vertices = List[GPUVertex]()
        self.indices = List[UInt16]()

    def __init__(out self, *, deinit move: Self):
        self.vertices = move.vertices^
        self.indices = move.indices^

    def vertex_byte_size(self) -> Int:
        return len(self.vertices) * 32  # sizeof(GPUVertex)

    def index_byte_size(self) -> Int:
        return len(self.indices) * 2  # sizeof(UInt16)


struct MeshHandle(Copyable, Movable):
    """GPU-side mesh reference.

    Mojo nightly removed nullable Pointer, so there is no longer a
    no-arg constructor that produces a sentinel "empty" handle. Callers
    that need a deferred-init field should wrap this in
    ``Optional[MeshHandle]`` and assign once a real mesh is uploaded.
    """

    var vertex_buffer: Ptr[GPUBuffer, MutUntrackedOrigin]
    var index_buffer: Ptr[GPUBuffer, MutUntrackedOrigin]
    var num_indices: UInt32
    var num_vertices: UInt32

    def __init__[
        vb_o: MutOrigin, ib_o: MutOrigin, //
    ](
        out self,
        vertex_buffer: Ptr[GPUBuffer, vb_o],
        index_buffer: Ptr[GPUBuffer, ib_o],
        num_indices: UInt32,
        num_vertices: UInt32,
    ):
        self.vertex_buffer = untracked(vertex_buffer)
        self.index_buffer = untracked(index_buffer)
        self.num_indices = num_indices
        self.num_vertices = num_vertices

    def __init__(out self, *, copy: Self):
        self.vertex_buffer = copy.vertex_buffer
        self.index_buffer = copy.index_buffer
        self.num_indices = copy.num_indices
        self.num_vertices = copy.num_vertices

    def __init__(out self, *, deinit move: Self):
        self.vertex_buffer = move.vertex_buffer
        self.index_buffer = move.index_buffer
        self.num_indices = move.num_indices
        self.num_vertices = move.num_vertices


struct CapsuleCacheEntry(Copyable, Movable):
    """Cached capsule mesh keyed by approximate (radius, half_height)."""

    var radius: Float32
    var half_height: Float32
    var mesh: MeshHandle

    def __init__(
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

    def __init__(out self, *, copy: Self):
        self.radius = copy.radius
        self.half_height = copy.half_height
        self.mesh = MeshHandle(
            copy.mesh.vertex_buffer,
            copy.mesh.index_buffer,
            copy.mesh.num_indices,
            copy.mesh.num_vertices,
        )

    def __init__(out self, *, deinit move: Self):
        self.radius = move.radius
        self.half_height = move.half_height
        self.mesh = move.mesh^

    def matches(self, radius: Float32, half_height: Float32) -> Bool:
        """Check if this entry matches the given dimensions (within tolerance).
        """
        var eps = Float32(0.001)
        return (
            abs(self.radius - radius) < eps
            and abs(self.half_height - half_height) < eps
        )


struct MeshCacheEntry(Copyable, Movable):
    """Cached STL mesh keyed by name string."""

    var name: String
    var mesh: MeshHandle

    def __init__(out self, name: String, mesh: MeshHandle):
        self.name = name
        self.mesh = MeshHandle(
            mesh.vertex_buffer,
            mesh.index_buffer,
            mesh.num_indices,
            mesh.num_vertices,
        )

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.mesh = MeshHandle(
            copy.mesh.vertex_buffer,
            copy.mesh.index_buffer,
            copy.mesh.num_indices,
            copy.mesh.num_vertices,
        )

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.mesh = move.mesh^

    def matches(self, name: String) -> Bool:
        """Check if this entry matches the given name."""
        return self.name == name


struct TextureCacheEntry(Copyable, Movable):
    """Cached GPU texture keyed by name string."""

    var name: String
    var texture: Ptr[GPUTexture, MutUntrackedOrigin]
    var sampler: Ptr[GPUSampler, MutUntrackedOrigin]
    var width: UInt32
    var height: UInt32

    def __init__[
        tex_o: MutOrigin, samp_o: MutOrigin, //
    ](
        out self,
        name: String,
        texture: Ptr[GPUTexture, tex_o],
        sampler: Ptr[GPUSampler, samp_o],
        width: UInt32,
        height: UInt32,
    ):
        self.name = name
        self.texture = untracked(texture)
        self.sampler = untracked(sampler)
        self.width = width
        self.height = height

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.texture = copy.texture
        self.sampler = copy.sampler
        self.width = copy.width
        self.height = copy.height

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.texture = move.texture
        self.sampler = move.sampler
        self.width = move.width
        self.height = move.height

    def matches(self, name: String) -> Bool:
        """Check if this entry matches the given name."""
        return self.name == name


struct SolidDrawCommand(ImplicitlyCopyable, Movable):
    """Deferred draw command for solid objects."""

    var mesh_idx: Int  # 0=sphere, 1=box, or capsule cache index + 100
    var uniforms: ObjectUniforms
    var is_capsule: Bool
    var capsule_cache_idx: Int
    var is_cylinder: Bool
    var cylinder_cache_idx: Int
    var is_mesh: Bool
    var mesh_cache_idx: Int
    var texture_cache_idx: Int  # -1 = no texture (use default white)

    def __init__(
        out self,
        mesh_idx: Int,
        uniforms: ObjectUniforms,
        is_capsule: Bool = False,
        capsule_cache_idx: Int = 0,
        is_cylinder: Bool = False,
        cylinder_cache_idx: Int = 0,
        is_mesh: Bool = False,
        mesh_cache_idx: Int = 0,
        texture_cache_idx: Int = -1,
    ):
        self.mesh_idx = mesh_idx
        self.uniforms = uniforms
        self.is_capsule = is_capsule
        self.capsule_cache_idx = capsule_cache_idx
        self.is_cylinder = is_cylinder
        self.cylinder_cache_idx = cylinder_cache_idx
        self.is_mesh = is_mesh
        self.mesh_cache_idx = mesh_cache_idx
        self.texture_cache_idx = texture_cache_idx

    def __init__(out self, *, copy: Self):
        self.mesh_idx = copy.mesh_idx
        self.uniforms = copy.uniforms
        self.is_capsule = copy.is_capsule
        self.capsule_cache_idx = copy.capsule_cache_idx
        self.is_cylinder = copy.is_cylinder
        self.cylinder_cache_idx = copy.cylinder_cache_idx
        self.is_mesh = copy.is_mesh
        self.mesh_cache_idx = copy.mesh_cache_idx
        self.texture_cache_idx = copy.texture_cache_idx

    def __init__(out self, *, deinit move: Self):
        self.mesh_idx = move.mesh_idx
        self.uniforms = move.uniforms
        self.is_capsule = move.is_capsule
        self.capsule_cache_idx = move.capsule_cache_idx
        self.is_cylinder = move.is_cylinder
        self.cylinder_cache_idx = move.cylinder_cache_idx
        self.is_mesh = move.is_mesh
        self.mesh_cache_idx = move.mesh_cache_idx
        self.texture_cache_idx = move.texture_cache_idx


# --- Helper functions ---


def mat4_to_gpu_f32(m: Mat4) -> InlineArray[Float32, 16]:
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


def perspective_projection(
    fov_y: Float64, aspect: Float64, near: Float64, far: Float64
) -> Mat4:
    """Perspective projection with Z in [0, 1].

    Compatible with Metal, Vulkan, and D3D12 clip space (all use Z in [0, 1]).

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

    # Row-major perspective matrix with Z range [0, 1]
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


def color_to_vec4(color: Color) -> InlineArray[Float32, 4]:
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


def color_to_vec4(r: UInt8, g: UInt8, b: UInt8) -> InlineArray[Float32, 4]:
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


def make_identity_f32() -> InlineArray[Float32, 16]:
    """Create a Float32 identity matrix in column-major order."""
    var out = InlineArray[Float32, 16](fill=Float32(0))
    out[0] = 1.0
    out[5] = 1.0
    out[10] = 1.0
    out[15] = 1.0
    return out^


def ortho_projection(
    left: Float64,
    right: Float64,
    bottom: Float64,
    top: Float64,
    near: Float64,
    far: Float64,
) -> Mat4:
    """Orthographic projection with Z in [0, 1].

    Compatible with Metal, Vulkan, and D3D12 clip space.

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
    m.m22 = -1.0 / (far - near)  # Z range [0, 1]
    m.m23 = -near / (far - near)

    m.m30 = 0.0
    m.m31 = 0.0
    m.m32 = 0.0
    m.m33 = 1.0

    return m


# --- Text rendering types ---


struct TextVertex(TrivialRegisterPassable):
    """32-byte vertex for 2D text rendering: pos(float2) + uv(float2) + color(float4).
    """

    var px: Float32  # screen x (pixels)
    var py: Float32  # screen y (pixels)
    var u: Float32  # atlas UV x
    var v: Float32  # atlas UV y
    var cr: Float32  # color red
    var cg: Float32  # color green
    var cb: Float32  # color blue
    var ca: Float32  # color alpha

    def __init__(
        out self,
        px: Float32,
        py: Float32,
        u: Float32,
        v: Float32,
        cr: Float32,
        cg: Float32,
        cb: Float32,
        ca: Float32,
    ):
        self.px = px
        self.py = py
        self.u = u
        self.v = v
        self.cr = cr
        self.cg = cg
        self.cb = cb
        self.ca = ca


struct TextUniforms(ImplicitlyCopyable, Movable):
    """64-byte uniform block for text shader: column-major ortho projection mat4.
    """

    var ortho_proj: InlineArray[Float32, 16]

    def __init__(out self):
        self.ortho_proj = make_identity_f32()

    def __init__(out self, *, copy: Self):
        self.ortho_proj = copy.ortho_proj.copy()

    def __init__(out self, *, deinit move: Self):
        self.ortho_proj = move.ortho_proj^
