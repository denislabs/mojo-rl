"""GPU-Accelerated 3D Renderer.

Uses SDL3's GPU API with Metal (MSL shaders) for true 3D rendering
with Blinn-Phong lighting, depth buffering, and procedural checkerboard ground.
"""

from memory import UnsafePointer, memcpy, alloc
from math import sqrt, sin, cos
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric, Mat4 as Mat4Generic
from ffi import _get_dylib_function
from .sdl import (
    Ptr,
    AnyOrigin,
    lib,
    c_char,
    c_float,
    c_int,
    # GPU types
    GPUDevice,
    GPUBuffer,
    GPUTransferBuffer,
    GPUTexture,
    GPUSampler,
    GPUShader,
    GPUGraphicsPipeline,
    GPUCommandBuffer,
    GPURenderPass,
    GPUCopyPass,
    # Create info structs
    GPUShaderCreateInfo,
    GPUGraphicsPipelineCreateInfo,
    GPUGraphicsPipelineTargetInfo,
    GPUBufferCreateInfo,
    GPUTransferBufferCreateInfo,
    GPUTextureCreateInfo,
    GPUSamplerCreateInfo,
    GPUTextureSamplerBinding,
    GPUColorTargetDescription,
    GPUColorTargetBlendState,
    GPUColorTargetInfo,
    GPUDepthStencilTargetInfo,
    GPUDepthStencilState,
    GPURasterizerState,
    GPUMultisampleState,
    GPUVertexInputState,
    GPUVertexBufferDescription,
    GPUVertexAttribute,
    GPUBufferBinding,
    GPUBufferRegion,
    GPUTransferBufferLocation,
    GPUViewport,
    # Enums
    GPUPrimitiveType,
    GPULoadOp,
    GPUStoreOp,
    GPUShaderStage,
    GPUShaderFormat,
    GPUTextureFormat,
    GPUTextureType,
    GPUTextureUsageFlags,
    GPUBufferUsageFlags,
    GPUTransferBufferUsage,
    GPUVertexElementFormat,
    GPUVertexInputRate,
    GPUFillMode,
    GPUCullMode,
    GPUFrontFace,
    GPUCompareOp,
    GPUStencilOp,
    GPUStencilOpState,
    GPUSampleCount,
    GPUIndexElementSize,
    GPUBlendFactor,
    GPUBlendOp,
    GPUColorComponentFlags,
    GPUFilter,
    GPUSamplerMipmapMode,
    GPUSamplerAddressMode,
    PropertiesID,
    # SDL core
    Window,
    WindowFlags,
    Event,
    EventType,
    KeyboardEvent,
    Keycode,
    FColor,
    InitFlags,
    # Functions
    init,
    quit,
    create_window,
    destroy_window,
    poll_event,
    delay,
    destroy_gpu_device,
    claim_window_for_gpu_device,
    release_window_from_gpu_device,
    get_gpu_swapchain_texture_format,
    create_gpu_shader,
    create_gpu_graphics_pipeline,
    create_gpu_buffer,
    create_gpu_transfer_buffer,
    create_gpu_texture,
    create_gpu_sampler,
    acquire_gpu_command_buffer,
    wait_and_acquire_gpu_swapchain_texture,
    begin_gpu_render_pass,
    end_gpu_render_pass,
    begin_gpu_copy_pass,
    end_gpu_copy_pass,
    submit_gpu_command_buffer,
    bind_gpu_graphics_pipeline,
    set_gpu_viewport,
    bind_gpu_vertex_buffers,
    bind_gpu_index_buffer,
    draw_gpu_indexed_primitives,
    draw_gpu_primitives,
    push_gpu_vertex_uniform_data,
    push_gpu_fragment_uniform_data,
    bind_gpu_fragment_samplers,
    upload_to_gpu_buffer,
    map_gpu_transfer_buffer,
    unmap_gpu_transfer_buffer,
    release_gpu_buffer,
    release_gpu_transfer_buffer,
    release_gpu_texture,
    release_gpu_sampler,
    release_gpu_shader,
    release_gpu_graphics_pipeline,
)
from .camera3d import Camera3D
from .types import Color
from .light import Light
from .gpu_types import (
    GPUVertex,
    SceneUniforms,
    ObjectUniforms,
    LineUniforms,
    ShadowUniforms,
    SkyboxUniforms,
    MeshData,
    MeshHandle,
    CapsuleCacheEntry,
    SolidDrawCommand,
    mat4_to_gpu_f32,
    perspective_metal,
    ortho_metal,
    color_to_vec4,
    make_identity_f32,
)
from .gpu_mesh import (
    generate_sphere,
    generate_box,
    generate_capsule,
    generate_ground,
)
from .gpu_shaders import (
    SOLID_VERTEX_MSL,
    SOLID_FRAGMENT_MSL,
    GROUND_VERTEX_MSL,
    GROUND_FRAGMENT_MSL,
    LINE_VERTEX_MSL,
    LINE_FRAGMENT_MSL,
    SHADOW_VERTEX_MSL,
    SHADOW_FRAGMENT_MSL,
    REFLECTION_FRAGMENT_MSL,
    SKYBOX_VERTEX_MSL,
    SKYBOX_FRAGMENT_MSL,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]
comptime Mat4 = Mat4Generic[DType.float64]


# Maximum line vertices per frame
comptime MAX_LINE_VERTICES = 512


# --- Line color entry for list storage ---


struct LineColorEntry(Copyable, Movable):
    """Stores RGBA color for a line segment."""

    var r: Float32
    var g: Float32
    var b: Float32
    var a: Float32

    fn __init__(out self, color: InlineArray[Float32, 4]):
        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]

    fn __copyinit__(out self, read other: Self):
        self.r = other.r
        self.g = other.g
        self.b = other.b
        self.a = other.a

    fn __moveinit__(out self, deinit other: Self):
        self.r = other.r
        self.g = other.g
        self.b = other.b
        self.a = other.a

    fn to_inline_array(self) -> InlineArray[Float32, 4]:
        var out = InlineArray[Float32, 4](fill=Float32(0))
        out[0] = self.r
        out[1] = self.g
        out[2] = self.b
        out[3] = self.a
        return out^


struct Renderer3D(Movable):
    """GPU-accelerated 3D renderer using SDL3 GPU API.

    Uses Metal (MSL) shaders for Blinn-Phong lit solid rendering with
    procedural checkerboard ground and flat-color line drawing.
    """

    # SDL3 handles
    var window: Ptr[Window, AnyOrigin[True]]
    var device: Ptr[GPUDevice, AnyOrigin[True]]

    # Pipelines
    var solid_pipeline: Ptr[GPUGraphicsPipeline, AnyOrigin[True]]
    var ground_pipeline: Ptr[GPUGraphicsPipeline, AnyOrigin[True]]
    var line_pipeline: Ptr[GPUGraphicsPipeline, AnyOrigin[True]]
    var shadow_pipeline: Ptr[GPUGraphicsPipeline, AnyOrigin[True]]
    var reflection_pipeline: Ptr[GPUGraphicsPipeline, AnyOrigin[True]]
    var skybox_pipeline: Ptr[GPUGraphicsPipeline, AnyOrigin[True]]

    # Depth buffer
    var depth_texture: Ptr[GPUTexture, AnyOrigin[True]]

    # Shadow mapping resources
    var shadow_map: Ptr[GPUTexture, AnyOrigin[True]]
    var shadow_sampler: Ptr[GPUSampler, AnyOrigin[True]]
    var shadow_uniforms: ShadowUniforms
    var ground_z: Float64

    # Cached static meshes
    var sphere_mesh: MeshHandle
    var box_mesh: MeshHandle
    var ground_mesh: MeshHandle
    var capsule_cache: List[CapsuleCacheEntry]

    # Dynamic line buffer
    var line_vertex_data: List[Float32]  # x,y,z per vertex
    var line_colors: List[LineColorEntry]  # color per segment (2 verts)
    var line_vertex_buffer: Ptr[GPUBuffer, AnyOrigin[True]]
    var line_transfer_buffer: Ptr[GPUTransferBuffer, AnyOrigin[True]]

    # Deferred draw commands
    var solid_draws: List[SolidDrawCommand]
    var ground_uniforms: ObjectUniforms
    var has_ground: Bool

    # Camera and scene
    var camera: Camera3D
    var width: Int
    var height: Int
    var background_color: Color
    var scene_uniforms: SceneUniforms

    # Skybox
    var skybox_uniforms: SkyboxUniforms
    var draw_skybox: Bool

    # Configurable light parameters (up to 4 lights)
    var lights: List[Light]

    # Swapchain format
    var swapchain_format: GPUTextureFormat

    # State
    var initialized: Bool
    var should_quit: Bool
    var draw_grid: Bool
    var draw_axes: Bool

    # Camera switching (set by check_quit, read by ModelRenderer)
    var camera_switch_request: Int  # -1 = none, 0-8 = switch to camera N

    fn __init__(
        out self,
        width: Int = 800,
        height: Int = 450,
        camera: Camera3D = Camera3D(),
        draw_grid: Bool = True,
        draw_axes: Bool = True,
        lights: List[Light] = List[Light](),
        light_dir_x: Float32 = 0.3,
        light_dir_y: Float32 = -0.4,
        light_dir_z: Float32 = -0.8,
        light_color_r: Float32 = 1.0,
        light_color_g: Float32 = 0.98,
        light_color_b: Float32 = 0.95,
        light_ambient: Float32 = 0.25,
    ) raises:
        self.width = width
        self.height = height
        self.background_color = Color(32, 32, 48, 255)
        self.draw_grid = draw_grid
        self.draw_axes = draw_axes
        self.should_quit = False
        self.initialized = False

        # Copy camera
        self.camera = Camera3D(
            eye=camera.eye,
            target=camera.target,
            up=camera.up,
            fov=camera.fov * 180.0 / 3.14159265358979,
            aspect=Float64(width) / Float64(height),
            near=camera.near,
            far=camera.far,
            screen_width=width,
            screen_height=height,
        )

        # Null handles
        self.window = Ptr[Window, AnyOrigin[True]]()
        self.device = Ptr[GPUDevice, AnyOrigin[True]]()
        self.solid_pipeline = Ptr[GPUGraphicsPipeline, AnyOrigin[True]]()
        self.ground_pipeline = Ptr[GPUGraphicsPipeline, AnyOrigin[True]]()
        self.line_pipeline = Ptr[GPUGraphicsPipeline, AnyOrigin[True]]()
        self.shadow_pipeline = Ptr[GPUGraphicsPipeline, AnyOrigin[True]]()
        self.reflection_pipeline = Ptr[GPUGraphicsPipeline, AnyOrigin[True]]()
        self.skybox_pipeline = Ptr[GPUGraphicsPipeline, AnyOrigin[True]]()
        self.depth_texture = Ptr[GPUTexture, AnyOrigin[True]]()
        self.shadow_map = Ptr[GPUTexture, AnyOrigin[True]]()
        self.shadow_sampler = Ptr[GPUSampler, AnyOrigin[True]]()
        self.shadow_uniforms = ShadowUniforms()
        self.ground_z = 0.0
        self.line_vertex_buffer = Ptr[GPUBuffer, AnyOrigin[True]]()
        self.line_transfer_buffer = Ptr[GPUTransferBuffer, AnyOrigin[True]]()
        self.swapchain_format = (
            GPUTextureFormat.GPU_TEXTUREFORMAT_B8G8R8A8_UNORM
        )

        # Meshes
        self.sphere_mesh = MeshHandle()
        self.box_mesh = MeshHandle()
        self.ground_mesh = MeshHandle()
        self.capsule_cache = List[CapsuleCacheEntry]()

        # Line data
        self.line_vertex_data = List[Float32]()
        self.line_colors = List[LineColorEntry]()

        # Draw commands
        self.solid_draws = List[SolidDrawCommand]()
        self.ground_uniforms = ObjectUniforms()
        self.has_ground = False

        self.scene_uniforms = SceneUniforms()
        self.skybox_uniforms = SkyboxUniforms()
        self.draw_skybox = False

        # Store configurable light parameters (up to 4 lights)
        self.camera_switch_request = -1
        if len(lights) > 0:
            self.lights = lights.copy()
        else:
            # Create default light from individual params (backward compatibility)
            self.lights = List[Light]()
            self.lights.append(
                Light(
                    mode=0,  # LIGHT_DIRECTIONAL
                    dir_x=Float64(light_dir_x),
                    dir_y=Float64(light_dir_y),
                    dir_z=Float64(light_dir_z),
                    color_r=Float64(light_color_r),
                    color_g=Float64(light_color_g),
                    color_b=Float64(light_color_b),
                    ambient=Float64(light_ambient),
                    specular_intensity=0.3,
                    specular_exponent=32.0,
                    cast_shadow=True,
                )
            )

    fn __moveinit__(out self, deinit other: Self):
        self.window = other.window
        self.device = other.device
        self.solid_pipeline = other.solid_pipeline
        self.ground_pipeline = other.ground_pipeline
        self.line_pipeline = other.line_pipeline
        self.shadow_pipeline = other.shadow_pipeline
        self.reflection_pipeline = other.reflection_pipeline
        self.skybox_pipeline = other.skybox_pipeline
        self.depth_texture = other.depth_texture
        self.shadow_map = other.shadow_map
        self.shadow_sampler = other.shadow_sampler
        self.shadow_uniforms = other.shadow_uniforms
        self.ground_z = other.ground_z
        self.sphere_mesh = other.sphere_mesh^
        self.box_mesh = other.box_mesh^
        self.ground_mesh = other.ground_mesh^
        self.capsule_cache = other.capsule_cache^
        self.line_vertex_data = other.line_vertex_data^
        self.line_colors = other.line_colors^
        self.line_vertex_buffer = other.line_vertex_buffer
        self.line_transfer_buffer = other.line_transfer_buffer
        self.solid_draws = other.solid_draws^
        self.ground_uniforms = other.ground_uniforms
        self.has_ground = other.has_ground
        self.camera = other.camera^
        self.width = other.width
        self.height = other.height
        self.background_color = other.background_color
        self.scene_uniforms = other.scene_uniforms
        self.skybox_uniforms = other.skybox_uniforms
        self.draw_skybox = other.draw_skybox
        self.swapchain_format = other.swapchain_format
        self.lights = other.lights^
        self.camera_switch_request = other.camera_switch_request
        self.initialized = other.initialized
        self.should_quit = other.should_quit
        self.draw_grid = other.draw_grid
        self.draw_axes = other.draw_axes

    fn init(mut self, mut title: String) raises:
        """Initialize SDL3, GPU device, pipelines, and static meshes."""
        # 1. Init SDL3
        init(InitFlags.INIT_VIDEO)

        # 2. Create window
        self.window = create_window(
            title, c_int(self.width), c_int(self.height), WindowFlags(0)
        )

        # 3. Create GPU device (MSL shaders, debug mode)
        # Must pass NULL (not empty string) for driver name to auto-select
        self.device = _get_dylib_function[
            lib,
            "SDL_CreateGPUDevice",
            fn(
                GPUShaderFormat, Bool, Ptr[c_char, AnyOrigin[False]]
            ) -> Ptr[GPUDevice, AnyOrigin[True]],
        ]()(
            GPUShaderFormat.GPU_SHADERFORMAT_MSL,
            True,
            Ptr[c_char, AnyOrigin[False]](),  # NULL = auto-select driver
        )

        # 4. Claim window
        claim_window_for_gpu_device(self.device, self.window)

        # 5. Get swapchain format
        self.swapchain_format = get_gpu_swapchain_texture_format(
            self.device, self.window
        )

        # 6. Create shaders and pipelines
        self._create_pipelines()

        # 7. Create depth texture
        self._create_depth_texture()

        # 8. Create shadow map resources
        self._create_shadow_resources()

        # 9. Generate and upload static meshes
        self._upload_static_meshes()

        # 10. Allocate line buffers
        self._create_line_buffers()

        self.initialized = True

    fn _create_shader(
        self,
        source: String,
        stage: GPUShaderStage,
        num_uniform_buffers: UInt32,
        entrypoint: String,
        num_samplers: UInt32 = 0,
    ) raises -> Ptr[GPUShader, AnyOrigin[True]]:
        """Compile an MSL shader from source string."""
        var code_bytes = source.as_bytes()
        var ep = entrypoint

        var info = GPUShaderCreateInfo(
            code_size=len(code_bytes),
            code=code_bytes.unsafe_ptr(),
            entrypoint=ep.as_c_string_slice().unsafe_ptr(),
            format=GPUShaderFormat.GPU_SHADERFORMAT_MSL,
            stage=stage,
            num_samplers=num_samplers,
            num_storage_textures=0,
            num_storage_buffers=0,
            num_uniform_buffers=num_uniform_buffers,
            props=PropertiesID(0),
        )

        return create_gpu_shader(self.device, Ptr(to=info))

    fn _no_stencil_op(self) -> GPUStencilOpState:
        """Return a zeroed-out stencil op state."""
        return GPUStencilOpState(
            fail_op=GPUStencilOp.GPU_STENCILOP_KEEP,
            pass_op=GPUStencilOp.GPU_STENCILOP_KEEP,
            depth_fail_op=GPUStencilOp.GPU_STENCILOP_KEEP,
            compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
        )

    fn _create_pipelines(mut self) raises:
        """Create solid, ground, line, shadow, and reflection GPU pipelines."""
        # --- Solid pipeline ---
        var solid_vs = self._create_shader(
            SOLID_VERTEX_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
            entrypoint=String("solid_vertex"),
        )
        var solid_fs = self._create_shader(
            SOLID_FRAGMENT_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=2,
            entrypoint=String("solid_fragment"),
            num_samplers=1,
        )

        # Vertex input - allocate attributes contiguously on heap
        var solid_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        var solid_attrs = alloc[GPUVertexAttribute](3)
        solid_attrs[0] = GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        )
        solid_attrs[1] = GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        )
        solid_attrs[2] = GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        )
        var solid_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=solid_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=solid_attrs,
            num_vertex_attributes=3,
        )

        # Color target - no blending
        var solid_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=False,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        var solid_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=solid_vs,
            fragment_shader=solid_fs,
            vertex_input_state=solid_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_BACK,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=True,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=solid_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.solid_pipeline = create_gpu_graphics_pipeline(
            self.device, Ptr(to=solid_pi)
        )

        # --- Ground pipeline (alpha blend for distance fade) ---
        var ground_vs = self._create_shader(
            GROUND_VERTEX_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
            entrypoint=String("ground_vertex"),
        )
        var ground_fs = self._create_shader(
            GROUND_FRAGMENT_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=2,
            entrypoint=String("ground_fragment"),
            num_samplers=1,
        )

        var ground_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_SRC_ALPHA,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=True,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Ground uses same vertex layout as solid - allocate contiguously
        var ground_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        var ground_attrs = alloc[GPUVertexAttribute](3)
        ground_attrs[0] = GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        )
        ground_attrs[1] = GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        )
        ground_attrs[2] = GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        )
        var ground_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=ground_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=ground_attrs,
            num_vertex_attributes=3,
        )

        var ground_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=ground_vs,
            fragment_shader=ground_fs,
            vertex_input_state=ground_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS_OR_EQUAL,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=True,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=ground_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.ground_pipeline = create_gpu_graphics_pipeline(
            self.device, Ptr(to=ground_pi)
        )

        # --- Line pipeline ---
        var line_vs = self._create_shader(
            LINE_VERTEX_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=1,
            entrypoint=String("line_vertex"),
        )
        var line_fs = self._create_shader(
            LINE_FRAGMENT_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=1,
            entrypoint=String("line_fragment"),
        )

        var line_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=False,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Line vertex input - single attribute (position only)
        var line_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=12,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        var line_attr = GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        )
        var line_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=line_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=Ptr(to=line_attr),
            num_vertex_attributes=1,
        )

        var line_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=line_vs,
            fragment_shader=line_fs,
            vertex_input_state=line_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_LINELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS_OR_EQUAL,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=False,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=line_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.line_pipeline = create_gpu_graphics_pipeline(
            self.device, Ptr(to=line_pi)
        )

        # --- Shadow pipeline (depth-only, from light POV) ---
        var shadow_vs = self._create_shader(
            SHADOW_VERTEX_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
            entrypoint=String("shadow_vertex"),
        )
        var shadow_fs = self._create_shader(
            SHADOW_FRAGMENT_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=0,
            entrypoint=String("shadow_fragment"),
        )

        # Shadow uses same vertex layout as solid
        var shadow_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        var shadow_attrs = alloc[GPUVertexAttribute](3)
        shadow_attrs[0] = GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        )
        shadow_attrs[1] = GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        )
        shadow_attrs[2] = GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        )
        var shadow_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=shadow_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=shadow_attrs,
            num_vertex_attributes=3,
        )

        var shadow_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=shadow_vs,
            fragment_shader=shadow_fs,
            vertex_input_state=shadow_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_BACK,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=1.5,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=2.0,
                enable_depth_bias=True,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=True,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr[
                    GPUColorTargetDescription, AnyOrigin[False]
                ](),
                num_color_targets=0,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.shadow_pipeline = create_gpu_graphics_pipeline(
            self.device, Ptr(to=shadow_pi)
        )

        # --- Reflection pipeline (alpha-blended, front-cull, no depth write) ---
        var refl_fs = self._create_shader(
            REFLECTION_FRAGMENT_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=1,
            entrypoint=String("reflection_fragment"),
        )

        # Reuse solid_vs for reflection (same vertex output struct)
        # Need a second reference - create another shader
        var refl_vs = self._create_shader(
            SOLID_VERTEX_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
            entrypoint=String("solid_vertex"),
        )

        var refl_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_SRC_ALPHA,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=True,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Reflection uses same vertex layout as solid
        var refl_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        var refl_attrs = alloc[GPUVertexAttribute](3)
        refl_attrs[0] = GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        )
        refl_attrs[1] = GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        )
        refl_attrs[2] = GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        )
        var refl_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=refl_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=refl_attrs,
            num_vertex_attributes=3,
        )

        var refl_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=refl_vs,
            fragment_shader=refl_fs,
            vertex_input_state=refl_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_FRONT,  # Front-cull (Z-flip reverses winding)
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=False,  # Don't write depth (ground renders on top)
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=refl_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.reflection_pipeline = create_gpu_graphics_pipeline(
            self.device, Ptr(to=refl_pi)
        )

        # --- Skybox pipeline (fullscreen gradient, no depth write, no vertex input) ---
        var skybox_vs = self._create_shader(
            SKYBOX_VERTEX_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=0,
            entrypoint=String("skybox_vertex"),
        )
        var skybox_fs = self._create_shader(
            SKYBOX_FRAGMENT_MSL,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=1,
            entrypoint=String("skybox_fragment"),
        )

        var skybox_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=False,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Skybox has NO vertex input (uses vertex_id to generate fullscreen triangle)
        var skybox_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr[GPUVertexBufferDescription, AnyOrigin[False]](),
            num_vertex_buffers=0,
            vertex_attributes=Ptr[GPUVertexAttribute, AnyOrigin[False]](),
            num_vertex_attributes=0,
        )

        var skybox_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=skybox_vs,
            fragment_shader=skybox_fs,
            vertex_input_state=skybox_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=False,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS_OR_EQUAL,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=False,
                enable_depth_write=False,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=skybox_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.skybox_pipeline = create_gpu_graphics_pipeline(
            self.device, Ptr(to=skybox_pi)
        )

        # Free heap-allocated vertex attribute arrays
        solid_attrs.free()
        ground_attrs.free()
        shadow_attrs.free()
        refl_attrs.free()

        # Release shader objects (pipelines retain them)
        release_gpu_shader(self.device, solid_vs)
        release_gpu_shader(self.device, solid_fs)
        release_gpu_shader(self.device, ground_vs)
        release_gpu_shader(self.device, ground_fs)
        release_gpu_shader(self.device, line_vs)
        release_gpu_shader(self.device, line_fs)
        release_gpu_shader(self.device, shadow_vs)
        release_gpu_shader(self.device, shadow_fs)
        release_gpu_shader(self.device, refl_vs)
        release_gpu_shader(self.device, refl_fs)
        release_gpu_shader(self.device, skybox_vs)
        release_gpu_shader(self.device, skybox_fs)

    fn _create_depth_texture(mut self) raises:
        """Create the depth buffer texture."""
        var info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
            width=UInt32(self.width),
            height=UInt32(self.height),
            layer_count_or_depth=1,
            num_levels=1,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        self.depth_texture = create_gpu_texture(self.device, Ptr(to=info))

    fn _create_shadow_resources(mut self) raises:
        """Create shadow map texture and comparison sampler."""
        # Shadow map: D32_FLOAT, usable as both depth target and sampler source
        var sm_info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET
            | GPUTextureUsageFlags.GPU_TEXTUREUSAGE_SAMPLER,
            width=1024,
            height=1024,
            layer_count_or_depth=1,
            num_levels=1,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        self.shadow_map = create_gpu_texture(self.device, Ptr(to=sm_info))

        # Comparison sampler for shadow mapping
        var sampler_info = GPUSamplerCreateInfo(
            min_filter=GPUFilter.GPU_FILTER_LINEAR,
            mag_filter=GPUFilter.GPU_FILTER_LINEAR,
            mipmap_mode=GPUSamplerMipmapMode.GPU_SAMPLERMIPMAPMODE_NEAREST,
            address_mode_u=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_v=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_w=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            mip_lod_bias=0.0,
            max_anisotropy=1.0,
            compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
            min_lod=0.0,
            max_lod=0.0,
            enable_anisotropy=False,
            enable_compare=True,
            padding1=0,
            padding2=0,
            props=PropertiesID(0),
        )
        self.shadow_sampler = create_gpu_sampler(
            self.device, Ptr(to=sampler_info)
        )

    fn _upload_mesh(self, mesh_data: MeshData) raises -> MeshHandle:
        """Upload mesh data to GPU buffers via transfer buffer."""
        var vb_size = UInt32(mesh_data.vertex_byte_size())
        var ib_size = UInt32(mesh_data.index_byte_size())
        var total_size = vb_size + ib_size

        # Create transfer buffer
        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=total_size,
            props=PropertiesID(0),
        )
        var transfer_buf = create_gpu_transfer_buffer(
            self.device, Ptr(to=tb_info)
        )

        # Map and copy data
        var mapped = map_gpu_transfer_buffer(self.device, transfer_buf, False)
        var mapped_ptr = mapped.bitcast[UInt8]()

        # Copy vertices
        memcpy(
            dest=mapped_ptr,
            src=UnsafePointer(to=mesh_data.vertices[0]).bitcast[UInt8](),
            count=Int(vb_size),
        )
        # Copy indices after vertices
        memcpy(
            dest=mapped_ptr + Int(vb_size),
            src=UnsafePointer(to=mesh_data.indices[0]).bitcast[UInt8](),
            count=Int(ib_size),
        )

        unmap_gpu_transfer_buffer(self.device, transfer_buf)

        # Create GPU buffers
        var vb_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_VERTEX,
            size=vb_size,
            props=PropertiesID(0),
        )
        var vertex_buffer = create_gpu_buffer(self.device, Ptr(to=vb_info))

        var ib_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_INDEX,
            size=ib_size,
            props=PropertiesID(0),
        )
        var index_buffer = create_gpu_buffer(self.device, Ptr(to=ib_info))

        # Upload via copy pass
        var cmd_buf = acquire_gpu_command_buffer(self.device)
        var copy_pass = begin_gpu_copy_pass(cmd_buf)

        var vb_src = GPUTransferBufferLocation(
            transfer_buffer=transfer_buf, offset=0
        )
        var vb_dst = GPUBufferRegion(
            buffer=vertex_buffer, offset=0, size=vb_size
        )
        upload_to_gpu_buffer(copy_pass, Ptr(to=vb_src), Ptr(to=vb_dst), False)

        var ib_src = GPUTransferBufferLocation(
            transfer_buffer=transfer_buf, offset=vb_size
        )
        var ib_dst = GPUBufferRegion(
            buffer=index_buffer, offset=0, size=ib_size
        )
        upload_to_gpu_buffer(copy_pass, Ptr(to=ib_src), Ptr(to=ib_dst), False)

        end_gpu_copy_pass(copy_pass)
        submit_gpu_command_buffer(cmd_buf)

        # Release transfer buffer
        release_gpu_transfer_buffer(self.device, transfer_buf)

        return MeshHandle(
            vertex_buffer,
            index_buffer,
            UInt32(len(mesh_data.indices)),
            UInt32(len(mesh_data.vertices)),
        )

    fn _upload_static_meshes(mut self) raises:
        """Generate and upload sphere, box, and ground meshes."""
        var sphere_data = generate_sphere(16, 12)
        self.sphere_mesh = self._upload_mesh(sphere_data)

        var box_data = generate_box()
        self.box_mesh = self._upload_mesh(box_data)

        var ground_data = generate_ground(12.0)
        self.ground_mesh = self._upload_mesh(ground_data)

    fn _create_line_buffers(mut self) raises:
        """Allocate GPU and transfer buffers for dynamic line rendering."""
        var line_buf_size = UInt32(
            MAX_LINE_VERTICES * 12
        )  # 12 bytes per vertex (float3)

        var vb_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_VERTEX,
            size=line_buf_size,
            props=PropertiesID(0),
        )
        self.line_vertex_buffer = create_gpu_buffer(
            self.device, Ptr(to=vb_info)
        )

        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=line_buf_size,
            props=PropertiesID(0),
        )
        self.line_transfer_buffer = create_gpu_transfer_buffer(
            self.device, Ptr(to=tb_info)
        )

    # --- Public Drawing API ---

    fn begin_frame(mut self):
        """Begin a new frame: clear draw command lists."""
        self.solid_draws.clear()
        self.line_vertex_data.clear()
        self.line_colors.clear()
        self.has_ground = False

    fn draw_sphere(
        mut self,
        center: Vec3,
        radius: Float64,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
    ):
        """Draw a solid sphere.

        Args:
            center: Sphere center in world space.
            radius: Sphere radius.
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
        """
        var model = Mat4.compose(
            center, Quat.identity(), Vec3(radius, radius, radius)
        )
        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = reflectance
        uniforms.material[3] = emission

        self.solid_draws.append(SolidDrawCommand(0, uniforms))

    fn draw_capsule(
        mut self,
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
    ) raises:
        """Draw a solid capsule.

        Args:
            center: Capsule center in world space.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical section.
            axis: Local axis (0=X, 1=Y, 2=Z).
            color: Surface color.
        """
        var f_radius = Float32(radius)
        var f_half = Float32(half_height)

        # Look up or create capsule mesh
        var cache_idx = -1
        for i in range(len(self.capsule_cache)):
            if self.capsule_cache[i].matches(f_radius, f_half):
                cache_idx = i
                break

        if cache_idx < 0:
            # Generate new capsule mesh
            var mesh_data = generate_capsule(f_radius, f_half)
            var handle = self._upload_mesh(mesh_data)
            self.capsule_cache.append(
                CapsuleCacheEntry(f_radius, f_half, handle^)
            )
            cache_idx = len(self.capsule_cache) - 1

        # Build model matrix
        # Capsule geometry is along Z-axis. Apply pre-rotation if axis != 2.
        var pre_rot = Quat.identity()
        if axis == 0:
            # Rotate Z -> X: 90 degrees around Y
            pre_rot = Quat.from_axis_angle(Vec3.unit_y(), 1.5707963267949)
        elif axis == 1:
            # Rotate Z -> Y: -90 degrees around X
            pre_rot = Quat.from_axis_angle(Vec3.unit_x(), -1.5707963267949)

        var final_quat = orientation
        # Apply pre-rotation: rotate local capsule geometry, then apply orientation
        var model = Mat4.from_quat(final_quat, center) @ Mat4.from_quat(
            pre_rot, Vec3.zero()
        )

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = reflectance
        uniforms.material[3] = emission

        self.solid_draws.append(
            SolidDrawCommand(
                0, uniforms, is_capsule=True, capsule_cache_idx=cache_idx
            )
        )

    fn draw_box(
        mut self,
        center: Vec3,
        orientation: Quat,
        half_extents: Vec3,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
    ):
        """Draw a solid box.

        Args:
            center: Box center in world space.
            orientation: Box orientation.
            half_extents: Half-extents along local X, Y, Z.
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
        """
        # Unit box is [-0.5, 0.5], so scale by 2 * half_extents
        var scale = Vec3(
            half_extents.x * 2.0,
            half_extents.y * 2.0,
            half_extents.z * 2.0,
        )
        var model = Mat4.compose(center, orientation, scale)

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = reflectance
        uniforms.material[3] = emission

        self.solid_draws.append(SolidDrawCommand(1, uniforms))

    fn set_skybox(
        mut self,
        top_r: Float32 = 0.8,
        top_g: Float32 = 0.85,
        top_b: Float32 = 0.95,
        bottom_r: Float32 = 0.3,
        bottom_g: Float32 = 0.35,
        bottom_b: Float32 = 0.5,
    ):
        """Enable skybox gradient background.

        Args:
            top_r/g/b: Gradient top color (RGB, 0-1).
            bottom_r/g/b: Gradient bottom color (RGB, 0-1).
        """
        self.draw_skybox = True
        self.skybox_uniforms.top_color[0] = top_r
        self.skybox_uniforms.top_color[1] = top_g
        self.skybox_uniforms.top_color[2] = top_b
        self.skybox_uniforms.top_color[3] = 1.0
        self.skybox_uniforms.bottom_color[0] = bottom_r
        self.skybox_uniforms.bottom_color[1] = bottom_g
        self.skybox_uniforms.bottom_color[2] = bottom_b
        self.skybox_uniforms.bottom_color[3] = 1.0

    fn set_ground_checker_colors(
        mut self,
        r: Float32 = 0.22,
        g: Float32 = 0.22,
        b: Float32 = 0.25,
    ):
        """Set ground checker secondary color (stored in scene ground_params.xyz).

        Args:
            r/g/b: Checker dark tile color (RGB, 0-1). Light tile is 1.6x brighter.
        """
        self.scene_uniforms.ground_params[0] = r
        self.scene_uniforms.ground_params[1] = g
        self.scene_uniforms.ground_params[2] = b

    fn draw_ground_grid(
        mut self,
        center_x: Float64 = 0.0,
        size: Float64 = 10.0,
        height: Float64 = 0.0,
    ):
        """Draw the ground plane with procedural checkerboard.

        Args:
            center_x: X-coordinate to center the ground on (for scrolling envs).
            size: Unused (ground mesh is pre-sized).
            height: Z-coordinate of the ground plane.
        """
        var model = Mat4.from_translation(Vec3(center_x, 0.0, height))
        self.ground_uniforms = ObjectUniforms()
        self.ground_uniforms.model = mat4_to_gpu_f32(model)
        self.ground_uniforms.color = color_to_vec4(255, 255, 255)
        self.has_ground = True
        self.ground_z = height

    fn draw_coordinate_axes(
        mut self,
        origin: Vec3 = Vec3.zero(),
        length: Float64 = 1.0,
    ):
        """Draw coordinate axes: X=red, Y=green, Z=blue.

        Args:
            origin: Origin point.
            length: Length of each axis.
        """
        # X axis - red
        self._add_line(
            origin,
            origin + Vec3(length, 0.0, 0.0),
            color_to_vec4(255, 50, 50),
        )
        # Y axis - green
        self._add_line(
            origin,
            origin + Vec3(0.0, length, 0.0),
            color_to_vec4(50, 255, 50),
        )
        # Z axis - blue
        self._add_line(
            origin,
            origin + Vec3(0.0, 0.0, length),
            color_to_vec4(80, 80, 255),
        )

    fn draw_line_3d(
        mut self,
        start: Vec3,
        end: Vec3,
        color: Color,
    ):
        """Draw a 3D line segment.

        Args:
            start: Start point in world space.
            end: End point in world space.
            color: Line color.
        """
        self._add_line(
            start,
            end,
            color_to_vec4(color),
        )

    fn _add_line(
        mut self,
        start: Vec3,
        end: Vec3,
        color: InlineArray[Float32, 4],
    ):
        """Add a line segment to the line accumulator."""
        if len(self.line_vertex_data) + 6 > MAX_LINE_VERTICES * 3:
            return  # Buffer full

        self.line_vertex_data.append(Float32(start.x))
        self.line_vertex_data.append(Float32(start.y))
        self.line_vertex_data.append(Float32(start.z))

        self.line_vertex_data.append(Float32(end.x))
        self.line_vertex_data.append(Float32(end.y))
        self.line_vertex_data.append(Float32(end.z))

        self.line_colors.append(LineColorEntry(color))

    fn render_scene(mut self):
        """Render default scene elements (grid and axes)."""
        if self.draw_grid:
            self.draw_ground_grid()

        if self.draw_axes:
            self.draw_coordinate_axes()

    fn _select_and_draw(
        self,
        render_pass: Ptr[GPURenderPass, AnyOrigin[True]],
        draw: SolidDrawCommand,
    ) raises:
        """Select mesh buffers for a draw command, bind, and draw."""
        var vb: Ptr[GPUBuffer, AnyOrigin[True]]
        var ib: Ptr[GPUBuffer, AnyOrigin[True]]
        var n_idx: UInt32

        if draw.is_capsule:
            var ci = draw.capsule_cache_idx
            vb = self.capsule_cache[ci].mesh.vertex_buffer
            ib = self.capsule_cache[ci].mesh.index_buffer
            n_idx = self.capsule_cache[ci].mesh.num_indices
        elif draw.mesh_idx == 0:
            vb = self.sphere_mesh.vertex_buffer
            ib = self.sphere_mesh.index_buffer
            n_idx = self.sphere_mesh.num_indices
        else:
            vb = self.box_mesh.vertex_buffer
            ib = self.box_mesh.index_buffer
            n_idx = self.box_mesh.num_indices

        var vb_binding = GPUBufferBinding(buffer=vb, offset=0)
        bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=vb_binding), 1)
        var ib_binding = GPUBufferBinding(buffer=ib, offset=0)
        bind_gpu_index_buffer(
            render_pass,
            Ptr(to=ib_binding),
            GPUIndexElementSize.GPU_INDEXELEMENTSIZE_16BIT,
        )
        draw_gpu_indexed_primitives(render_pass, n_idx, 1, 0, 0, 0)

    fn end_frame(mut self) raises:
        """End frame: shadow pass, then main pass with reflections, ground, solids, lines.
        """
        # Acquire command buffer
        var cmd_buf = acquire_gpu_command_buffer(self.device)

        # Upload line data if any
        var num_line_verts = len(self.line_vertex_data) // 3
        if num_line_verts > 0:
            var mapped = map_gpu_transfer_buffer(
                self.device, self.line_transfer_buffer, True
            )
            var mapped_f32 = mapped.bitcast[Float32]()
            for i in range(len(self.line_vertex_data)):
                (mapped_f32 + i)[] = self.line_vertex_data[i]
            unmap_gpu_transfer_buffer(self.device, self.line_transfer_buffer)

            var copy_pass = begin_gpu_copy_pass(cmd_buf)
            var src = GPUTransferBufferLocation(
                transfer_buffer=self.line_transfer_buffer, offset=0
            )
            var dst = GPUBufferRegion(
                buffer=self.line_vertex_buffer,
                offset=0,
                size=UInt32(len(self.line_vertex_data) * 4),
            )
            upload_to_gpu_buffer(copy_pass, Ptr(to=src), Ptr(to=dst), False)
            end_gpu_copy_pass(copy_pass)

        # Build scene uniforms and shadow uniforms
        self._build_scene_uniforms()
        self._build_light_view_proj()

        # Store ground_z in scene_uniforms ground_params.w for reflection clipping
        self.scene_uniforms.ground_params[3] = Float32(self.ground_z)

        # ====================================================================
        # SHADOW PASS (depth-only, from light POV)
        # ====================================================================
        if len(self.solid_draws) > 0:
            var shadow_depth_info = GPUDepthStencilTargetInfo(
                texture=self.shadow_map,
                clear_depth=1.0,
                load_op=GPULoadOp.GPU_LOADOP_CLEAR,
                store_op=GPUStoreOp.GPU_STOREOP_STORE,
                stencil_load_op=GPULoadOp.GPU_LOADOP_DONT_CARE,
                stencil_store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
                cycle=True,
                clear_stencil=0,
                padding1=0,
                padding2=0,
            )

            var shadow_pass = begin_gpu_render_pass(
                cmd_buf,
                Ptr[GPUColorTargetInfo, AnyOrigin[False]](),  # No color targets
                0,
                Ptr(to=shadow_depth_info),
            )

            var shadow_viewport = GPUViewport(
                x=0.0,
                y=0.0,
                w=1024.0,
                h=1024.0,
                min_depth=0.0,
                max_depth=1.0,
            )
            set_gpu_viewport(shadow_pass, Ptr(to=shadow_viewport))

            bind_gpu_graphics_pipeline(shadow_pass, self.shadow_pipeline)

            # Push light VP as SceneUniforms (reuse struct layout)
            var light_scene = SceneUniforms()
            light_scene.view_proj = self.shadow_uniforms.light_view_proj.copy()
            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=light_scene).bitcast[NoneType](),
                224,
            )

            for i in range(len(self.solid_draws)):
                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    1,
                    Ptr(to=self.solid_draws[i].uniforms).bitcast[NoneType](),
                    96,
                )
                self._select_and_draw(shadow_pass, self.solid_draws[i])

            end_gpu_render_pass(shadow_pass)

        # ====================================================================
        # Acquire swapchain texture
        # ====================================================================
        var swapchain_tex = Ptr[GPUTexture, AnyOrigin[True]]()
        var sc_w = UInt32(0)
        var sc_h = UInt32(0)
        wait_and_acquire_gpu_swapchain_texture(
            cmd_buf,
            self.window,
            Ptr(to=swapchain_tex),
            Ptr(to=sc_w),
            Ptr(to=sc_h),
        )

        if not swapchain_tex:
            submit_gpu_command_buffer(cmd_buf)
            return

        # Handle resize
        if Int(sc_w) != self.width or Int(sc_h) != self.height:
            self.width = Int(sc_w)
            self.height = Int(sc_h)
            self.camera.set_screen_size(self.width, self.height)
            release_gpu_texture(self.device, self.depth_texture)
            self._create_depth_texture()

        # ====================================================================
        # MAIN RENDER PASS
        # ====================================================================
        var bg_r = Float32(self.background_color.r) / 255.0
        var bg_g = Float32(self.background_color.g) / 255.0
        var bg_b = Float32(self.background_color.b) / 255.0

        var color_info = GPUColorTargetInfo(
            texture=swapchain_tex,
            mip_level=0,
            layer_or_depth_plane=0,
            clear_color=FColor(bg_r, bg_g, bg_b, 1.0),
            load_op=GPULoadOp.GPU_LOADOP_CLEAR,
            store_op=GPUStoreOp.GPU_STOREOP_STORE,
            resolve_texture=Ptr[GPUTexture, AnyOrigin[True]](),
            resolve_mip_level=0,
            resolve_layer=0,
            cycle=True,
            cycle_resolve_texture=False,
            padding1=0,
            padding2=0,
        )

        var depth_info = GPUDepthStencilTargetInfo(
            texture=self.depth_texture,
            clear_depth=1.0,
            load_op=GPULoadOp.GPU_LOADOP_CLEAR,
            store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
            stencil_load_op=GPULoadOp.GPU_LOADOP_DONT_CARE,
            stencil_store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
            cycle=True,
            clear_stencil=0,
            padding1=0,
            padding2=0,
        )

        var render_pass = begin_gpu_render_pass(
            cmd_buf, Ptr(to=color_info), 1, Ptr(to=depth_info)
        )

        var viewport = GPUViewport(
            x=0.0,
            y=0.0,
            w=c_float(sc_w),
            h=c_float(sc_h),
            min_depth=0.0,
            max_depth=1.0,
        )
        set_gpu_viewport(render_pass, Ptr(to=viewport))

        # Shadow map texture+sampler binding (reused by solid and ground passes)
        var shadow_binding = GPUTextureSamplerBinding(
            texture=self.shadow_map,
            sampler=self.shadow_sampler,
        )

        # ------------------------------------------------------------------
        # Phase 0: SKYBOX (fullscreen gradient, drawn first)
        # ------------------------------------------------------------------
        if self.draw_skybox:
            bind_gpu_graphics_pipeline(render_pass, self.skybox_pipeline)
            push_gpu_fragment_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.skybox_uniforms).bitcast[NoneType](),
                32,
            )
            # Draw fullscreen triangle (3 vertices, no vertex buffer)
            draw_gpu_primitives(render_pass, 3, 1, 0, 0)

        # ------------------------------------------------------------------
        # Phase A: REFLECTIONS (Z-flipped solid objects, semi-transparent)
        # ------------------------------------------------------------------
        if self.has_ground and len(self.solid_draws) > 0:
            bind_gpu_graphics_pipeline(render_pass, self.reflection_pipeline)

            # Push scene uniforms (fragment slot 0 for reflection clipping)
            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).bitcast[NoneType](),
                224,
            )
            push_gpu_fragment_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).bitcast[NoneType](),
                224,
            )

            for i in range(len(self.solid_draws)):
                # Build mirrored model matrix: flip Z around ground_z
                var mirrored_uniforms = self.solid_draws[i].uniforms
                # M_reflected = T(0,0,2*gz) * S(1,1,-1) * M
                # This negates row 2 of the model matrix (m20,m21,m22,m23)
                # In column-major storage: m20=[2], m21=[6], m22=[10], m23=[14]
                var gz = Float32(self.ground_z)
                mirrored_uniforms.model[2] = -mirrored_uniforms.model[2]  # m20
                mirrored_uniforms.model[6] = -mirrored_uniforms.model[6]  # m21
                mirrored_uniforms.model[10] = -mirrored_uniforms.model[
                    10
                ]  # m22
                mirrored_uniforms.model[14] = (
                    -mirrored_uniforms.model[14] + 2.0 * gz
                )  # m23

                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    1,
                    Ptr(to=mirrored_uniforms).bitcast[NoneType](),
                    96,
                )
                self._select_and_draw(render_pass, self.solid_draws[i])

        # ------------------------------------------------------------------
        # Phase B: GROUND (checkerboard with shadow map sampling)
        # ------------------------------------------------------------------
        if self.has_ground:
            bind_gpu_graphics_pipeline(render_pass, self.ground_pipeline)

            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).bitcast[NoneType](),
                224,
            )
            push_gpu_fragment_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).bitcast[NoneType](),
                224,
            )
            # Push shadow uniforms to fragment slot 1
            push_gpu_fragment_uniform_data(
                cmd_buf,
                1,
                Ptr(to=self.shadow_uniforms).bitcast[NoneType](),
                80,
            )
            push_gpu_vertex_uniform_data(
                cmd_buf,
                1,
                Ptr(to=self.ground_uniforms).bitcast[NoneType](),
                96,
            )

            # Bind shadow map + sampler to fragment sampler slot 0
            bind_gpu_fragment_samplers(
                render_pass, 0, Ptr(to=shadow_binding), 1
            )

            var gvb = GPUBufferBinding(
                buffer=self.ground_mesh.vertex_buffer, offset=0
            )
            bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=gvb), 1)

            var gib = GPUBufferBinding(
                buffer=self.ground_mesh.index_buffer, offset=0
            )
            bind_gpu_index_buffer(
                render_pass,
                Ptr(to=gib),
                GPUIndexElementSize.GPU_INDEXELEMENTSIZE_16BIT,
            )

            draw_gpu_indexed_primitives(
                render_pass,
                self.ground_mesh.num_indices,
                1,
                0,
                0,
                0,
            )

        # ------------------------------------------------------------------
        # Phase C: SOLID OBJECTS (with shadow map sampling)
        # ------------------------------------------------------------------
        if len(self.solid_draws) > 0:
            bind_gpu_graphics_pipeline(render_pass, self.solid_pipeline)

            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).bitcast[NoneType](),
                224,
            )
            push_gpu_fragment_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).bitcast[NoneType](),
                224,
            )
            # Push shadow uniforms to fragment slot 1
            push_gpu_fragment_uniform_data(
                cmd_buf,
                1,
                Ptr(to=self.shadow_uniforms).bitcast[NoneType](),
                80,
            )

            # Bind shadow map + sampler
            bind_gpu_fragment_samplers(
                render_pass, 0, Ptr(to=shadow_binding), 1
            )

            for i in range(len(self.solid_draws)):
                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    1,
                    Ptr(to=self.solid_draws[i].uniforms).bitcast[NoneType](),
                    96,
                )
                self._select_and_draw(render_pass, self.solid_draws[i])

        # ------------------------------------------------------------------
        # Phase D: LINES (unchanged)
        # ------------------------------------------------------------------
        if num_line_verts >= 2:
            bind_gpu_graphics_pipeline(render_pass, self.line_pipeline)

            var line_offset = 0
            for seg_idx in range(len(self.line_colors)):
                var lu = LineUniforms()
                lu.view_proj = self.scene_uniforms.view_proj.copy()
                lu.color = self.line_colors[seg_idx].to_inline_array()

                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    0,
                    Ptr(to=lu).bitcast[NoneType](),
                    80,
                )
                push_gpu_fragment_uniform_data(
                    cmd_buf,
                    0,
                    Ptr(to=lu).bitcast[NoneType](),
                    80,
                )

                var lb = GPUBufferBinding(
                    buffer=self.line_vertex_buffer,
                    offset=UInt32(line_offset * 12),
                )
                bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=lb), 1)

                draw_gpu_primitives(render_pass, 2, 1, 0, 0)

                line_offset += 2

        # End render pass and submit
        end_gpu_render_pass(render_pass)
        submit_gpu_command_buffer(cmd_buf)

    fn _build_scene_uniforms(mut self):
        """Build scene uniforms from current camera state."""
        var view = self.camera.get_view_matrix()
        var proj = perspective_metal(
            self.camera.fov,
            self.camera.aspect,
            self.camera.near,
            self.camera.far,
        )
        var view_proj = proj @ view

        self.scene_uniforms.view_proj = mat4_to_gpu_f32(view_proj)

        # Camera position + num_active_lights in w
        var num_lights = len(self.lights)
        if num_lights < 1:
            num_lights = 1
        if num_lights > 4:
            num_lights = 4
        self.scene_uniforms.camera_pos[0] = Float32(self.camera.eye.x)
        self.scene_uniforms.camera_pos[1] = Float32(self.camera.eye.y)
        self.scene_uniforms.camera_pos[2] = Float32(self.camera.eye.z)
        self.scene_uniforms.camera_pos[3] = Float32(num_lights)

        # Fill light slots from self.lights (up to 4)
        for li in range(num_lights):
            var light = self.lights[li].copy()
            var lx = Float32(light.dir_x)
            var ly = Float32(light.dir_y)
            var lz = Float32(light.dir_z)
            var ll = sqrt(lx * lx + ly * ly + lz * lz)
            if ll < 1e-6:
                ll = 1.0

            if li == 0:
                self.scene_uniforms.light0_dir[0] = lx / ll
                self.scene_uniforms.light0_dir[1] = ly / ll
                self.scene_uniforms.light0_dir[2] = lz / ll
                self.scene_uniforms.light0_dir[3] = Float32(light.ambient)
                self.scene_uniforms.light0_color[0] = Float32(light.color_r)
                self.scene_uniforms.light0_color[1] = Float32(light.color_g)
                self.scene_uniforms.light0_color[2] = Float32(light.color_b)
                self.scene_uniforms.light0_color[3] = Float32(
                    1.0 if light.cast_shadow else 0.0
                )
            elif li == 1:
                self.scene_uniforms.light1_dir[0] = lx / ll
                self.scene_uniforms.light1_dir[1] = ly / ll
                self.scene_uniforms.light1_dir[2] = lz / ll
                self.scene_uniforms.light1_dir[3] = Float32(light.ambient)
                self.scene_uniforms.light1_color[0] = Float32(light.color_r)
                self.scene_uniforms.light1_color[1] = Float32(light.color_g)
                self.scene_uniforms.light1_color[2] = Float32(light.color_b)
                self.scene_uniforms.light1_color[3] = Float32(
                    1.0 if light.cast_shadow else 0.0
                )
            elif li == 2:
                self.scene_uniforms.light2_dir[0] = lx / ll
                self.scene_uniforms.light2_dir[1] = ly / ll
                self.scene_uniforms.light2_dir[2] = lz / ll
                self.scene_uniforms.light2_dir[3] = Float32(light.ambient)
                self.scene_uniforms.light2_color[0] = Float32(light.color_r)
                self.scene_uniforms.light2_color[1] = Float32(light.color_g)
                self.scene_uniforms.light2_color[2] = Float32(light.color_b)
                self.scene_uniforms.light2_color[3] = Float32(
                    1.0 if light.cast_shadow else 0.0
                )
            elif li == 3:
                self.scene_uniforms.light3_dir[0] = lx / ll
                self.scene_uniforms.light3_dir[1] = ly / ll
                self.scene_uniforms.light3_dir[2] = lz / ll
                self.scene_uniforms.light3_dir[3] = Float32(light.ambient)
                self.scene_uniforms.light3_color[0] = Float32(light.color_r)
                self.scene_uniforms.light3_color[1] = Float32(light.color_g)
                self.scene_uniforms.light3_color[2] = Float32(light.color_b)
                self.scene_uniforms.light3_color[3] = Float32(
                    1.0 if light.cast_shadow else 0.0
                )

    fn _build_light_view_proj(mut self):
        """Build light's orthographic view-projection matrix for shadow mapping.

        Uses the first shadow-casting light, or light 0 as fallback.
        """
        # Find first shadow-casting light direction
        var light_dir = Vec3(
            Float64(self.scene_uniforms.light0_dir[0]),
            Float64(self.scene_uniforms.light0_dir[1]),
            Float64(self.scene_uniforms.light0_dir[2]),
        )
        for li in range(len(self.lights)):
            if self.lights[li].cast_shadow:
                var lx = self.lights[li].dir_x
                var ly = self.lights[li].dir_y
                var lz = self.lights[li].dir_z
                var ll = sqrt(lx * lx + ly * ly + lz * lz)
                if ll < 1e-6:
                    ll = 1.0
                light_dir = Vec3(lx / ll, ly / ll, lz / ll)
                break

        # Light position: offset from camera target along negative light direction
        var target = self.camera.target
        var light_distance = 15.0
        var light_pos = target - light_dir * light_distance

        # Build look-at view matrix for the light using Mat4.look_at
        # which correctly handles the sign conventions (Z-row = -forward)
        var up = Vec3(0.0, 0.0, 1.0)
        # If light is nearly vertical, use a different up vector
        var abs_dot = abs(
            light_dir.x * up.x + light_dir.y * up.y + light_dir.z * up.z
        )
        if abs_dot > 0.99:
            up = Vec3(0.0, 1.0, 0.0)

        var light_view = Mat4.look_at(light_pos, target, up)

        # Orthographic projection covering the scene
        var ortho_size = 8.0
        var light_proj = ortho_metal(
            -ortho_size,
            ortho_size,
            -ortho_size,
            ortho_size,
            0.1,
            30.0,
        )

        var light_vp = light_proj @ light_view
        self.shadow_uniforms.light_view_proj = mat4_to_gpu_f32(light_vp)

    # --- Event Handling ---

    fn check_quit(mut self) -> Bool:
        """Check if user wants to quit and detect camera switch keys.

        Polls SDL events and returns True if quit event detected
        (window close or Escape key). Number keys 1-9 set
        camera_switch_request for ModelRenderer to pick up.

        Returns:
            True if quit event detected.
        """
        self.camera_switch_request = -1
        var event = Event()

        try:
            while poll_event(Ptr(to=event)):
                var event_type = event[UInt32]
                if EventType(Int(event_type)) == EventType.EVENT_QUIT:
                    self.should_quit = True
                    return True
                elif EventType(Int(event_type)) == EventType.EVENT_KEY_DOWN:
                    var key_event = event[KeyboardEvent]
                    var key_val = Int(key_event.key)
                    if key_val == Int(Keycode.SDLK_ESCAPE):
                        self.should_quit = True
                        return True
                    # Number keys 1-9 for camera switching
                    # SDLK_1=0x31 ... SDLK_9=0x39
                    elif key_val >= 0x31 and key_val <= 0x39:
                        self.camera_switch_request = key_val - 0x31  # 0-8
        except:
            pass

        return self.should_quit

    # --- Camera Controls ---

    fn set_camera_position(mut self, eye: Vec3, target: Vec3):
        """Set camera position and target.

        Args:
            eye: Camera position.
            target: Look-at target.
        """
        self.camera.eye = eye
        self.camera.target = target

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64):
        """Orbit camera around target.

        Args:
            delta_theta: Horizontal rotation (radians).
            delta_phi: Vertical rotation (radians).
        """
        self.camera.orbit(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64):
        """Zoom camera in/out.

        Args:
            delta: Zoom amount.
        """
        self.camera.zoom(delta)

    fn delay_ms(self, ms: Int) raises:
        """Delay for given milliseconds.

        Args:
            ms: Milliseconds to delay.
        """
        delay(UInt32(ms))

    # --- Cleanup ---

    fn close(mut self) raises:
        """Release all GPU resources and shutdown SDL3."""
        if not self.initialized:
            return

        # Release capsule cache meshes
        for i in range(len(self.capsule_cache)):
            release_gpu_buffer(
                self.device, self.capsule_cache[i].mesh.vertex_buffer
            )
            release_gpu_buffer(
                self.device, self.capsule_cache[i].mesh.index_buffer
            )

        # Release static mesh buffers
        release_gpu_buffer(self.device, self.sphere_mesh.vertex_buffer)
        release_gpu_buffer(self.device, self.sphere_mesh.index_buffer)
        release_gpu_buffer(self.device, self.box_mesh.vertex_buffer)
        release_gpu_buffer(self.device, self.box_mesh.index_buffer)
        release_gpu_buffer(self.device, self.ground_mesh.vertex_buffer)
        release_gpu_buffer(self.device, self.ground_mesh.index_buffer)

        # Release line buffers
        release_gpu_buffer(self.device, self.line_vertex_buffer)
        release_gpu_transfer_buffer(self.device, self.line_transfer_buffer)

        # Release depth texture
        release_gpu_texture(self.device, self.depth_texture)

        # Release shadow resources
        release_gpu_texture(self.device, self.shadow_map)
        release_gpu_sampler(self.device, self.shadow_sampler)

        # Release pipelines
        release_gpu_graphics_pipeline(self.device, self.solid_pipeline)
        release_gpu_graphics_pipeline(self.device, self.ground_pipeline)
        release_gpu_graphics_pipeline(self.device, self.line_pipeline)
        release_gpu_graphics_pipeline(self.device, self.shadow_pipeline)
        release_gpu_graphics_pipeline(self.device, self.reflection_pipeline)
        release_gpu_graphics_pipeline(self.device, self.skybox_pipeline)

        # Release window and device
        release_window_from_gpu_device(self.device, self.window)
        destroy_window(self.window)
        destroy_gpu_device(self.device)
        quit()

        self.initialized = False
