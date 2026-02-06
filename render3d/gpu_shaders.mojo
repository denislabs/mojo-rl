"""MSL Shader Source Strings for GPU 3D Renderer.

Three shader pairs as comptime string constants:
  1. Solid object shaders (Blinn-Phong lighting)
  2. Ground shaders (procedural checkerboard)
  3. Line shaders (flat color, no lighting)

SDL_GPU MSL binding convention:
  - [[buffer(0)]] = uniform slot 0
  - [[buffer(1)]] = uniform slot 1
  - Vertex buffers auto-bound at [[buffer(14+)]] by SDL_GPU
  - [[stage_in]] for vertex attributes from pipeline layout
"""


# --- Solid Object Shaders (Blinn-Phong) ---

comptime SOLID_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
    float4 obj_color;
};

struct SceneUniforms {
    float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;  // w = ambient
    float4 padding;
};

struct ObjectUniforms {
    float4x4 model;
    float4 color;
};

vertex VertexOut solid_vertex(
    VertexIn in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]],
    constant ObjectUniforms &obj [[buffer(1)]]
) {
    VertexOut out;
    float4 world = obj.model * float4(in.position, 1.0);
    out.position = scene.view_proj * world;
    out.world_pos = world.xyz;
    // Transform normal by upper 3x3 of model matrix
    out.world_normal = (obj.model * float4(in.normal, 0.0)).xyz;
    out.obj_color = obj.color;
    return out;
}
"""

comptime SOLID_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
    float4 obj_color;
};

struct SceneUniforms {
    float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;  // w = ambient
    float4 padding;
};

fragment float4 solid_fragment(
    VertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]]
) {
    float3 N = normalize(in.world_normal);
    float3 L = normalize(-scene.light_dir.xyz);
    float3 V = normalize(scene.camera_pos.xyz - in.world_pos);
    float3 H = normalize(L + V);

    float ambient = scene.light_color.w;
    float diffuse = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 32.0) * 0.3;

    float3 light_col = scene.light_color.xyz;
    float3 color = in.obj_color.rgb * (ambient + diffuse * light_col) + specular * light_col;

    return float4(color, in.obj_color.a);
}
"""

# --- Ground Shaders (Procedural Checkerboard) ---

comptime GROUND_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
};

struct SceneUniforms {
    float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;
    float4 padding;
};

struct ObjectUniforms {
    float4x4 model;
    float4 color;
};

vertex VertexOut ground_vertex(
    VertexIn in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]],
    constant ObjectUniforms &obj [[buffer(1)]]
) {
    VertexOut out;
    float4 world = obj.model * float4(in.position, 1.0);
    out.position = scene.view_proj * world;
    out.world_pos = world.xyz;
    out.world_normal = float3(0.0, 0.0, 1.0);
    return out;
}
"""

comptime GROUND_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
};

struct SceneUniforms {
    float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;
    float4 padding;
};

fragment float4 ground_fragment(
    VertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]]
) {
    // Checkerboard pattern
    float tile_size = 1.0;
    float2 tile = floor(in.world_pos.xy / tile_size);
    float checker = fmod(tile.x + tile.y, 2.0);
    checker = abs(checker);

    float3 color1 = float3(0.35, 0.35, 0.38);  // Light tile
    float3 color2 = float3(0.22, 0.22, 0.25);  // Dark tile
    float3 base_color = mix(color1, color2, checker);

    // Subtle directional lighting
    float3 N = float3(0.0, 0.0, 1.0);
    float3 L = normalize(-scene.light_dir.xyz);
    float diffuse = max(dot(N, L), 0.0) * 0.3 + 0.7;
    base_color *= diffuse;

    // Distance fade for smooth ground edge
    float dist = length(in.world_pos.xy - scene.camera_pos.xy);
    float fade = 1.0 - smoothstep(8.0, 12.0, dist);

    return float4(base_color, fade);
}
"""

# --- Line Shaders (Flat Color) ---

comptime LINE_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
};

struct LineUniforms {
    float4x4 view_proj;
    float4 color;
};

vertex VertexOut line_vertex(
    VertexIn in [[stage_in]],
    constant LineUniforms &uniforms [[buffer(0)]]
) {
    VertexOut out;
    out.position = uniforms.view_proj * float4(in.position, 1.0);
    return out;
}
"""

comptime LINE_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

struct LineUniforms {
    float4x4 view_proj;
    float4 color;
};

fragment float4 line_fragment(
    constant LineUniforms &uniforms [[buffer(0)]]
) {
    return uniforms.color;
}
"""
