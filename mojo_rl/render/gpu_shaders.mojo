"""MSL Shader Source Strings for GPU 3D Renderer.

Seven shader pairs as comptime string constants:
  1. Solid object shaders (Blinn-Phong lighting + shadow sampling + per-object material)
  2. Ground shaders (procedural checkerboard + shadow sampling + material)
  3. Line shaders (flat color, no lighting)
  4. Shadow map shaders (depth-only pass from light POV)
  5. Reflection shaders (Z-flipped, darkened, semi-transparent)
  6. Skybox shaders (fullscreen vertical gradient)

SDL_GPU MSL binding convention:
  - [[buffer(0)]] = uniform slot 0
  - [[buffer(1)]] = uniform slot 1
  - Vertex buffers auto-bound at [[buffer(14+)]] by SDL_GPU
  - [[stage_in]] for vertex attributes from pipeline layout
  - [[texture(0)]], [[sampler(0)]] = fragment sampler slot 0
"""


# --- Shared SceneUniforms MSL struct definition (used in multiple shaders) ---
# 240B: view_proj(64) + camera_pos(16) + 4 lights*2 vec4(128) + ground_params(16) + fog_params(16)

comptime _SCENE_UNIFORMS_MSL = """
struct SceneUniforms {
    float4x4 view_proj;
    float4 camera_pos;      // w = num_active_lights
    float4 light0_dir;      // w = ambient0
    float4 light0_color;    // w = cast_shadow (0/1)
    float4 light1_dir;
    float4 light1_color;
    float4 light2_dir;
    float4 light2_color;
    float4 light3_dir;
    float4 light3_color;
    float4 ground_params;   // xyz = checker_color2, w = ground_z
    float4 fog_params;      // x = fogstart, y = fogend, z = 0, w = 0
};
"""

# Helper to get light dir/color by index in MSL
comptime _LIGHT_ACCESS_MSL = """
float4 get_light_dir(constant SceneUniforms &scene, int i) {
    if (i == 0) return scene.light0_dir;
    if (i == 1) return scene.light1_dir;
    if (i == 2) return scene.light2_dir;
    return scene.light3_dir;
}

float4 get_light_color(constant SceneUniforms &scene, int i) {
    if (i == 0) return scene.light0_color;
    if (i == 1) return scene.light1_color;
    if (i == 2) return scene.light2_color;
    return scene.light3_color;
}
"""


# --- Solid Object Shaders (Blinn-Phong + Shadows + Material) ---

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
    float2 uv;
    float4 obj_color;
    float4 obj_material;
};

""" + _SCENE_UNIFORMS_MSL + """

struct ObjectUniforms {
    float4x4 model;
    float4 color;
    float4 material;  // x=shininess, y=specular, z=reflectance (>0 = has texture), w=emission
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
    out.uv = in.uv;
    out.obj_color = obj.color;
    out.obj_material = obj.material;
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
    float2 uv;
    float4 obj_color;
    float4 obj_material;
};

""" + _SCENE_UNIFORMS_MSL + _LIGHT_ACCESS_MSL + """

struct ShadowUniforms {
    float4x4 light_view_proj;
    float4 params;  // x=shadow_intensity, y=bias
};

float compute_shadow(float3 world_pos,
                     constant ShadowUniforms &shadow,
                     depth2d<float> shadow_map,
                     sampler shadow_sampler) {
    float4 light_pos = shadow.light_view_proj * float4(world_pos, 1.0);
    float3 proj = light_pos.xyz / light_pos.w;

    // Map NDC [-1,1] XY to UV [0,1]
    float2 shadow_uv = proj.xy * 0.5 + 0.5;
    shadow_uv.y = 1.0 - shadow_uv.y;  // Metal Y-flip

    // Check if outside shadow map bounds
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
        return 1.0;  // Lit (outside shadow frustum)
    }

    float bias = shadow.params.y;
    float current_depth = proj.z - bias;

    // 3x3 PCF for soft shadows
    float shadow_val = 0.0;
    float texel_size = 1.0 / 4096.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 offset = float2(float(x), float(y)) * texel_size;
            shadow_val += shadow_map.sample_compare(shadow_sampler, shadow_uv + offset, current_depth);
        }
    }
    shadow_val /= 9.0;

    // Mix between full shadow and lit based on intensity
    float intensity = shadow.params.x;
    return 1.0 - intensity * (1.0 - shadow_val);
}

fragment float4 solid_fragment(
    VertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]],
    constant ShadowUniforms &shadow [[buffer(1)]],
    depth2d<float> shadow_map [[texture(0)]],
    sampler shadow_sampler [[sampler(0)]],
    texture2d<float> obj_texture [[texture(1)]],
    sampler obj_sampler [[sampler(1)]]
) {
    float3 N = normalize(in.world_normal);
    float3 V = normalize(scene.camera_pos.xyz - in.world_pos);

    // Per-object material properties
    float mat_shininess = in.obj_material.x;  // 0-1, maps to specular exponent
    float mat_specular = in.obj_material.y;    // 0-1, specular intensity
    float has_texture = in.obj_material.z;     // >0 = sample obj_texture
    float mat_emission = in.obj_material.w;    // 0-1, emissive intensity

    // Map shininess [0,1] to specular exponent: 0.0->4, 0.5->32, 1.0->128
    float spec_exp = mix(4.0, 128.0, mat_shininess);

    // Sample texture if enabled (material.z > 0)
    float4 base_color = in.obj_color;
    if (has_texture > 0.5) {
        float4 tex_color = obj_texture.sample(obj_sampler, in.uv);
        base_color = float4(base_color.rgb * tex_color.rgb, base_color.a * tex_color.a);
    }

    int num_lights = int(scene.camera_pos.w);
    if (num_lights < 1) num_lights = 1;
    if (num_lights > 4) num_lights = 4;

    float3 total_color = float3(0.0);
    float total_ambient = 0.0;

    for (int li = 0; li < num_lights; li++) {
        float4 l_dir = get_light_dir(scene, li);
        float4 l_color = get_light_color(scene, li);

        float3 L = normalize(-l_dir.xyz);
        float3 H = normalize(L + V);

        float ambient = l_dir.w;
        float diffuse = max(dot(N, L), 0.0);
        float specular = pow(max(dot(N, H), 0.0), spec_exp) * mat_specular;

        // Shadow only for first shadow-casting light
        float shadow_factor = 1.0;
        if (li == 0 && l_color.w > 0.5) {
            shadow_factor = compute_shadow(in.world_pos, shadow, shadow_map, shadow_sampler);
        }

        float3 light_col = l_color.xyz;
        total_color += base_color.rgb * shadow_factor * diffuse * light_col
                     + shadow_factor * specular * light_col;
        total_ambient += ambient;
    }

    // Clamp ambient to avoid over-brightening with multiple lights
    total_ambient = min(total_ambient, 1.0);

    float3 color = base_color.rgb * total_ambient + total_color
                 + base_color.rgb * mat_emission;

    // Linear fog: blend towards fog color (use skybox-like grey) based on distance
    float fog_start = scene.fog_params.x;
    float fog_end = scene.fog_params.y;
    if (fog_end > fog_start) {
        float dist = length(in.world_pos - scene.camera_pos.xyz);
        float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        float3 fog_color = float3(0.5, 0.495, 0.48);  // match typical skybox
        color = mix(color, fog_color, fog_factor);
    }

    return float4(color, base_color.a);
}
"""

# --- Ground Shaders (Procedural Checkerboard + Shadows) ---

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
    float2 uv;
};

""" + _SCENE_UNIFORMS_MSL + """

struct ObjectUniforms {
    float4x4 model;
    float4 color;
    float4 material;  // x=shininess, y=specular, z=reflectance, w=emission
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
    out.uv = in.uv;
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
    float2 uv;
};

""" + _SCENE_UNIFORMS_MSL + _LIGHT_ACCESS_MSL + """

struct ShadowUniforms {
    float4x4 light_view_proj;
    float4 params;  // x=shadow_intensity, y=bias
};

float compute_shadow_ground(float3 world_pos,
                            constant ShadowUniforms &shadow,
                            depth2d<float> shadow_map,
                            sampler shadow_sampler) {
    float4 light_pos = shadow.light_view_proj * float4(world_pos, 1.0);
    float3 proj = light_pos.xyz / light_pos.w;

    // Map NDC [-1,1] XY to UV [0,1]
    float2 shadow_uv = proj.xy * 0.5 + 0.5;
    shadow_uv.y = 1.0 - shadow_uv.y;  // Metal Y-flip

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
        return 1.0;
    }

    float bias = shadow.params.y;
    float current_depth = proj.z - bias;

    // 3x3 PCF
    float shadow_val = 0.0;
    float texel_size = 1.0 / 4096.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 offset = float2(float(x), float(y)) * texel_size;
            shadow_val += shadow_map.sample_compare(shadow_sampler, shadow_uv + offset, current_depth);
        }
    }
    shadow_val /= 9.0;

    float intensity = shadow.params.x;
    return 1.0 - intensity * (1.0 - shadow_val);
}

fragment float4 ground_fragment(
    VertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]],
    constant ShadowUniforms &shadow [[buffer(1)]],
    depth2d<float> shadow_map [[texture(0)]],
    sampler shadow_sampler [[sampler(0)]],
    texture2d<float> ground_texture [[texture(1)]],
    sampler ground_tex_sampler [[sampler(1)]]
) {
    // Ground color — three modes based on ground_params encoding:
    //   ground_params.z > 1.5: texture mode (xy = texrepeat), sample ground_texture
    //   ground_params.x < 0: solid color mode, color = abs(ground_params.xyz)
    //   else: checker mode, light tile = ground_params.xyz
    // Note: ground_params.w is reserved for ground_z (reflection clipping)
    float3 base_color;

    if (scene.ground_params.z > 1.5) {
        // Texture mode: tile the texture using world-space XY coordinates
        // ground_params.xy = texrepeat_u, texrepeat_v (tiles across ground extent)
        float tex_repeat_u = scene.ground_params.x;
        float tex_repeat_v = scene.ground_params.y;
        // Map UVs: use mesh UVs scaled by texrepeat
        float2 tex_uv = in.uv * float2(tex_repeat_u, tex_repeat_v);
        float4 tex_color = ground_texture.sample(ground_tex_sampler, tex_uv);
        base_color = tex_color.rgb;
    } else if (scene.ground_params.x < -0.001) {
        // Solid color mode (no texture defined in XML, use geom rgba)
        base_color = -scene.ground_params.xyz;
    } else {
        // Checkerboard pattern
        float3 checker_color1 = float3(0.35, 0.35, 0.38);  // Light tile (default)
        float3 checker_color2 = float3(0.22, 0.22, 0.25);  // Dark tile (default)

        // Use ground_params.xyz as light tile color (rgb2), dark tile = black (rgb1)
        // Matches MuJoCo checker: rgb1=(0,0,0) black, rgb2=(0.8,0.8,0.8) grey
        if (scene.ground_params.x > 0.001 || scene.ground_params.y > 0.001 || scene.ground_params.z > 0.001) {
            checker_color1 = scene.ground_params.xyz;  // Light tile = rgb2
            checker_color2 = float3(0.0, 0.0, 0.0);   // Dark tile = black (rgb1)
        }

        float tile_size = 1.0;
        float2 tile = floor(in.world_pos.xy / tile_size);
        float checker = fmod(tile.x + tile.y, 2.0);
        checker = abs(checker);

        base_color = mix(checker_color1, checker_color2, checker);
    }

    bool is_textured = (scene.ground_params.z > 1.5);

    // Apply shadow (first shadow-casting light)
    float shadow_factor = compute_shadow_ground(in.world_pos, shadow, shadow_map, shadow_sampler);

    if (is_textured) {
        // Textured ground: texture contains its own shading, only apply shadows
        base_color *= shadow_factor;
    } else {
        // Procedural ground: apply multi-light shading + shadows + fog
        int num_lights = int(scene.camera_pos.w);
        if (num_lights < 1) num_lights = 1;
        if (num_lights > 4) num_lights = 4;

        float3 N = float3(0.0, 0.0, 1.0);
        float lighting = 0.0;
        for (int li = 0; li < num_lights; li++) {
            float4 l_dir = get_light_dir(scene, li);
            float3 L = normalize(-l_dir.xyz);
            float diffuse = max(dot(N, L), 0.0) * 0.3 + 0.7 / float(num_lights);
            lighting += diffuse / float(num_lights);
        }
        base_color *= lighting;
        base_color *= shadow_factor;

        // Linear fog for procedural ground only
        float fog_start = scene.fog_params.x;
        float fog_end = scene.fog_params.y;
        if (fog_end > fog_start) {
            float fog_dist = length(in.world_pos - scene.camera_pos.xyz);
            float fog_factor = clamp((fog_dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
            float3 fog_color = float3(0.5, 0.495, 0.48);
            base_color = mix(base_color, fog_color, fog_factor);
        }
    }

    // Distance fade for a smooth ground edge. This one STAYS: without it the
    // finite ground quad ends in a hard line against the sky.
    float dist = length(in.world_pos.xy - scene.camera_pos.xy);
    float edge_fade = 1.0 - smoothstep(8.0, 12.0, dist);

    // ⚠ THE GROUND IS OPAQUE. It used to be alpha 0.55 (0.95 textured) so the
    // reflection pass, drawn UNDERNEATH it, would show through — and what
    // showed through was not only the reflection. Where no reflected geometry
    // existed, the remaining 45% was the SKYBOX, so the starfield was visible
    // THROUGH THE FLOOR. MuJoCo's floor is opaque and its `reflectance` blends
    // the reflection ON TOP; `render_frame` now does the same, so nothing here
    // needs to be see-through.
    return float4(base_color, edge_fade);
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

# --- Shadow Map Shaders (Depth-Only Pass) ---

comptime SHADOW_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
};

""" + _SCENE_UNIFORMS_MSL + """

struct ObjectUniforms {
    float4x4 model;
    float4 color;
    float4 material;
};

vertex VertexOut shadow_vertex(
    VertexIn in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]],
    constant ObjectUniforms &obj [[buffer(1)]]
) {
    VertexOut out;
    float4 world = obj.model * float4(in.position, 1.0);
    out.position = scene.view_proj * world;
    return out;
}
"""

comptime SHADOW_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

// Minimal fragment shader for depth-only pass
fragment void shadow_fragment() {
    // Depth is written automatically; no color output needed
}
"""

# --- Reflection Shaders (Z-Flipped, Darkened, Semi-Transparent) ---

comptime REFLECTION_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
    float4 obj_color;
    float4 obj_material;
};

""" + _SCENE_UNIFORMS_MSL + _LIGHT_ACCESS_MSL + """

fragment float4 reflection_fragment(
    VertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]]
) {
    // Discard fragments above the ground plane
    float ground_z = scene.ground_params.w;
    if (in.world_pos.z > ground_z + 0.001) {
        discard_fragment();
    }

    // Multi-light reflection shading
    float3 N = normalize(in.world_normal);
    float3 V = normalize(scene.camera_pos.xyz - in.world_pos);

    float mat_shininess = in.obj_material.x;
    float mat_specular = in.obj_material.y;
    float spec_exp = mix(4.0, 128.0, mat_shininess);

    int num_lights = int(scene.camera_pos.w);
    if (num_lights < 1) num_lights = 1;
    if (num_lights > 4) num_lights = 4;

    float3 total_color = float3(0.0);
    float total_ambient = 0.0;

    for (int li = 0; li < num_lights; li++) {
        float4 l_dir = get_light_dir(scene, li);
        float4 l_color = get_light_color(scene, li);

        float3 L = normalize(-l_dir.xyz);
        float3 H = normalize(L + V);

        float ambient = l_dir.w;
        float diffuse = max(dot(N, L), 0.0);
        float specular = pow(max(dot(N, H), 0.0), spec_exp) * mat_specular * 0.5;

        float3 light_col = l_color.xyz;
        total_color += in.obj_color.rgb * diffuse * light_col + specular * light_col;
        total_ambient += ambient;
    }

    total_ambient = min(total_ambient, 1.0);
    float3 color = in.obj_color.rgb * total_ambient + total_color;

    // ⚠ ALPHA IS THE REFLECTANCE, and it is the ONLY attenuation. The colour
    // used to be pre-darkened (`color *= 0.35`) as well as blended at 0.35,
    // which double-counted: MuJoCo's mirror term is
    // `floor*(1-reflectance) + reflected*reflectance`, one factor, not two.
    //
    // 0.2 is dm_control's own number — `<material name="grid" reflectance=".2">`
    // in `suite/common/materials.xml`, which every suite floor uses. It is a
    // constant here rather than a uniform because no model we ship differs; a
    // model that did would need it threaded through SceneUniforms.
    float alpha = 0.2;

    // Fade out near the edges of the ground. ⚠ LOAD-BEARING NOW THAT THIS PASS
    // RUNS WITH THE DEPTH TEST OFF (see `render_frame` Phase B2): nothing else
    // stops a reflection from painting onto the sky past the ground's rim. Its
    // 6→10 fade sits INSIDE the ground's 8→12, so the reflection is always gone
    // before the floor it is supposed to be lying on is.
    float dist = length(in.world_pos.xy - scene.camera_pos.xy);
    alpha *= 1.0 - smoothstep(6.0, 10.0, dist);

    return float4(color, alpha);
}
"""

# --- Skybox Shaders (Fullscreen Vertical Gradient) ---

comptime SKYBOX_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle: 3 vertices cover the entire screen
vertex VertexOut skybox_vertex(uint vid [[vertex_id]]) {
    VertexOut out;
    // Generate fullscreen triangle from vertex ID
    float2 pos = float2((vid << 1) & 2, vid & 2);
    out.position = float4(pos * 2.0 - 1.0, 0.999, 1.0);  // Near far plane
    out.uv = float2(pos.x, 1.0 - pos.y);  // UV: (0,0) bottom-left, (1,1) top-right
    return out;
}
"""

comptime SKYBOX_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct SkyboxUniforms {
    float4 top_color;     // Gradient top color (rgb + alpha)
    float4 bottom_color;  // Gradient bottom color (rgb + alpha)
    float4 mark_color;    // Starfield rgb, .w = density (0 disables)
    float4 cam_right;     // Camera right basis, .w = tan(fovy/2)
    float4 cam_up;        // Camera up basis,    .w = aspect
    float4 cam_fwd;       // Camera forward basis
};

// Cheap 3D value hash. Stars must be a pure function of DIRECTION so they sit
// still in the world while the camera moves; anything seeded by screen
// position would slide across the sky and look like a camera bug.
static inline float sky_hash(float3 p) {
    p = fract(p * 0.3183099 + float3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

fragment float4 skybox_fragment(
    VertexOut in [[stage_in]],
    constant SkyboxUniforms &sky [[buffer(0)]]
) {
    // ⚠ uv.y=0 is the TOP of the screen, not the bottom. `skybox_vertex`
    // writes `uv.y = 1.0 - pos.y` while Metal NDC has y=+1 at the top, so the
    // two cancel: uv.y runs 0 at the top to 1 at the bottom. This line used to
    // read `t = in.uv.y` against a comment claiming the opposite, which put
    // rgb2 at the zenith and rgb1 at the horizon — dm_control's sky
    // (rgb1=".4 .6 .8", rgb2="0 0 0") came out black overhead and blue at the
    // horizon, the exact inverse of MuJoCo, where rgb1 is the top.
    float t = 1.0 - in.uv.y;
    float3 color = mix(sky.bottom_color.rgb, sky.top_color.rgb, t);

    // MuJoCo's `mark="random"`: dots baked into the skybox texture, which over
    // a dark gradient is a starfield. Rebuild the world-space view ray from
    // the camera basis, then hash a coarse grid on the unit sphere so each
    // cell holds at most one star and every star stays put in the world.
    float density = sky.mark_color.w;
    if (density > 0.0) {
        float2 ndc = float2(in.uv.x, 1.0 - in.uv.y) * 2.0 - 1.0;
        float tan_h = sky.cam_right.w;
        float aspect = sky.cam_up.w;
        float3 dir = normalize(
            sky.cam_fwd.xyz
            + sky.cam_right.xyz * (ndc.x * tan_h * aspect)
            + sky.cam_up.xyz * (ndc.y * tan_h)
        );
        // 260 cells across the sphere's diameter: fine enough that stars read
        // as points, coarse enough that neighbouring pixels share a cell and
        // the dot has a body rather than aliasing to nothing.
        float3 g = dir * 260.0;
        float3 cell = floor(g);
        if (sky_hash(cell) < density) {
            float3 star = float3(sky_hash(cell + 11.3),
                                 sky_hash(cell + 27.7),
                                 sky_hash(cell + 43.1));
            float d = length((g - cell) - star);
            // Fade rather than cut, so a star does not pop as it crosses a
            // pixel boundary.
            float b = smoothstep(0.42, 0.0, d);
            float mag = 0.35 + 0.65 * sky_hash(cell + 59.9);
            color += sky.mark_color.rgb * (b * mag);
        }
    }
    return float4(color, 1.0);
}
"""

comptime TEXT_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct TextVertIn {
    float2 pos   [[attribute(0)]];
    float2 uv    [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct TextVertOut {
    float4 pos   [[position]];
    float2 uv;
    float4 color;
};

vertex TextVertOut text_vertex(
    TextVertIn in [[stage_in]],
    constant float4x4& ortho_proj [[buffer(0)]]
) {
    TextVertOut out;
    out.pos   = ortho_proj * float4(in.pos, 0.0, 1.0);
    out.uv    = in.uv;
    out.color = in.color;
    return out;
}
"""

comptime TEXT_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

struct TextVertOut {
    float4 pos   [[position]];
    float2 uv;
    float4 color;
};

fragment float4 text_fragment(
    TextVertOut in        [[stage_in]],
    texture2d<float> atlas [[texture(0)]],
    sampler samp           [[sampler(0)]]
) {
    float alpha = atlas.sample(samp, in.uv).r;
    return float4(in.color.rgb, in.color.a * alpha);
}
"""
