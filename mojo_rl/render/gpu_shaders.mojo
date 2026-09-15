"""MSL Shader Source Strings for GPU 3D Renderer.

Seven shader pairs as comptime string constants:
  1. Solid object shaders (MuJoCo's fixed-function light model + shadow sampling + per-object material)
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
# 560B — `gpu_types.SCENE_UNIFORMS_BYTES`. ⚠ THREE HAND-WRITTEN COPIES OF
# THIS STRUCT EXIST: here, `gpu_types.SceneUniforms`, and every
# `shaders/*.glsl`. Move them together.

comptime _SCENE_UNIFORMS_MSL = """
struct SceneUniforms {
    float4x4 view_proj;
    float4 camera_pos;          // xyz eye; w = number of model lights (0..4)
    float4 light_pos[4];        // xyz world; w = 1 spot, 0 directional
    float4 light_dir[4];        // xyz unit; w = cos(cutoff), -2 = no cone
    float4 light_diffuse[4];    // rgb; w = cast_shadow
    float4 light_specular[4];   // rgb; w = spot exponent
    float4 light_ambient[4];    // rgb
    float4 light_atten[4];      // constant, linear, quadratic
    float4 headlight_ambient;   // rgb; w = active
    float4 headlight_diffuse;   // rgb; w = global ambient
    float4 headlight_specular;  // rgb
    float4 camera_fwd;          // xyz unit view direction
    float4 ground_params;       // see draw_ground_grid
    float4 fog_params;          // x = fogstart, y = fogend, z = ground reflectance
};
"""

# --- MuJoCo's light model, shared by the solid, ground and reflection passes ---
#
# ⚠ A TRANSCRIPTION, NOT A DESIGN. `render_gl3.c:initLights` / `adjustLight`
# program OpenGL's fixed-function lighting and this is that equation
# (GL 2.1 spec 2.14.1) with the values MuJoCo feeds it:
#
#   * material: ambient = diffuse = rgba (GL_COLOR_MATERIAL), specular =
#     (mat.specular,)*3, shininess = mat.shininess * 128, emission =
#     mat.emission * rgba  (`render_gl3.c:302-316`)
#   * per light: ambient/diffuse/specular RGB; a DIRECTIONAL light has no
#     position and no cone; a SPOT has position, cutoff, exponent and
#     constant/linear/quadratic attenuation  (`initLights`)
#   * the headlight is one more DIRECTIONAL light at the eye, aimed along the
#     view, with `<visual headlight>`'s colours  (`mjv_makeLights`)
#   * global ambient is 0.3 only when there is no light at all
#     (`initLights`, "create some ambient light if no supported lights")
#   * LOCAL_VIEWER = 1, so the half-vector uses the true eye direction
#   * the specular term is zero where N.L <= 0 (the spec's "f")
#
# Shadow: MuJoCo's shadow map removes the shadow-casting light's DIFFUSE and
# SPECULAR where the fragment is occluded and keeps its ambient; the first
# light with `castshadow` is the one with a map here, as before.
#
# The old shader was a Blinn-Phong of its own — every light directional,
# ambient summed as a scalar, shininess remapped 4..128 — and on a scene lit
# by two 45-degree spots it put both at full strength on the table (1.6 =
# flat white) and neither on the walls (black).

comptime _MJ_SHADE_MSL = """
static inline float3 mj_light_term(constant SceneUniforms &scene, int li,
                                   float3 N, float3 V, float3 P,
                                   float3 base_rgb, float3 mat_spec,
                                   float spec_exp, float shadow_factor) {
    float4 lpos = scene.light_pos[li];
    float4 ldir = scene.light_dir[li];
    float3 L;
    float att = 1.0;
    float spot = 1.0;
    if (lpos.w > 0.5) {
        float3 d = lpos.xyz - P;
        float dist = length(d);
        L = d / max(dist, 1e-6);
        float4 a = scene.light_atten[li];
        att = 1.0 / max(a.x + a.y * dist + a.z * dist * dist, 1e-6);
        if (ldir.w > -1.5) {
            float sd = dot(-L, ldir.xyz);
            spot = sd < ldir.w ? 0.0 : pow(max(sd, 1e-6), scene.light_specular[li].w);
        }
    } else {
        L = normalize(-ldir.xyz);
    }
    float ndl = max(dot(N, L), 0.0);
    float3 H = normalize(L + V);
    float spec = ndl > 0.0 ? pow(max(dot(N, H), 1e-6), spec_exp) : 0.0;
    float sh = (li == 0 && scene.light_diffuse[li].w > 0.5) ? shadow_factor : 1.0;
    return att * spot * (scene.light_ambient[li].rgb * base_rgb
                         + sh * (ndl * scene.light_diffuse[li].rgb * base_rgb
                                 + spec * scene.light_specular[li].rgb * mat_spec));
}

static inline float3 mj_shade(constant SceneUniforms &scene,
                              float3 N, float3 V, float3 P, float3 base_rgb,
                              float mat_specular, float mat_shininess,
                              float mat_emission, float shadow_factor) {
    float spec_exp = mat_shininess * 128.0;
    float3 mat_spec = float3(mat_specular);
    float3 color = base_rgb * (mat_emission + scene.headlight_diffuse.w);
    if (scene.headlight_ambient.w > 0.5) {
        float3 L = normalize(-scene.camera_fwd.xyz);
        float ndl = max(dot(N, L), 0.0);
        float3 H = normalize(L + V);
        float spec = ndl > 0.0 ? pow(max(dot(N, H), 1e-6), spec_exp) : 0.0;
        color += scene.headlight_ambient.rgb * base_rgb
               + ndl * scene.headlight_diffuse.rgb * base_rgb
               + spec * scene.headlight_specular.rgb * mat_spec;
    }
    int n = clamp(int(scene.camera_pos.w), 0, 4);
    for (int li = 0; li < n; li++) {
        color += mj_light_term(scene, li, N, V, P, base_rgb, mat_spec, spec_exp, shadow_factor);
    }
    return color;
}

// GL_TEXTURE_CUBE_MAP's face selection (GL 2.1 spec table 3.19) on an
// object-space position, then the face's (s, t) as a 2D lookup — MuJoCo
// uploads a square `type="cube"` image to all six faces
// (`render_context.c`: "assign data: repeated"), so one 2D texture is the
// whole cube.
static inline float2 mj_cube_uv(float3 p) {
    float3 a = abs(p);
    float ma;
    float sc;
    float tc;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        sc = p.x > 0.0 ? -p.z : p.z;
        tc = -p.y;
    } else if (a.y >= a.z) {
        ma = a.y;
        sc = p.x;
        tc = p.y > 0.0 ? p.z : -p.z;
    } else {
        ma = a.z;
        sc = p.z > 0.0 ? p.x : -p.x;
        tc = -p.y;
    }
    ma = max(ma, 1e-6);
    return float2(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0));
}
"""

# The per-object block, shared by every vertex shader that draws geometry.
# 128B — `gpu_types.OBJECT_UNIFORMS_BYTES`.
comptime _OBJECT_UNIFORMS_MSL = """
struct ObjectUniforms {
    float4x4 model;
    float4 color;
    float4 material;    // x=shininess, y=specular, z=has_texture (>0), w=emission
    float4 tex_params;  // xy = uv repeat, z = 1 for cube mapping
    float4 tex_scale;   // xyz scales the object-space position for the cube lookup
};
"""

# What the solid vertex shader hands the solid AND reflection fragments.
comptime _SOLID_VERTEX_OUT_MSL = """
struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
    float3 local_pos;
    float2 uv;
    float4 obj_color;
    float4 obj_material;
    float4 tex_params;
};
"""


# --- Solid Object Shaders (MuJoCo lighting + Shadows + Material) ---

comptime SOLID_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

""" + _SOLID_VERTEX_OUT_MSL + _SCENE_UNIFORMS_MSL + _OBJECT_UNIFORMS_MSL + """

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
    out.local_pos = in.position * obj.tex_scale.xyz;
    out.uv = in.uv;
    out.obj_color = obj.color;
    out.obj_material = obj.material;
    out.tex_params = obj.tex_params;
    return out;
}
"""

comptime _SHADOW_SAMPLE_MSL = """
struct ShadowUniforms {
    float4x4 light_view_proj;
    float4 params;  // x=shadow_intensity, y=bias, z=shadow map size
};

static inline float compute_shadow(float3 world_pos,
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
    // ⚠ THE REAL MAP SIZE, from `<visual quality shadowsize=>`. This was
    // hardcoded to 4096 while `quadruped escape` asks for 2048, so every PCF
    // tap landed half a texel from where it meant to. Zero means "not set" —
    // fall back rather than divide by it.
    float smap = shadow.params.z > 0.5 ? shadow.params.z : 4096.0;
    float texel_size = 1.0 / smap;

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
"""

comptime SOLID_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

""" + _SOLID_VERTEX_OUT_MSL + _SCENE_UNIFORMS_MSL + _MJ_SHADE_MSL + _SHADOW_SAMPLE_MSL + """

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
    float mat_shininess = in.obj_material.x;
    float mat_specular = in.obj_material.y;
    float has_texture = in.obj_material.z;     // >0 = sample obj_texture
    float mat_emission = in.obj_material.w;

    // ⚠ LIGHT FIRST, TEXTURE SECOND — `GL_MODULATE` with the default
    // single-colour specular (`render_gl3.c:699`). OpenGL lights the vertex
    // colour, clamps the sum (highlight included) to [0,1], and only then
    // multiplies by the texel. Texturing first and adding the highlight on
    // top made every textured surface brighter than MuJoCo's by its
    // specular term — ~7% on LIBERO's wood table.
    float4 base_color = in.obj_color;
    float shadow_factor = compute_shadow(in.world_pos, shadow, shadow_map, shadow_sampler);
    float3 color = mj_shade(scene, N, V, in.world_pos, base_color.rgb,
                            mat_specular, mat_shininess, mat_emission, shadow_factor);
    if (has_texture > 0.5) {
        float2 uv = in.uv * in.tex_params.xy;
        if (in.tex_params.z > 0.5) {
            uv = mj_cube_uv(in.local_pos);
        }
        float4 tex_color = obj_texture.sample(obj_sampler, uv);
        color = clamp(color, 0.0, 1.0) * tex_color.rgb;
        base_color.a *= tex_color.a;
    }

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

# --- Ground Shaders (Procedural Checkerboard / texture + Shadows) ---

comptime _GROUND_VERTEX_OUT_MSL = """
struct VertexOut {
    float4 position  [[position]];
    float3 world_pos;
    float3 world_normal;
    float2 uv;
    float4 obj_material;
};
"""

comptime GROUND_VERTEX_MSL = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

""" + _GROUND_VERTEX_OUT_MSL + _SCENE_UNIFORMS_MSL + _OBJECT_UNIFORMS_MSL + """

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
    out.obj_material = obj.material;
    return out;
}
"""

comptime GROUND_FRAGMENT_MSL = """
#include <metal_stdlib>
using namespace metal;

""" + _GROUND_VERTEX_OUT_MSL + _SCENE_UNIFORMS_MSL + _MJ_SHADE_MSL + _SHADOW_SAMPLE_MSL + """

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
    //   ground_params.z > 1.5: texture mode, xy = texture repeats PER METRE
    //   ground_params.x < 0: solid color mode, color = abs(ground_params.xyz)
    //   else: checker mode, light tile = ground_params.xyz
    // Note: ground_params.w is reserved for ground_z (reflection clipping)
    float3 base_color;

    if (scene.ground_params.z > 1.5) {
        // Texture mode. ⚠ WORLD-SPACE, NOT MESH UV: MuJoCo tiles a plane in
        // the plane's own units (`texrepeat` over its `size`, or per unit
        // with `texuniform`), and the ground quad here is a 24 m mesh whose
        // extent has nothing to do with the plane's. `draw_ground_grid`
        // converts the material to repeats per metre.
        float2 tex_uv = in.world_pos.xy * scene.ground_params.xy;
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

    // The same light model as every other surface. ⚠ A TEXTURED FLOOR IS LIT
    // TOO: it used to draw its texels unshaded ("the texture contains its own
    // shading"), which made a light-grey plank tile glow under a scene whose
    // spots never reach the floor — MuJoCo shows it dark.
    // Lit white, clamped, then modulated — `GL_MODULATE`, as for solids.
    float shadow_factor = compute_shadow(in.world_pos, shadow, shadow_map, shadow_sampler);
    float3 N = float3(0.0, 0.0, 1.0);
    float3 V = normalize(scene.camera_pos.xyz - in.world_pos);
    float3 lit = mj_shade(scene, N, V, in.world_pos, float3(1.0),
                          in.obj_material.y, in.obj_material.x, in.obj_material.w,
                          shadow_factor);
    base_color = clamp(lit, 0.0, 1.0) * base_color;

    // Linear fog
    float fog_start = scene.fog_params.x;
    float fog_end = scene.fog_params.y;
    if (fog_end > fog_start) {
        float fog_dist = length(in.world_pos - scene.camera_pos.xyz);
        float fog_factor = clamp((fog_dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        float3 fog_color = float3(0.5, 0.495, 0.48);
        base_color = mix(base_color, fog_color, fog_factor);
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

""" + _SCENE_UNIFORMS_MSL + _OBJECT_UNIFORMS_MSL + """

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

""" + _SOLID_VERTEX_OUT_MSL + _SCENE_UNIFORMS_MSL + _MJ_SHADE_MSL + """

fragment float4 reflection_fragment(
    VertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]]
) {
    // Discard fragments above the ground plane
    float ground_z = scene.ground_params.w;
    if (in.world_pos.z > ground_z + 0.001) {
        discard_fragment();
    }

    float3 N = normalize(in.world_normal);
    float3 V = normalize(scene.camera_pos.xyz - in.world_pos);
    // Unshadowed and untextured — the reflection is a hint, not a second
    // render, and it is blended at 0.2 below.
    float3 color = mj_shade(scene, N, V, in.world_pos, in.obj_color.rgb,
                            in.obj_material.y, in.obj_material.x,
                            in.obj_material.w, 1.0);

    // ⚠ ALPHA IS THE REFLECTANCE, and it is the ONLY attenuation. The colour
    // used to be pre-darkened (`color *= 0.35`) as well as blended at 0.35,
    // which double-counted: MuJoCo's mirror term is
    // `floor*(1-reflectance) + reflected*reflectance`, one factor, not two.
    //
    // The plane material's `reflectance`, carried in `fog_params.z`. It was a
    // constant 0.2 (dm_control's `grid` material) until LIBERO's floor, whose
    // material says 0, showed a ghost robot under the table.
    float alpha = scene.fog_params.z;

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
