#version 450
// GLSL twin of gpu_shaders.mojo:GROUND_FRAGMENT_MSL — keep the two in step.
// (The texture mode was missing here for a year while the MSL had it.)

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 obj_material;

layout(location = 0) out vec4 fragColor;

layout(std140, set = 3, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_pos;          // xyz eye; w = number of model lights (0..4)
    vec4 light_pos[4];        // xyz world; w = 1 spot, 0 directional
    vec4 light_dir[4];        // xyz unit; w = cos(cutoff), -2 = no cone
    vec4 light_diffuse[4];    // rgb; w = cast_shadow
    vec4 light_specular[4];   // rgb; w = spot exponent
    vec4 light_ambient[4];    // rgb
    vec4 light_atten[4];      // constant, linear, quadratic
    vec4 headlight_ambient;   // rgb; w = active
    vec4 headlight_diffuse;   // rgb; w = global ambient
    vec4 headlight_specular;  // rgb
    vec4 camera_fwd;          // xyz unit view direction
    vec4 ground_params;
    vec4 fog_params;
} scene;

layout(std140, set = 3, binding = 1) uniform ShadowUniforms {
    mat4 light_view_proj;
    vec4 params;  // x=shadow_intensity, y=bias, z=shadow map size
} shadow;

layout(set = 2, binding = 0) uniform sampler2DShadow shadow_map;

float compute_shadow(vec3 wp) {
    vec4 light_pos = shadow.light_view_proj * vec4(wp, 1.0);
    vec3 proj = light_pos.xyz / light_pos.w;

    // Map NDC [-1,1] XY to UV [0,1]
    vec2 shadow_uv = proj.xy * 0.5 + 0.5;

    // Check if outside shadow map bounds
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
        return 1.0;  // Lit (outside shadow frustum)
    }

    float bias = shadow.params.y;
    float current_depth = proj.z - bias;

    // 3x3 PCF for soft shadows. The real map size, from
    // `<visual quality shadowsize=>`; zero means "not set".
    float shadow_val = 0.0;
    float smap = shadow.params.z > 0.5 ? shadow.params.z : 4096.0;
    float texel_size = 1.0 / smap;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            shadow_val += texture(shadow_map, vec3(shadow_uv + offset, current_depth));
        }
    }
    shadow_val /= 9.0;

    // Mix between full shadow and lit based on intensity
    float intensity = shadow.params.x;
    return 1.0 - intensity * (1.0 - shadow_val);
}

layout(set = 2, binding = 1) uniform sampler2D ground_texture;

vec3 mj_light_term(int li, vec3 N, vec3 V, vec3 P, vec3 base_rgb, vec3 mat_spec,
                   float spec_exp, float shadow_factor) {
    vec4 lpos = scene.light_pos[li];
    vec4 ldir = scene.light_dir[li];
    vec3 L;
    float att = 1.0;
    float spot = 1.0;
    if (lpos.w > 0.5) {
        vec3 d = lpos.xyz - P;
        float dist = length(d);
        L = d / max(dist, 1e-6);
        vec4 a = scene.light_atten[li];
        att = 1.0 / max(a.x + a.y * dist + a.z * dist * dist, 1e-6);
        if (ldir.w > -1.5) {
            float sd = dot(-L, ldir.xyz);
            spot = sd < ldir.w ? 0.0 : pow(max(sd, 1e-6), scene.light_specular[li].w);
        }
    } else {
        L = normalize(-ldir.xyz);
    }
    float ndl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = ndl > 0.0 ? pow(max(dot(N, H), 1e-6), spec_exp) : 0.0;
    float sh = (li == 0 && scene.light_diffuse[li].w > 0.5) ? shadow_factor : 1.0;
    return att * spot * (scene.light_ambient[li].rgb * base_rgb
                         + sh * (ndl * scene.light_diffuse[li].rgb * base_rgb
                                 + spec * scene.light_specular[li].rgb * mat_spec));
}

vec3 mj_shade(vec3 N, vec3 V, vec3 P, vec3 base_rgb, float mat_specular,
              float mat_shininess, float mat_emission, float shadow_factor) {
    float spec_exp = mat_shininess * 128.0;
    vec3 mat_spec = vec3(mat_specular);
    vec3 color = base_rgb * (mat_emission + scene.headlight_diffuse.w);
    if (scene.headlight_ambient.w > 0.5) {
        vec3 L = normalize(-scene.camera_fwd.xyz);
        float ndl = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = ndl > 0.0 ? pow(max(dot(N, H), 1e-6), spec_exp) : 0.0;
        color += scene.headlight_ambient.rgb * base_rgb
               + ndl * scene.headlight_diffuse.rgb * base_rgb
               + spec * scene.headlight_specular.rgb * mat_spec;
    }
    int n = clamp(int(scene.camera_pos.w), 0, 4);
    for (int li = 0; li < n; li++) {
        color += mj_light_term(li, N, V, P, base_rgb, mat_spec, spec_exp, shadow_factor);
    }
    return color;
}

// GL_TEXTURE_CUBE_MAP face selection on an object-space position, then the
// face (s, t) as a 2D lookup — see the MSL twin.
vec2 mj_cube_uv(vec3 p) {
    vec3 a = abs(p);
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
    return vec2(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0));
}

void main() {
    // Ground color — three modes based on ground_params encoding:
    //   ground_params.z > 1.5: texture mode, xy = texture repeats PER METRE
    //   ground_params.x < 0: solid color mode, color = abs(ground_params.xyz)
    //   else: checker mode, light tile = ground_params.xyz
    vec3 base_color;

    if (scene.ground_params.z > 1.5) {
        vec2 tex_uv = world_pos.xy * scene.ground_params.xy;
        base_color = texture(ground_texture, tex_uv).rgb;
    } else if (scene.ground_params.x < -0.001) {
        base_color = -scene.ground_params.xyz;
    } else {
        vec3 checker_color1 = vec3(0.35, 0.35, 0.38);
        vec3 checker_color2 = vec3(0.22, 0.22, 0.25);
        if (scene.ground_params.x > 0.001 || scene.ground_params.y > 0.001 || scene.ground_params.z > 0.001) {
            checker_color1 = scene.ground_params.xyz;
            checker_color2 = vec3(0.0, 0.0, 0.0);
        }
        float tile_size = 1.0;
        vec2 tile = floor(world_pos.xy / tile_size);
        float checker = abs(mod(tile.x + tile.y, 2.0));
        base_color = mix(checker_color1, checker_color2, checker);
    }

    float shadow_factor = compute_shadow(world_pos);
    vec3 N = vec3(0.0, 0.0, 1.0);
    vec3 V = normalize(scene.camera_pos.xyz - world_pos);
    vec3 lit = mj_shade(N, V, world_pos, vec3(1.0), obj_material.y,
                        obj_material.x, obj_material.w, shadow_factor);
    base_color = clamp(lit, 0.0, 1.0) * base_color;

    // Linear fog
    float fog_start = scene.fog_params.x;
    float fog_end = scene.fog_params.y;
    if (fog_end > fog_start) {
        float fog_dist = length(world_pos - scene.camera_pos.xyz);
        float fog_factor = clamp((fog_dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        vec3 fog_color = vec3(0.5, 0.495, 0.48);
        base_color = mix(base_color, fog_color, fog_factor);
    }

    // Distance fade
    float dist = length(world_pos.xy - scene.camera_pos.xy);
    float edge_fade = 1.0 - smoothstep(8.0, 12.0, dist);

    // ⚠ THE GROUND IS OPAQUE — see the MSL twin in gpu_shaders.mojo.
    fragColor = vec4(base_color, edge_fade);
}
