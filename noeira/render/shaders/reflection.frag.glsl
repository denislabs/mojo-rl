#version 450
// GLSL twin of gpu_shaders.mojo:REFLECTION_FRAGMENT_MSL — keep the two in step.

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 in_obj_color;
layout(location = 4) in vec4 in_obj_material;
layout(location = 5) in vec3 local_pos;
layout(location = 6) in vec4 tex_params;

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
    // Discard fragments above the ground plane
    float ground_z = scene.ground_params.w;
    if (world_pos.z > ground_z + 0.001) {
        discard;
    }

    vec3 N = normalize(world_normal);
    vec3 V = normalize(scene.camera_pos.xyz - world_pos);
    // Unshadowed and untextured — the reflection is a hint, blended at 0.2.
    vec3 color = mj_shade(N, V, world_pos, in_obj_color.rgb, in_obj_material.y,
                          in_obj_material.x, in_obj_material.w, 1.0);

    // ⚠ ALPHA IS THE REFLECTANCE, and the only attenuation — see the MSL twin.
    // The plane material's reflectance, carried in fog_params.z.
    float alpha = scene.fog_params.z;

    // Fade out near the edges of the ground. ⚠ LOAD-BEARING with the depth test
    // off (render_frame Phase B2): it is what keeps a reflection off the sky.
    float dist = length(world_pos.xy - scene.camera_pos.xy);
    alpha *= 1.0 - smoothstep(6.0, 10.0, dist);

    fragColor = vec4(color, alpha);
}
