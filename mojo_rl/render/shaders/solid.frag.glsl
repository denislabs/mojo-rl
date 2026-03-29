#version 450

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 obj_color;
layout(location = 4) in vec4 obj_material;

layout(location = 0) out vec4 fragColor;

layout(std140, set = 3, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light0_dir;
    vec4 light0_color;
    vec4 light1_dir;
    vec4 light1_color;
    vec4 light2_dir;
    vec4 light2_color;
    vec4 light3_dir;
    vec4 light3_color;
    vec4 ground_params;
    vec4 fog_params;
} scene;

layout(std140, set = 3, binding = 1) uniform ShadowUniforms {
    mat4 light_view_proj;
    vec4 params;  // x=shadow_intensity, y=bias
} shadow;

layout(set = 2, binding = 0) uniform sampler2DShadow shadow_map;
layout(set = 2, binding = 1) uniform sampler2D obj_texture;

vec4 get_light_dir(int i) {
    if (i == 0) return scene.light0_dir;
    if (i == 1) return scene.light1_dir;
    if (i == 2) return scene.light2_dir;
    return scene.light3_dir;
}

vec4 get_light_color(int i) {
    if (i == 0) return scene.light0_color;
    if (i == 1) return scene.light1_color;
    if (i == 2) return scene.light2_color;
    return scene.light3_color;
}

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

    // 3x3 PCF for soft shadows
    float shadow_val = 0.0;
    float texel_size = 1.0 / 4096.0;
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

void main() {
    vec3 N = normalize(world_normal);
    vec3 V = normalize(scene.camera_pos.xyz - world_pos);

    // Per-object material properties
    float mat_shininess = obj_material.x;
    float mat_specular = obj_material.y;
    float has_texture = obj_material.z;
    float mat_emission = obj_material.w;

    // Map shininess [0,1] to specular exponent: 0.0->4, 0.5->32, 1.0->128
    float spec_exp = mix(4.0, 128.0, mat_shininess);

    // Sample texture if enabled (material.z > 0)
    vec4 base_color = obj_color;
    if (has_texture > 0.5) {
        vec4 tex_color = texture(obj_texture, frag_uv);
        base_color = vec4(base_color.rgb * tex_color.rgb, base_color.a * tex_color.a);
    }

    int num_lights = int(scene.camera_pos.w);
    if (num_lights < 1) num_lights = 1;
    if (num_lights > 4) num_lights = 4;

    vec3 total_color = vec3(0.0);
    float total_ambient = 0.0;

    for (int li = 0; li < num_lights; li++) {
        vec4 l_dir = get_light_dir(li);
        vec4 l_color = get_light_color(li);

        vec3 L = normalize(-l_dir.xyz);
        vec3 H = normalize(L + V);

        float ambient = l_dir.w;
        float diffuse = max(dot(N, L), 0.0);
        float specular = pow(max(dot(N, H), 0.0), spec_exp) * mat_specular;

        // Shadow only for first shadow-casting light
        float shadow_factor = 1.0;
        if (li == 0 && l_color.w > 0.5) {
            shadow_factor = compute_shadow(world_pos);
        }

        vec3 light_col = l_color.xyz;
        total_color += base_color.rgb * shadow_factor * diffuse * light_col
                     + shadow_factor * specular * light_col;
        total_ambient += ambient;
    }

    // Clamp ambient to avoid over-brightening with multiple lights
    total_ambient = min(total_ambient, 1.0);

    vec3 color = base_color.rgb * total_ambient + total_color
                 + base_color.rgb * mat_emission;

    // Linear fog
    float fog_start = scene.fog_params.x;
    float fog_end = scene.fog_params.y;
    if (fog_end > fog_start) {
        float dist = length(world_pos - scene.camera_pos.xyz);
        float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        vec3 fog_color = vec3(0.5, 0.495, 0.48);
        color = mix(color, fog_color, fog_factor);
    }

    fragColor = vec4(color, base_color.a);
}
