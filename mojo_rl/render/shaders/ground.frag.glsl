#version 450

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;

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

vec4 get_light_dir(int i) {
    if (i == 0) return scene.light0_dir;
    if (i == 1) return scene.light1_dir;
    if (i == 2) return scene.light2_dir;
    return scene.light3_dir;
}

float compute_shadow_ground(vec3 wp) {
    vec4 light_pos = shadow.light_view_proj * vec4(wp, 1.0);
    vec3 proj = light_pos.xyz / light_pos.w;

    vec2 shadow_uv = proj.xy * 0.5 + 0.5;

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
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            shadow_val += texture(shadow_map, vec3(shadow_uv + offset, current_depth));
        }
    }
    shadow_val /= 9.0;

    float intensity = shadow.params.x;
    return 1.0 - intensity * (1.0 - shadow_val);
}

void main() {
    // Ground color — solid or checkerboard
    vec3 base_color;

    if (scene.ground_params.x < -0.001) {
        // Solid color mode
        base_color = -scene.ground_params.xyz;
    } else {
        // Checkerboard pattern
        vec3 checker_color1 = vec3(0.35, 0.35, 0.38);
        vec3 checker_color2 = vec3(0.22, 0.22, 0.25);

        if (scene.ground_params.x > 0.001 || scene.ground_params.y > 0.001 || scene.ground_params.z > 0.001) {
            checker_color1 = scene.ground_params.xyz;
            checker_color2 = vec3(0.0, 0.0, 0.0);
        }

        float tile_size = 1.0;
        vec2 tile = floor(world_pos.xy / tile_size);
        float checker = mod(tile.x + tile.y, 2.0);
        checker = abs(checker);

        base_color = mix(checker_color1, checker_color2, checker);
    }

    // Multi-light ground shading
    int num_lights = int(scene.camera_pos.w);
    if (num_lights < 1) num_lights = 1;
    if (num_lights > 4) num_lights = 4;

    vec3 N = vec3(0.0, 0.0, 1.0);
    float lighting = 0.0;
    for (int li = 0; li < num_lights; li++) {
        vec4 l_dir = get_light_dir(li);
        vec3 L = normalize(-l_dir.xyz);
        float diffuse = max(dot(N, L), 0.0) * 0.3 + 0.7 / float(num_lights);
        lighting += diffuse / float(num_lights);
    }
    base_color *= lighting;

    // Apply shadow
    float shadow_factor = compute_shadow_ground(world_pos);
    base_color *= shadow_factor;

    // Distance fade
    float dist = length(world_pos.xy - scene.camera_pos.xy);
    float edge_fade = 1.0 - smoothstep(8.0, 12.0, dist);

    // Linear fog
    float fog_start = scene.fog_params.x;
    float fog_end = scene.fog_params.y;
    if (fog_end > fog_start) {
        float fog_dist = length(world_pos - scene.camera_pos.xyz);
        float fog_factor = clamp((fog_dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        vec3 fog_color = vec3(0.5, 0.495, 0.48);
        base_color = mix(base_color, fog_color, fog_factor);
    }

    float alpha = 0.55 * edge_fade;
    fragColor = vec4(base_color, alpha);
}
