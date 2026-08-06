#version 450

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 in_obj_color;
layout(location = 4) in vec4 in_obj_material;

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

void main() {
    // Discard fragments above the ground plane
    float ground_z = scene.ground_params.w;
    if (world_pos.z > ground_z + 0.001) {
        discard;
    }

    // Multi-light reflection shading
    vec3 N = normalize(world_normal);
    vec3 V = normalize(scene.camera_pos.xyz - world_pos);

    float mat_shininess = in_obj_material.x;
    float mat_specular = in_obj_material.y;
    float spec_exp = mix(4.0, 128.0, mat_shininess);

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
        float specular = pow(max(dot(N, H), 0.0), spec_exp) * mat_specular * 0.5;

        vec3 light_col = l_color.xyz;
        total_color += in_obj_color.rgb * diffuse * light_col + specular * light_col;
        total_ambient += ambient;
    }

    total_ambient = min(total_ambient, 1.0);
    vec3 color = in_obj_color.rgb * total_ambient + total_color;

    // ⚠ ALPHA IS THE REFLECTANCE, and the only attenuation — see the MSL twin.
    // 0.2 is dm_control's `<material name="grid" reflectance=".2">`.
    float alpha = 0.2;

    // Fade out near the edges of the ground. ⚠ LOAD-BEARING with the depth test
    // off (render_frame Phase B2): it is what keeps a reflection off the sky.
    float dist = length(world_pos.xy - scene.camera_pos.xy);
    alpha *= 1.0 - smoothstep(6.0, 10.0, dist);

    fragColor = vec4(color, alpha);
}
