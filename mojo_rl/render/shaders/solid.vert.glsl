#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out vec4 obj_color;
layout(location = 4) out vec4 obj_material;

layout(std140, set = 1, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_pos;      // w = num_active_lights
    vec4 light0_dir;      // w = ambient0
    vec4 light0_color;    // w = cast_shadow (0/1)
    vec4 light1_dir;
    vec4 light1_color;
    vec4 light2_dir;
    vec4 light2_color;
    vec4 light3_dir;
    vec4 light3_color;
    vec4 ground_params;   // xyz = checker_color2, w = ground_z
    vec4 fog_params;      // x = fogstart, y = fogend, z = 0, w = 0
} scene;

layout(std140, set = 1, binding = 1) uniform ObjectUniforms {
    mat4 model;
    vec4 color;
    vec4 material;  // x=shininess, y=specular, z=reflectance (>0 = has texture), w=emission
} obj;

void main() {
    vec4 world = obj.model * vec4(position, 1.0);
    gl_Position = scene.view_proj * world;
    world_pos = world.xyz;
    // Transform normal by upper 3x3 of model matrix
    world_normal = (obj.model * vec4(normal, 0.0)).xyz;
    frag_uv = uv;
    obj_color = obj.color;
    obj_material = obj.material;
}
