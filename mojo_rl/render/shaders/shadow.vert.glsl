#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(std140, set = 1, binding = 0) uniform SceneUniforms {
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

layout(std140, set = 1, binding = 1) uniform ObjectUniforms {
    mat4 model;
    vec4 color;
    vec4 material;
} obj;

void main() {
    vec4 world = obj.model * vec4(position, 1.0);
    gl_Position = scene.view_proj * world;
}
