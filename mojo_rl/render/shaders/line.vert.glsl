#version 450

layout(location = 0) in vec3 position;

layout(std140, set = 1, binding = 0) uniform LineUniforms {
    mat4 view_proj;
    vec4 color;
} uniforms;

void main() {
    gl_Position = uniforms.view_proj * vec4(position, 1.0);
}
