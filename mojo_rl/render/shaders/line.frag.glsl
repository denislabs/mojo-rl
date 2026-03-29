#version 450

layout(location = 0) out vec4 fragColor;

layout(std140, set = 3, binding = 0) uniform LineUniforms {
    mat4 view_proj;
    vec4 color;
} uniforms;

void main() {
    fragColor = uniforms.color;
}
