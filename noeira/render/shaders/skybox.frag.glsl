#version 450

layout(location = 0) in vec2 frag_uv;

layout(location = 0) out vec4 fragColor;

layout(std140, set = 3, binding = 0) uniform SkyboxUniforms {
    vec4 top_color;     // Gradient top color (rgb + alpha)
    vec4 bottom_color;  // Gradient bottom color (rgb + alpha)
} sky;

void main() {
    // Vertical gradient: uv.y=1 is top, uv.y=0 is bottom
    float t = frag_uv.y;
    vec3 color = mix(sky.bottom_color.rgb, sky.top_color.rgb, t);
    fragColor = vec4(color, 1.0);
}
