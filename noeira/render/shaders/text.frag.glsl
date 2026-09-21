#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(location = 0) out vec4 fragColor;

layout(set = 2, binding = 0) uniform sampler2D atlas;

void main() {
    float alpha = texture(atlas, frag_uv).r;
    fragColor = vec4(frag_color.rgb, frag_color.a * alpha);
}
