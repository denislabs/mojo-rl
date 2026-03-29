#version 450

layout(location = 0) out vec2 frag_uv;

// Fullscreen triangle: 3 vertices cover the entire screen
void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.999, 1.0);  // Near far plane
    frag_uv = vec2(pos.x, 1.0 - pos.y);  // UV: (0,0) bottom-left, (1,1) top-right
}
