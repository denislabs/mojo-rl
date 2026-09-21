#version 450
// GLSL twin of gpu_shaders.mojo:SOLID_VERTEX_MSL — keep the two in step.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out vec4 obj_color;
layout(location = 4) out vec4 obj_material;
layout(location = 5) out vec3 local_pos;
layout(location = 6) out vec4 tex_params;

layout(std140, set = 1, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_pos;          // xyz eye; w = number of model lights (0..4)
    vec4 light_pos[4];        // xyz world; w = 1 spot, 0 directional
    vec4 light_dir[4];        // xyz unit; w = cos(cutoff), -2 = no cone
    vec4 light_diffuse[4];    // rgb; w = cast_shadow
    vec4 light_specular[4];   // rgb; w = spot exponent
    vec4 light_ambient[4];    // rgb
    vec4 light_atten[4];      // constant, linear, quadratic
    vec4 headlight_ambient;   // rgb; w = active
    vec4 headlight_diffuse;   // rgb; w = global ambient
    vec4 headlight_specular;  // rgb
    vec4 camera_fwd;          // xyz unit view direction
    vec4 ground_params;
    vec4 fog_params;
} scene;

layout(std140, set = 1, binding = 1) uniform ObjectUniforms {
    mat4 model;
    vec4 color;
    vec4 material;    // x=shininess, y=specular, z=has_texture (>0), w=emission
    vec4 tex_params;  // xy = uv repeat, z = 1 for cube mapping
    vec4 tex_scale;   // xyz scales the object-space position for the cube lookup
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
    local_pos = position * obj.tex_scale.xyz;
    tex_params = obj.tex_params;
}
