"""Light source for 3D rendering.

Defines light direction, color, ambient intensity, and specular parameters.
Used by ModelRenderer to configure the 3D renderer at construction time.
"""


struct LightMode:
    comptime DIRECTIONAL: Int = 0
    comptime POINT: Int = 1


@fieldwise_init
struct Light(Copyable, Movable):
    var mode: Int
    var dir_x: Float64
    var dir_y: Float64
    var dir_z: Float64
    var color_r: Float64
    var color_g: Float64
    var color_b: Float64
    var ambient: Float64
    var specular_intensity: Float64
    var specular_exponent: Float64
    var cast_shadow: Bool
