"""Light source for 3D rendering — MuJoCo's `mjvLight`, minus the image type.

`ModelRenderer` builds one per `<light>` from `RenderFields`; `Renderer3D`
packs up to four into `SceneUniforms` every frame. The shading model is the
one `render_gl3.c` sets up on the fixed-function pipeline (see
`gpu_shaders.mojo:_MJ_SHADE_MSL`), so the fields here are OpenGL's light
parameters by other names.

⚠ TWO KINDS, NOT ONE. A `directional="false"` MJCF light — the DEFAULT — is a
SPOT: it has a position, a 45-degree cone (`cutoff`), an `exponent` and a
distance attenuation, and it lights NOTHING outside its cone. Until
2026-09-15 every light was packed as directional, so a LIBERO arena's two
`diffuse=".8"` spots summed to 1.6 on the table (flat white) while the walls
their cones never reached went black.

`ambient` and `specular_intensity` are kept as the scalar averages older
callers pass; the RGB fields beside them are what the shader reads, and the
constructor fills the RGB from the scalar when only the scalar is given.
"""


struct LightMode:
    comptime DIRECTIONAL: Int = 0
    comptime SPOT: Int = 1
    comptime POINT: Int = 1
    """Kept for callers that spelled it this way; MuJoCo's fixed-function
    renderer has no point light, `directional="false"` is a spot."""


struct Light(Copyable, Movable):
    var mode: Int
    var dir_x: Float64
    var dir_y: Float64
    var dir_z: Float64
    var color_r: Float64
    """Diffuse RGB (`<light diffuse>`)."""
    var color_g: Float64
    var color_b: Float64
    var ambient: Float64
    var specular_intensity: Float64
    var specular_exponent: Float64
    """Unused by the shader since 2026-09-15 — the specular exponent is the
    MATERIAL's (`shininess * 128`, `render_gl3.c:314`), never the light's.
    Kept so older constructors compile."""
    var cast_shadow: Bool
    var pos_x: Float64
    """World position; meaningful for `mode == SPOT` only."""
    var pos_y: Float64
    var pos_z: Float64
    var cutoff: Float64
    """Spot cone half-angle in DEGREES. 180 disables the cone (OpenGL's
    special case, which MuJoCo uses for directional lights)."""
    var exponent: Float64
    """Spot exponent: the cone falls off as `cos(angle) ** exponent`."""
    var attenuation_0: Float64
    var attenuation_1: Float64
    var attenuation_2: Float64
    var ambient_r: Float64
    var ambient_g: Float64
    var ambient_b: Float64
    var specular_r: Float64
    var specular_g: Float64
    var specular_b: Float64

    def __init__(
        out self,
        mode: Int = 0,
        dir_x: Float64 = 0.0,
        dir_y: Float64 = 0.0,
        dir_z: Float64 = -1.0,
        color_r: Float64 = 0.7,
        color_g: Float64 = 0.7,
        color_b: Float64 = 0.7,
        ambient: Float64 = 0.0,
        specular_intensity: Float64 = 0.3,
        specular_exponent: Float64 = 10.0,
        cast_shadow: Bool = True,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        cutoff: Float64 = 180.0,
        exponent: Float64 = 0.0,
        attenuation_0: Float64 = 1.0,
        attenuation_1: Float64 = 0.0,
        attenuation_2: Float64 = 0.0,
        ambient_r: Float64 = -1.0,
        ambient_g: Float64 = -1.0,
        ambient_b: Float64 = -1.0,
        specular_r: Float64 = -1.0,
        specular_g: Float64 = -1.0,
        specular_b: Float64 = -1.0,
    ):
        self.mode = mode
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.ambient = ambient
        self.specular_intensity = specular_intensity
        self.specular_exponent = specular_exponent
        self.cast_shadow = cast_shadow
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.cutoff = cutoff
        self.exponent = exponent
        self.attenuation_0 = attenuation_0
        self.attenuation_1 = attenuation_1
        self.attenuation_2 = attenuation_2
        # A negative RGB means "not given": fall back to the scalar, so a
        # caller that only knows `ambient=0.3` still gets a grey ambient.
        self.ambient_r = ambient_r if ambient_r >= 0.0 else ambient
        self.ambient_g = ambient_g if ambient_g >= 0.0 else ambient
        self.ambient_b = ambient_b if ambient_b >= 0.0 else ambient
        self.specular_r = (
            specular_r if specular_r >= 0.0 else specular_intensity
        )
        self.specular_g = (
            specular_g if specular_g >= 0.0 else specular_intensity
        )
        self.specular_b = (
            specular_b if specular_b >= 0.0 else specular_intensity
        )
