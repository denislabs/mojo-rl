"""MaterialSpec trait and concrete material types for surface properties.

Materials control Blinn-Phong shading: shininess (specular exponent),
specular (intensity), reflectance, and emission. Each material can optionally
reference a TextureSpec by name.

Usage:
    from physics3d.model.material_spec import MaterialSpec, Material, DefaultMaterial

    comptime MyMat = Material[
        name="shiny",
        shininess=0.8,
        specular=0.7,
        reflectance=0.3,
    ]
"""


trait MaterialSpec:
    """Compile-time specification for a material."""

    comptime NAME: String
    comptime SHININESS: Float64       # Specular exponent scaling (0-1, maps to pow range)
    comptime SPECULAR: Float64        # Specular intensity (0-1)
    comptime REFLECTANCE: Float64     # Reflectance coefficient (0-1)
    comptime EMISSION: Float64        # Emissive intensity (0-1)
    comptime HAS_TEXTURE: Bool        # Whether a texture is assigned
    comptime TEXTURE_NAME: String  # Reference to texture by name


@fieldwise_init
struct Material[
    name: String = "default",
    shininess: Float64 = 0.5,
    specular: Float64 = 0.5,
    reflectance: Float64 = 0.0,
    emission: Float64 = 0.0,
    has_texture: Bool = False,
    texture_name: String = "",
](MaterialSpec):
    """General-purpose material with configurable properties.

    MuJoCo defaults: shininess=0.5, specular=0.5, reflectance=0.0, emission=0.0
    """

    comptime NAME: String = Self.name
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime EMISSION: Float64 = Self.emission
    comptime HAS_TEXTURE: Bool = Self.has_texture
    comptime TEXTURE_NAME: String = Self.texture_name


# --- Pre-built materials matching common MuJoCo configurations ---

comptime DefaultMaterial = Material[
    name="default",
    shininess=0.5,
    specular=0.5,
]

comptime PlaneMaterial = Material[
    name="MatPlane",
    shininess=1.0,
    specular=1.0,
    reflectance=0.5,
    has_texture=True,
    texture_name="checker",
]

comptime GeomMaterial = Material[
    name="geom",
    shininess=1.0,
    specular=0.5,
    reflectance=0.5,
    has_texture=True,
    texture_name="flat",
]
