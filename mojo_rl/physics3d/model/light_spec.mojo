"""LightSpec trait and concrete light types for model-defined lights.

MuJoCo XML defines lights per-model (e.g., <light cutoff="100" diffuse="1 1 1"
dir="0 0 -1.3" directional="true" pos="0 0 1.3" specular=".1 .1 .1"/>).
This module provides compile-time light specifications that environments use
to configure their renderers.

Light modes:
  - LIGHT_DIRECTIONAL (0): Parallel rays from a direction (sun-like).
  - LIGHT_POINT (1): Emits from a position (not yet used by renderer).

Usage:
    from mojo_rl.physics3d.model.light_spec import LightSpec, DirectionalLight

    # Default light (matches current Renderer3D hardcoded values)
    comptime MyLight = DirectionalLight[]

    # Custom warm light from above
    comptime WarmLight = DirectionalLight[
        dir_x=0.0, dir_y=0.0, dir_z=-1.0,
        color_r=1.0, color_g=0.9, color_b=0.8,
        ambient=0.3,
    ]
"""


from mojo_rl.render import Light

# Light mode constants
comptime LIGHT_DIRECTIONAL: Int = 0
comptime LIGHT_POINT: Int = 1


trait LightSpec:
    """Compile-time specification for a light source.

    Defines light direction, color, ambient intensity, and specular parameters.
    Used by ModelRenderer to configure the 3D renderer at construction time.
    """

    comptime MODE: Int  # LIGHT_DIRECTIONAL or LIGHT_POINT
    comptime DIR_X: Float64
    comptime DIR_Y: Float64
    comptime DIR_Z: Float64
    comptime COLOR_R: Float64
    comptime COLOR_G: Float64
    comptime COLOR_B: Float64
    comptime AMBIENT: Float64
    comptime SPECULAR_INTENSITY: Float64
    comptime SPECULAR_EXPONENT: Float64
    comptime CAST_SHADOW: Bool


@fieldwise_init
struct DirectionalLight[
    dir_x: Float64 = 0.3,
    dir_y: Float64 = -0.4,
    dir_z: Float64 = -0.8,
    color_r: Float64 = 1.0,
    color_g: Float64 = 0.98,
    color_b: Float64 = 0.95,
    ambient: Float64 = 0.25,
    specular_intensity: Float64 = 0.3,
    specular_exponent: Float64 = 32.0,
    cast_shadow: Bool = True,
](LightSpec):
    """Directional light with parallel rays from a given direction.

    Default values match the current Renderer3D hardcoded lighting:
    direction (0.3, -0.4, -0.8), warm white color, 0.25 ambient.
    """

    comptime MODE: Int = LIGHT_DIRECTIONAL
    comptime DIR_X: Float64 = Self.dir_x
    comptime DIR_Y: Float64 = Self.dir_y
    comptime DIR_Z: Float64 = Self.dir_z
    comptime COLOR_R: Float64 = Self.color_r
    comptime COLOR_G: Float64 = Self.color_g
    comptime COLOR_B: Float64 = Self.color_b
    comptime AMBIENT: Float64 = Self.ambient
    comptime SPECULAR_INTENSITY: Float64 = Self.specular_intensity
    comptime SPECULAR_EXPONENT: Float64 = Self.specular_exponent
    comptime CAST_SHADOW: Bool = Self.cast_shadow


# =============================================================================
# Lights — variadic light list (purely visual, no setup_model)
# =============================================================================


trait LightsLike:
    """Trait for compile-time light container types."""

    comptime N: Int

    @staticmethod
    def setup_lights() -> List[Light]:
        ...


@fieldwise_init
struct Lights[*L: LightSpec](LightsLike):
    """Compile-time list of light specifications.

    Provides N (light count) and type-level access to each light via light_types[i].
    Lights are purely visual — no setup_model needed.
    """

    comptime light_types = Self.L
    comptime N: Int = Self.light_types.size

    @staticmethod
    def setup_lights() -> List[Light]:
        var lights = List[Light]()

        comptime for i in range(Self.N):
            comptime L = Self.light_types[i]
            lights.append(
                Light(
                    mode=L.MODE,
                    dir_x=L.DIR_X,
                    dir_y=L.DIR_Y,
                    dir_z=L.DIR_Z,
                    color_r=L.COLOR_R,
                    color_g=L.COLOR_G,
                    color_b=L.COLOR_B,
                    ambient=L.AMBIENT,
                    specular_intensity=L.SPECULAR_INTENSITY,
                    specular_exponent=L.SPECULAR_EXPONENT,
                    cast_shadow=L.CAST_SHADOW,
                )
            )
        return lights^


@fieldwise_init
struct _EmptyLights(LightsLike):
    comptime N: Int = 0

    @staticmethod
    def setup_lights() -> List[Light]:
        return []
