"""TextureSpec trait and concrete texture types for material appearance.

Supports MuJoCo-style builtin textures:
  - CheckerTexture: Two-color checkerboard pattern (ground planes)
  - FlatTexture: Single solid color (geom surfaces)
  - GradientTexture: Vertical gradient between two colors (skybox)

Usage:
    from physics3d.model.texture_spec import TextureSpec, CheckerTexture, GradientTexture

    comptime MyChecker = CheckerTexture[
        rgb1_r=0.2, rgb1_g=0.3, rgb1_b=0.4,
        rgb2_r=0.8, rgb2_g=0.8, rgb2_b=0.8,
        repeat_x=60.0, repeat_y=60.0,
    ]
"""

# Texture builtin types
comptime TEX_CHECKER: Int = 0
comptime TEX_FLAT: Int = 1
comptime TEX_GRADIENT: Int = 2


trait TextureSpec:
    """Compile-time specification for a texture."""

    comptime NAME: String
    comptime BUILTIN: Int  # TEX_CHECKER, TEX_FLAT, TEX_GRADIENT
    comptime RGB1_R: Float64  # Primary color
    comptime RGB1_G: Float64
    comptime RGB1_B: Float64
    comptime RGB2_R: Float64  # Secondary color
    comptime RGB2_G: Float64
    comptime RGB2_B: Float64
    comptime TEX_REPEAT_X: Float64  # Tiling repeat
    comptime TEX_REPEAT_Y: Float64


@fieldwise_init
struct CheckerTexture[
    name: String = "checker",
    rgb1_r: Float64 = 0.0,
    rgb1_g: Float64 = 0.0,
    rgb1_b: Float64 = 0.0,
    rgb2_r: Float64 = 0.8,
    rgb2_g: Float64 = 0.8,
    rgb2_b: Float64 = 0.8,
    repeat_x: Float64 = 60.0,
    repeat_y: Float64 = 60.0,
](TextureSpec):
    """Checkerboard texture for ground planes.

    Defaults match MuJoCo HalfCheetah XML:
      rgb1="0 0 0" rgb2=".8 .8 .8" repeat="60 60"
    """

    comptime NAME: String = Self.name
    comptime BUILTIN: Int = TEX_CHECKER
    comptime RGB1_R: Float64 = Self.rgb1_r
    comptime RGB1_G: Float64 = Self.rgb1_g
    comptime RGB1_B: Float64 = Self.rgb1_b
    comptime RGB2_R: Float64 = Self.rgb2_r
    comptime RGB2_G: Float64 = Self.rgb2_g
    comptime RGB2_B: Float64 = Self.rgb2_b
    comptime TEX_REPEAT_X: Float64 = Self.repeat_x
    comptime TEX_REPEAT_Y: Float64 = Self.repeat_y


@fieldwise_init
struct FlatTexture[
    name: String = "flat",
    rgb1_r: Float64 = 1.0,
    rgb1_g: Float64 = 1.0,
    rgb1_b: Float64 = 1.0,
](TextureSpec):
    """Single solid color texture for geom surfaces."""

    comptime NAME: String = Self.name
    comptime BUILTIN: Int = TEX_FLAT
    comptime RGB1_R: Float64 = Self.rgb1_r
    comptime RGB1_G: Float64 = Self.rgb1_g
    comptime RGB1_B: Float64 = Self.rgb1_b
    comptime RGB2_R: Float64 = Self.rgb1_r  # Same as primary
    comptime RGB2_G: Float64 = Self.rgb1_g
    comptime RGB2_B: Float64 = Self.rgb1_b
    comptime TEX_REPEAT_X: Float64 = 1.0
    comptime TEX_REPEAT_Y: Float64 = 1.0


@fieldwise_init
struct GradientTexture[
    name: String = "gradient",
    rgb1_r: Float64 = 1.0,
    rgb1_g: Float64 = 1.0,
    rgb1_b: Float64 = 1.0,
    rgb2_r: Float64 = 0.0,
    rgb2_g: Float64 = 0.0,
    rgb2_b: Float64 = 0.0,
](TextureSpec):
    """Vertical gradient texture for skybox.

    Defaults match MuJoCo HalfCheetah XML:
      rgb1="1 1 1" rgb2="0 0 0" (white top, black bottom)
    """

    comptime NAME: String = Self.name
    comptime BUILTIN: Int = TEX_GRADIENT
    comptime RGB1_R: Float64 = Self.rgb1_r
    comptime RGB1_G: Float64 = Self.rgb1_g
    comptime RGB1_B: Float64 = Self.rgb1_b
    comptime RGB2_R: Float64 = Self.rgb2_r
    comptime RGB2_G: Float64 = Self.rgb2_g
    comptime RGB2_B: Float64 = Self.rgb2_b
    comptime TEX_REPEAT_X: Float64 = 1.0
    comptime TEX_REPEAT_Y: Float64 = 1.0
