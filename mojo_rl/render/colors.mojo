"""Color utilities and palettes for rendering.

Provides predefined color palettes for consistent visual styling,
color interpolation, and common color operations.
"""

from .types import Color


# =============================================================================
# Color Constants - Common Colors
# =============================================================================


struct Colors:
    comptime white: Color = Color(255, 255, 255, 255)
    comptime black: Color = Color(0, 0, 0, 255)
    comptime red: Color = Color(255, 0, 0, 255)
    comptime green: Color = Color(0, 255, 0, 255)
    comptime blue: Color = Color(0, 0, 255, 255)
    comptime yellow: Color = Color(255, 255, 0, 255)
    comptime cyan: Color = Color(0, 255, 255, 255)
    comptime magenta: Color = Color(255, 0, 255, 255)
    comptime orange: Color = Color(255, 165, 0, 255)
    comptime purple: Color = Color(128, 0, 128, 255)
    comptime gray: Color = Color(128, 128, 128, 255)
    comptime light_gray: Color = Color(192, 192, 192, 255)
    comptime dark_gray: Color = Color(64, 64, 64, 255)
    comptime transparent: Color = Color(0, 0, 0, 0)


def white() -> Color:
    return Color(255, 255, 255, 255)


def black() -> Color:
    return Color(0, 0, 0, 255)


def red() -> Color:
    return Color(255, 0, 0, 255)


def green() -> Color:
    return Color(0, 255, 0, 255)


def blue() -> Color:
    return Color(0, 0, 255, 255)


def yellow() -> Color:
    return Color(255, 255, 0, 255)


def cyan() -> Color:
    return Color(0, 255, 255, 255)


def magenta() -> Color:
    return Color(255, 0, 255, 255)


def orange() -> Color:
    return Color(255, 165, 0, 255)


def purple() -> Color:
    return Color(128, 0, 128, 255)


def gray() -> Color:
    return Color(128, 128, 128, 255)


def light_gray() -> Color:
    return Color(192, 192, 192, 255)


def dark_gray() -> Color:
    return Color(64, 64, 64, 255)


def transparent() -> Color:
    return Color(0, 0, 0, 0)


# =============================================================================
# Environment-Specific Colors
# =============================================================================


# Sky/Background colors
def sky_blue() -> Color:
    """Light blue sky background."""
    return Color(135, 206, 235, 255)


def night_sky() -> Color:
    """Dark night sky background."""
    return Color(25, 25, 112, 255)


def space_black() -> Color:
    """Deep space background (LunarLander)."""
    return Color(0, 0, 0, 255)


# Ground/Terrain colors
def ground_brown() -> Color:
    """Brown ground/terrain."""
    return Color(139, 119, 101, 255)


def grass_green() -> Color:
    """Grass/field green."""
    return Color(34, 139, 34, 255)


def dark_grass() -> Color:
    """Darker grass for checkerboard patterns."""
    return Color(28, 107, 28, 255)


def sand() -> Color:
    """Sandy/tan color."""
    return Color(210, 180, 140, 255)


def mountain_brown() -> Color:
    """Mountain terrain (MountainCar)."""
    return Color(139, 90, 43, 255)


def moon_gray() -> Color:
    """Lunar surface gray."""
    return Color(102, 102, 102, 255)


# Vehicle/Object colors
def cart_blue() -> Color:
    """Blue cart color (CartPole)."""
    return Color(31, 119, 180, 255)


def pole_tan() -> Color:
    """Tan pole color (CartPole)."""
    return Color(204, 153, 102, 255)


def axle_purple() -> Color:
    """Purple axle color (CartPole)."""
    return Color(127, 127, 204, 255)


def car_red() -> Color:
    """Red car color (MountainCar)."""
    return Color(204, 51, 51, 255)


def lander_gray() -> Color:
    """Lunar lander body gray."""
    return Color(128, 128, 128, 255)


def hull_purple() -> Color:
    """Hull/body purple (BipedalWalker)."""
    return Color(127, 51, 127, 255)


# State indicator colors
def contact_green() -> Color:
    """Green for ground contact."""
    return Color(0, 255, 0, 255)


def no_contact_blue() -> Color:
    """Blue for no ground contact."""
    return Color(77, 166, 255, 255)


def active_green() -> Color:
    """Active state indicator."""
    return Color(0, 200, 0, 255)


def inactive_gray() -> Color:
    """Inactive state indicator."""
    return Color(128, 128, 128, 255)


def warning_orange() -> Color:
    """Warning indicator."""
    return Color(255, 165, 0, 255)


def danger_red() -> Color:
    """Danger indicator."""
    return Color(255, 50, 50, 255)


# Visualization colors
def velocity_orange() -> Color:
    """Velocity arrow/indicator."""
    return Color(255, 165, 0, 255)


def force_red() -> Color:
    """Force arrow/indicator."""
    return Color(255, 0, 0, 255)


def torque_blue() -> Color:
    """Torque indicator."""
    return Color(0, 100, 255, 255)


def target_gold() -> Color:
    """Goal/target marker."""
    return Color(255, 215, 0, 255)


def flag_red() -> Color:
    """Flag marker."""
    return Color(255, 0, 0, 255)


def helipad_yellow() -> Color:
    """Helipad marker."""
    return Color(255, 255, 0, 255)


# Track/Road colors
def track_gray() -> Color:
    """Track/road surface."""
    return Color(107, 107, 107, 255)


def track_visited() -> Color:
    """Visited track tile (CarRacing)."""
    return Color(107, 137, 107, 255)


def curb_red() -> Color:
    """Track curb red."""
    return Color(255, 0, 0, 255)


def curb_white() -> Color:
    """Track curb white."""
    return Color(255, 255, 255, 255)


# =============================================================================
# Color Utilities
# =============================================================================


def rgb(r: Int, g: Int, b: Int) -> Color:
    """Create an opaque RGB color.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        Color with alpha=255.
    """
    return Color(UInt8(r), UInt8(g), UInt8(b), 255)


def rgba(r: Int, g: Int, b: Int, a: Int) -> Color:
    """Create an RGBA color.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).
        a: Alpha component (0-255).

    Returns:
        Color.
    """
    return Color(UInt8(r), UInt8(g), UInt8(b), UInt8(a))


def with_alpha(color: Color, alpha: Int) -> Color:
    """Return color with modified alpha.

    Args:
        color: Source color.
        alpha: New alpha value (0-255).

    Returns:
        Color with new alpha.
    """
    return Color(color.r, color.g, color.b, UInt8(alpha))


def lerp_color(c1: Color, c2: Color, t: Float64) -> Color:
    """Linearly interpolate between two colors.

    Args:
        c1: Start color (t=0).
        c2: End color (t=1).
        t: Interpolation factor (0-1, clamped).

    Returns:
        Interpolated color.
    """
    var tt = max(0.0, min(1.0, t))
    var inv_t = 1.0 - tt
    return Color(
        UInt8(Int(Float64(Int(c1.r)) * inv_t + Float64(Int(c2.r)) * tt)),
        UInt8(Int(Float64(Int(c1.g)) * inv_t + Float64(Int(c2.g)) * tt)),
        UInt8(Int(Float64(Int(c1.b)) * inv_t + Float64(Int(c2.b)) * tt)),
        UInt8(Int(Float64(Int(c1.a)) * inv_t + Float64(Int(c2.a)) * tt)),
    )


def brighten(color: Color, factor: Float64) -> Color:
    """Brighten a color.

    Args:
        color: Source color.
        factor: Brightness factor (1.0 = unchanged, >1 brighter).

    Returns:
        Brightened color.
    """
    return Color(
        UInt8(min(255, Int(Float64(Int(color.r)) * factor))),
        UInt8(min(255, Int(Float64(Int(color.g)) * factor))),
        UInt8(min(255, Int(Float64(Int(color.b)) * factor))),
        color.a,
    )


def darken(color: Color, factor: Float64) -> Color:
    """Darken a color.

    Args:
        color: Source color.
        factor: Darkness factor (1.0 = unchanged, <1 darker).

    Returns:
        Darkened color.
    """
    return Color(
        UInt8(Int(Float64(Int(color.r)) * factor)),
        UInt8(Int(Float64(Int(color.g)) * factor)),
        UInt8(Int(Float64(Int(color.b)) * factor)),
        color.a,
    )


def grayscale(color: Color) -> Color:
    """Convert color to grayscale.

    Args:
        color: Source color.

    Returns:
        Grayscale color (using luminance formula).
    """
    # Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
    var gray = Int(
        Float64(Int(color.r)) * 0.299
        + Float64(Int(color.g)) * 0.587
        + Float64(Int(color.b)) * 0.114
    )
    return Color(UInt8(gray), UInt8(gray), UInt8(gray), color.a)


# =============================================================================
# Particle/Effect Colors
# =============================================================================


def flame_color(lifetime_ratio: Float64) -> Color:
    """Get flame particle color based on lifetime.

    Transitions from yellow -> orange -> red as lifetime decreases.

    Args:
        lifetime_ratio: Remaining lifetime (0-1, 1=just spawned).

    Returns:
        Flame color.
    """
    var t = max(0.0, min(1.0, lifetime_ratio))
    return Color(
        255,  # Red always max
        UInt8(Int(200.0 * t + 50.0)),  # Green fades
        UInt8(Int(50.0 * t)),  # Blue fades faster
        255,
    )


def smoke_color(lifetime_ratio: Float64) -> Color:
    """Get smoke particle color based on lifetime.

    Transitions from gray -> lighter gray as lifetime decreases.

    Args:
        lifetime_ratio: Remaining lifetime (0-1, 1=just spawned).

    Returns:
        Smoke color with fading alpha.
    """
    var t = max(0.0, min(1.0, lifetime_ratio))
    var gray_val = Int(100.0 + 80.0 * (1.0 - t))
    return Color(
        UInt8(gray_val),
        UInt8(gray_val),
        UInt8(gray_val),
        UInt8(Int(200.0 * t)),  # Fade out
    )


def spark_color(lifetime_ratio: Float64) -> Color:
    """Get spark particle color based on lifetime.

    Transitions from white -> yellow -> orange.

    Args:
        lifetime_ratio: Remaining lifetime (0-1, 1=just spawned).

    Returns:
        Spark color.
    """
    var t = max(0.0, min(1.0, lifetime_ratio))
    return Color(
        255,
        UInt8(Int(255.0 * t)),  # Green fades from white to yellow to orange
        UInt8(Int(200.0 * t * t)),  # Blue fades faster
        255,
    )


# =============================================================================
# Color Gradients
# =============================================================================


def heat_gradient(value: Float64) -> Color:
    """Get color from heat gradient (blue -> green -> yellow -> red).

    Args:
        value: Value in range 0-1 (0=cold/blue, 1=hot/red).

    Returns:
        Gradient color.
    """
    var t = max(0.0, min(1.0, value))

    if t < 0.25:
        # Blue to Cyan
        var local_t = t / 0.25
        return lerp_color(blue(), cyan(), local_t)
    elif t < 0.5:
        # Cyan to Green
        var local_t = (t - 0.25) / 0.25
        return lerp_color(cyan(), green(), local_t)
    elif t < 0.75:
        # Green to Yellow
        var local_t = (t - 0.5) / 0.25
        return lerp_color(green(), yellow(), local_t)
    else:
        # Yellow to Red
        var local_t = (t - 0.75) / 0.25
        return lerp_color(yellow(), red(), local_t)


def rainbow_gradient(value: Float64) -> Color:
    """Get color from rainbow gradient.

    Args:
        value: Value in range 0-1.

    Returns:
        Rainbow color.
    """
    var t = max(0.0, min(1.0, value))
    var segment = t * 6.0
    var local_t = segment - Float64(Int(segment))

    if segment < 1:
        return lerp_color(red(), yellow(), local_t)
    elif segment < 2:
        return lerp_color(yellow(), green(), local_t)
    elif segment < 3:
        return lerp_color(green(), cyan(), local_t)
    elif segment < 4:
        return lerp_color(cyan(), blue(), local_t)
    elif segment < 5:
        return lerp_color(blue(), magenta(), local_t)
    else:
        return lerp_color(magenta(), red(), local_t)
