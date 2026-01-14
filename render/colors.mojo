"""Color utilities and palettes for rendering.

Provides predefined color palettes for consistent visual styling,
color interpolation, and common color operations.
"""

from .sdl2 import SDL_Color


# =============================================================================
# Color Constants - Common Colors
# =============================================================================


fn white() -> SDL_Color:
    return SDL_Color(255, 255, 255, 255)


fn black() -> SDL_Color:
    return SDL_Color(0, 0, 0, 255)


fn red() -> SDL_Color:
    return SDL_Color(255, 0, 0, 255)


fn green() -> SDL_Color:
    return SDL_Color(0, 255, 0, 255)


fn blue() -> SDL_Color:
    return SDL_Color(0, 0, 255, 255)


fn yellow() -> SDL_Color:
    return SDL_Color(255, 255, 0, 255)


fn cyan() -> SDL_Color:
    return SDL_Color(0, 255, 255, 255)


fn magenta() -> SDL_Color:
    return SDL_Color(255, 0, 255, 255)


fn orange() -> SDL_Color:
    return SDL_Color(255, 165, 0, 255)


fn purple() -> SDL_Color:
    return SDL_Color(128, 0, 128, 255)


fn gray() -> SDL_Color:
    return SDL_Color(128, 128, 128, 255)


fn light_gray() -> SDL_Color:
    return SDL_Color(192, 192, 192, 255)


fn dark_gray() -> SDL_Color:
    return SDL_Color(64, 64, 64, 255)


fn transparent() -> SDL_Color:
    return SDL_Color(0, 0, 0, 0)


# =============================================================================
# Environment-Specific Colors
# =============================================================================


# Sky/Background colors
fn sky_blue() -> SDL_Color:
    """Light blue sky background."""
    return SDL_Color(135, 206, 235, 255)


fn night_sky() -> SDL_Color:
    """Dark night sky background."""
    return SDL_Color(25, 25, 112, 255)


fn space_black() -> SDL_Color:
    """Deep space background (LunarLander)."""
    return SDL_Color(0, 0, 0, 255)


# Ground/Terrain colors
fn ground_brown() -> SDL_Color:
    """Brown ground/terrain."""
    return SDL_Color(139, 119, 101, 255)


fn grass_green() -> SDL_Color:
    """Grass/field green."""
    return SDL_Color(34, 139, 34, 255)


fn dark_grass() -> SDL_Color:
    """Darker grass for checkerboard patterns."""
    return SDL_Color(28, 107, 28, 255)


fn sand() -> SDL_Color:
    """Sandy/tan color."""
    return SDL_Color(210, 180, 140, 255)


fn mountain_brown() -> SDL_Color:
    """Mountain terrain (MountainCar)."""
    return SDL_Color(139, 90, 43, 255)


fn moon_gray() -> SDL_Color:
    """Lunar surface gray."""
    return SDL_Color(102, 102, 102, 255)


# Vehicle/Object colors
fn cart_blue() -> SDL_Color:
    """Blue cart color (CartPole)."""
    return SDL_Color(31, 119, 180, 255)


fn pole_tan() -> SDL_Color:
    """Tan pole color (CartPole)."""
    return SDL_Color(204, 153, 102, 255)


fn axle_purple() -> SDL_Color:
    """Purple axle color (CartPole)."""
    return SDL_Color(127, 127, 204, 255)


fn car_red() -> SDL_Color:
    """Red car color (MountainCar)."""
    return SDL_Color(204, 51, 51, 255)


fn lander_gray() -> SDL_Color:
    """Lunar lander body gray."""
    return SDL_Color(128, 128, 128, 255)


fn hull_purple() -> SDL_Color:
    """Hull/body purple (BipedalWalker)."""
    return SDL_Color(127, 51, 127, 255)


# State indicator colors
fn contact_green() -> SDL_Color:
    """Green for ground contact."""
    return SDL_Color(0, 255, 0, 255)


fn no_contact_blue() -> SDL_Color:
    """Blue for no ground contact."""
    return SDL_Color(77, 166, 255, 255)


fn active_green() -> SDL_Color:
    """Active state indicator."""
    return SDL_Color(0, 200, 0, 255)


fn inactive_gray() -> SDL_Color:
    """Inactive state indicator."""
    return SDL_Color(128, 128, 128, 255)


fn warning_orange() -> SDL_Color:
    """Warning indicator."""
    return SDL_Color(255, 165, 0, 255)


fn danger_red() -> SDL_Color:
    """Danger indicator."""
    return SDL_Color(255, 50, 50, 255)


# Visualization colors
fn velocity_orange() -> SDL_Color:
    """Velocity arrow/indicator."""
    return SDL_Color(255, 165, 0, 255)


fn force_red() -> SDL_Color:
    """Force arrow/indicator."""
    return SDL_Color(255, 0, 0, 255)


fn torque_blue() -> SDL_Color:
    """Torque indicator."""
    return SDL_Color(0, 100, 255, 255)


fn target_gold() -> SDL_Color:
    """Goal/target marker."""
    return SDL_Color(255, 215, 0, 255)


fn flag_red() -> SDL_Color:
    """Flag marker."""
    return SDL_Color(255, 0, 0, 255)


fn helipad_yellow() -> SDL_Color:
    """Helipad marker."""
    return SDL_Color(255, 255, 0, 255)


# Track/Road colors
fn track_gray() -> SDL_Color:
    """Track/road surface."""
    return SDL_Color(107, 107, 107, 255)


fn track_visited() -> SDL_Color:
    """Visited track tile (CarRacing)."""
    return SDL_Color(107, 137, 107, 255)


fn curb_red() -> SDL_Color:
    """Track curb red."""
    return SDL_Color(255, 0, 0, 255)


fn curb_white() -> SDL_Color:
    """Track curb white."""
    return SDL_Color(255, 255, 255, 255)


# =============================================================================
# Color Utilities
# =============================================================================


fn rgb(r: Int, g: Int, b: Int) -> SDL_Color:
    """Create an opaque RGB color.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        SDL_Color with alpha=255.
    """
    return SDL_Color(UInt8(r), UInt8(g), UInt8(b), 255)


fn rgba(r: Int, g: Int, b: Int, a: Int) -> SDL_Color:
    """Create an RGBA color.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).
        a: Alpha component (0-255).

    Returns:
        SDL_Color.
    """
    return SDL_Color(UInt8(r), UInt8(g), UInt8(b), UInt8(a))


fn with_alpha(color: SDL_Color, alpha: Int) -> SDL_Color:
    """Return color with modified alpha.

    Args:
        color: Source color.
        alpha: New alpha value (0-255).

    Returns:
        Color with new alpha.
    """
    return SDL_Color(color.r, color.g, color.b, UInt8(alpha))


fn lerp_color(c1: SDL_Color, c2: SDL_Color, t: Float64) -> SDL_Color:
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
    return SDL_Color(
        UInt8(Int(Float64(Int(c1.r)) * inv_t + Float64(Int(c2.r)) * tt)),
        UInt8(Int(Float64(Int(c1.g)) * inv_t + Float64(Int(c2.g)) * tt)),
        UInt8(Int(Float64(Int(c1.b)) * inv_t + Float64(Int(c2.b)) * tt)),
        UInt8(Int(Float64(Int(c1.a)) * inv_t + Float64(Int(c2.a)) * tt)),
    )


fn brighten(color: SDL_Color, factor: Float64) -> SDL_Color:
    """Brighten a color.

    Args:
        color: Source color.
        factor: Brightness factor (1.0 = unchanged, >1 brighter).

    Returns:
        Brightened color.
    """
    return SDL_Color(
        UInt8(min(255, Int(Float64(Int(color.r)) * factor))),
        UInt8(min(255, Int(Float64(Int(color.g)) * factor))),
        UInt8(min(255, Int(Float64(Int(color.b)) * factor))),
        color.a,
    )


fn darken(color: SDL_Color, factor: Float64) -> SDL_Color:
    """Darken a color.

    Args:
        color: Source color.
        factor: Darkness factor (1.0 = unchanged, <1 darker).

    Returns:
        Darkened color.
    """
    return SDL_Color(
        UInt8(Int(Float64(Int(color.r)) * factor)),
        UInt8(Int(Float64(Int(color.g)) * factor)),
        UInt8(Int(Float64(Int(color.b)) * factor)),
        color.a,
    )


fn grayscale(color: SDL_Color) -> SDL_Color:
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
    return SDL_Color(UInt8(gray), UInt8(gray), UInt8(gray), color.a)


# =============================================================================
# Particle/Effect Colors
# =============================================================================


fn flame_color(lifetime_ratio: Float64) -> SDL_Color:
    """Get flame particle color based on lifetime.

    Transitions from yellow -> orange -> red as lifetime decreases.

    Args:
        lifetime_ratio: Remaining lifetime (0-1, 1=just spawned).

    Returns:
        Flame color.
    """
    var t = max(0.0, min(1.0, lifetime_ratio))
    return SDL_Color(
        255,  # Red always max
        UInt8(Int(200.0 * t + 50.0)),  # Green fades
        UInt8(Int(50.0 * t)),  # Blue fades faster
        255,
    )


fn smoke_color(lifetime_ratio: Float64) -> SDL_Color:
    """Get smoke particle color based on lifetime.

    Transitions from gray -> lighter gray as lifetime decreases.

    Args:
        lifetime_ratio: Remaining lifetime (0-1, 1=just spawned).

    Returns:
        Smoke color with fading alpha.
    """
    var t = max(0.0, min(1.0, lifetime_ratio))
    var gray_val = Int(100.0 + 80.0 * (1.0 - t))
    return SDL_Color(
        UInt8(gray_val),
        UInt8(gray_val),
        UInt8(gray_val),
        UInt8(Int(200.0 * t)),  # Fade out
    )


fn spark_color(lifetime_ratio: Float64) -> SDL_Color:
    """Get spark particle color based on lifetime.

    Transitions from white -> yellow -> orange.

    Args:
        lifetime_ratio: Remaining lifetime (0-1, 1=just spawned).

    Returns:
        Spark color.
    """
    var t = max(0.0, min(1.0, lifetime_ratio))
    return SDL_Color(
        255,
        UInt8(Int(255.0 * t)),  # Green fades from white to yellow to orange
        UInt8(Int(200.0 * t * t)),  # Blue fades faster
        255,
    )


# =============================================================================
# Color Gradients
# =============================================================================


fn heat_gradient(value: Float64) -> SDL_Color:
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


fn rainbow_gradient(value: Float64) -> SDL_Color:
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
