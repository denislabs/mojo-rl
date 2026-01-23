# =============================================================================
# Helper Functions - Observation Normalization & Shaping
# =============================================================================

from .constants import LLConstants
from math import sqrt


@always_inline
fn normalize_position[
    T: DType
](x: Scalar[T], y: Scalar[T]) -> Tuple[Scalar[T], Scalar[T]]:
    """Normalize position relative to helipad center.

    Args:
        x: Raw x position in world units.
        y: Raw y position in world units.

    Returns:
        Tuple of (x_norm, y_norm) in range approximately [-1, 1].
    """
    var x_norm = (x - Scalar[T](LLConstants.HELIPAD_X)) / Scalar[T](
        LLConstants.W_UNITS / 2.0
    )
    var y_norm = (
        y
        - Scalar[T](
            LLConstants.HELIPAD_Y + LLConstants.LEG_DOWN / LLConstants.SCALE
        )
    ) / Scalar[T](LLConstants.H_UNITS / 2.0)
    return (x_norm, y_norm)


@always_inline
fn normalize_velocity[
    T: DType
](vx: Scalar[T], vy: Scalar[T]) -> Tuple[Scalar[T], Scalar[T]]:
    """Normalize velocity for observation.

    Args:
        vx: Raw x velocity.
        vy: Raw y velocity.

    Returns:
        Tuple of (vx_norm, vy_norm) scaled by viewport and FPS.
    """
    var vx_norm = (
        vx * Scalar[T](LLConstants.W_UNITS / 2.0) / Scalar[T](LLConstants.FPS)
    )
    var vy_norm = (
        vy * Scalar[T](LLConstants.H_UNITS / 2.0) / Scalar[T](LLConstants.FPS)
    )
    return (vx_norm, vy_norm)


@always_inline
fn normalize_angular_velocity[T: DType](omega: Scalar[T]) -> Scalar[T]:
    """Normalize angular velocity for observation.

    Args:
        omega: Raw angular velocity in rad/s.

    Returns:
        Normalized angular velocity.
    """
    return Scalar[T](20.0) * omega / Scalar[T](LLConstants.FPS)


@always_inline
fn get_terrain_height_at_x[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Get terrain height at a given x position using simplified chunk lookup.

    This matches the CPU version's _get_terrain_height() behavior, which returns
    the height at the start of the chunk containing x (no interpolation).

    Note: For GPU, the actual terrain edges need to be passed in. This function
    returns helipad_y for the helipad region and a basic approximation otherwise.
    Use get_terrain_height_from_edges for actual terrain lookup in GPU kernels.

    Args:
        x: X position in world units.

    Returns:
        Approximate terrain height (helipad_y for simplicity).
    """
    # For the helipad region, terrain is always at HELIPAD_Y
    # For non-helipad regions, this is an approximation
    return Scalar[T](LLConstants.HELIPAD_Y)


@always_inline
fn compute_shaping[
    T: DType
](
    x_norm: Scalar[T],
    y_norm: Scalar[T],
    vx_norm: Scalar[T],
    vy_norm: Scalar[T],
    angle: Scalar[T],
    left_contact: Scalar[T],
    right_contact: Scalar[T],
) -> Scalar[T]:
    """Compute shaping potential for reward calculation.

    The shaping reward encourages:
    - Being close to landing pad (low distance)
    - Moving slowly (low speed)
    - Being upright (low angle)
    - Having legs in contact with ground

    Args:
        x_norm: Normalized x position.
        y_norm: Normalized y position.
        vx_norm: Normalized x velocity.
        vy_norm: Normalized y velocity.
        angle: Angle in radians.
        left_contact: 1.0 if left leg touching, 0.0 otherwise.
        right_contact: 1.0 if right leg touching, 0.0 otherwise.

    Returns:
        Shaping potential value.
    """
    var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
    var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
    var abs_angle = angle if angle >= Scalar[T](0.0) else -angle

    return (
        Scalar[T](-100.0) * dist
        - Scalar[T](100.0) * speed
        - Scalar[T](100.0) * abs_angle
        + Scalar[T](10.0) * left_contact
        + Scalar[T](10.0) * right_contact
    )
