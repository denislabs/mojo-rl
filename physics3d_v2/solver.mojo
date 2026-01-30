"""Physics3D v2 constraint solver - Contact impulse resolution.

Phase 2: Simple impulse-based contact solver with Baumgarte stabilization.
"""

from .types import Model, Data


fn max_scalar[
    DTYPE: DType
](a: Scalar[DTYPE], b: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """Return the maximum of two scalars."""
    if a > b:
        return a
    return b


fn solve_contact[DTYPE: DType](model: Model[DTYPE], mut data: Data[DTYPE]):
    """Apply contact impulse to prevent penetration.

    Uses impulse-based collision response:
    1. If object is approaching ground (vn < 0), apply impulse
    2. Impulse magnitude: j = -(1+e) * m * vn
    3. Apply Baumgarte position correction for residual penetration

    Args:
        model: Static model with restitution coefficient.
        data: Mutable simulation state (velocities will be modified).
    """
    if not data.contact.active:
        return

    var m = model.body.mass

    # Velocity toward ground (negative = approaching)
    # Contact normal is [0, 0, 1], so vn = qvel[2]
    var vn = data.qvel[2]

    # Only apply impulse if approaching ground
    if vn < Scalar[DTYPE](0):
        # Impulse magnitude: j = -(1+e) * m * vn
        # For inelastic collision (e=0): j = -m * vn
        # For elastic collision (e=1): j = -2 * m * vn
        var e = model.restitution
        var j = -(Scalar[DTYPE](1) + e) * m * vn

        # Apply impulse: Δv = j/m (only in z direction for flat ground)
        data.qvel[2] += j / m

    # Position correction (Baumgarte stabilization)
    # Pushes the sphere out of the ground to prevent sinking
    # correction = max(depth - slop, 0) * beta
    var beta = Scalar[DTYPE](0.2)  # Correction factor (0.1-0.3 typical)
    var slop = Scalar[DTYPE](0.001)  # Allowed penetration to prevent jitter
    var depth = data.contact.depth
    var correction = max_scalar(depth - slop, Scalar[DTYPE](0)) * beta

    # Apply correction directly to position
    data.qpos[2] += correction
