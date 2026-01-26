"""Joint Constraint Solver for 3D Physics.

Implements velocity and position constraint solving for joint types
used in articulated bodies (hinge, ball joints).
"""

from math import sqrt, cos, sin
from math3d import Vec3, Quat
from ..constants import (
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    IDX_VX, IDX_VY, IDX_VZ,
    IDX_WX, IDX_WY, IDX_WZ,
    IDX_INV_MASS,
    IDX_IXX, IDX_IYY, IDX_IZZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    DEFAULT_BAUMGARTE_3D,
)


fn _get_world_anchor(
    state: List[Float64],
    body_idx: Int,
    local_anchor: Vec3,
) -> Vec3:
    """Transform local anchor point to world space.

    Args:
        state: Physics state array.
        body_idx: Body index.
        local_anchor: Anchor in body local space.

    Returns:
        Anchor in world space.
    """
    var base = body_idx * BODY_STATE_SIZE_3D

    var pos = Vec3(
        state[base + IDX_PX],
        state[base + IDX_PY],
        state[base + IDX_PZ],
    )

    var q = Quat(
        state[base + IDX_QW],
        state[base + IDX_QX],
        state[base + IDX_QY],
        state[base + IDX_QZ],
    )

    return pos + q.rotate_vec(local_anchor)


fn _get_world_axis(
    state: List[Float64],
    body_idx: Int,
    local_axis: Vec3,
) -> Vec3:
    """Transform local axis to world space.

    Args:
        state: Physics state array.
        body_idx: Body index.
        local_axis: Axis in body local space.

    Returns:
        Axis in world space (normalized).
    """
    var base = body_idx * BODY_STATE_SIZE_3D

    var q = Quat(
        state[base + IDX_QW],
        state[base + IDX_QX],
        state[base + IDX_QY],
        state[base + IDX_QZ],
    )

    return q.rotate_vec(local_axis).normalized()


fn _apply_impulse_at_point(
    mut state: List[Float64],
    body_idx: Int,
    impulse: Vec3,
    world_point: Vec3,
):
    """Apply impulse to body at world-space point."""
    if body_idx < 0:
        return

    var base = body_idx * BODY_STATE_SIZE_3D

    var body_type = Int(state[base + IDX_BODY_TYPE])
    if body_type != BODY_DYNAMIC:
        return

    var inv_mass = state[base + IDX_INV_MASS]

    # Linear velocity change
    state[base + IDX_VX] += impulse.x * inv_mass
    state[base + IDX_VY] += impulse.y * inv_mass
    state[base + IDX_VZ] += impulse.z * inv_mass

    # Angular velocity change
    var pos = Vec3(
        state[base + IDX_PX],
        state[base + IDX_PY],
        state[base + IDX_PZ],
    )
    var r = world_point - pos
    var torque = r.cross(impulse)

    var inv_ixx = 1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    var inv_iyy = 1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    var inv_izz = 1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0

    state[base + IDX_WX] += torque.x * inv_ixx
    state[base + IDX_WY] += torque.y * inv_iyy
    state[base + IDX_WZ] += torque.z * inv_izz


fn _apply_angular_impulse(
    mut state: List[Float64],
    body_idx: Int,
    angular_impulse: Vec3,
):
    """Apply pure angular impulse to body."""
    if body_idx < 0:
        return

    var base = body_idx * BODY_STATE_SIZE_3D

    var body_type = Int(state[base + IDX_BODY_TYPE])
    if body_type != BODY_DYNAMIC:
        return

    var inv_ixx = 1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    var inv_iyy = 1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    var inv_izz = 1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0

    state[base + IDX_WX] += angular_impulse.x * inv_ixx
    state[base + IDX_WY] += angular_impulse.y * inv_iyy
    state[base + IDX_WZ] += angular_impulse.z * inv_izz


fn solve_hinge_velocity(
    mut state: List[Float64],
    body_a: Int,
    body_b: Int,
    anchor_a: Vec3,
    anchor_b: Vec3,
    axis_a: Vec3,
):
    """Solve hinge joint velocity constraint.

    Constrains:
    1. Anchor points to coincide (ball joint constraint)
    2. Angular velocities to be equal except around hinge axis

    Args:
        state: Physics state array.
        body_a: First body index.
        body_b: Second body index (-1 for world).
        anchor_a: Local anchor on body A.
        anchor_b: Local anchor on body B.
        axis_a: Hinge axis in body A's local frame.
    """
    # Get world-space anchors
    var world_anchor_a = _get_world_anchor(state, body_a, anchor_a)
    var world_anchor_b = _get_world_anchor(state, body_b, anchor_b) if body_b >= 0 else anchor_b
    var world_axis = _get_world_axis(state, body_a, axis_a)

    # --- Point-to-point constraint (ball joint part) ---
    var base_a = body_a * BODY_STATE_SIZE_3D

    var vel_a = Vec3(
        state[base_a + IDX_VX],
        state[base_a + IDX_VY],
        state[base_a + IDX_VZ],
    )
    var omega_a = Vec3(
        state[base_a + IDX_WX],
        state[base_a + IDX_WY],
        state[base_a + IDX_WZ],
    )

    var pos_a = Vec3(
        state[base_a + IDX_PX],
        state[base_a + IDX_PY],
        state[base_a + IDX_PZ],
    )
    var r_a = world_anchor_a - pos_a

    # Velocity at anchor on A
    var vel_anchor_a = vel_a + omega_a.cross(r_a)

    var vel_anchor_b = Vec3.zero()
    var r_b = Vec3.zero()

    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D

        var vel_b = Vec3(
            state[base_b + IDX_VX],
            state[base_b + IDX_VY],
            state[base_b + IDX_VZ],
        )
        var omega_b = Vec3(
            state[base_b + IDX_WX],
            state[base_b + IDX_WY],
            state[base_b + IDX_WZ],
        )
        var pos_b = Vec3(
            state[base_b + IDX_PX],
            state[base_b + IDX_PY],
            state[base_b + IDX_PZ],
        )
        r_b = world_anchor_b - pos_b
        vel_anchor_b = vel_b + omega_b.cross(r_b)

    # Velocity error
    var dv = vel_anchor_a - vel_anchor_b

    # Compute effective mass (simplified: use inverse masses)
    var inv_mass_a = state[base_a + IDX_INV_MASS]
    var inv_mass_b = 0.0
    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D
        inv_mass_b = state[base_b + IDX_INV_MASS]

    var k = inv_mass_a + inv_mass_b
    if k <= 0.0:
        return

    # Simple impulse (without full K matrix)
    var impulse = dv * (-1.0 / k)

    # Apply impulse
    _apply_impulse_at_point(state, body_a, impulse, world_anchor_a)
    _apply_impulse_at_point(state, body_b, impulse * -1.0, world_anchor_b)

    # --- Angular constraint (restrict rotation to hinge axis) ---
    # Get angular velocities
    omega_a = Vec3(
        state[base_a + IDX_WX],
        state[base_a + IDX_WY],
        state[base_a + IDX_WZ],
    )

    var omega_b = Vec3.zero()
    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D
        omega_b = Vec3(
            state[base_b + IDX_WX],
            state[base_b + IDX_WY],
            state[base_b + IDX_WZ],
        )

    # Relative angular velocity
    var rel_omega = omega_a - omega_b

    # Component perpendicular to hinge axis (should be zero)
    var omega_along_axis = world_axis * rel_omega.dot(world_axis)
    var omega_perp = rel_omega - omega_along_axis

    # Create angular impulse to cancel perpendicular component
    if omega_perp.length_squared() > 1e-10:
        # Get inverse inertias
        var inv_i_a = Vec3(
            1.0 / state[base_a + IDX_IXX] if state[base_a + IDX_IXX] > 0.0 else 0.0,
            1.0 / state[base_a + IDX_IYY] if state[base_a + IDX_IYY] > 0.0 else 0.0,
            1.0 / state[base_a + IDX_IZZ] if state[base_a + IDX_IZZ] > 0.0 else 0.0,
        )

        var k_angular = inv_i_a.x + inv_i_a.y + inv_i_a.z

        if body_b >= 0:
            var base_b = body_b * BODY_STATE_SIZE_3D
            k_angular += 1.0 / state[base_b + IDX_IXX] if state[base_b + IDX_IXX] > 0.0 else 0.0
            k_angular += 1.0 / state[base_b + IDX_IYY] if state[base_b + IDX_IYY] > 0.0 else 0.0
            k_angular += 1.0 / state[base_b + IDX_IZZ] if state[base_b + IDX_IZZ] > 0.0 else 0.0

        if k_angular > 0.0:
            var angular_impulse = omega_perp * (-0.5 / k_angular)
            _apply_angular_impulse(state, body_a, angular_impulse)
            _apply_angular_impulse(state, body_b, angular_impulse * -1.0)


fn solve_hinge_position(
    mut state: List[Float64],
    body_a: Int,
    body_b: Int,
    anchor_a: Vec3,
    anchor_b: Vec3,
    baumgarte: Float64 = DEFAULT_BAUMGARTE_3D,
):
    """Solve hinge joint position constraint.

    Corrects anchor point drift.

    Args:
        state: Physics state array.
        body_a: First body index.
        body_b: Second body index (-1 for world).
        anchor_a: Local anchor on body A.
        anchor_b: Local anchor on body B (or world position if body_b=-1).
        baumgarte: Stabilization factor.
    """
    # Get world-space anchors
    var world_anchor_a = _get_world_anchor(state, body_a, anchor_a)
    var world_anchor_b = _get_world_anchor(state, body_b, anchor_b) if body_b >= 0 else anchor_b

    # Position error
    var error = world_anchor_a - world_anchor_b
    var error_len = error.length()

    if error_len < 1e-6:
        return

    var base_a = body_a * BODY_STATE_SIZE_3D

    var inv_mass_a = state[base_a + IDX_INV_MASS]
    var inv_mass_b = 0.0
    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D
        inv_mass_b = state[base_b + IDX_INV_MASS]

    var total_inv_mass = inv_mass_a + inv_mass_b
    if total_inv_mass <= 0.0:
        return

    # Position correction
    var correction = error * (-baumgarte / total_inv_mass)

    # Apply correction
    state[base_a + IDX_PX] += correction.x * inv_mass_a
    state[base_a + IDX_PY] += correction.y * inv_mass_a
    state[base_a + IDX_PZ] += correction.z * inv_mass_a

    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D
        state[base_b + IDX_PX] -= correction.x * inv_mass_b
        state[base_b + IDX_PY] -= correction.y * inv_mass_b
        state[base_b + IDX_PZ] -= correction.z * inv_mass_b


struct JointSolver3D:
    """Joint constraint solver for 3D articulated bodies."""

    var baumgarte: Float64
    var velocity_iterations: Int
    var position_iterations: Int

    fn __init__(
        out self,
        baumgarte: Float64 = DEFAULT_BAUMGARTE_3D,
        velocity_iterations: Int = 10,
        position_iterations: Int = 5,
    ):
        """Initialize joint solver.

        Args:
            baumgarte: Stabilization factor.
            velocity_iterations: Number of velocity iterations.
            position_iterations: Number of position iterations.
        """
        self.baumgarte = baumgarte
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    fn solve_hinge_velocities(
        self,
        mut state: List[Float64],
        body_a: Int,
        body_b: Int,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
    ):
        """Solve hinge joint velocity constraints.

        Args:
            state: Physics state array.
            body_a: First body index.
            body_b: Second body index.
            anchor_a: Local anchor on body A.
            anchor_b: Local anchor on body B.
            axis_a: Hinge axis in body A's frame.
        """
        for _ in range(self.velocity_iterations):
            solve_hinge_velocity(state, body_a, body_b, anchor_a, anchor_b, axis_a)

    fn solve_hinge_positions(
        self,
        mut state: List[Float64],
        body_a: Int,
        body_b: Int,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ):
        """Solve hinge joint position constraints.

        Args:
            state: Physics state array.
            body_a: First body index.
            body_b: Second body index.
            anchor_a: Local anchor on body A.
            anchor_b: Local anchor on body B.
        """
        for _ in range(self.position_iterations):
            solve_hinge_position(state, body_a, body_b, anchor_a, anchor_b, self.baumgarte)
