"""Sequential Impulse Solver for 3D Contact Constraints.

Implements position-based and velocity-based contact constraint solving
using the sequential impulse method.
"""

from math import sqrt
from math3d import Vec3
from ..constants import (
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    IDX_VX, IDX_VY, IDX_VZ,
    IDX_WX, IDX_WY, IDX_WZ,
    IDX_MASS, IDX_INV_MASS,
    IDX_IXX, IDX_IYY, IDX_IZZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    DEFAULT_BAUMGARTE_3D,
    DEFAULT_SLOP_3D,
    DEFAULT_FRICTION_3D,
)
from ..collision import Contact3D


fn _get_inverse_mass_matrix(
    state: List[Float64],
    body_idx: Int,
    point: Vec3,
    normal: Vec3,
) -> Float64:
    """Compute effective inverse mass for contact constraint.

    Returns: K = 1/m_a + 1/m_b + r_a x n * I_a^-1 * r_a x n + r_b x n * I_b^-1 * r_b x n

    Args:
        state: Physics state array.
        body_idx: Index of body A (body B is -1 for ground).
        point: Contact point (world space).
        normal: Contact normal.

    Returns:
        Effective inverse mass for normal direction.
    """
    var base = body_idx * BODY_STATE_SIZE_3D

    var inv_mass = state[base + IDX_INV_MASS]

    # Get body position
    var pos = Vec3(
        state[base + IDX_PX],
        state[base + IDX_PY],
        state[base + IDX_PZ],
    )

    # r = contact point - body center
    var r = point - pos

    # r x n
    var r_cross_n = r.cross(normal)

    # Get inverse inertia (diagonal approximation)
    var inv_ixx = 1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    var inv_iyy = 1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    var inv_izz = 1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0

    # K contribution from rotation: (r x n) * I^-1 * (r x n)
    var rot_contrib = (
        r_cross_n.x * r_cross_n.x * inv_ixx
        + r_cross_n.y * r_cross_n.y * inv_iyy
        + r_cross_n.z * r_cross_n.z * inv_izz
    )

    return inv_mass + rot_contrib


fn _get_velocity_at_point(
    state: List[Float64],
    body_idx: Int,
    point: Vec3,
) -> Vec3:
    """Get velocity of body at contact point."""
    if body_idx < 0:
        return Vec3.zero()

    var base = body_idx * BODY_STATE_SIZE_3D

    var pos = Vec3(
        state[base + IDX_PX],
        state[base + IDX_PY],
        state[base + IDX_PZ],
    )
    var vel = Vec3(
        state[base + IDX_VX],
        state[base + IDX_VY],
        state[base + IDX_VZ],
    )
    var omega = Vec3(
        state[base + IDX_WX],
        state[base + IDX_WY],
        state[base + IDX_WZ],
    )

    var r = point - pos
    return vel + omega.cross(r)


fn _apply_impulse(
    mut state: List[Float64],
    body_idx: Int,
    impulse: Vec3,
    point: Vec3,
):
    """Apply impulse to body at contact point."""
    if body_idx < 0:
        return

    var base = body_idx * BODY_STATE_SIZE_3D

    # Check if dynamic
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
    var r = point - pos
    var torque_impulse = r.cross(impulse)

    var inv_ixx = 1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    var inv_iyy = 1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    var inv_izz = 1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0

    state[base + IDX_WX] += torque_impulse.x * inv_ixx
    state[base + IDX_WY] += torque_impulse.y * inv_iyy
    state[base + IDX_WZ] += torque_impulse.z * inv_izz


fn solve_contact_velocity(
    mut state: List[Float64],
    mut contact: Contact3D,
    friction: Float64 = DEFAULT_FRICTION_3D,
    restitution: Float64 = 0.0,
):
    """Solve contact velocity constraint using sequential impulses.

    Applies impulses to resolve penetrating contacts and friction.

    Args:
        state: Physics state array (modified in place).
        contact: Contact information (modified to store impulses).
        friction: Friction coefficient.
        restitution: Coefficient of restitution (bounciness).
    """
    if not contact.is_valid():
        return

    var body_a = contact.body_a
    var body_b = contact.body_b

    # Get velocities at contact point
    var vel_a = _get_velocity_at_point(state, body_a, contact.point)
    var vel_b = _get_velocity_at_point(state, body_b, contact.point)

    # Relative velocity
    var rel_vel = vel_a - vel_b

    # Normal component of relative velocity
    var vn = rel_vel.dot(contact.normal)

    # Skip if separating
    if vn > 0.0:
        return

    # Compute effective mass for normal direction
    var k_normal = _get_inverse_mass_matrix(state, body_a, contact.point, contact.normal)
    if body_b >= 0:
        k_normal += _get_inverse_mass_matrix(state, body_b, contact.point, contact.normal)

    if k_normal <= 0.0:
        return

    # Normal impulse
    var j_n = -(1.0 + restitution) * vn / k_normal

    # Clamp to be non-negative (push apart, not pull together)
    var old_impulse_n = contact.impulse_n
    contact.impulse_n = max(old_impulse_n + j_n, 0.0)
    j_n = contact.impulse_n - old_impulse_n

    # Apply normal impulse
    var normal_impulse = contact.normal * j_n
    _apply_impulse(state, body_a, normal_impulse, contact.point)
    _apply_impulse(state, body_b, normal_impulse * -1.0, contact.point)

    # --- Friction ---
    # Recompute relative velocity after normal impulse
    vel_a = _get_velocity_at_point(state, body_a, contact.point)
    vel_b = _get_velocity_at_point(state, body_b, contact.point)
    rel_vel = vel_a - vel_b

    # Tangential velocity
    var vt1 = rel_vel.dot(contact.tangent1)
    var vt2 = rel_vel.dot(contact.tangent2)

    # Effective mass for tangent directions
    var k_t1 = _get_inverse_mass_matrix(state, body_a, contact.point, contact.tangent1)
    var k_t2 = _get_inverse_mass_matrix(state, body_a, contact.point, contact.tangent2)
    if body_b >= 0:
        k_t1 += _get_inverse_mass_matrix(state, body_b, contact.point, contact.tangent1)
        k_t2 += _get_inverse_mass_matrix(state, body_b, contact.point, contact.tangent2)

    # Tangent impulses
    var j_t1 = -vt1 / k_t1 if k_t1 > 0.0 else 0.0
    var j_t2 = -vt2 / k_t2 if k_t2 > 0.0 else 0.0

    # Friction cone clamping (Coulomb friction)
    var max_friction = friction * contact.impulse_n

    var old_impulse_t1 = contact.impulse_t1
    var old_impulse_t2 = contact.impulse_t2

    contact.impulse_t1 += j_t1
    contact.impulse_t2 += j_t2

    # Clamp to friction cone
    var tangent_mag = sqrt(contact.impulse_t1 * contact.impulse_t1 + contact.impulse_t2 * contact.impulse_t2)
    if tangent_mag > max_friction and tangent_mag > 0.0:
        var scale = max_friction / tangent_mag
        contact.impulse_t1 *= scale
        contact.impulse_t2 *= scale

    j_t1 = contact.impulse_t1 - old_impulse_t1
    j_t2 = contact.impulse_t2 - old_impulse_t2

    # Apply friction impulses
    var friction_impulse = contact.tangent1 * j_t1 + contact.tangent2 * j_t2
    _apply_impulse(state, body_a, friction_impulse, contact.point)
    _apply_impulse(state, body_b, friction_impulse * -1.0, contact.point)


fn solve_contact_position(
    mut state: List[Float64],
    contact: Contact3D,
    baumgarte: Float64 = DEFAULT_BAUMGARTE_3D,
    slop: Float64 = DEFAULT_SLOP_3D,
):
    """Solve contact position constraint (Baumgarte stabilization).

    Pushes bodies apart to resolve penetration.

    Args:
        state: Physics state array (modified in place).
        contact: Contact information.
        baumgarte: Baumgarte stabilization factor (0-1).
        slop: Penetration slop (allows small overlap).
    """
    if not contact.is_valid():
        return

    # Only correct if penetration exceeds slop
    var correction_depth = contact.depth - slop
    if correction_depth <= 0.0:
        return

    var body_a = contact.body_a
    var body_b = contact.body_b

    # Compute effective mass (linear only for position correction)
    var inv_mass_a = 0.0
    var inv_mass_b = 0.0

    if body_a >= 0:
        var base_a = body_a * BODY_STATE_SIZE_3D
        inv_mass_a = state[base_a + IDX_INV_MASS]

    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D
        inv_mass_b = state[base_b + IDX_INV_MASS]

    var total_inv_mass = inv_mass_a + inv_mass_b
    if total_inv_mass <= 0.0:
        return

    # Position correction
    var correction = contact.normal * (baumgarte * correction_depth / total_inv_mass)

    # Apply position correction
    if body_a >= 0:
        var base_a = body_a * BODY_STATE_SIZE_3D
        state[base_a + IDX_PX] += correction.x * inv_mass_a
        state[base_a + IDX_PY] += correction.y * inv_mass_a
        state[base_a + IDX_PZ] += correction.z * inv_mass_a

    if body_b >= 0:
        var base_b = body_b * BODY_STATE_SIZE_3D
        state[base_b + IDX_PX] -= correction.x * inv_mass_b
        state[base_b + IDX_PY] -= correction.y * inv_mass_b
        state[base_b + IDX_PZ] -= correction.z * inv_mass_b


struct ContactSolver3D:
    """Contact constraint solver for 3D physics.

    Uses sequential impulses with warm starting for stable contact resolution.
    """

    var friction: Float64
    var restitution: Float64
    var baumgarte: Float64
    var slop: Float64
    var velocity_iterations: Int
    var position_iterations: Int

    fn __init__(
        out self,
        friction: Float64 = DEFAULT_FRICTION_3D,
        restitution: Float64 = 0.0,
        baumgarte: Float64 = DEFAULT_BAUMGARTE_3D,
        slop: Float64 = DEFAULT_SLOP_3D,
        velocity_iterations: Int = 10,
        position_iterations: Int = 5,
    ):
        """Initialize contact solver.

        Args:
            friction: Friction coefficient.
            restitution: Coefficient of restitution.
            baumgarte: Baumgarte stabilization factor.
            slop: Penetration slop.
            velocity_iterations: Number of velocity solver iterations.
            position_iterations: Number of position solver iterations.
        """
        self.friction = friction
        self.restitution = restitution
        self.baumgarte = baumgarte
        self.slop = slop
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    fn solve_velocities(
        self,
        mut state: List[Float64],
        mut contacts: List[Contact3D],
    ):
        """Solve velocity constraints for all contacts.

        Args:
            state: Physics state array.
            contacts: List of contacts (modified for warm starting).
        """
        for _ in range(self.velocity_iterations):
            for i in range(len(contacts)):
                solve_contact_velocity(
                    state, contacts[i], self.friction, self.restitution
                )

    fn solve_positions(
        self,
        mut state: List[Float64],
        contacts: List[Contact3D],
    ):
        """Solve position constraints for all contacts.

        Args:
            state: Physics state array.
            contacts: List of contacts.
        """
        for _ in range(self.position_iterations):
            for i in range(len(contacts)):
                solve_contact_position(
                    state, contacts[i], self.baumgarte, self.slop
                )

    fn solve(
        self,
        mut state: List[Float64],
        mut contacts: List[Contact3D],
    ):
        """Solve all contact constraints (velocity then position).

        Args:
            state: Physics state array.
            contacts: List of contacts.
        """
        self.solve_velocities(state, contacts)
        self.solve_positions(state, contacts)
