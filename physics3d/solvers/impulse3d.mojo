"""Sequential Impulse Solver for 3D Contact Constraints.

Implements position-based and velocity-based contact constraint solving
using the sequential impulse method.

GPU support follows the three-method hierarchy pattern for solving
contacts directly from flat buffer arrays.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from math3d import Vec3
from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE_3D,
    CONTACT_DATA_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    DEFAULT_BAUMGARTE_3D,
    DEFAULT_SLOP_3D,
    DEFAULT_FRICTION_3D,
    CONTACT_BODY_A_3D,
    CONTACT_BODY_B_3D,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_POINT_Z,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_NORMAL_Z,
    CONTACT_DEPTH_3D,
    CONTACT_IMPULSE_N,
    CONTACT_IMPULSE_T1,
    CONTACT_IMPULSE_T2,
    CONTACT_TANGENT1_X,
    CONTACT_TANGENT1_Y,
    CONTACT_TANGENT1_Z,
)
from ..collision import Contact3D


fn _get_inverse_mass_matrix[
    DTYPE: DType
](
    state: List[Scalar[DTYPE]],
    body_idx: Int,
    point: Vec3[DTYPE],
    normal: Vec3[DTYPE],
) -> Scalar[DTYPE]:
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
    var pos = Vec3[DTYPE](
        state[base + IDX_PX],
        state[base + IDX_PY],
        state[base + IDX_PZ],
    )

    # r = contact point - body center
    var r = point - pos

    # r x n
    var r_cross_n = r.cross(normal)

    # Get inverse inertia (diagonal approximation)
    var inv_ixx = (
        1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    )
    var inv_iyy = (
        1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    )
    var inv_izz = (
        1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0
    )

    # K contribution from rotation: (r x n) * I^-1 * (r x n)
    var rot_contrib = (
        r_cross_n.x * r_cross_n.x * inv_ixx
        + r_cross_n.y * r_cross_n.y * inv_iyy
        + r_cross_n.z * r_cross_n.z * inv_izz
    )

    return inv_mass + rot_contrib


fn _get_velocity_at_point[
    DTYPE: DType
](
    state: List[Scalar[DTYPE]],
    body_idx: Int,
    point: Vec3[DTYPE],
) -> Vec3[
    DTYPE
]:
    """Get velocity of body at contact point."""
    if body_idx < 0:
        return Vec3[DTYPE].zero()

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


fn _apply_impulse[
    DTYPE: DType
](
    mut state: List[Scalar[DTYPE]],
    body_idx: Int,
    impulse: Vec3[DTYPE],
    point: Vec3[DTYPE],
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

    var inv_ixx = (
        1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    )
    var inv_iyy = (
        1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    )
    var inv_izz = (
        1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0
    )

    state[base + IDX_WX] += torque_impulse.x * inv_ixx
    state[base + IDX_WY] += torque_impulse.y * inv_iyy
    state[base + IDX_WZ] += torque_impulse.z * inv_izz


fn solve_contact_velocity[
    DTYPE: DType
](
    mut state: List[Scalar[DTYPE]],
    mut contact: Contact3D[DTYPE],
    friction: Scalar[DTYPE] = Scalar[DTYPE](DEFAULT_FRICTION_3D),
    restitution: Scalar[DTYPE] = Scalar[DTYPE](0.0),
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
    var k_normal = _get_inverse_mass_matrix(
        state, body_a, contact.point, contact.normal
    )
    if body_b >= 0:
        k_normal += _get_inverse_mass_matrix(
            state, body_b, contact.point, contact.normal
        )

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
    var k_t1 = _get_inverse_mass_matrix(
        state, body_a, contact.point, contact.tangent1
    )
    var k_t2 = _get_inverse_mass_matrix(
        state, body_a, contact.point, contact.tangent2
    )
    if body_b >= 0:
        k_t1 += _get_inverse_mass_matrix(
            state, body_b, contact.point, contact.tangent1
        )
        k_t2 += _get_inverse_mass_matrix(
            state, body_b, contact.point, contact.tangent2
        )

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
    var tangent_mag = sqrt(
        contact.impulse_t1 * contact.impulse_t1
        + contact.impulse_t2 * contact.impulse_t2
    )
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


fn solve_contact_position[
    DTYPE: DType
](
    mut state: List[Scalar[DTYPE]],
    contact: Contact3D[DTYPE],
    baumgarte: Scalar[DTYPE] = Scalar[DTYPE](DEFAULT_BAUMGARTE_3D),
    slop: Scalar[DTYPE] = Scalar[DTYPE](DEFAULT_SLOP_3D),
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
    var inv_mass_a = Scalar[DTYPE](0.0)
    var inv_mass_b = Scalar[DTYPE](0.0)

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
    var correction = contact.normal * (
        baumgarte * correction_depth / total_inv_mass
    )

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

    var friction: Scalar[dtype]
    var restitution: Scalar[dtype]
    var baumgarte: Scalar[dtype]
    var slop: Scalar[dtype]
    var velocity_iterations: Int
    var position_iterations: Int

    fn __init__(
        out self,
        friction: Scalar[dtype] = Scalar[dtype](DEFAULT_FRICTION_3D),
        restitution: Scalar[dtype] = Scalar[dtype](0.0),
        baumgarte: Scalar[dtype] = Scalar[dtype](DEFAULT_BAUMGARTE_3D),
        slop: Scalar[dtype] = Scalar[dtype](DEFAULT_SLOP_3D),
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
        mut state: List[Scalar[dtype]],
        mut contacts: List[Contact3D[dtype]],
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
        mut state: List[Scalar[dtype]],
        contacts: List[Contact3D[dtype]],
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
        mut state: List[Scalar[dtype]],
        mut contacts: List[Contact3D[dtype]],
    ):
        """Solve all contact constraints (velocity then position).

        Args:
            state: Physics state array.
            contacts: List of contacts.
        """
        self.solve_velocities(state, contacts)
        self.solve_positions(state, contacts)


# =============================================================================
# GPU Implementation
# =============================================================================


struct ImpulseSolver3DGPU:
    """GPU-compatible Sequential Impulse Solver for 3D contacts.

    Solves contact constraints directly from flat state buffer arrays.
    Designed for use in fused physics kernels.
    """

    # =========================================================================
    # Single-Environment Methods (can be called from fused kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    fn solve_velocity_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_count: Int,
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
    ):
        """Solve velocity constraints for a single environment.

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE].
            contacts: Contact buffer [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D].
            contact_count: Number of valid contacts.
            friction: Friction coefficient.
            restitution: Restitution coefficient.
        """
        for c in range(contact_count):
            var body_a_idx = Int(contacts[env, c, CONTACT_BODY_A_3D])
            var body_b_idx = Int(contacts[env, c, CONTACT_BODY_B_3D])

            var body_a_off = BODIES_OFFSET + body_a_idx * BODY_STATE_SIZE_3D

            # Contact geometry
            var point_x = contacts[env, c, CONTACT_POINT_X]
            var point_y = contacts[env, c, CONTACT_POINT_Y]
            var point_z = contacts[env, c, CONTACT_POINT_Z]
            var normal_x = contacts[env, c, CONTACT_NORMAL_X]
            var normal_y = contacts[env, c, CONTACT_NORMAL_Y]
            var normal_z = contacts[env, c, CONTACT_NORMAL_Z]

            # Get body A state
            var pos_a_x = state[env, body_a_off + IDX_PX]
            var pos_a_y = state[env, body_a_off + IDX_PY]
            var pos_a_z = state[env, body_a_off + IDX_PZ]
            var vel_a_x = state[env, body_a_off + IDX_VX]
            var vel_a_y = state[env, body_a_off + IDX_VY]
            var vel_a_z = state[env, body_a_off + IDX_VZ]
            var omega_a_x = state[env, body_a_off + IDX_WX]
            var omega_a_y = state[env, body_a_off + IDX_WY]
            var omega_a_z = state[env, body_a_off + IDX_WZ]
            var inv_mass_a = state[env, body_a_off + IDX_INV_MASS]

            # Get inverse inertia for body A (diagonal)
            var ixx_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX])
            var iyy_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY])
            var izz_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ])
            var inv_ixx_a = Scalar[dtype](0)
            var inv_iyy_a = Scalar[dtype](0)
            var inv_izz_a = Scalar[dtype](0)
            if ixx_a > Scalar[dtype](1e-10):
                inv_ixx_a = Scalar[dtype](1.0) / ixx_a
            if iyy_a > Scalar[dtype](1e-10):
                inv_iyy_a = Scalar[dtype](1.0) / iyy_a
            if izz_a > Scalar[dtype](1e-10):
                inv_izz_a = Scalar[dtype](1.0) / izz_a

            # Ground properties (body_b_idx == -1)
            var inv_mass_b = Scalar[dtype](0)
            var vel_b_x = Scalar[dtype](0)
            var vel_b_y = Scalar[dtype](0)
            var vel_b_z = Scalar[dtype](0)
            var omega_b_x = Scalar[dtype](0)
            var omega_b_y = Scalar[dtype](0)
            var omega_b_z = Scalar[dtype](0)
            var pos_b_x = point_x
            var pos_b_y = point_y
            var pos_b_z = point_z
            var inv_ixx_b = Scalar[dtype](0)
            var inv_iyy_b = Scalar[dtype](0)
            var inv_izz_b = Scalar[dtype](0)
            var body_b_off = 0

            if body_b_idx >= 0:
                body_b_off = BODIES_OFFSET + body_b_idx * BODY_STATE_SIZE_3D
                pos_b_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PX])
                pos_b_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PY])
                pos_b_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PZ])
                vel_b_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_VX])
                vel_b_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_VY])
                vel_b_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_VZ])
                omega_b_x = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_WX]
                )
                omega_b_y = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_WY]
                )
                omega_b_z = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_WZ]
                )
                inv_mass_b = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_INV_MASS]
                )
                var ixx_b = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_IXX]
                )
                var iyy_b = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_IYY]
                )
                var izz_b = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_IZZ]
                )
                if ixx_b > Scalar[dtype](1e-10):
                    inv_ixx_b = Scalar[dtype](1.0) / ixx_b
                if iyy_b > Scalar[dtype](1e-10):
                    inv_iyy_b = Scalar[dtype](1.0) / iyy_b
                if izz_b > Scalar[dtype](1e-10):
                    inv_izz_b = Scalar[dtype](1.0) / izz_b

            # r vectors (contact point relative to body centers)
            var ra_x = point_x - pos_a_x
            var ra_y = point_y - pos_a_y
            var ra_z = point_z - pos_a_z
            var rb_x = point_x - pos_b_x
            var rb_y = point_y - pos_b_y
            var rb_z = point_z - pos_b_z

            # Velocity at contact points: v + omega x r
            # omega x r = (wy*rz - wz*ry, wz*rx - wx*rz, wx*ry - wy*rx)
            var vel_at_a_x = vel_a_x + (omega_a_y * ra_z - omega_a_z * ra_y)
            var vel_at_a_y = vel_a_y + (omega_a_z * ra_x - omega_a_x * ra_z)
            var vel_at_a_z = vel_a_z + (omega_a_x * ra_y - omega_a_y * ra_x)
            var vel_at_b_x = vel_b_x + (omega_b_y * rb_z - omega_b_z * rb_y)
            var vel_at_b_y = vel_b_y + (omega_b_z * rb_x - omega_b_x * rb_z)
            var vel_at_b_z = vel_b_z + (omega_b_x * rb_y - omega_b_y * rb_x)

            # Relative velocity
            var rel_vel_x = vel_at_a_x - vel_at_b_x
            var rel_vel_y = vel_at_a_y - vel_at_b_y
            var rel_vel_z = vel_at_a_z - vel_at_b_z

            # Normal component of relative velocity
            var vel_normal = (
                rel_vel_x * normal_x
                + rel_vel_y * normal_y
                + rel_vel_z * normal_z
            )

            # Only resolve if approaching
            if vel_normal < Scalar[dtype](0):
                # r x n
                var ra_cross_n_x = ra_y * normal_z - ra_z * normal_y
                var ra_cross_n_y = ra_z * normal_x - ra_x * normal_z
                var ra_cross_n_z = ra_x * normal_y - ra_y * normal_x
                var rb_cross_n_x = rb_y * normal_z - rb_z * normal_y
                var rb_cross_n_y = rb_z * normal_x - rb_x * normal_z
                var rb_cross_n_z = rb_x * normal_y - rb_y * normal_x

                # Effective mass: K = 1/m_a + 1/m_b + (r_a x n)·I_a^-1·(r_a x n) + ...
                var k = inv_mass_a + inv_mass_b
                k = k + ra_cross_n_x * ra_cross_n_x * inv_ixx_a
                k = k + ra_cross_n_y * ra_cross_n_y * inv_iyy_a
                k = k + ra_cross_n_z * ra_cross_n_z * inv_izz_a
                k = k + rb_cross_n_x * rb_cross_n_x * inv_ixx_b
                k = k + rb_cross_n_y * rb_cross_n_y * inv_iyy_b
                k = k + rb_cross_n_z * rb_cross_n_z * inv_izz_b

                if k <= Scalar[dtype](0):
                    continue

                # Normal impulse: j = -(1+e) * v_n / K
                var j_normal = (
                    -(Scalar[dtype](1) + restitution) * vel_normal / k
                )

                # Clamp accumulated impulse
                var old_impulse = contacts[env, c, CONTACT_IMPULSE_N]
                var new_impulse = old_impulse + j_normal
                if new_impulse < Scalar[dtype](0):
                    new_impulse = Scalar[dtype](0)
                contacts[env, c, CONTACT_IMPULSE_N] = new_impulse
                j_normal = new_impulse - old_impulse

                # Apply normal impulse
                var impulse_x = j_normal * normal_x
                var impulse_y = j_normal * normal_y
                var impulse_z = j_normal * normal_z

                # Linear velocity change: v += j/m
                state[env, body_a_off + IDX_VX] = (
                    vel_a_x + impulse_x * inv_mass_a
                )
                state[env, body_a_off + IDX_VY] = (
                    vel_a_y + impulse_y * inv_mass_a
                )
                state[env, body_a_off + IDX_VZ] = (
                    vel_a_z + impulse_z * inv_mass_a
                )

                # Angular velocity change: w += I^-1 * (r x j)
                var torque_a_x = ra_y * impulse_z - ra_z * impulse_y
                var torque_a_y = ra_z * impulse_x - ra_x * impulse_z
                var torque_a_z = ra_x * impulse_y - ra_y * impulse_x
                state[env, body_a_off + IDX_WX] = (
                    omega_a_x + torque_a_x * inv_ixx_a
                )
                state[env, body_a_off + IDX_WY] = (
                    omega_a_y + torque_a_y * inv_iyy_a
                )
                state[env, body_a_off + IDX_WZ] = (
                    omega_a_z + torque_a_z * inv_izz_a
                )

                if body_b_idx >= 0:
                    state[env, body_b_off + IDX_VX] = (
                        vel_b_x - impulse_x * inv_mass_b
                    )
                    state[env, body_b_off + IDX_VY] = (
                        vel_b_y - impulse_y * inv_mass_b
                    )
                    state[env, body_b_off + IDX_VZ] = (
                        vel_b_z - impulse_z * inv_mass_b
                    )
                    var torque_b_x = rb_y * impulse_z - rb_z * impulse_y
                    var torque_b_y = rb_z * impulse_x - rb_x * impulse_z
                    var torque_b_z = rb_x * impulse_y - rb_y * impulse_x
                    state[env, body_b_off + IDX_WX] = (
                        omega_b_x - torque_b_x * inv_ixx_b
                    )
                    state[env, body_b_off + IDX_WY] = (
                        omega_b_y - torque_b_y * inv_iyy_b
                    )
                    state[env, body_b_off + IDX_WZ] = (
                        omega_b_z - torque_b_z * inv_izz_b
                    )

                # ============================================================
                # Friction
                # ============================================================

                # Re-read velocities after normal impulse
                vel_a_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_VX])
                vel_a_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_VY])
                vel_a_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_VZ])
                omega_a_x = rebind[Scalar[dtype]](
                    state[env, body_a_off + IDX_WX]
                )
                omega_a_y = rebind[Scalar[dtype]](
                    state[env, body_a_off + IDX_WY]
                )
                omega_a_z = rebind[Scalar[dtype]](
                    state[env, body_a_off + IDX_WZ]
                )
                if body_b_idx >= 0:
                    vel_b_x = rebind[Scalar[dtype]](
                        state[env, body_b_off + IDX_VX]
                    )
                    vel_b_y = rebind[Scalar[dtype]](
                        state[env, body_b_off + IDX_VY]
                    )
                    vel_b_z = rebind[Scalar[dtype]](
                        state[env, body_b_off + IDX_VZ]
                    )
                    omega_b_x = rebind[Scalar[dtype]](
                        state[env, body_b_off + IDX_WX]
                    )
                    omega_b_y = rebind[Scalar[dtype]](
                        state[env, body_b_off + IDX_WY]
                    )
                    omega_b_z = rebind[Scalar[dtype]](
                        state[env, body_b_off + IDX_WZ]
                    )

                # Recompute relative velocity for friction
                vel_at_a_x = vel_a_x + (omega_a_y * ra_z - omega_a_z * ra_y)
                vel_at_a_y = vel_a_y + (omega_a_z * ra_x - omega_a_x * ra_z)
                vel_at_a_z = vel_a_z + (omega_a_x * ra_y - omega_a_y * ra_x)
                vel_at_b_x = vel_b_x + (omega_b_y * rb_z - omega_b_z * rb_y)
                vel_at_b_y = vel_b_y + (omega_b_z * rb_x - omega_b_x * rb_z)
                vel_at_b_z = vel_b_z + (omega_b_x * rb_y - omega_b_y * rb_x)
                rel_vel_x = vel_at_a_x - vel_at_b_x
                rel_vel_y = vel_at_a_y - vel_at_b_y
                rel_vel_z = vel_at_a_z - vel_at_b_z

                # Get tangent basis
                var tangent1_x = contacts[env, c, CONTACT_TANGENT1_X]
                var tangent1_y = contacts[env, c, CONTACT_TANGENT1_Y]
                var tangent1_z = contacts[env, c, CONTACT_TANGENT1_Z]

                # Compute tangent2 = normal x tangent1
                var tangent2_x = normal_y * tangent1_z - normal_z * tangent1_y
                var tangent2_y = normal_z * tangent1_x - normal_x * tangent1_z
                var tangent2_z = normal_x * tangent1_y - normal_y * tangent1_x

                # Tangential velocity components
                var vel_t1 = rebind[Scalar[dtype]](
                    rel_vel_x * tangent1_x
                    + rel_vel_y * tangent1_y
                    + rel_vel_z * tangent1_z
                )
                var vel_t2 = rebind[Scalar[dtype]](
                    rel_vel_x * tangent2_x
                    + rel_vel_y * tangent2_y
                    + rel_vel_z * tangent2_z
                )

                # Effective mass for tangent directions
                var ra_cross_t1_x = ra_y * tangent1_z - ra_z * tangent1_y
                var ra_cross_t1_y = ra_z * tangent1_x - ra_x * tangent1_z
                var ra_cross_t1_z = ra_x * tangent1_y - ra_y * tangent1_x

                var k_t1 = rebind[Scalar[dtype]](inv_mass_a + inv_mass_b)
                k_t1 = k_t1 + rebind[Scalar[dtype]](
                    ra_cross_t1_x * ra_cross_t1_x * inv_ixx_a
                )
                k_t1 = k_t1 + rebind[Scalar[dtype]](
                    ra_cross_t1_y * ra_cross_t1_y * inv_iyy_a
                )
                k_t1 = k_t1 + rebind[Scalar[dtype]](
                    ra_cross_t1_z * ra_cross_t1_z * inv_izz_a
                )
                if body_b_idx >= 0:
                    var rb_cross_t1_x = rb_y * tangent1_z - rb_z * tangent1_y
                    var rb_cross_t1_y = rb_z * tangent1_x - rb_x * tangent1_z
                    var rb_cross_t1_z = rb_x * tangent1_y - rb_y * tangent1_x
                    k_t1 = k_t1 + rebind[Scalar[dtype]](
                        rb_cross_t1_x * rb_cross_t1_x * inv_ixx_b
                    )
                    k_t1 = k_t1 + rebind[Scalar[dtype]](
                        rb_cross_t1_y * rb_cross_t1_y * inv_iyy_b
                    )
                    k_t1 = k_t1 + rebind[Scalar[dtype]](
                        rb_cross_t1_z * rb_cross_t1_z * inv_izz_b
                    )

                var ra_cross_t2_x = ra_y * tangent2_z - ra_z * tangent2_y
                var ra_cross_t2_y = ra_z * tangent2_x - ra_x * tangent2_z
                var ra_cross_t2_z = ra_x * tangent2_y - ra_y * tangent2_x

                var k_t2 = rebind[Scalar[dtype]](inv_mass_a + inv_mass_b)
                k_t2 = k_t2 + rebind[Scalar[dtype]](
                    ra_cross_t2_x * ra_cross_t2_x * inv_ixx_a
                )
                k_t2 = k_t2 + rebind[Scalar[dtype]](
                    ra_cross_t2_y * ra_cross_t2_y * inv_iyy_a
                )
                k_t2 = k_t2 + rebind[Scalar[dtype]](
                    ra_cross_t2_z * ra_cross_t2_z * inv_izz_a
                )
                if body_b_idx >= 0:
                    var rb_cross_t2_x = rb_y * tangent2_z - rb_z * tangent2_y
                    var rb_cross_t2_y = rb_z * tangent2_x - rb_x * tangent2_z
                    var rb_cross_t2_z = rb_x * tangent2_y - rb_y * tangent2_x
                    k_t2 = k_t2 + rebind[Scalar[dtype]](
                        rb_cross_t2_x * rb_cross_t2_x * inv_ixx_b
                    )
                    k_t2 = k_t2 + rebind[Scalar[dtype]](
                        rb_cross_t2_y * rb_cross_t2_y * inv_iyy_b
                    )
                    k_t2 = k_t2 + rebind[Scalar[dtype]](
                        rb_cross_t2_z * rb_cross_t2_z * inv_izz_b
                    )

                # Friction impulses
                var j_t1 = Scalar[dtype](0)
                var j_t2 = Scalar[dtype](0)
                if k_t1 > Scalar[dtype](0):
                    j_t1 = -vel_t1 / k_t1
                if k_t2 > Scalar[dtype](0):
                    j_t2 = -vel_t2 / k_t2

                # Friction cone clamping
                var max_friction = (
                    friction * contacts[env, c, CONTACT_IMPULSE_N]
                )
                var old_t1 = contacts[env, c, CONTACT_IMPULSE_T1]
                var old_t2 = contacts[env, c, CONTACT_IMPULSE_T2]
                var new_t1 = old_t1 + j_t1
                var new_t2 = old_t2 + j_t2

                var friction_mag = sqrt(new_t1 * new_t1 + new_t2 * new_t2)
                if friction_mag > max_friction and friction_mag > Scalar[dtype](
                    0
                ):
                    var scale = max_friction / friction_mag
                    new_t1 = new_t1 * scale
                    new_t2 = new_t2 * scale

                contacts[env, c, CONTACT_IMPULSE_T1] = new_t1
                contacts[env, c, CONTACT_IMPULSE_T2] = new_t2
                j_t1 = rebind[Scalar[dtype]](new_t1 - old_t1)
                j_t2 = rebind[Scalar[dtype]](new_t2 - old_t2)

                # Apply friction impulse
                var friction_impulse_x = j_t1 * tangent1_x + j_t2 * tangent2_x
                var friction_impulse_y = j_t1 * tangent1_y + j_t2 * tangent2_y
                var friction_impulse_z = j_t1 * tangent1_z + j_t2 * tangent2_z

                state[env, body_a_off + IDX_VX] = (
                    state[env, body_a_off + IDX_VX]
                    + friction_impulse_x * inv_mass_a
                )
                state[env, body_a_off + IDX_VY] = (
                    state[env, body_a_off + IDX_VY]
                    + friction_impulse_y * inv_mass_a
                )
                state[env, body_a_off + IDX_VZ] = (
                    state[env, body_a_off + IDX_VZ]
                    + friction_impulse_z * inv_mass_a
                )

                var friction_torque_a_x = (
                    ra_y * friction_impulse_z - ra_z * friction_impulse_y
                )
                var friction_torque_a_y = (
                    ra_z * friction_impulse_x - ra_x * friction_impulse_z
                )
                var friction_torque_a_z = (
                    ra_x * friction_impulse_y - ra_y * friction_impulse_x
                )
                state[env, body_a_off + IDX_WX] = (
                    state[env, body_a_off + IDX_WX]
                    + friction_torque_a_x * inv_ixx_a
                )
                state[env, body_a_off + IDX_WY] = (
                    state[env, body_a_off + IDX_WY]
                    + friction_torque_a_y * inv_iyy_a
                )
                state[env, body_a_off + IDX_WZ] = (
                    state[env, body_a_off + IDX_WZ]
                    + friction_torque_a_z * inv_izz_a
                )

                if body_b_idx >= 0:
                    state[env, body_b_off + IDX_VX] = (
                        state[env, body_b_off + IDX_VX]
                        - friction_impulse_x * inv_mass_b
                    )
                    state[env, body_b_off + IDX_VY] = (
                        state[env, body_b_off + IDX_VY]
                        - friction_impulse_y * inv_mass_b
                    )
                    state[env, body_b_off + IDX_VZ] = (
                        state[env, body_b_off + IDX_VZ]
                        - friction_impulse_z * inv_mass_b
                    )
                    var friction_torque_b_x = (
                        rb_y * friction_impulse_z - rb_z * friction_impulse_y
                    )
                    var friction_torque_b_y = (
                        rb_z * friction_impulse_x - rb_x * friction_impulse_z
                    )
                    var friction_torque_b_z = (
                        rb_x * friction_impulse_y - rb_y * friction_impulse_x
                    )
                    state[env, body_b_off + IDX_WX] = (
                        state[env, body_b_off + IDX_WX]
                        - friction_torque_b_x * inv_ixx_b
                    )
                    state[env, body_b_off + IDX_WY] = (
                        state[env, body_b_off + IDX_WY]
                        - friction_torque_b_y * inv_iyy_b
                    )
                    state[env, body_b_off + IDX_WZ] = (
                        state[env, body_b_off + IDX_WZ]
                        - friction_torque_b_z * inv_izz_b
                    )

    @always_inline
    @staticmethod
    fn solve_position_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_count: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints for a single environment.

        Applies Baumgarte stabilization to push penetrating bodies apart.

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE].
            contacts: Contact buffer [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D].
            contact_count: Number of valid contacts.
            baumgarte: Position correction factor (0.1-0.3 typical).
            slop: Penetration allowance before correction.
        """
        for c in range(contact_count):
            var body_a_idx = Int(contacts[env, c, CONTACT_BODY_A_3D])
            var body_b_idx = Int(contacts[env, c, CONTACT_BODY_B_3D])

            var body_a_off = BODIES_OFFSET + body_a_idx * BODY_STATE_SIZE_3D

            var normal_x = contacts[env, c, CONTACT_NORMAL_X]
            var normal_y = contacts[env, c, CONTACT_NORMAL_Y]
            var normal_z = contacts[env, c, CONTACT_NORMAL_Z]
            var penetration = contacts[env, c, CONTACT_DEPTH_3D]

            # Skip if within slop
            var correction = penetration - slop
            if correction <= Scalar[dtype](0):
                continue

            correction = baumgarte * correction

            var inv_mass_a = rebind[Scalar[dtype]](
                state[env, body_a_off + IDX_INV_MASS]
            )
            var inv_mass_b = Scalar[dtype](0)
            var body_b_off = 0
            if body_b_idx >= 0:
                body_b_off = BODIES_OFFSET + body_b_idx * BODY_STATE_SIZE_3D
                inv_mass_b = rebind[Scalar[dtype]](
                    state[env, body_b_off + IDX_INV_MASS]
                )

            var total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass <= Scalar[dtype](0):
                continue

            var correction_a = correction * inv_mass_a / total_inv_mass
            var correction_b = correction * inv_mass_b / total_inv_mass

            # Apply position correction
            state[env, body_a_off + IDX_PX] = (
                state[env, body_a_off + IDX_PX] + normal_x * correction_a
            )
            state[env, body_a_off + IDX_PY] = (
                state[env, body_a_off + IDX_PY] + normal_y * correction_a
            )
            state[env, body_a_off + IDX_PZ] = (
                state[env, body_a_off + IDX_PZ] + normal_z * correction_a
            )

            if body_b_idx >= 0:
                state[env, body_b_off + IDX_PX] = (
                    state[env, body_b_off + IDX_PX] - normal_x * correction_b
                )
                state[env, body_b_off + IDX_PY] = (
                    state[env, body_b_off + IDX_PY] - normal_y * correction_b
                )
                state[env, body_b_off + IDX_PZ] = (
                    state[env, body_b_off + IDX_PZ] - normal_z * correction_b
                )

    # =========================================================================
    # GPU Kernel Entry Points
    # =========================================================================

    @always_inline
    @staticmethod
    fn solve_velocity_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
    ):
        """GPU kernel for velocity constraint solving."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var count = Int(contact_counts[env])
        ImpulseSolver3DGPU.solve_velocity_single_env[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](env, state, contacts, count, friction, restitution)

    @always_inline
    @staticmethod
    fn solve_position_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel for position constraint solving."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var count = Int(contact_counts[env])
        ImpulseSolver3DGPU.solve_position_single_env[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](env, state, contacts, count, baumgarte, slop)

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn solve_velocity_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
    ) raises:
        """Launch velocity constraint solver kernel on GPU."""
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ](state_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
        ):
            ImpulseSolver3DGPU.solve_velocity_kernel[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](state, contacts, contact_counts, friction, restitution)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            contacts,
            contact_counts,
            friction,
            restitution,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn solve_position_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Launch position constraint solver kernel on GPU."""
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ](state_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            ImpulseSolver3DGPU.solve_position_kernel[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](state, contacts, contact_counts, baumgarte, slop)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            contacts,
            contact_counts,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
