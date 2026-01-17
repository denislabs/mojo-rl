"""Simple impulse-based constraint solver.

This solver resolves contact constraints using sequential impulses:
1. Compute relative velocity at contact point
2. Apply normal impulse to stop penetration
3. Apply friction impulse (Coulomb model)
4. Apply position correction (Baumgarte stabilization)

Matches Box2D's constraint solving approach for compatibility.
"""

from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer


from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    CONTACT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
)
from ..traits.solver import ConstraintSolver


struct ImpulseSolver(ConstraintSolver):
    """Simple impulse-based contact solver.

    Features:
    - Normal impulse (stops penetration)
    - Friction impulse (Coulomb model)
    - Position correction (Baumgarte stabilization)
    - Warm starting from previous frame

    This is a sequential impulse solver - each contact is solved in order,
    and the process is iterated multiple times for convergence.
    """

    comptime VELOCITY_ITERATIONS: Int = DEFAULT_VELOCITY_ITERATIONS
    comptime POSITION_ITERATIONS: Int = DEFAULT_POSITION_ITERATIONS

    var friction: Scalar[dtype]
    var restitution: Scalar[dtype]
    var baumgarte: Scalar[dtype]
    var slop: Scalar[dtype]

    fn __init__(
        out self,
        friction: Float64 = DEFAULT_FRICTION,
        restitution: Float64 = DEFAULT_RESTITUTION,
        baumgarte: Float64 = DEFAULT_BAUMGARTE,
        slop: Float64 = DEFAULT_SLOP,
    ):
        """Initialize solver with contact physics parameters.

        Args:
            friction: Coulomb friction coefficient.
            restitution: Bounce coefficient (0 = no bounce, 1 = perfect bounce).
            baumgarte: Position correction factor (0.1-0.3 typical).
            slop: Penetration allowance before correction.
        """
        self.friction = Scalar[dtype](friction)
        self.restitution = Scalar[dtype](restitution)
        self.baumgarte = Scalar[dtype](baumgarte)
        self.slop = Scalar[dtype](slop)

    # =========================================================================
    # CPU Implementation
    # =========================================================================

    fn solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Solve velocity constraints for one iteration."""
        for env in range(BATCH):
            var count = Int(contact_counts[env])

            for c in range(count):
                var body_a_idx = Int(contacts[env, c, CONTACT_BODY_A])
                var body_b_idx = Int(
                    contacts[env, c, CONTACT_BODY_B]
                )  # -1 for ground

                # Contact geometry
                var point_x = contacts[env, c, CONTACT_POINT_X]
                var point_y = contacts[env, c, CONTACT_POINT_Y]
                var normal_x = contacts[env, c, CONTACT_NORMAL_X]
                var normal_y = contacts[env, c, CONTACT_NORMAL_Y]

                # Get body A state
                var pos_a_x = bodies[env, body_a_idx, IDX_X]
                var pos_a_y = bodies[env, body_a_idx, IDX_Y]
                var vel_a_x = bodies[env, body_a_idx, IDX_VX]
                var vel_a_y = bodies[env, body_a_idx, IDX_VY]
                var omega_a = bodies[env, body_a_idx, IDX_OMEGA]
                var inv_mass_a = bodies[env, body_a_idx, IDX_INV_MASS]
                var inv_inertia_a = bodies[env, body_a_idx, IDX_INV_INERTIA]

                # Ground properties (body_b_idx == -1)
                var inv_mass_b = Scalar[dtype](0)
                var inv_inertia_b = Scalar[dtype](0)
                var vel_b_x = Scalar[dtype](0)
                var vel_b_y = Scalar[dtype](0)
                var omega_b = Scalar[dtype](0)
                var pos_b_x = point_x  # Contact point IS ground position
                var pos_b_y = point_y

                if body_b_idx >= 0:
                    pos_b_x = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_X]
                    )
                    pos_b_y = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_Y]
                    )
                    vel_b_x = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_VX]
                    )
                    vel_b_y = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_VY]
                    )
                    omega_b = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_OMEGA]
                    )
                    inv_mass_b = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_INV_MASS]
                    )
                    inv_inertia_b = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_INV_INERTIA]
                    )

                # Compute r vectors (contact point relative to body centers)
                var ra_x = point_x - pos_a_x
                var ra_y = point_y - pos_a_y
                var rb_x = point_x - pos_b_x
                var rb_y = point_y - pos_b_y

                # Compute velocity at contact points
                # v_at_contact = v + omega x r = (vx - omega*ry, vy + omega*rx)
                var vel_at_a_x = vel_a_x - omega_a * ra_y
                var vel_at_a_y = vel_a_y + omega_a * ra_x
                var vel_at_b_x = vel_b_x - omega_b * rb_y
                var vel_at_b_y = vel_b_y + omega_b * rb_x

                # Relative velocity
                var rel_vel_x = vel_at_a_x - vel_at_b_x
                var rel_vel_y = vel_at_a_y - vel_at_b_y

                # Normal component of relative velocity
                var vel_normal = rel_vel_x * normal_x + rel_vel_y * normal_y

                # Only resolve if objects are approaching
                if vel_normal < Scalar[dtype](0):
                    # Compute effective mass for normal impulse
                    # K = 1/m_a + 1/m_b + (r_a x n)^2 / I_a + (r_b x n)^2 / I_b
                    var ra_cross_n = ra_x * normal_y - ra_y * normal_x
                    var rb_cross_n = rb_x * normal_y - rb_y * normal_x

                    var k = inv_mass_a + inv_mass_b
                    k = k + inv_inertia_a * ra_cross_n * ra_cross_n
                    k = k + inv_inertia_b * rb_cross_n * rb_cross_n

                    # Normal impulse magnitude: j = -(1+e) * v_n / K
                    var j_normal = (
                        -(Scalar[dtype](1) + self.restitution) * vel_normal / k
                    )

                    # Clamp accumulated impulse (sequential impulse method)
                    var old_impulse = contacts[env, c, CONTACT_NORMAL_IMPULSE]
                    var new_impulse = old_impulse + j_normal
                    if new_impulse < Scalar[dtype](0):
                        new_impulse = Scalar[dtype](0)
                    contacts[env, c, CONTACT_NORMAL_IMPULSE] = new_impulse
                    j_normal = new_impulse - old_impulse

                    # Apply normal impulse
                    var impulse_x = j_normal * normal_x
                    var impulse_y = j_normal * normal_y

                    bodies[env, body_a_idx, IDX_VX] = (
                        vel_a_x + impulse_x * inv_mass_a
                    )
                    bodies[env, body_a_idx, IDX_VY] = (
                        vel_a_y + impulse_y * inv_mass_a
                    )
                    bodies[env, body_a_idx, IDX_OMEGA] = (
                        omega_a
                        + (ra_x * impulse_y - ra_y * impulse_x) * inv_inertia_a
                    )

                    if body_b_idx >= 0:
                        bodies[env, body_b_idx, IDX_VX] = (
                            vel_b_x - impulse_x * inv_mass_b
                        )
                        bodies[env, body_b_idx, IDX_VY] = (
                            vel_b_y - impulse_y * inv_mass_b
                        )
                        bodies[env, body_b_idx, IDX_OMEGA] = (
                            omega_b
                            - (rb_x * impulse_y - rb_y * impulse_x)
                            * inv_inertia_b
                        )

                    # Update velocities for friction calculation
                    vel_a_x = rebind[Scalar[dtype]](
                        bodies[env, body_a_idx, IDX_VX]
                    )
                    vel_a_y = rebind[Scalar[dtype]](
                        bodies[env, body_a_idx, IDX_VY]
                    )
                    omega_a = rebind[Scalar[dtype]](
                        bodies[env, body_a_idx, IDX_OMEGA]
                    )
                    if body_b_idx >= 0:
                        vel_b_x = rebind[Scalar[dtype]](
                            bodies[env, body_b_idx, IDX_VX]
                        )
                        vel_b_y = rebind[Scalar[dtype]](
                            bodies[env, body_b_idx, IDX_VY]
                        )
                        omega_b = rebind[Scalar[dtype]](
                            bodies[env, body_b_idx, IDX_OMEGA]
                        )

                    # Recompute relative velocity for friction
                    vel_at_a_x = vel_a_x - omega_a * ra_y
                    vel_at_a_y = vel_a_y + omega_a * ra_x
                    vel_at_b_x = vel_b_x - omega_b * rb_y
                    vel_at_b_y = vel_b_y + omega_b * rb_x
                    rel_vel_x = vel_at_a_x - vel_at_b_x
                    rel_vel_y = vel_at_a_y - vel_at_b_y

                    # Friction impulse (tangent direction)
                    var tangent_x = -normal_y
                    var tangent_y = normal_x
                    var vel_tangent = (
                        rel_vel_x * tangent_x + rel_vel_y * tangent_y
                    )

                    var ra_cross_t = ra_x * tangent_y - ra_y * tangent_x
                    var rb_cross_t = rb_x * tangent_y - rb_y * tangent_x
                    var k_t = inv_mass_a + inv_mass_b
                    k_t = k_t + inv_inertia_a * ra_cross_t * ra_cross_t
                    k_t = k_t + inv_inertia_b * rb_cross_t * rb_cross_t

                    var j_tangent = -vel_tangent / k_t

                    # Clamp by friction cone
                    var max_friction = (
                        self.friction * contacts[env, c, CONTACT_NORMAL_IMPULSE]
                    )
                    var old_tangent = contacts[env, c, CONTACT_TANGENT_IMPULSE]
                    var new_tangent = old_tangent + j_tangent
                    if new_tangent > max_friction:
                        new_tangent = max_friction
                    elif new_tangent < -max_friction:
                        new_tangent = -max_friction
                    contacts[env, c, CONTACT_TANGENT_IMPULSE] = new_tangent
                    j_tangent = new_tangent - old_tangent

                    # Apply friction impulse
                    var friction_x = j_tangent * tangent_x
                    var friction_y = j_tangent * tangent_y

                    bodies[env, body_a_idx, IDX_VX] = (
                        bodies[env, body_a_idx, IDX_VX]
                        + friction_x * inv_mass_a
                    )
                    bodies[env, body_a_idx, IDX_VY] = (
                        bodies[env, body_a_idx, IDX_VY]
                        + friction_y * inv_mass_a
                    )
                    bodies[env, body_a_idx, IDX_OMEGA] = (
                        bodies[env, body_a_idx, IDX_OMEGA]
                        + (ra_x * friction_y - ra_y * friction_x)
                        * inv_inertia_a
                    )

                    if body_b_idx >= 0:
                        bodies[env, body_b_idx, IDX_VX] = (
                            bodies[env, body_b_idx, IDX_VX]
                            - friction_x * inv_mass_b
                        )
                        bodies[env, body_b_idx, IDX_VY] = (
                            bodies[env, body_b_idx, IDX_VY]
                            - friction_y * inv_mass_b
                        )
                        bodies[env, body_b_idx, IDX_OMEGA] = (
                            bodies[env, body_b_idx, IDX_OMEGA]
                            - (rb_x * friction_y - rb_y * friction_x)
                            * inv_inertia_b
                        )

    fn solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Solve position constraints (push bodies apart)."""
        for env in range(BATCH):
            var count = Int(contact_counts[env])

            for c in range(count):
                var body_a_idx = Int(contacts[env, c, CONTACT_BODY_A])
                var body_b_idx = Int(contacts[env, c, CONTACT_BODY_B])

                var normal_x = contacts[env, c, CONTACT_NORMAL_X]
                var normal_y = contacts[env, c, CONTACT_NORMAL_Y]
                var penetration = contacts[env, c, CONTACT_DEPTH]

                # Skip if within slop
                var correction = penetration - self.slop
                if correction <= Scalar[dtype](0):
                    continue

                # Position correction
                correction = self.baumgarte * correction

                var inv_mass_a = rebind[Scalar[dtype]](
                    bodies[env, body_a_idx, IDX_INV_MASS]
                )
                var inv_mass_b = Scalar[dtype](0)
                if body_b_idx >= 0:
                    inv_mass_b = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_INV_MASS]
                    )

                var total_inv_mass = inv_mass_a + inv_mass_b
                if total_inv_mass == Scalar[dtype](0):
                    continue

                var correction_a = correction * inv_mass_a / total_inv_mass
                var correction_b = correction * inv_mass_b / total_inv_mass

                bodies[env, body_a_idx, IDX_X] = (
                    bodies[env, body_a_idx, IDX_X] + normal_x * correction_a
                )
                bodies[env, body_a_idx, IDX_Y] = (
                    bodies[env, body_a_idx, IDX_Y] + normal_y * correction_a
                )

                if body_b_idx >= 0:
                    bodies[env, body_b_idx, IDX_X] = (
                        bodies[env, body_b_idx, IDX_X] - normal_x * correction_b
                    )
                    bodies[env, body_b_idx, IDX_Y] = (
                        bodies[env, body_b_idx, IDX_Y] - normal_y * correction_b
                    )

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================

    @always_inline
    @staticmethod
    fn solve_velocity_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
    ):
        """GPU kernel: one thread per environment."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var count = Int(contact_counts[env])

        for c in range(count):
            var body_a_idx = Int(contacts[env, c, CONTACT_BODY_A])
            var body_b_idx = Int(contacts[env, c, CONTACT_BODY_B])

            var point_x = contacts[env, c, CONTACT_POINT_X]
            var point_y = contacts[env, c, CONTACT_POINT_Y]
            var normal_x = contacts[env, c, CONTACT_NORMAL_X]
            var normal_y = contacts[env, c, CONTACT_NORMAL_Y]

            var pos_a_x = bodies[env, body_a_idx, IDX_X]
            var pos_a_y = bodies[env, body_a_idx, IDX_Y]
            var vel_a_x = bodies[env, body_a_idx, IDX_VX]
            var vel_a_y = bodies[env, body_a_idx, IDX_VY]
            var omega_a = bodies[env, body_a_idx, IDX_OMEGA]
            var inv_mass_a = bodies[env, body_a_idx, IDX_INV_MASS]
            var inv_inertia_a = bodies[env, body_a_idx, IDX_INV_INERTIA]

            var inv_mass_b = Scalar[dtype](0)
            var inv_inertia_b = Scalar[dtype](0)
            var vel_b_x = Scalar[dtype](0)
            var vel_b_y = Scalar[dtype](0)
            var omega_b = Scalar[dtype](0)
            var pos_b_x = point_x
            var pos_b_y = point_y

            if body_b_idx >= 0:
                pos_b_x = rebind[Scalar[dtype]](bodies[env, body_b_idx, IDX_X])
                pos_b_y = rebind[Scalar[dtype]](bodies[env, body_b_idx, IDX_Y])
                vel_b_x = rebind[Scalar[dtype]](bodies[env, body_b_idx, IDX_VX])
                vel_b_y = rebind[Scalar[dtype]](bodies[env, body_b_idx, IDX_VY])
                omega_b = rebind[Scalar[dtype]](
                    bodies[env, body_b_idx, IDX_OMEGA]
                )
                inv_mass_b = rebind[Scalar[dtype]](
                    bodies[env, body_b_idx, IDX_INV_MASS]
                )
                inv_inertia_b = rebind[Scalar[dtype]](
                    bodies[env, body_b_idx, IDX_INV_INERTIA]
                )

            var ra_x = point_x - pos_a_x
            var ra_y = point_y - pos_a_y
            var rb_x = point_x - pos_b_x
            var rb_y = point_y - pos_b_y

            var vel_at_a_x = vel_a_x - omega_a * ra_y
            var vel_at_a_y = vel_a_y + omega_a * ra_x
            var vel_at_b_x = vel_b_x - omega_b * rb_y
            var vel_at_b_y = vel_b_y + omega_b * rb_x

            var rel_vel_x = vel_at_a_x - vel_at_b_x
            var rel_vel_y = vel_at_a_y - vel_at_b_y
            var vel_normal = rel_vel_x * normal_x + rel_vel_y * normal_y

            if vel_normal < Scalar[dtype](0):
                var ra_cross_n = ra_x * normal_y - ra_y * normal_x
                var rb_cross_n = rb_x * normal_y - rb_y * normal_x

                var k = inv_mass_a + inv_mass_b
                k = k + inv_inertia_a * ra_cross_n * ra_cross_n
                k = k + inv_inertia_b * rb_cross_n * rb_cross_n

                var j_normal = (
                    -(Scalar[dtype](1) + restitution) * vel_normal / k
                )

                var old_impulse = contacts[env, c, CONTACT_NORMAL_IMPULSE]
                var new_impulse = old_impulse + j_normal
                if new_impulse < Scalar[dtype](0):
                    new_impulse = Scalar[dtype](0)
                contacts[env, c, CONTACT_NORMAL_IMPULSE] = new_impulse
                j_normal = new_impulse - old_impulse

                var impulse_x = j_normal * normal_x
                var impulse_y = j_normal * normal_y

                bodies[env, body_a_idx, IDX_VX] = (
                    vel_a_x + impulse_x * inv_mass_a
                )
                bodies[env, body_a_idx, IDX_VY] = (
                    vel_a_y + impulse_y * inv_mass_a
                )
                bodies[env, body_a_idx, IDX_OMEGA] = (
                    omega_a
                    + (ra_x * impulse_y - ra_y * impulse_x) * inv_inertia_a
                )

                if body_b_idx >= 0:
                    bodies[env, body_b_idx, IDX_VX] = (
                        vel_b_x - impulse_x * inv_mass_b
                    )
                    bodies[env, body_b_idx, IDX_VY] = (
                        vel_b_y - impulse_y * inv_mass_b
                    )
                    bodies[env, body_b_idx, IDX_OMEGA] = (
                        omega_b
                        - (rb_x * impulse_y - rb_y * impulse_x) * inv_inertia_b
                    )

                # Update velocities for friction calculation
                vel_a_x = rebind[Scalar[dtype]](bodies[env, body_a_idx, IDX_VX])
                vel_a_y = rebind[Scalar[dtype]](bodies[env, body_a_idx, IDX_VY])
                omega_a = rebind[Scalar[dtype]](
                    bodies[env, body_a_idx, IDX_OMEGA]
                )
                if body_b_idx >= 0:
                    vel_b_x = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_VX]
                    )
                    vel_b_y = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_VY]
                    )
                    omega_b = rebind[Scalar[dtype]](
                        bodies[env, body_b_idx, IDX_OMEGA]
                    )

                # Recompute relative velocity for friction
                vel_at_a_x = vel_a_x - omega_a * ra_y
                vel_at_a_y = vel_a_y + omega_a * ra_x
                vel_at_b_x = vel_b_x - omega_b * rb_y
                vel_at_b_y = vel_b_y + omega_b * rb_x
                rel_vel_x = vel_at_a_x - vel_at_b_x
                rel_vel_y = vel_at_a_y - vel_at_b_y

                # Friction impulse (tangent direction)
                var tangent_x = -normal_y
                var tangent_y = normal_x
                var vel_tangent = rel_vel_x * tangent_x + rel_vel_y * tangent_y

                var ra_cross_t = ra_x * tangent_y - ra_y * tangent_x
                var rb_cross_t = rb_x * tangent_y - rb_y * tangent_x
                var k_t = inv_mass_a + inv_mass_b
                k_t = k_t + inv_inertia_a * ra_cross_t * ra_cross_t
                k_t = k_t + inv_inertia_b * rb_cross_t * rb_cross_t

                var j_tangent = -vel_tangent / k_t

                # Clamp by friction cone
                var max_friction = (
                    friction * contacts[env, c, CONTACT_NORMAL_IMPULSE]
                )
                var old_tangent = contacts[env, c, CONTACT_TANGENT_IMPULSE]
                var new_tangent = old_tangent + j_tangent
                if new_tangent > max_friction:
                    new_tangent = max_friction
                elif new_tangent < -max_friction:
                    new_tangent = -max_friction
                contacts[env, c, CONTACT_TANGENT_IMPULSE] = new_tangent
                j_tangent = new_tangent - old_tangent

                # Apply friction impulse
                var friction_x = j_tangent * tangent_x
                var friction_y = j_tangent * tangent_y

                bodies[env, body_a_idx, IDX_VX] = (
                    bodies[env, body_a_idx, IDX_VX] + friction_x * inv_mass_a
                )
                bodies[env, body_a_idx, IDX_VY] = (
                    bodies[env, body_a_idx, IDX_VY] + friction_y * inv_mass_a
                )
                bodies[env, body_a_idx, IDX_OMEGA] = (
                    bodies[env, body_a_idx, IDX_OMEGA]
                    + (ra_x * friction_y - ra_y * friction_x) * inv_inertia_a
                )

                if body_b_idx >= 0:
                    bodies[env, body_b_idx, IDX_VX] = (
                        bodies[env, body_b_idx, IDX_VX]
                        - friction_x * inv_mass_b
                    )
                    bodies[env, body_b_idx, IDX_VY] = (
                        bodies[env, body_b_idx, IDX_VY]
                        - friction_y * inv_mass_b
                    )
                    bodies[env, body_b_idx, IDX_OMEGA] = (
                        bodies[env, body_b_idx, IDX_OMEGA]
                        - (rb_x * friction_y - rb_y * friction_x)
                        * inv_inertia_b
                    )

    @always_inline
    @staticmethod
    fn solve_position_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel: one thread per environment."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var count = Int(contact_counts[env])

        for c in range(count):
            var body_a_idx = Int(contacts[env, c, CONTACT_BODY_A])
            var body_b_idx = Int(contacts[env, c, CONTACT_BODY_B])

            var normal_x = contacts[env, c, CONTACT_NORMAL_X]
            var normal_y = contacts[env, c, CONTACT_NORMAL_Y]
            var penetration = contacts[env, c, CONTACT_DEPTH]

            var correction = penetration - slop
            if correction <= Scalar[dtype](0):
                continue

            correction = baumgarte * correction

            var inv_mass_a = rebind[Scalar[dtype]](
                bodies[env, body_a_idx, IDX_INV_MASS]
            )
            var inv_mass_b = Scalar[dtype](0)
            if body_b_idx >= 0:
                inv_mass_b = rebind[Scalar[dtype]](
                    bodies[env, body_b_idx, IDX_INV_MASS]
                )

            var total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass == Scalar[dtype](0):
                continue

            var correction_a = correction * inv_mass_a / total_inv_mass
            var correction_b = correction * inv_mass_b / total_inv_mass

            bodies[env, body_a_idx, IDX_X] = (
                bodies[env, body_a_idx, IDX_X] + normal_x * correction_a
            )
            bodies[env, body_a_idx, IDX_Y] = (
                bodies[env, body_a_idx, IDX_Y] + normal_y * correction_a
            )

            if body_b_idx >= 0:
                bodies[env, body_b_idx, IDX_X] = (
                    bodies[env, body_b_idx, IDX_X] - normal_x * correction_b
                )
                bodies[env, body_b_idx, IDX_Y] = (
                    bodies[env, body_b_idx, IDX_Y] - normal_y * correction_b
                )

    # =========================================================================
    # GPU Kernel Launchers
    # =========================================================================

    @staticmethod
    fn solve_velocity_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
    ) raises:
        """Launch velocity constraint solver kernel."""
        var bodies = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            bodies: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
                MutAnyOrigin,
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
        ):
            ImpulseSolver.solve_velocity_kernel[
                BATCH, NUM_BODIES, MAX_CONTACTS
            ](bodies, contacts, contact_counts, friction, restitution)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            bodies,
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
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Launch position constraint solver kernel."""
        var bodies = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            bodies: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
                MutAnyOrigin,
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            ImpulseSolver.solve_position_kernel[
                BATCH, NUM_BODIES, MAX_CONTACTS
            ](bodies, contacts, contact_counts, baumgarte, slop)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            bodies,
            contacts,
            contact_counts,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
