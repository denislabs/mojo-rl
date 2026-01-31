"""Physics3D v2 Impulse-Based Solver (Bullet/Box2D style).

Sequential impulse solver with Split Impulse for position correction.

This uses the Split Impulse method (similar to Bullet Physics / Box2D):
- Velocity solver: Only handles velocity constraints (stopping/bouncing)
- Position solver: Uses pseudo-velocities that don't affect real velocities

This separation prevents position correction from adding energy to the system,
which is critical for stable stacking.

Reference: Erin Catto's GDC presentations on constraint solving.
"""

from ..types import Model, Data
from layout import LayoutTensor, Layout
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_IDX_INV_MASS,
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_VX,
    BODY_IDX_VY,
    BODY_IDX_VZ,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
    body_offset,
    contact_offset,
    metadata_offset,
)


fn solve_velocity_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    iterations: Int = 10,
):
    """Solve velocity constraints using sequential impulses.

    Only handles velocity-level constraints - making sure bodies don't
    interpenetrate faster. Does NOT add position correction bias here.

    Args:
        model: Static model configuration.
        data: Mutable simulation state.
        iterations: Number of solver iterations.
    """
    # Reset accumulated impulses at start of solve
    for c in range(data.num_contacts):
        data.contacts[c].impulse_n = Scalar[DTYPE](0)

    for _ in range(iterations):
        for c in range(data.num_contacts):
            _solve_single_contact_velocity(model, data, c)


fn _is_grounded[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS], body_idx: Int,) -> Bool:
    """Check if a body has ground contact."""
    for c in range(data.num_contacts):
        if (
            data.contacts[c].body_a == body_idx
            and data.contacts[c].body_b == -1
        ):
            return True
    return False


fn _solve_single_contact_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    contact_idx: Int,
):
    """Solve velocity constraint for one contact.

    Goal: Ensure relative normal velocity is non-negative (not approaching).
    For impacts, apply restitution to create bounce.
    """
    var contact = data.contacts[contact_idx]
    var body_a = contact.body_a
    var body_b = contact.body_b

    # Get contact normal (points from A to B)
    var nx = contact.normal_x
    var ny = contact.normal_y
    var nz = contact.normal_z

    # Get velocities
    var va_x = data.velocities[body_a * 3 + 0]
    var va_y = data.velocities[body_a * 3 + 1]
    var va_z = data.velocities[body_a * 3 + 2]

    var vb_x: Scalar[DTYPE]
    var vb_y: Scalar[DTYPE]
    var vb_z: Scalar[DTYPE]
    if body_b >= 0:
        vb_x = data.velocities[body_b * 3 + 0]
        vb_y = data.velocities[body_b * 3 + 1]
        vb_z = data.velocities[body_b * 3 + 2]
    else:
        vb_x = Scalar[DTYPE](0)
        vb_y = Scalar[DTYPE](0)
        vb_z = Scalar[DTYPE](0)

    # Relative velocity along normal
    # Convention for sphere-sphere: rel_vn = (v_A - v_B) · n where n points from A to B
    # rel_vn > 0 means A is moving toward B = APPROACHING
    #
    # For ground contacts: normal points UP, so we need to flip the sign
    # - Sphere falling (va_z < 0) should be "approaching" ground
    # - Sphere bouncing up (va_z > 0) should be "separating"
    var rel_vn = (va_x - vb_x) * nx + (va_y - vb_y) * ny + (va_z - vb_z) * nz

    # For ground contacts, flip sign so rel_vn > 0 means approaching ground (falling)
    if body_b < 0:
        rel_vn = -rel_vn

    # Effective mass (for point masses without rotation)
    var inv_mass_a = model.inv_masses[body_a]
    var inv_mass_b: Scalar[DTYPE]
    if body_b >= 0:
        inv_mass_b = model.inv_masses[body_b]
        # For sphere-sphere: only treat ONE as grounded to avoid K=0 deadlock
        # This allows grounded spheres to still push each other apart
        var a_grounded = _is_grounded(data, body_a)
        var b_grounded = _is_grounded(data, body_b)
        if a_grounded and not b_grounded:
            # Only A is grounded - treat it as immovable
            inv_mass_a = Scalar[DTYPE](0)
        elif b_grounded and not a_grounded:
            # Only B is grounded - treat it as immovable
            inv_mass_b = Scalar[DTYPE](0)
        # If both are grounded, use normal masses so they can push each other
    else:
        inv_mass_b = Scalar[DTYPE](0)

    var K = inv_mass_a + inv_mass_b
    if K < Scalar[DTYPE](1e-10):
        return

    # Only process if approaching (rel_vn > 0)
    if rel_vn <= Scalar[DTYPE](0):
        return  # Already separating, no impulse needed

    # Velocity-dependent restitution: no bounce for slow impacts (resting contact)
    # Use lower threshold for sphere-sphere to allow horizontal bouncing
    var restitution = model.restitution
    var vel_threshold: Scalar[DTYPE]
    if body_b >= 0:
        # Sphere-sphere: lower threshold to allow more bouncing
        vel_threshold = Scalar[DTYPE](0.1)
    else:
        # Ground contact: higher threshold for stable resting
        vel_threshold = Scalar[DTYPE](0.5)
    if rel_vn < vel_threshold:
        restitution = Scalar[DTYPE](0)

    # Target: stop (rel_vn = 0) or bounce (rel_vn = -e * current)
    # Since rel_vn > 0 means approaching, we want to reverse it
    var target_vn = -restitution * rel_vn

    # Impulse needed to change velocity from rel_vn to target_vn
    # j = (rel_vn - target_vn) / K (positive for approaching contact)
    # This is positive because rel_vn > 0 and target_vn <= 0
    var delta_j = (rel_vn - target_vn) / K

    # Accumulated impulse clamping (total impulse must be >= 0)
    var old_impulse = data.contacts[contact_idx].impulse_n
    var new_impulse = max(old_impulse + delta_j, Scalar[DTYPE](0))
    delta_j = new_impulse - old_impulse
    data.contacts[contact_idx].impulse_n = new_impulse

    # Apply impulse to velocities
    # For sphere-sphere: Body A receives impulse in -normal direction (pushed back)
    #                    Body B receives impulse in +normal direction (pushed forward)
    # For ground contacts: We flipped rel_vn, so we need to flip the impulse direction
    #                     Sphere should be pushed UP (+normal direction)
    if body_b < 0:
        # Ground contact: push sphere in +normal direction (up)
        data.velocities[body_a * 3 + 0] += delta_j * nx * inv_mass_a
        data.velocities[body_a * 3 + 1] += delta_j * ny * inv_mass_a
        data.velocities[body_a * 3 + 2] += delta_j * nz * inv_mass_a
    else:
        # Sphere-sphere: standard convention
        data.velocities[body_a * 3 + 0] -= delta_j * nx * inv_mass_a
        data.velocities[body_a * 3 + 1] -= delta_j * ny * inv_mass_a
        data.velocities[body_a * 3 + 2] -= delta_j * nz * inv_mass_a

        data.velocities[body_b * 3 + 0] += delta_j * nx * inv_mass_b
        data.velocities[body_b * 3 + 1] += delta_j * ny * inv_mass_b
        data.velocities[body_b * 3 + 2] += delta_j * nz * inv_mass_b


fn solve_position_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    baumgarte: Scalar[DTYPE] = 0.2,
    slop: Scalar[DTYPE] = 0.001,
):
    """Direct position correction for penetration.

    Uses pseudo-positions (Split Impulse): corrects positions directly
    without affecting velocities. This prevents position correction
    from adding energy to the system.

    Normal conventions:
    - Ground contact: Normal points UP from ground, body A (sphere) should move UP
    - Sphere-sphere: Normal points from A to B

    Args:
        model: Static model configuration.
        data: Mutable simulation state.
        baumgarte: Correction factor (0.2-0.8 typical).
        slop: Allowed penetration to prevent jitter.
    """
    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body_a = contact.body_a
        var body_b = contact.body_b
        var dist = contact.dist  # Negative for penetration

        # Only correct if penetrating beyond slop
        var penetration = -dist - slop
        if penetration <= Scalar[DTYPE](0):
            continue

        var nx = contact.normal_x
        var ny = contact.normal_y
        var nz = contact.normal_z

        var inv_mass_a = model.inv_masses[body_a]
        var inv_mass_b: Scalar[DTYPE]
        if body_b >= 0:
            inv_mass_b = model.inv_masses[body_b]
        else:
            inv_mass_b = Scalar[DTYPE](0)

        var total_inv_mass = inv_mass_a + inv_mass_b

        if total_inv_mass < Scalar[DTYPE](1e-10):
            continue

        # Calculate correction magnitude
        var correction = baumgarte * penetration

        # Distribute by mass ratio
        var ratio_a = inv_mass_a / total_inv_mass
        var ratio_b = inv_mass_b / total_inv_mass

        if body_b < 0:
            # Ground contact: normal points UP from ground surface
            # Sphere should move in +normal direction (UP) to exit ground
            data.positions[body_a * 3 + 0] += correction * nx
            data.positions[body_a * 3 + 1] += correction * ny
            data.positions[body_a * 3 + 2] += correction * nz
        else:
            # Sphere-sphere contact: normal points from A to B
            # To separate: A moves in -normal, B moves in +normal
            data.positions[body_a * 3 + 0] -= correction * nx * ratio_a
            data.positions[body_a * 3 + 1] -= correction * ny * ratio_a
            data.positions[body_a * 3 + 2] -= correction * nz * ratio_a

            data.positions[body_b * 3 + 0] += correction * nx * ratio_b
            data.positions[body_b * 3 + 1] += correction * ny * ratio_b
            data.positions[body_b * 3 + 2] += correction * nz * ratio_b


fn solve_resting_contacts[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
):
    """Apply gravity cancellation for bodies in resting contact.

    For stable stacking, bodies that are resting on support should have
    their gravity effectively cancelled by contact forces. This function
    detects which bodies are part of a resting stack and zeroes their
    downward velocities.

    This is a simplification of proper constraint force computation.
    """
    # Build support graph: which bodies are supporting which
    # A body is "supported" if it has a contact below it (normal.z > 0.5)
    var is_supported = InlineArray[Bool, NUM_BODIES](uninitialized=True)
    for i in range(NUM_BODIES):
        is_supported[i] = False

    # Check each contact
    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body_a = contact.body_a
        var body_b = contact.body_b

        # Normal points from A to B
        # If nz > 0.5, the contact is mostly vertical
        # Body A is "above" body B (or ground)
        if contact.normal_z > Scalar[DTYPE](0.5):
            is_supported[body_a] = True
        elif contact.normal_z < Scalar[DTYPE](-0.5):
            # Normal points down, so body B is above body A
            if body_b >= 0:
                is_supported[body_b] = True

    # For supported bodies with very low velocity, clamp downward velocity
    # Use a small threshold to allow bounces while preventing micro-drift
    for i in range(NUM_BODIES):
        if is_supported[i]:
            var vz = data.velocities[i * 3 + 2]
            # Only stop if moving VERY slowly (allows bounces)
            if vz > Scalar[DTYPE](-0.05) and vz < Scalar[DTYPE](0.05):
                data.velocities[i * 3 + 2] = Scalar[DTYPE](0)
                # Also zero out very small horizontal velocities
                var vx = data.velocities[i * 3 + 0]
                var vy = data.velocities[i * 3 + 1]
                if vx > Scalar[DTYPE](-0.02) and vx < Scalar[DTYPE](0.02):
                    data.velocities[i * 3 + 0] = Scalar[DTYPE](0)
                if vy > Scalar[DTYPE](-0.02) and vy < Scalar[DTYPE](0.02):
                    data.velocities[i * 3 + 1] = Scalar[DTYPE](0)


# =========================================================================
# Impulse Solver
# =========================================================================


@always_inline
fn solve_velocity_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    restitution: Scalar[DTYPE],
    iterations: Int,
):
    """Solve velocity constraints using sequential impulses."""
    var meta_off = metadata_offset[NUM_BODIES, MAX_CONTACTS]()
    var num_contacts = Int(
        rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
    )

    for _ in range(iterations):
        for c in range(num_contacts):
            var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS](c)
            var body_a = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A])
            )
            var body_b = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B])
            )

            var nx = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NX])
            var ny = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NY])
            var nz = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NZ])

            # Get velocities
            var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS](body_a)
            var vx_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VX])
            var vy_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VY])
            var vz_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VZ])

            var vx_b: Scalar[DTYPE] = 0
            var vy_b: Scalar[DTYPE] = 0
            var vz_b: Scalar[DTYPE] = 0
            if body_b >= 0:
                var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS](body_b)
                vx_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VX])
                vy_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VY])
                vz_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VZ])

            # Relative velocity along normal
            # Convention: rel_vn = (v_a - v_b) · n where n points from A to B
            # rel_vn > 0 means A is moving toward B = APPROACHING
            var rel_vx = vx_a - vx_b
            var rel_vy = vy_a - vy_b
            var rel_vz = vz_a - vz_b
            var rel_vn = rel_vx * nx + rel_vy * ny + rel_vz * nz

            # For ground contacts (body_b < 0), normal points UP
            # Sphere falling has va_z < 0, so rel_vn = va_z * 1 < 0
            # But we want rel_vn > 0 to mean "approaching", so flip for ground
            if body_b < 0:
                rel_vn = -rel_vn

            # Only solve if approaching (rel_vn > 0)
            if rel_vn <= Scalar[DTYPE](0):
                continue

            # Compute effective mass
            var inv_mass_a = rebind[Scalar[DTYPE]](
                model[body_a, MODEL_IDX_INV_MASS]
            )
            var inv_mass_b: Scalar[DTYPE] = 0
            if body_b >= 0:
                inv_mass_b = rebind[Scalar[DTYPE]](
                    model[body_b, MODEL_IDX_INV_MASS]
                )
            var K = inv_mass_a + inv_mass_b

            # Target: stop (rel_vn = 0) or bounce (rel_vn = -e * current)
            # j = (rel_vn - target_vn) / K = (rel_vn + e*rel_vn) / K = (1+e)*rel_vn / K
            var j = (Scalar[DTYPE](1) + restitution) * rel_vn / K

            # Apply impulse to velocities
            # For sphere-sphere: Body A receives impulse in -normal direction
            #                    Body B receives impulse in +normal direction
            # For ground contacts: We flipped rel_vn, so flip impulse direction
            #                     Sphere should be pushed UP (+normal direction)
            if body_b < 0:
                # Ground contact: push sphere in +normal direction (up)
                state[env, b_off_a + BODY_IDX_VX] = vx_a + j * nx * inv_mass_a
                state[env, b_off_a + BODY_IDX_VY] = vy_a + j * ny * inv_mass_a
                state[env, b_off_a + BODY_IDX_VZ] = vz_a + j * nz * inv_mass_a
            else:
                # Sphere-sphere: A pushed back, B pushed forward
                state[env, b_off_a + BODY_IDX_VX] = vx_a - j * nx * inv_mass_a
                state[env, b_off_a + BODY_IDX_VY] = vy_a - j * ny * inv_mass_a
                state[env, b_off_a + BODY_IDX_VZ] = vz_a - j * nz * inv_mass_a

                var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS](body_b)
                state[env, b_off_b + BODY_IDX_VX] = vx_b + j * nx * inv_mass_b
                state[env, b_off_b + BODY_IDX_VY] = vy_b + j * ny * inv_mass_b
                state[env, b_off_b + BODY_IDX_VZ] = vz_b + j * nz * inv_mass_b


@always_inline
fn solve_position_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    baumgarte: Scalar[DTYPE],
    slop: Scalar[DTYPE],
):
    """Baumgarte position correction for penetration."""
    var meta_off = metadata_offset[NUM_BODIES, MAX_CONTACTS]()
    var num_contacts = Int(
        rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
    )

    for c in range(num_contacts):
        var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS](c)
        var body_a = Int(
            rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A])
        )
        var body_b = Int(
            rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B])
        )
        var dist = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_DIST])

        # Only correct if penetrating beyond slop
        var penetration = -dist - slop
        if penetration <= Scalar[DTYPE](0):
            continue

        var nx = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NX])
        var ny = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NY])
        var nz = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NZ])

        var inv_mass_a = rebind[Scalar[DTYPE]](
            model[body_a, MODEL_IDX_INV_MASS]
        )
        var inv_mass_b: Scalar[DTYPE] = 0
        if body_b >= 0:
            inv_mass_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_MASS]
            )
        var total_inv_mass = inv_mass_a + inv_mass_b

        var correction = baumgarte * penetration / total_inv_mass

        # Push bodies apart along normal
        # Normal conventions:
        # - Ground contact: Normal points UP, body A (sphere) should move UP (+normal)
        # - Sphere-sphere: Normal points from A to B, so A moves in -normal, B in +normal
        var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS](body_a)
        var px_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PX])
        var py_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PY])
        var pz_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PZ])

        if body_b < 0:
            # Ground contact: push sphere UP (+normal direction)
            state[env, b_off_a + BODY_IDX_PX] = px_a + correction * nx * inv_mass_a
            state[env, b_off_a + BODY_IDX_PY] = py_a + correction * ny * inv_mass_a
            state[env, b_off_a + BODY_IDX_PZ] = pz_a + correction * nz * inv_mass_a
        else:
            # Sphere-sphere: A moves in -normal, B moves in +normal
            var ratio_a = inv_mass_a / total_inv_mass
            var ratio_b = inv_mass_b / total_inv_mass
            var corr_a = correction * ratio_a * total_inv_mass
            var corr_b = correction * ratio_b * total_inv_mass

            state[env, b_off_a + BODY_IDX_PX] = px_a - corr_a * nx
            state[env, b_off_a + BODY_IDX_PY] = py_a - corr_a * ny
            state[env, b_off_a + BODY_IDX_PZ] = pz_a - corr_a * nz

            var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS](body_b)
            var px_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PX])
            var py_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PY])
            var pz_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PZ])
            state[env, b_off_b + BODY_IDX_PX] = px_b + corr_b * nx
            state[env, b_off_b + BODY_IDX_PY] = py_b + corr_b * ny
            state[env, b_off_b + BODY_IDX_PZ] = pz_b + corr_b * nz
