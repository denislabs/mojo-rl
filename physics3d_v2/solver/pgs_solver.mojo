"""Physics3D v2 Projected Gauss-Seidel Solver (MuJoCo-style).

This implements a PGS solver following MuJoCo's constraint formulation:

Key Design Decisions:
1. Ground contacts resolved FIRST to get proper grounded body velocities
2. Then sphere-sphere impacts handled with restitution
3. Finally PGS iterations for soft constraint resolution

MuJoCo's approach:
1. Contacts are soft constraints with spring-damper dynamics
2. solref = [timeconst, dampratio] controls stiffness
3. Reference acceleration: aref = -k * pos - b * vel
4. D (effective mass) includes impedance regularization
5. Constraint forces are clamped to >= 0 (unilateral)

Key equations from MuJoCo:
  k = 1 / (timeconst^2 * dampratio^2)
  b = 2 / timeconst
  aref = -k * penetration - b * velocity
  D = 1 / (invweight * (1 - imp) / imp)  (simplified: D = 1/invweight for imp=0.5)

Phase 6: Added Coulomb friction support.

Reference: MuJoCo Warp constraint.py and solver.py
"""

from math import sqrt
from ..types import Model, Data
from ..collision.collision_primitives import compute_tangent_basis
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

# =============================================================================
# MuJoCo-style Constraint Parameters
# =============================================================================

# Default solver reference parameters (MuJoCo style)
# solref = [timeconst, dampratio]
# - timeconst: time constant for the spring (smaller = stiffer)
# - dampratio: damping ratio (1.0 = critical damping)

comptime DEFAULT_TIMECONST: Float64 = 0.02  # 20ms - fairly stiff
comptime DEFAULT_DAMPRATIO: Float64 = 1.0  # Critical damping

# Solver impedance parameters (MuJoCo solimp)
comptime DEFAULT_IMPEDANCE: Float64 = 0.9  # High impedance = soft contact
comptime MIN_IMPEDANCE: Float64 = 0.001
comptime MAX_IMPEDANCE: Float64 = 0.999


# =============================================================================
# Constraint Computation Functions
# =============================================================================


fn compute_spring_damper_params[
    DTYPE: DType
](
    timeconst: Scalar[DTYPE],
    dampratio: Scalar[DTYPE],
    dt: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute spring (k) and damper (b) coefficients from MuJoCo solref.

    MuJoCo formula:
      k = 1 / (timeconst^2 * dampratio^2)
      b = 2 / timeconst

    Also clamp timeconst to at least 2*dt for stability.
    """
    # Clamp timeconst for stability (MuJoCo's refsafe)
    var tc = max(timeconst, Scalar[DTYPE](2) * dt)
    var dr = max(dampratio, Scalar[DTYPE](0.01))

    var k = Scalar[DTYPE](1) / (tc * tc * dr * dr)
    var b = Scalar[DTYPE](2) / tc

    return (k, b)


fn compute_effective_mass[
    DTYPE: DType
](
    inv_mass_a: Scalar[DTYPE],
    inv_mass_b: Scalar[DTYPE],
    impedance: Scalar[DTYPE] = 0.9,
) -> Scalar[DTYPE]:
    """Compute effective mass for constraint (MuJoCo-style with impedance).

    MuJoCo formula:
      invweight = 1/m_a + 1/m_b (for point masses)
      D = 1 / max(invweight * (1-imp)/imp, minval)

    High impedance (close to 1) = softer contact, lower D
    Low impedance (close to 0) = stiffer contact, higher D
    """
    var invweight = inv_mass_a + inv_mass_b
    if invweight < Scalar[DTYPE](1e-10):
        return Scalar[DTYPE](0)

    # Clamp impedance
    var imp = max(
        min(impedance, Scalar[DTYPE](MAX_IMPEDANCE)),
        Scalar[DTYPE](MIN_IMPEDANCE),
    )

    # MuJoCo effective mass formula
    var D_inv = invweight * (Scalar[DTYPE](1) - imp) / imp
    D_inv = max(D_inv, Scalar[DTYPE](1e-10))

    return Scalar[DTYPE](1) / D_inv


fn compute_reference_acceleration[
    DTYPE: DType
](
    penetration: Scalar[DTYPE],  # Positive when penetrating
    velocity: Scalar[DTYPE],  # Positive when approaching
    k: Scalar[DTYPE],  # Spring coefficient
    b: Scalar[DTYPE],  # Damper coefficient
    restitution: Scalar[DTYPE],
    dt: Scalar[DTYPE],  # Timestep for velocity->acceleration conversion
) -> Scalar[DTYPE]:
    """Compute reference acceleration (MuJoCo-style with restitution fix).

    MuJoCo formula: aref = -k * pos - b * vel

    For contacts:
    - pos = -penetration (negative distance)
    - vel = constraint velocity (positive = approaching)

    So: aref = k * penetration - b * velocity

    For high-velocity impacts with restitution, we compute the acceleration
    needed to achieve the target bounce velocity.
    """
    var aref = Scalar[DTYPE](0)

    # Spring term: push out of penetration
    if penetration > Scalar[DTYPE](0):
        aref += k * penetration

    # Handle approaching velocity
    if velocity > Scalar[DTYPE](0):
        # For high velocity impacts, compute acceleration to achieve bounce
        var vel_threshold = Scalar[DTYPE](0.3)
        if velocity > vel_threshold and restitution > Scalar[DTYPE](0):
            # Target velocity change: from +v to -e*v
            # delta_v = -(1+e)*v
            # Required acceleration: a = delta_v / dt
            var target_delta_v = -(Scalar[DTYPE](1) + restitution) * velocity
            aref += -target_delta_v / dt  # Positive acceleration pushes back
        else:
            # Soft damping for low velocity (prevents jitter)
            aref += b * velocity

    return aref


fn compute_constraint_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
](
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    body_a: Int,
    body_b: Int,
    normal_x: Scalar[DTYPE],
    normal_y: Scalar[DTYPE],
    normal_z: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Compute relative velocity in constraint direction (J @ qvel).

    Convention: positive = approaching, negative = separating
    """
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

    return (
        (va_x - vb_x) * normal_x
        + (va_y - vb_y) * normal_y
        + (va_z - vb_z) * normal_z
    )


# =============================================================================
# PGS Solver (MuJoCo-style)
# =============================================================================


fn solve_constraints_pgs[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    dt: Scalar[DTYPE],
    iterations: Int = 20,
):
    """Projected Gauss-Seidel constraint solver (MuJoCo-style).

    Algorithm:
    1. Compute spring-damper parameters from solref
    2. For each contact, compute effective mass D and reference acceleration aref
    3. Iteratively solve for constraint forces (lambda) using PGS:
       - Compute delta_lambda to match aref
       - Clamp accumulated lambda >= 0
       - Apply velocity changes
    """
    if data.num_contacts == 0:
        return

    # Solver parameters
    var timeconst = Scalar[DTYPE](DEFAULT_TIMECONST)
    var dampratio = Scalar[DTYPE](DEFAULT_DAMPRATIO)
    var impedance = Scalar[DTYPE](DEFAULT_IMPEDANCE)

    # Compute spring-damper coefficients
    var params = compute_spring_damper_params(timeconst, dampratio, dt)
    var k = params[0]
    var b = params[1]

    # Pre-compute constraint data
    var D = InlineArray[Scalar[DTYPE], MAX_CONTACTS](uninitialized=True)
    var lambda_n = InlineArray[Scalar[DTYPE], MAX_CONTACTS](uninitialized=True)
    var inv_mass_a_arr = InlineArray[Scalar[DTYPE], MAX_CONTACTS](
        uninitialized=True
    )
    var inv_mass_b_arr = InlineArray[Scalar[DTYPE], MAX_CONTACTS](
        uninitialized=True
    )
    var had_impact = InlineArray[Bool, MAX_CONTACTS](uninitialized=True)

    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body_a = contact.body_a
        var body_b = contact.body_b

        var inv_mass_a = model.inv_masses[body_a]
        var inv_mass_b: Scalar[DTYPE]
        if body_b >= 0:
            inv_mass_b = model.inv_masses[body_b]
        else:
            inv_mass_b = Scalar[DTYPE](0)

        # For sphere-sphere contacts, check if bodies are grounded
        # If so, only treat ONE as grounded to avoid K=0 deadlock
        if body_b >= 0:
            var a_grounded = _is_grounded(data, body_a)
            var b_grounded = _is_grounded(data, body_b)
            if a_grounded and not b_grounded:
                inv_mass_a = Scalar[DTYPE](0)
            elif b_grounded and not a_grounded:
                inv_mass_b = Scalar[DTYPE](0)
            # If both grounded, use normal masses

        inv_mass_a_arr[c] = inv_mass_a
        inv_mass_b_arr[c] = inv_mass_b
        D[c] = compute_effective_mass(inv_mass_a, inv_mass_b, impedance)
        lambda_n[c] = Scalar[DTYPE](0)
        had_impact[c] = False

    # FIRST: Resolve ground contacts to get proper grounded body velocities
    # This is critical - ground contacts must be resolved before checking
    # sphere-sphere relative velocities
    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body_a = contact.body_a
        var body_b = contact.body_b

        # Only process ground contacts (body_b == -1)
        if body_b >= 0:
            continue

        var nz = contact.normal_z

        # Get velocity toward ground (negative = approaching ground for nz=+1)
        var va_z = data.velocities[body_a * 3 + 2]
        var vel_n = (
            va_z * nz
        )  # Velocity in normal direction (ground normal is +z)

        if vel_n < Scalar[DTYPE](0):
            # Approaching ground - stop or bounce
            var restitution = model.restitution
            # Low restitution for slow impacts (resting contact)
            if va_z > Scalar[DTYPE](-0.5):
                restitution = Scalar[DTYPE](0)

            # Cancel velocity + add bounce
            var target_vz = -restitution * va_z

            # Compute and store the normal impulse for friction calculation
            # j = m * delta_v = m * (target_vz - va_z)
            # For ground: inv_mass_b = 0, so j = delta_v / inv_mass_a
            var inv_mass_a = inv_mass_a_arr[c]
            if inv_mass_a > Scalar[DTYPE](1e-10):
                var delta_vz = target_vz - va_z
                var j = abs(delta_vz) / inv_mass_a  # Positive impulse magnitude
                lambda_n[c] = j
                had_impact[c] = True

            data.velocities[body_a * 3 + 2] = target_vz

    # SECOND: Handle high-velocity sphere-sphere impacts
    for c in range(data.num_contacts):
        var inv_mass_a = inv_mass_a_arr[c]
        var inv_mass_b = inv_mass_b_arr[c]
        var K = inv_mass_a + inv_mass_b

        if K < Scalar[DTYPE](1e-10):
            continue

        var contact = data.contacts[c]
        var body_a = contact.body_a
        var body_b = contact.body_b

        # Skip ground contacts (already handled)
        if body_b < 0:
            continue

        var nx = contact.normal_x
        var ny = contact.normal_y
        var nz = contact.normal_z

        # Current constraint velocity (positive = approaching)
        var vel = compute_constraint_velocity(data, body_a, body_b, nx, ny, nz)

        # Only handle high-velocity impacts in first pass
        var vel_threshold = Scalar[DTYPE](0.5)
        if vel > vel_threshold:
            # Impulse for velocity reversal: j = -(1+e) * vel * m_eff
            # where m_eff = 1/K
            var j = (Scalar[DTYPE](1) + model.restitution) * vel / K

            # Apply impulse directly (not accumulated)
            data.velocities[body_a * 3 + 0] -= j * nx * inv_mass_a
            data.velocities[body_a * 3 + 1] -= j * ny * inv_mass_a
            data.velocities[body_a * 3 + 2] -= j * nz * inv_mass_a

            if body_b >= 0:
                data.velocities[body_b * 3 + 0] += j * nx * inv_mass_b
                data.velocities[body_b * 3 + 1] += j * ny * inv_mass_b
                data.velocities[body_b * 3 + 2] += j * nz * inv_mass_b

            # Mark this contact as handled for impact
            lambda_n[c] = j  # Store for warm-start
            had_impact[c] = True

    # Initialize friction impulse arrays
    var lambda_t1 = InlineArray[Scalar[DTYPE], MAX_CONTACTS](uninitialized=True)
    var lambda_t2 = InlineArray[Scalar[DTYPE], MAX_CONTACTS](uninitialized=True)
    for c in range(data.num_contacts):
        lambda_t1[c] = Scalar[DTYPE](0)
        lambda_t2[c] = Scalar[DTYPE](0)

    # PGS iterations for soft constraints (resting/slow contacts)
    for iteration in range(iterations):
        for c in range(data.num_contacts):
            # Skip contacts that had impact impulse for NORMAL constraints only
            # (they already got their bounce impulse)
            if had_impact[c]:
                continue

            var inv_mass_a = inv_mass_a_arr[c]
            var inv_mass_b = inv_mass_b_arr[c]
            var K = inv_mass_a + inv_mass_b

            if K < Scalar[DTYPE](1e-10):
                continue

            var contact = data.contacts[c]
            var body_a = contact.body_a
            var body_b = contact.body_b
            var nx = contact.normal_x
            var ny = contact.normal_y
            var nz = contact.normal_z

            # Current constraint velocity
            # Convention differs for ground vs sphere-sphere:
            # - Ground (body_b < 0): vel > 0 means SEPARATING (moving up)
            # - Sphere-sphere: vel > 0 means APPROACHING
            var vel = compute_constraint_velocity(
                data, body_a, body_b, nx, ny, nz
            )

            # Penetration (positive when penetrating)
            var penetration = -contact.dist

            # For ground contacts: flip the velocity sign to match sphere-sphere convention
            # (after flip: vel > 0 means approaching ground = moving down)
            var approach_vel = vel
            if body_b < 0:
                approach_vel = -vel  # Flip for ground contacts

            # Skip if separating and not penetrating
            if approach_vel <= Scalar[DTYPE](0) and penetration <= Scalar[
                DTYPE
            ](0):
                continue

            # Soft constraint: spring-damper (MuJoCo style)
            # Only apply damping when approaching (approach_vel > 0)
            var aref = Scalar[DTYPE](0)
            if penetration > Scalar[DTYPE](0):
                aref += k * penetration
            if approach_vel > Scalar[DTYPE](0):
                aref += b * approach_vel

            # Convert to impulse
            var delta_j = aref * dt / K

            # Accumulate and clamp (unilateral constraint)
            var old_lambda = lambda_n[c]
            var new_lambda = max(old_lambda + delta_j, Scalar[DTYPE](0))
            delta_j = new_lambda - old_lambda
            lambda_n[c] = new_lambda

            # Apply velocity change
            # For ground contacts: push sphere UP in +normal direction
            # For sphere-sphere: push body_a in -normal, body_b in +normal
            if body_b < 0:
                # Ground contact: push sphere up (+normal)
                data.velocities[body_a * 3 + 0] += delta_j * nx * inv_mass_a
                data.velocities[body_a * 3 + 1] += delta_j * ny * inv_mass_a
                data.velocities[body_a * 3 + 2] += delta_j * nz * inv_mass_a
            else:
                # Sphere-sphere: body_a moves in -normal, body_b in +normal
                data.velocities[body_a * 3 + 0] -= delta_j * nx * inv_mass_a
                data.velocities[body_a * 3 + 1] -= delta_j * ny * inv_mass_a
                data.velocities[body_a * 3 + 2] -= delta_j * nz * inv_mass_a

                data.velocities[body_b * 3 + 0] += delta_j * nx * inv_mass_b
                data.velocities[body_b * 3 + 1] += delta_j * ny * inv_mass_b
                data.velocities[body_b * 3 + 2] += delta_j * nz * inv_mass_b

        # =========================================================
        # FRICTION SOLVING (Coulomb friction) - separate loop
        # Applies to ALL contacts with positive normal impulse,
        # including impact contacts (they still have tangential velocity)
        # =========================================================
        for c in range(data.num_contacts):
            var jn = lambda_n[c]
            if jn <= Scalar[DTYPE](0) or model.friction <= Scalar[DTYPE](0):
                continue

            var inv_mass_a = inv_mass_a_arr[c]
            var inv_mass_b = inv_mass_b_arr[c]
            var K = inv_mass_a + inv_mass_b

            if K < Scalar[DTYPE](1e-10):
                continue

            var contact = data.contacts[c]
            var body_a = contact.body_a
            var body_b = contact.body_b
            var nx = contact.normal_x
            var ny = contact.normal_y
            var nz = contact.normal_z

            # Compute tangent basis from normal
            var tangents = compute_tangent_basis(nx, ny, nz)
            var t1x = tangents[0]
            var t1y = tangents[1]
            var t1z = tangents[2]
            var t2x = tangents[3]
            var t2y = tangents[4]
            var t2z = tangents[5]

            # Get current velocities
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

            # Compute relative tangent velocities
            var rel_vt1 = (
                (va_x - vb_x) * t1x + (va_y - vb_y) * t1y + (va_z - vb_z) * t1z
            )
            var rel_vt2 = (
                (va_x - vb_x) * t2x + (va_y - vb_y) * t2y + (va_z - vb_z) * t2z
            )

            # Compute tangent impulses to stop sliding
            var delta_jt1 = rel_vt1 / K
            var delta_jt2 = rel_vt2 / K

            # Accumulate tangent impulses
            var old_jt1 = lambda_t1[c]
            var old_jt2 = lambda_t2[c]
            var new_jt1 = old_jt1 + delta_jt1
            var new_jt2 = old_jt2 + delta_jt2

            # Clamp to friction cone: |j_t| <= μ * j_n
            var jt_mag = sqrt(new_jt1 * new_jt1 + new_jt2 * new_jt2)
            var max_friction = model.friction * jn

            if jt_mag > max_friction and jt_mag > Scalar[DTYPE](1e-10):
                var scale = max_friction / jt_mag
                new_jt1 = new_jt1 * scale
                new_jt2 = new_jt2 * scale

            # Compute actual delta after clamping
            delta_jt1 = new_jt1 - old_jt1
            delta_jt2 = new_jt2 - old_jt2
            lambda_t1[c] = new_jt1
            lambda_t2[c] = new_jt2

            # Apply tangent impulses to velocities
            if body_b < 0:
                # Ground contact
                data.velocities[body_a * 3 + 0] -= (
                    delta_jt1 * t1x + delta_jt2 * t2x
                ) * inv_mass_a
                data.velocities[body_a * 3 + 1] -= (
                    delta_jt1 * t1y + delta_jt2 * t2y
                ) * inv_mass_a
                data.velocities[body_a * 3 + 2] -= (
                    delta_jt1 * t1z + delta_jt2 * t2z
                ) * inv_mass_a
            else:
                # Sphere-sphere
                data.velocities[body_a * 3 + 0] -= (
                    delta_jt1 * t1x + delta_jt2 * t2x
                ) * inv_mass_a
                data.velocities[body_a * 3 + 1] -= (
                    delta_jt1 * t1y + delta_jt2 * t2y
                ) * inv_mass_a
                data.velocities[body_a * 3 + 2] -= (
                    delta_jt1 * t1z + delta_jt2 * t2z
                ) * inv_mass_a

                data.velocities[body_b * 3 + 0] += (
                    delta_jt1 * t1x + delta_jt2 * t2x
                ) * inv_mass_b
                data.velocities[body_b * 3 + 1] += (
                    delta_jt1 * t1y + delta_jt2 * t2y
                ) * inv_mass_b
                data.velocities[body_b * 3 + 2] += (
                    delta_jt1 * t1z + delta_jt2 * t2z
                ) * inv_mass_b

    # Store final impulses in contact data for warm starting
    for c in range(data.num_contacts):
        data.contacts[c].impulse_n = lambda_n[c]
        data.contacts[c].impulse_t1 = lambda_t1[c]
        data.contacts[c].impulse_t2 = lambda_t2[c]


fn _is_grounded[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
](
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    body_idx: Int,
) -> Bool:
    """Check if a body has ground contact."""
    for c in range(data.num_contacts):
        if (
            data.contacts[c].body_a == body_idx
            and data.contacts[c].body_b == -1
        ):
            return True
    return False


# =============================================================================
# Position Correction (Baumgarte)
# =============================================================================


fn correct_positions[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    baumgarte: Scalar[DTYPE] = 0.8,
    slop: Scalar[DTYPE] = 0.0001,
):
    """Direct position correction for remaining penetration.

    Applied after velocity solving to clean up residual penetration.
    Uses Baumgarte stabilization with configurable correction factor.
    """
    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body_a = contact.body_a
        var body_b = contact.body_b
        var dist = contact.dist

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

        var correction = baumgarte * penetration
        var ratio_a = inv_mass_a / total_inv_mass
        var ratio_b = inv_mass_b / total_inv_mass

        if body_b < 0:
            # Ground contact: push sphere up
            data.positions[body_a * 3 + 0] += correction * nx
            data.positions[body_a * 3 + 1] += correction * ny
            data.positions[body_a * 3 + 2] += correction * nz
        else:
            # Sphere-sphere: push apart by mass ratio
            data.positions[body_a * 3 + 0] -= correction * nx * ratio_a
            data.positions[body_a * 3 + 1] -= correction * ny * ratio_a
            data.positions[body_a * 3 + 2] -= correction * nz * ratio_a

            data.positions[body_b * 3 + 0] += correction * nx * ratio_b
            data.positions[body_b * 3 + 1] += correction * ny * ratio_b
            data.positions[body_b * 3 + 2] += correction * nz * ratio_b


# =========================================================================
# PGS Solver GPU (MuJoCo-style)
# =========================================================================


@always_inline
fn solve_constraints_pgs_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
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
    dt: Scalar[DTYPE],
    restitution: Scalar[DTYPE],
    friction: Scalar[DTYPE],
    iterations: Int,
):
    """PGS constraint solver with spring-damper model and friction."""
    var meta_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    var num_contacts = Int(
        rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
    )

    # Spring-damper parameters (MuJoCo defaults)
    var stiffness: Scalar[DTYPE] = 2000.0
    var damping: Scalar[DTYPE] = 100.0

    # Reset impulses at start
    for c in range(num_contacts):
        var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](c)
        state[env, c_off + CONTACT_IDX_IMPULSE_N] = Scalar[DTYPE](0)
        state[env, c_off + CONTACT_IDX_IMPULSE_T1] = Scalar[DTYPE](0)
        state[env, c_off + CONTACT_IDX_IMPULSE_T2] = Scalar[DTYPE](0)

    for _ in range(iterations):
        for c in range(num_contacts):
            var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](c)
            var body_a = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A])
            )
            var body_b = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B])
            )
            var dist = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_DIST]
            )

            if dist >= Scalar[DTYPE](0):
                continue

            var nx = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NX])
            var ny = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NY])
            var nz = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NZ])

            # Get velocities
            var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_a)
            var vx_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VX])
            var vy_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VY])
            var vz_a = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VZ])

            var vx_b: Scalar[DTYPE] = 0
            var vy_b: Scalar[DTYPE] = 0
            var vz_b: Scalar[DTYPE] = 0
            var b_off_b: Int = 0
            if body_b >= 0:
                b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_b)
                vx_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VX])
                vy_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VY])
                vz_b = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VZ])

            var rel_vx = vx_a - vx_b
            var rel_vy = vy_a - vy_b
            var rel_vz = vz_a - vz_b
            var rel_vn = rel_vx * nx + rel_vy * ny + rel_vz * nz

            # Spring-damper constraint force
            var penetration = -dist
            var bias = stiffness * penetration - damping * rel_vn

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

            # Normal constraint impulse
            var old_impulse = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )
            var delta_impulse = (bias * dt) / K
            var new_impulse = max(old_impulse + delta_impulse, Scalar[DTYPE](0))
            delta_impulse = new_impulse - old_impulse
            state[env, c_off + CONTACT_IDX_IMPULSE_N] = new_impulse

            # Apply normal impulse
            vx_a = vx_a + delta_impulse * nx * inv_mass_a
            vy_a = vy_a + delta_impulse * ny * inv_mass_a
            vz_a = vz_a + delta_impulse * nz * inv_mass_a
            state[env, b_off_a + BODY_IDX_VX] = vx_a
            state[env, b_off_a + BODY_IDX_VY] = vy_a
            state[env, b_off_a + BODY_IDX_VZ] = vz_a

            if body_b >= 0:
                vx_b = vx_b - delta_impulse * nx * inv_mass_b
                vy_b = vy_b - delta_impulse * ny * inv_mass_b
                vz_b = vz_b - delta_impulse * nz * inv_mass_b
                state[env, b_off_b + BODY_IDX_VX] = vx_b
                state[env, b_off_b + BODY_IDX_VY] = vy_b
                state[env, b_off_b + BODY_IDX_VZ] = vz_b

            # =========================================================
            # FRICTION SOLVING
            # =========================================================
            var jn = new_impulse
            if jn <= Scalar[DTYPE](0) or friction <= Scalar[DTYPE](0):
                continue

            # Compute tangent basis (inline for GPU)
            var abs_nx = abs(nx)
            var abs_ny = abs(ny)
            var abs_nz = abs(nz)
            var ax: Scalar[DTYPE]
            var ay: Scalar[DTYPE]
            var az: Scalar[DTYPE]

            if abs_nx < abs_ny and abs_nx < abs_nz:
                ax = Scalar[DTYPE](1.0)
                ay = Scalar[DTYPE](0.0)
                az = Scalar[DTYPE](0.0)
            elif abs_ny < abs_nz:
                ax = Scalar[DTYPE](0.0)
                ay = Scalar[DTYPE](1.0)
                az = Scalar[DTYPE](0.0)
            else:
                ax = Scalar[DTYPE](0.0)
                ay = Scalar[DTYPE](0.0)
                az = Scalar[DTYPE](1.0)

            # t1 = normalize(a - (a·n)*n)
            var dot = ax * nx + ay * ny + az * nz
            var t1x = ax - dot * nx
            var t1y = ay - dot * ny
            var t1z = az - dot * nz
            var t1_len = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
            if t1_len > Scalar[DTYPE](1e-10):
                t1x = t1x / t1_len
                t1y = t1y / t1_len
                t1z = t1z / t1_len

            # t2 = n × t1
            var t2x = ny * t1z - nz * t1y
            var t2y = nz * t1x - nx * t1z
            var t2z = nx * t1y - ny * t1x

            # Compute relative tangent velocities
            rel_vx = vx_a - vx_b
            rel_vy = vy_a - vy_b
            rel_vz = vz_a - vz_b
            var rel_vt1 = rel_vx * t1x + rel_vy * t1y + rel_vz * t1z
            var rel_vt2 = rel_vx * t2x + rel_vy * t2y + rel_vz * t2z

            # Compute tangent impulses
            var delta_jt1 = rel_vt1 / K
            var delta_jt2 = rel_vt2 / K

            # Accumulate and clamp to friction cone
            var old_jt1 = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_T1]
            )
            var old_jt2 = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_T2]
            )
            var new_jt1 = old_jt1 + delta_jt1
            var new_jt2 = old_jt2 + delta_jt2

            var jt_mag = sqrt(new_jt1 * new_jt1 + new_jt2 * new_jt2)
            var max_friction = friction * jn

            if jt_mag > max_friction and jt_mag > Scalar[DTYPE](1e-10):
                var scale = max_friction / jt_mag
                new_jt1 = new_jt1 * scale
                new_jt2 = new_jt2 * scale

            delta_jt1 = new_jt1 - old_jt1
            delta_jt2 = new_jt2 - old_jt2
            state[env, c_off + CONTACT_IDX_IMPULSE_T1] = new_jt1
            state[env, c_off + CONTACT_IDX_IMPULSE_T2] = new_jt2

            # Apply tangent impulses
            var dx = delta_jt1 * t1x + delta_jt2 * t2x
            var dy = delta_jt1 * t1y + delta_jt2 * t2y
            var dz = delta_jt1 * t1z + delta_jt2 * t2z

            state[env, b_off_a + BODY_IDX_VX] = vx_a - dx * inv_mass_a
            state[env, b_off_a + BODY_IDX_VY] = vy_a - dy * inv_mass_a
            state[env, b_off_a + BODY_IDX_VZ] = vz_a - dz * inv_mass_a

            if body_b >= 0:
                state[env, b_off_b + BODY_IDX_VX] = vx_b + dx * inv_mass_b
                state[env, b_off_b + BODY_IDX_VY] = vy_b + dy * inv_mass_b
                state[env, b_off_b + BODY_IDX_VZ] = vz_b + dz * inv_mass_b
